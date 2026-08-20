#!/usr/bin/env python3
"""
00_prepare_baselines.py — train the seven CIFAR-100 baselines of the single-TPU axis.

WHAT THIS PRODUCES
    For each requested architecture, two files in --output_dir:
        <name>.pt          the whole model (architecture + weights), pickled
        <name>.train.json  the recipe used, the best epoch, and the full
                           per-epoch accuracy curve

    These baselines are the reference point of everything downstream: every
    pruned variant is compared against the accuracy recorded here, so they must
    exist before 01_prune.py can run.

WHY THE MODELS ARE PICKLED WHOLE RATHER THAN AS STATE DICTS
    Structured pruning physically removes channels, so a pruned model has a
    different architecture from the one that produced it, and a state dict alone
    could not rebuild it. The whole-model pickle keeps architecture and weights
    together. The cost is that unpickling needs the defining modules importable
    under their original names (cifar_resnet, cifar_vgg, wrn), which is why those
    files sit next to this one and must not be moved into a subpackage.

TRAINING RECIPE
    Uniform across all seven architectures, taken from PruningBench (Li et al.,
    2024, arXiv:2406.12315; reference implementation in VainF/Torch-Pruning under
    reproduce/):

        optimiser     SGD, momentum 0.9, weight decay 5e-4
        initial LR    0.1
        schedule      MultiStepLR, milestones [120, 150, 180], gamma 0.1
        epochs        200
        batch size    128
        no warmup, no head/backbone parameter group split

    Augmentation is the CIFAR standard: RandomCrop(32, padding=4), random
    horizontal flip, normalisation by CIFAR-100 channel statistics.

WHY THE RECIPE IS UNIFORM AND WHY TRAINING IS FROM SCRATCH
    Uniformity is deliberate rather than convenient. The object of study is how a
    pruning criterion behaves across architectures, so per-architecture tuning
    would confound the comparison and would also break comparability with the
    PruningBench leaderboard. Training starts from random init for the same
    reason: ImageNet weights exist for three of the seven (mobilenetv2,
    googlenet, squeezenet1_1) but not for the CIFAR-native four, and an LR of 0.1
    would wash those weights out within a few epochs anyway.

CIFAR ADAPTATIONS OF THE IMAGENET ARCHITECTURES
    The four CIFAR-native architectures (resnet18, resnet50, vgg19, wrn_28_10)
    come from the sibling modules in this directory. The three torchvision ones
    are ImageNet designs, whose stems downsample far too aggressively for a 32x32
    input, so each has its stem stride relaxed to 1 and its classifier resized to
    100 classes. See build_model() for the per-architecture details.

REPRODUCIBILITY
    The seed is propagated to random, numpy and torch (CPU and CUDA), and the
    DataLoader shuffling is seeded from it too. All published results use seed 42.

USAGE (pytorch-env)
    # full recipe, all seven models
    python 00_prepare_baselines.py --output_dir ../models --data_dir /path/to/cifar100

    # a subset, or a shortened run for debugging
    python 00_prepare_baselines.py --models resnet18 --epochs 2
    python 00_prepare_baselines.py --force            # retrain even if .pt exists

    Options: --models, --epochs, --batch_size, --seed (default 42),
             --device cuda|cpu, --force
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# The architecture modules must be importable under their bare names for the
# whole-model pickles to load. Accept both a flat layout (modules next to this
# script) and a nested one (modules one level up).
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for _p in (_HERE, _PARENT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cifar_resnet import resnet18 as cifar_resnet18
from cifar_resnet import resnet50 as cifar_resnet50
from cifar_vgg import vgg19 as cifar_vgg19
from wrn import wrn_28_10 as cifar_wrn_28_10


CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)


# ─────────────────────────────────────────────
# PruningBench recipe, identical for all seven architectures
# ─────────────────────────────────────────────
RECIPE = {
    "optimizer": "sgd", "momentum": 0.9, "weight_decay": 5e-4,
    "lr": 0.1,
    "scheduler": "multistep",
    "lr_decay_epochs": [120, 150, 180], "lr_decay_rate": 0.1,
    "epochs": 200,
    "batch_size": 128,
    "ref": "PruningBench (Li et al. 2024, arXiv:2406.12315) — recette uniforme via VainF/Torch-Pruning/reproduce/main.py",
}

ALL_MODEL_NAMES = ["resnet18", "resnet50", "vgg19", "wrn_28_10",
                   "mobilenetv2", "googlenet", "squeezenet1_1"]


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def seed_everything(seed: int):
    """Seed random, numpy and torch (CPU and CUDA) from a single integer.

    Called once per model rather than once per process, so that training model N
    does not depend on how many models were trained before it in the same run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_workers():
    """Pick a DataLoader worker count from the visible CPU budget.

    Honours the SLURM allocation when present, since os.cpu_count() reports the
    whole node and would oversubscribe a job that was given only part of it.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return 4


# ─────────────────────────────────────────────
# Data CIFAR-100
# ─────────────────────────────────────────────
def get_dataloaders(data_dir, batch_size, seed):
    """Build the CIFAR-100 train and test loaders with standard augmentation.

    Train gets RandomCrop(32, padding=4) and a horizontal flip; test gets neither.
    Both are normalised by CIFAR-100 channel statistics. The generator is seeded so
    that shuffling is reproducible across runs.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_ds = datasets.CIFAR100(root=str(data_dir), train=True,
                                 download=False, transform=train_transform)
    val_ds = datasets.CIFAR100(root=str(data_dir), train=False,
                               download=False, transform=val_transform)
    nw = get_num_workers()
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=(nw > 0), generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True,
                            persistent_workers=(nw > 0))
    return train_loader, val_loader


# ─────────────────────────────────────────────
# Architecture construction (random init for all seven)
# ─────────────────────────────────────────────
def build_model(name, num_classes=100):
    """Instantiate one architecture by name, adapted to CIFAR-100 at 32x32.

    The four CIFAR-native architectures come straight from the sibling modules. The
    three torchvision ones are ImageNet designs and need their stem stride relaxed
    from 2 to 1, because at 32x32 the original stem would have thrown away most of
    the spatial resolution before the first block. googlenet additionally loses its
    auxiliary classifiers, which would otherwise complicate pruning downstream, and
    its initial max-pool. All classifiers are resized to `num_classes`.

    Weights are always random: see the module docstring on why no ImageNet init.
    """
    if name == "resnet18":
        return cifar_resnet18(num_classes=num_classes)
    if name == "resnet50":
        return cifar_resnet50(num_classes=num_classes)
    if name == "vgg19":
        return cifar_vgg19(num_classes=num_classes)
    if name == "wrn_28_10":
        return cifar_wrn_28_10(num_classes=num_classes)
    if name == "mobilenetv2":
        import torchvision.models as tv
        # Random init, PruningBench style: no ImageNet weights.
        m = tv.mobilenet_v2(weights=None)
        m.features[0][0].stride = (1, 1)  # CIFAR: keep 32x32 through the stem
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if name == "mnasnet1_0":
        # Kept in the dispatch so the archived .pt can still be reloaded, but
        # dropped from ALL_MODEL_NAMES: it was redundant with mobilenetv2.
        import torchvision.models as tv
        m = tv.mnasnet1_0(weights=None)
        m.layers[0].stride = (1, 1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if name == "googlenet":
        import torchvision.models as tv
        # aux_logits=False: the auxiliary classifiers would add branches that
        # complicate structured pruning downstream for no benefit here.
        # CIFAR adaptation: stem stride 2 -> 1 and the initial max-pool removed,
        # so a 32x32 input is not reduced to 8x8 before the first block.
        m = tv.googlenet(weights=None, aux_logits=False, init_weights=True)
        m.conv1.conv.stride = (1, 1)
        m.maxpool1 = nn.Identity()
        m.fc = nn.Linear(1024, num_classes)
        return m
    if name == "squeezenet1_1":
        import torchvision.models as tv
        m = tv.squeezenet1_1(weights=None)
        m.features[0].stride = (1, 1)
        m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        m.num_classes = num_classes
        return m
    raise ValueError(f"Unknown model: {name}")


# ─────────────────────────────────────────────
# Optimiser and scheduler (single PruningBench recipe)
# ─────────────────────────────────────────────
def build_optimizer(model, recipe):
    """SGD with the momentum and weight decay fixed by the PruningBench recipe."""
    return optim.SGD(model.parameters(), lr=recipe["lr"],
                     momentum=recipe["momentum"], weight_decay=recipe["weight_decay"])


def build_scheduler(optimizer, recipe, total_epochs):
    """MultiStepLR, keeping only the recipe milestones that fall inside the run.

    On the full 200-epoch recipe this is the identity. It matters when --epochs
    shortens the run: milestones of [120, 150, 180] would never fire in a 2-epoch
    debugging run, leaving the learning rate at 0.1 throughout. When no milestone
    survives the filter, a single one is placed near the end so the run still ends
    with a decay rather than at the peak rate.

    Note that the console banner echoes the recipe's own milestones, not these.
    """
    # Keep only the milestones that fall within the (possibly shortened) run
    ms = [m for m in recipe["lr_decay_epochs"] if m < total_epochs]
    if not ms:
        ms = [max(1, total_epochs - 1)]
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms,
                                          gamma=recipe["lr_decay_rate"])


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def evaluate(model, loader, device):
    """Return top-1 accuracy in percent over a loader, in eval mode and no-grad."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def train_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch; return mean loss and top-1 accuracy in percent."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total


def train_loop(model, train_loader, val_loader, device, optimizer, scheduler,
               total_epochs, name):
    """Train for `total_epochs`, keeping the weights of the best validation epoch.

    Returns (best_acc, best_epoch, history). Selection is on validation accuracy
    rather than final-epoch accuracy, which matters because the recipe has no early
    stopping and the last epochs after the final LR drop can drift slightly down.
    """
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state, best_epoch = 0.0, None, -1
    history = []

    for epoch in range(total_epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        improved = val_acc > best_acc
        if improved:
            best_acc, best_epoch = val_acc, epoch + 1
            # Preserve both the mapping type (OrderedDict) and its _metadata
            # attribute. Dropping _metadata makes MNASNet's _load_from_state_dict
            # fail later with "version should be 1 or 2": it reads the version
            # tag from there to decide how to interpret the checkpoint.
            sd = model.state_dict()
            best_state = type(sd)((k, v.cpu().clone()) for k, v in sd.items())
            if hasattr(sd, "_metadata"):
                best_state._metadata = sd._metadata

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "train_acc": train_acc, "val_acc": val_acc,
                        "lr": optimizer.param_groups[0]["lr"], "duration_s": dt})

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == total_epochs - 1 or improved:
            star = " *" if improved else ""
            print(f"    [{name}] Ep {epoch+1:3d}/{total_epochs}  loss={train_loss:.4f}  "
                  f"train={train_acc:5.2f}%  val={val_acc:5.2f}%  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  ({dt:.0f}s){star}",
                  flush=True)

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    return best_acc, best_epoch, history


# ─────────────────────────────────────────────
# Per-model pipeline
# ─────────────────────────────────────────────
def prepare_one(name, args, device, output_dir):
    """Train one architecture end to end and write its .pt and .train.json.

    Skips the model if the .pt already exists and --force was not given, which makes
    the script safe to re-run after an interruption or to extend to a new
    architecture without retraining the others.
    """
    dest = output_dir / f"{name}.pt"
    log_path = output_dir / f"{name}.train.json"
    if dest.exists() and not args.force:
        print(f"  [{name}] Already present ({dest}), skipping.")
        return None

    recipe = copy.deepcopy(RECIPE)
    epochs = args.epochs if args.epochs is not None else recipe["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else recipe["batch_size"]
    if args.epochs is not None:
        recipe["epochs_override"] = args.epochs
    if args.batch_size is not None:
        recipe["batch_size_override"] = args.batch_size

    print(f"\n{'━' * 70}")
    print(f"  {name.upper()}")
    print(f"  Recipe : {recipe['ref']}")
    print(f"{'━' * 70}", flush=True)

    seed_everything(args.seed)
    model = build_model(name, num_classes=100).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{name}] params={n_params:,}, batch_size={batch_size}, epochs={epochs}", flush=True)

    train_loader, val_loader = get_dataloaders(args.data_dir, batch_size, args.seed)
    optimizer = build_optimizer(model, recipe)
    scheduler = build_scheduler(optimizer, recipe, epochs)

    print(f"  [{name}] SGD lr={recipe['lr']} m={recipe['momentum']} wd={recipe['weight_decay']}, "
          f"MultiStep {recipe['lr_decay_epochs']} γ={recipe['lr_decay_rate']}", flush=True)

    t0 = time.time()
    best_acc, best_epoch, history = train_loop(
        model, train_loader, val_loader, device, optimizer, scheduler, epochs, name)
    dt_min = (time.time() - t0) / 60

    torch.save(model, str(dest))
    log = {
        "model": name, "recipe": recipe, "params": n_params,
        "best_acc": best_acc, "best_epoch": best_epoch,
        "total_epochs_run": len(history), "duration_min": dt_min,
        "seed": args.seed, "device": str(device),
        "history": history,
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"  [{name}] Best val : {best_acc:.2f}% @ ep {best_epoch}/{len(history)}  "
          f"→ {dest} ({dest.stat().st_size/1024/1024:.1f} MB, {dt_min:.1f} min)", flush=True)
    print(f"  [{name}] Log : {log_path}", flush=True)
    return best_acc


def main():
    """Parse arguments, resolve the device, then train each requested model in turn."""
    parser = argparse.ArgumentParser(
        description="Train the CIFAR-100 baselines (PruningBench recipe)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Dossier de sortie (ex: ../models)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dossier contenant le sous-dossier `cifar-100-python/` "
                             "(format pickle binaire d'origine de Krizhevsky). "
                             "Download it manually; see the README.")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=ALL_MODEL_NAMES,
                        help=f"Subset of models (default: all of {ALL_MODEL_NAMES})")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the epoch count (default 200, per the PruningBench recipe)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override the batch size (default 128)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Retrain even when the .pt already exists")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_list = args.models if args.models else ALL_MODEL_NAMES

    print("=" * 70)
    print("TRAINING CIFAR-100 BASELINES (PruningBench recipe)")
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Output dir   : {output_dir.resolve()}")
    print(f"  Data dir     : {Path(args.data_dir).resolve()}")
    print(f"  Models       : {model_list}")
    print(f"  Seed         : {args.seed}")
    print(f"  Workers      : {get_num_workers()}")
    if args.epochs:
        print(f"  ⚠ Override epochs={args.epochs}")
    if args.batch_size:
        print(f"  ⚠ Override batch_size={args.batch_size}")
    print("=" * 70, flush=True)

    results = {}
    t_start = time.time()
    for name in model_list:
        try:
            results[name] = prepare_one(name, args, device, output_dir)
        except Exception as e:
            print(f"\n  [{name}] [FAILED] {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            results[name] = None
    total_min = (time.time() - t_start) / 60

    print(f"\n{'=' * 70}")
    print(f"SUMMARY ({total_min:.1f} min total)")
    print(f"{'=' * 70}")
    for name, acc in results.items():
        status = f"{acc:.2f}%" if acc is not None else "FAILED"
        print(f"  {name:18s} {status}")

    print(f"\nFichiers dans {output_dir}/ :")
    for f in sorted(output_dir.iterdir()):
        if f.suffix in (".pt", ".json"):
            print(f"  {f.name:30s} {f.stat().st_size/1024/1024:8.2f} MB")


if __name__ == "__main__":
    main()

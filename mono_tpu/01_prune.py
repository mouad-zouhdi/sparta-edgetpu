#!/usr/bin/env python3
"""
01_prune.py — structured pruning of the CIFAR-100 baselines (single-TPU axis).

WHAT THIS PRODUCES
    One pruned-and-recovered model per (architecture, criterion, target) triple:
        pytorch_pruned/<name>_pruned<P>pct_<importance>.pt   whole-model pickle
        pruning_logs/<name>_<importance>_<P>pct.json         per-run log
        pruning_logs/pruning_summary.json                    run-level summary

    Each per-run log records top-1 and top-5 on the CIFAR-100 test set with
    bootstrap 95 % confidence intervals, the achieved parameter reduction (which
    is never exactly the requested one), the full fine-tuning history, and
    `layer_structure_post`: the surviving channel count of every layer. That last
    field is what later makes it possible to explain a latency result by the
    shape of the mask, rather than merely correlating with it.

    The published grid is 7 architectures x 7 criteria x 9 targets = 441 runs, of
    which 404 succeeded; the 37 failures are the expected criterion/architecture
    incompatibilities listed under LIMITATIONS below.

THE PROTOCOL, AND WHY EACH STEP IS THERE
    Runs are independent: every triple starts from a fresh copy of the baseline,
    never from a less-pruned model. Iterative pruning down a schedule would make
    the 30 % model depend on the path taken through 10 % and 20 %, and the
    comparison between criteria would then partly measure that path.

    For each (model, importance, target):

      1. Load a fresh baseline copy. The bn_scale criterion instead loads
         models/<name>_sparse.pt, produced by the sparsity-learning phase below.

      2. Build the pruner: BNScalePruner(reg=1e-5) for bn_scale, MetaPruner
         otherwise, with iterative_steps=400, global_pruning=True and
         max_pruning_ratio=0.9.

         global_pruning=True ranks channels across the whole network rather than
         within each layer, so the criterion is free to allocate sparsity
         unevenly; that allocation is precisely what differs between criteria and
         what changes the memory regime downstream. max_pruning_ratio=0.9 is
         PruningBench's "10 % group-wise protection": no layer may lose more than
         90 % of its channels, which prevents a layer from being pruned to zero
         width and disconnecting the network.

      3. Prune progressively: call pruner.step() in a loop until the target ratio
         is reached, with no fine-tuning in between. With 400 steps each removes
         about 1/400 of the budget, which keeps the importance ranking close to
         valid at every step: a single large step would rank channels using a
         network that no longer exists after the first removals.

      4. For the data-driven criteria (taylor, obdc), run forward and backward
         over 10 batches before each pruner.step(), so the accumulated gradients
         reflect the current network. For obdc the labels are sampled from the
         model's own softmax rather than taken from the dataset, which is what
         makes the accumulated quantity the empirical Fisher information rather
         than a plain gradient-squared sum.

      5. Recovery fine-tuning: 100 epochs, SGD lr=0.01, momentum 0.9,
         weight decay 5e-4, MultiStepLR [60, 80] gamma 0.1, batch 128, no warmup.

      6. Save the model and write the JSON log.

SPARSITY LEARNING, REQUIRED BY bn_scale ONLY
    BNScaleImportance ranks channels by the magnitude of their BatchNorm gamma.
    On a normally trained network that ranking is nearly meaningless: all gammas
    sit in the same range, so the criterion has nothing to discriminate on. The
    fix, from PruningBench's `--method slim --reg 1e-5`, is a preliminary phase
    of 100 epochs during which pruner.regularize(model) adds an L1 penalty on the
    gammas, pushing a subset of them towards zero and creating the separation the
    ranking needs.

    Recipe: SGD lr=0.01, momentum 0.9, weight decay 0, MultiStepLR [60, 80]
    gamma 0.1, reg=1e-5. The result is cached in models/<name>_sparse.pt and
    reused across all targets of that model, since it depends on the model only;
    the phase costs about 5-6 h per model on an RTX A6000. It is skipped for
    architectures without BatchNorm, which is why squeezenet1_1 has no bn_scale
    results at all.

AVAILABLE IMPORTANCE CRITERIA (Torch-Pruning 1.6)
    magnitude_l1   MagnitudeImportance(p=1)     data-free
    magnitude_l2   MagnitudeImportance(p=2)     data-free
    bn_scale       BNScaleImportance()          needs BatchNorm + sparsity learning
    fpgm           FPGMImportance()             filter pruning via geometric median
    taylor         TaylorImportance()           data-driven, 10 gradient batches
    obdc           OBDCImportance()             Fisher information via sampled labels
    random         RandomImportance()           control

    random is not filler: it is the control that tells you how much of a
    criterion's result comes from the criterion and how much from merely having
    removed capacity and retrained.

LIMITATIONS INHERITED FROM torch_pruning 1.6.0
    OBDCImportance._prepare_model crashes on models without attention layers; a
    local patch (see _patch_obdc_prepare_model) works around it. Even patched,
    obdc still fails on architectures with depthwise convolutions (mobilenetv2),
    with Fire modules (squeezenet1_1), and on vgg19 with a size mismatch. Those
    combinations are logged as failures and skipped rather than crashing the run,
    which is why the published grid has 404 models and not 441.

REPRODUCIBILITY
    The seed is propagated to random, numpy and torch (CPU and CUDA), and re-set
    at the start of every (model, importance, target) triple so that a run gives
    the same result whether it is executed alone or as part of a sweep. All
    published results use seed 42.

USAGE (pytorch-env)
    # a full sweep over nine targets
    python 01_prune.py --data_dir /path/to/cifar100 --num_classes 100 \\
        --checkpoints 10 20 30 40 50 60 70 80 90 --batch_size 128

    # smoke test: one target, two criteria, 1 fine-tuning epoch instead of 100
    python 01_prune.py --data_dir /path/to/cifar100 --num_classes 100 \\
        --checkpoints 30 --models resnet18 --importance random magnitude_l2 \\
        --final_epochs 1

    # single task, the form used by the SLURM array jobs
    python 01_prune.py --data_dir /path/to/cifar100 --num_classes 100 \\
        --checkpoints 30 --models resnet18 --importance taylor
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Required so torch.load(weights_only=False) can resolve the classes stored in
# the whole-model pickles. Keep these modules importable under these exact names.
import cifar_resnet  # noqa: F401
import cifar_vgg  # noqa: F401
import wrn  # noqa: F401


CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)

MODEL_SIZES = {
    "resnet18":      32,
    "resnet50":      32,
    "vgg19":         32,
    "mobilenetv2":   32,
    "mnasnet1_0":    32,  # archived: the .pt still loads, but it is out of the lineup
    "googlenet":     32,
    "squeezenet1_1": 32,
    "wrn_28_10":     32,
}

ALL_IMPORTANCES = ["magnitude_l1", "magnitude_l2", "bn_scale", "fpgm",
                   "taylor", "obdc", "random",
                   # Tier 1
                   "lamp", "fisher",
                   # Tier 2
                   "group_lasso", "hrank"]
# Data-driven : forward+backward (gradient) ou forward seul (activations) requis
# before each pruner.step(). See progressive_pruning_to_target().
DATA_DRIVEN = {"taylor", "obdc", "fisher", "hrank"}
# Criteria that need a prior sparsity-learning phase (cached in a dedicated .pt).
SPARSITY_WEIGHT_BASED = {"bn_scale", "group_lasso"}


# ─────────────────────────────────────────────
# PruningBench recipe, identical for every architecture
# ─────────────────────────────────────────────
RECIPE_FT = {
    "optimizer": "sgd", "momentum": 0.9, "weight_decay": 5e-4,
    "lr": 0.01,
    "scheduler": "multistep",
    "lr_decay_epochs": [60, 80], "lr_decay_rate": 0.1,
    "epochs": 100,
    "batch_size": 128,
    "ref": "PruningBench (Li et al. 2024) — recovery FT uniforme via VainF/Torch-Pruning/reproduce/main.py",
}

PRUNING_PROTOCOL = {
    "iterative_steps": 400,
    "global_pruning": True,
    "max_pruning_ratio": 0.9,         # = "10% group-wise protection" PruningBench
    "data_driven_batches": 10,
    "bn_scale_reg": 1e-5,             # PruningBench --reg 1e-5 pour BNScalePruner
    "ref": "PruningBench progressive_pruning + 10% protection + slim/bn_scale",
}

# Sparsity-learning recipe (bn_scale only; PruningBench "slim" mode)
SPARSITY_RECIPE = {
    "optimizer": "sgd", "momentum": 0.9, "weight_decay": 0,  # wd=0: the pruner supplies the regularisation
    "lr": 0.01,
    "scheduler": "multistep",
    "lr_decay_epochs": [60, 80], "lr_decay_rate": 0.1,
    "epochs": 100,
    "ref": "PruningBench --sl-total-epochs 100 --sl-lr 0.01 --sl-lr-decay-milestones 60,80",
}


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def seed_everything(seed: int):
    """Seed every source of randomness the pipeline uses.

    Four separate calls are needed because each library keeps its own internal RNG:
    seeding torch alone still lets random.shuffle and np.random.permutation, which
    drive the data augmentation, produce different sequences from run to run.

    Called twice: once at start-up, then again at the start of every
    (model, importance, target) triple in run_one(), so that a triple gives the same
    result whether it runs alone or in the middle of a sweep.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_workers():
    """Choose a DataLoader worker count from the CPU budget actually available.

    Order of preference:
        1. $SLURM_CPUS_PER_TASK when set, so a cluster job honours its allocation
           instead of contending with the other jobs sharing the node.
        2. os.sched_getaffinity(0), the cores this process is allowed to run on
           (respects cgroups and taskset). Linux only.
        3. Fall back to 4.

    Workers are separate processes that load and augment the next batches while the
    GPU works on the current one. With none, the GPU waits on the CPU at every batch
    and throughput collapses.
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
    """Build the CIFAR-100 train (50k) and test (10k) loaders.

    The two differ in ways that matter:

    - Train: augmentation (random crop with padding 4, horizontal flip) so the model
      learns to tolerate small variations. shuffle=True reshuffles each epoch, which
      SGD convergence depends on. drop_last=True discards a trailing partial batch,
      which would otherwise skew the BatchNorm running statistics.

    - Test: no augmentation and no shuffling, so the evaluation order is
      deterministic. This one loader serves both the per-epoch monitoring during
      fine-tuning (evaluate()) and the final publishable figures
      (evaluate_with_topk_ci()). CIFAR-100 ships no separate validation split, so the
      test set is used as the evaluation reference, as it is throughout this
      literature.

    pin_memory speeds up the host-to-device copy; persistent_workers avoids
    respawning the worker processes every epoch, which is a visible gain on a fast
    GPU where an epoch is short.
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
    # A seeded generator makes shuffle=True reproducible across runs.
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=(nw > 0), generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True,
                            persistent_workers=(nw > 0))
    return train_loader, val_loader


def evaluate(model, val_loader, device):
    """Top-1 accuracy over a loader, in percent.

    Called once per fine-tuning epoch, so roughly a hundred times per run: it stays
    deliberately cheap, with no top-5 and no confidence interval. The thorough
    version is evaluate_with_topk_ci(), called once at the end of run_one().

    model.eval() disables dropout and freezes the BatchNorm running statistics;
    torch.no_grad() drops the autograd tape, which saves memory and time.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_with_topk_ci(model, val_loader, device, n_bootstrap=1000, seed=0):
    """Final FP32 evaluation: top-1, top-5, and bootstrap 95 % confidence intervals.

    Called once per run, at the very end. It costs 5-10 s on a GPU for the 10k
    CIFAR-100 test images, negligible against the fine-tuning it follows, and yields
    numbers that can be published as they are.

    The interval comes from a bootstrap: resample the per-image correctness flags
    `n_bootstrap` times with replacement, take the mean accuracy of each resample,
    and read the 2.5 % and 97.5 % percentiles as the bounds. This is the standard
    non-parametric method (Efron, 1979): it assumes no distribution, which matters
    because accuracy near the ceiling is visibly non-normal.

    Returns a dict with top1_pct, top5_pct, top{1,5}_ci95_{lo,hi} and n_eval.
    """
    model.eval()
    # Collect a 0/1 correctness flag per image, separately for top-1 and top-5,
    # so the bootstrap below can resample them directly.
    # output.topk(5).indices returns the five predicted class indices ordered by
    # decreasing logit: the first column answers top-1, the whole row top-5.
    correct1, correct5 = [], []
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(images)
        top5 = out.topk(5, dim=1).indices  # (B, 5)
        correct1.append((top5[:, 0] == labels).cpu().numpy().astype(np.uint8))
        # Top-5: is the true class anywhere among the five predictions?
        correct5.append((top5 == labels.unsqueeze(1)).any(dim=1).cpu().numpy().astype(np.uint8))

    arr1 = np.concatenate(correct1) if correct1 else np.array([], dtype=np.uint8)
    arr5 = np.concatenate(correct5) if correct5 else np.array([], dtype=np.uint8)
    n = len(arr1)
    if n == 0:
        return {"top1_pct": 0.0, "top5_pct": 0.0,
                "top1_ci95_lo": 0.0, "top1_ci95_hi": 0.0,
                "top5_ci95_lo": 0.0, "top5_ci95_hi": 0.0, "n_eval": 0}

    rng = np.random.RandomState(seed)
    means1 = np.empty(n_bootstrap, dtype=np.float64)
    means5 = np.empty(n_bootstrap, dtype=np.float64)
    for k in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        means1[k] = arr1[idx].mean()
        means5[k] = arr5[idx].mean()
    return {
        "top1_pct": float(100.0 * arr1.mean()),
        "top5_pct": float(100.0 * arr5.mean()),
        "top1_ci95_lo": float(100.0 * np.percentile(means1, 2.5)),
        "top1_ci95_hi": float(100.0 * np.percentile(means1, 97.5)),
        "top5_ci95_lo": float(100.0 * np.percentile(means5, 2.5)),
        "top5_ci95_hi": float(100.0 * np.percentile(means5, 97.5)),
        "n_eval": int(n),
    }


def extract_layer_structure(model):
    """Record the output width of every Conv2d, Linear and BatchNorm2d, in order.

    Structured pruning physically removes channels, so a pruned checkpoint already
    carries the surviving width of each layer. Comparing that against the baseline
    structure gives the per-layer pruning rate.

    This is the field that turns a latency observation into an explanation. Without
    it, one can only say that two criteria at equal accuracy produce different
    speedups; with it, one can point at where each criterion put its sparsity, which
    is the actual mechanism.
    """
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            out.append({"layer": name, "kind": "Conv2d",
                        "out_channels": int(m.out_channels)})
        elif isinstance(m, nn.Linear):
            out.append({"layer": name, "kind": "Linear",
                        "out_channels": int(m.out_features)})
        elif isinstance(m, nn.BatchNorm2d):
            out.append({"layer": name, "kind": "BatchNorm2d",
                        "out_channels": int(m.num_features)})
    return out


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Run one SGD training epoch; return (mean loss, top-1 accuracy in percent).

    model.train() enables dropout and lets the BatchNorm layers keep updating their
    running statistics, which is the opposite of what model.eval() does and is easy
    to get wrong when alternating between training and evaluation in a loop.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
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


# ─────────────────────────────────────────────
# Recovery fine-tune (recette PruningBench unique)
# ─────────────────────────────────────────────
def finetune_pruningbench(model, train_loader, val_loader, device, total_epochs, label):
    """Recovery fine-tuning after pruning, following the PruningBench recipe.

    SGD lr=0.01, momentum 0.9, weight decay 5e-4, MultiStepLR [60, 80] gamma 0.1,
    100 epochs by default. Keeps the best-validation weights rather than the last
    ones. This is what recovers most of the accuracy lost to pruning, and holding it
    fixed across criteria is what makes the criteria comparable.
    """
    if total_epochs <= 0:
        return 0.0, []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=RECIPE_FT["lr"],
                          momentum=RECIPE_FT["momentum"],
                          weight_decay=RECIPE_FT["weight_decay"])
    # Adapter milestones si --final_epochs override < 100
    ms = [m for m in RECIPE_FT["lr_decay_epochs"] if m < total_epochs]
    if not ms:
        ms = [max(1, total_epochs - 1)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms,
                                               gamma=RECIPE_FT["lr_decay_rate"])

    print(f"  [{label}] {total_epochs} ep, SGD lr={RECIPE_FT['lr']} m={RECIPE_FT['momentum']} "
          f"wd={RECIPE_FT['weight_decay']}, MultiStep {ms} γ={RECIPE_FT['lr_decay_rate']}",
          flush=True)

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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "train_acc": train_acc, "val_acc": val_acc,
                        "lr": optimizer.param_groups[0]["lr"], "duration_s": dt})

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == total_epochs - 1 or improved:
            star = " *" if improved else ""
            print(f"    Ep {epoch+1:3d}/{total_epochs}  loss={train_loss:.4f}  "
                  f"train={train_acc:5.2f}%  val={val_acc:5.2f}%  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  ({dt:.0f}s){star}",
                  flush=True)

    if best_state:
        model.load_state_dict(best_state); model.to(device)
    print(f"  [{label}] Best val : {best_acc:.2f}% @ ep {best_epoch}/{len(history)}", flush=True)
    return best_acc, history


# ─────────────────────────────────────────────
# Pruning utilitaires
# ─────────────────────────────────────────────
def count_params(model):
    """Total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_macs(model, input_size):
    """Estimate multiply-accumulate operations for one forward pass.

    Used as the theoretical speedup reference: comparing the MAC reduction against
    the measured latency reduction is what exposes how much of a pruning gain the
    hardware actually realises, which on an Edge TPU is often much less than the
    arithmetic would suggest.
    """
    try:
        import torch_pruning as tp
        example = torch.randn(1, 3, input_size, input_size)
        macs, _ = tp.utils.count_ops_and_params(model, example)
        return macs
    except Exception:
        return 0


def _patch_obdc_prepare_model():
    """Work around a torch_pruning 1.6.0 crash in OBDCImportance._prepare_model.

    Upstream assumes the model contains attention layers and fails outright on the
    purely convolutional networks used here. The patch reimplements the method to
    skip that assumption. Applied at import time, and only to OBDC.
    """
    import torch_pruning as tp

    def _prepare_model_patched(self, model, pruner):
        """Replacement for the upstream method: tolerates models without attention layers."""
        from torch_pruning.pruner import function
        for group in pruner.DG.get_all_groups(
                ignored_layers=pruner.ignored_layers,
                root_module_types=pruner.root_module_types):
            new_group = pruner._downstream_node_as_root_if_attention(group)
            if new_group is not None:
                group = new_group
            for i, (dep, idxs) in enumerate(group):
                layer = dep.target.module
                if isinstance(layer, tuple(self.target_types)) and dep.handler in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    self.modules.append(layer)
                    layer.register_forward_pre_hook(self._save_input)
                    layer.register_full_backward_hook(self._save_grad_output)

    tp.importance.OBDCImportance._prepare_model = _prepare_model_patched


_patch_obdc_prepare_model()


# ─────────────────────────────────────────────
# Importance criteria custom (Tier 1 + Tier 2)
# ─────────────────────────────────────────────
def _build_fisher_importance_class():
    """Build FisherImportance as a variant of GroupTaylorImportance.

    The importance signal is the squared gradient |g|^2 instead of Taylor's |g.w|,
    which is the diagonal of the empirical Fisher information (Theis et al., 2018).
    It sits outside PruningBench but is a direct relative of Taylor, so it reuses the
    same fine-tuning recipe and the same 10 gradient batches.

    Implemented by reproducing the upstream __call__ control flow exactly, branch for
    branch (conv_out, conv_in, BN, LN, bias, transposed, grouped), substituting only
    the formula: (w*dw).abs() becomes dw^2 and (b*db).abs() becomes db^2. Copying the
    control flow rather than subclassing avoids silently diverging from upstream on
    the layer types this project does not exercise.
    """
    import torch_pruning as tp
    from torch_pruning.pruner import function

    class FisherImportance(tp.importance.GroupTaylorImportance):
        """Squared-gradient (empirical Fisher diagonal) channel importance."""
        @torch.no_grad()
        def __call__(self, group):
            """Accumulate the squared-gradient importance over a dependency group."""
            group_imp = []
            group_idxs = []
            for i, (dep, idxs) in enumerate(group):
                idxs.sort()
                layer = dep.target.module
                prune_fn = dep.handler
                root_idxs = group[i].root_idxs
                if not isinstance(layer, tuple(self.target_types)):
                    continue

                if prune_fn in [function.prune_conv_out_channels,
                                function.prune_linear_out_channels]:
                    if hasattr(layer, "transposed") and layer.transposed:
                        dw = layer.weight.grad.data.transpose(1, 0)[idxs].flatten(1)
                    else:
                        dw = layer.weight.grad.data[idxs].flatten(1)
                    local_imp = (dw ** 2).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None and layer.bias.grad is not None:
                        db = layer.bias.grad.data[idxs]
                        local_imp = db ** 2
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

                elif prune_fn in [function.prune_conv_in_channels,
                                  function.prune_linear_in_channels]:
                    # Suit l'ordre exact de GroupTaylorImportance : on aplatit
                    # Flatten BEFORE indexing by idxs, otherwise scatter_add_
                    # inside _reduce hits a dimension mismatch.
                    if hasattr(layer, "transposed") and layer.transposed:
                        dw = (layer.weight.grad).flatten(1)
                    else:
                        dw = (layer.weight.grad).transpose(0, 1).flatten(1)
                    local_imp = (dw ** 2).sum(1)
                    if (prune_fn == function.prune_conv_in_channels
                            and layer.groups != layer.in_channels
                            and layer.groups != 1):
                        local_imp = local_imp.repeat(layer.groups)
                    local_imp = local_imp[idxs]
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                elif prune_fn == function.prune_groupnorm_out_channels:
                    if layer.affine:
                        dw = layer.weight.grad.data[idxs]
                        local_imp = dw ** 2
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
                        if self.bias and layer.bias is not None and layer.bias.grad is not None:
                            db = layer.bias.grad.data[idxs]
                            local_imp = db ** 2
                            group_imp.append(local_imp)
                            group_idxs.append(root_idxs)

                elif prune_fn == function.prune_layernorm_out_channels:
                    if layer.elementwise_affine:
                        dw = layer.weight.grad.data[idxs]
                        local_imp = dw ** 2
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
                        if self.bias and layer.bias is not None and layer.bias.grad is not None:
                            db = layer.bias.grad.data[idxs]
                            local_imp = db ** 2
                            group_imp.append(local_imp)
                            group_idxs.append(root_idxs)

            if len(group_imp) == 0:
                return None
            group_imp = self._reduce(group_imp, group_idxs)
            group_imp = self._normalize(group_imp, self.normalizer)
            return group_imp

    return FisherImportance


def _build_hrank_importance_class():
    """Build HRankImportance (Lin et al., 2020): rank of the output feature maps.

    For each convolution, average the matrix rank of its (H, W) output feature maps
    over N batches. A high-rank feature map carries more independent information, so
    its channel is deemed more important.

    Follows the compute_importance() pattern of ActivationImportance: a forward hook
    on each Conv2d accumulates a per-channel score, which __call__ then reads.

    The scores must be recomputed at every pruning step. Reusing them after channels
    have been removed leaves the stored indices pointing at channels that no longer
    exist, which produces a silently wrong ranking rather than an error.
    """
    import torch_pruning as tp
    from torch_pruning.pruner import function
    from contextlib import contextmanager

    class HRankImportance(tp.importance.ActivationImportance):
        """Feature-map-rank channel importance."""
        def __init__(self, group_reduction="mean", normalizer="mean", bias=False,
                     target_types=(nn.Conv2d,)):
            """Set up the per-convolution score accumulators."""
            self.group_reduction = group_reduction
            self.normalizer = normalizer
            self.bias = bias
            self.target_types = tuple(target_types)

        @contextmanager
        def compute_importance(self, model):
            """Run N batches with forward hooks to accumulate the mean feature-map rank."""
            @torch.no_grad()
            def _hook(module, input, output):
                """Forward hook: accumulate the mean matrix rank of this layer's feature maps."""
                if not isinstance(module, nn.Conv2d):
                    return
                # output is (B, C, H, W). Matrix rank of each (H, W) feature map,
                # averaged per channel, then per batch.
                B, C, H, W = output.shape
                fm = output.detach().reshape(B * C, H, W).float()
                ranks = torch.linalg.matrix_rank(fm).float().reshape(B, C).mean(0)
                # Running mean across successive batches.
                if not hasattr(module, "_hrank_sum") or module._hrank_sum is None:
                    module._hrank_sum = ranks.clone()
                    module._hrank_n = 1
                else:
                    module._hrank_sum = module._hrank_sum + ranks
                    module._hrank_n += 1
                module._importance = module._hrank_sum / module._hrank_n

            hooks = []
            for m in model.modules():
                if isinstance(m, self.target_types):
                    m._hrank_sum = None
                    m._hrank_n = 0
                    hooks.append(m.register_forward_hook(_hook))
            yield
            for h in hooks:
                h.remove()

        @torch.no_grad()
        def __call__(self, group):
            """Return the accumulated rank scores for a dependency group."""
            group_imp = []
            group_idxs = []
            for i, (dep, idxs) in enumerate(group):
                idxs.sort()
                layer = dep.target.module
                prune_fn = dep.handler
                root_idxs = group[i].root_idxs
                if not isinstance(layer, self.target_types):
                    continue
                # HRank score of a conv output channel: the rank of its feature map
                if prune_fn == function.prune_conv_out_channels:
                    if not hasattr(layer, "_importance") or layer._importance is None:
                        continue
                    local_imp = layer._importance[idxs].to(layer.weight.device)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
            if len(group_imp) == 0:
                return None
            group_imp = self._reduce(group_imp, group_idxs)
            group_imp = self._normalize(group_imp, self.normalizer)
            return group_imp

    return HRankImportance


def get_importance(name, num_classes):
    """Instantiate the torch_pruning importance object for a criterion name.

    Maps the criterion names used on the command line to their implementations, and
    returns for each whether it is data-driven, since that decides whether
    progressive_pruning_to_target() must run gradient batches before each step.
    """
    import torch_pruning as tp
    if name == "magnitude_l1":
        return tp.importance.MagnitudeImportance(p=1)
    if name == "magnitude_l2":
        return tp.importance.MagnitudeImportance(p=2)
    if name == "bn_scale":
        return tp.importance.BNScaleImportance()
    if name == "fpgm":
        return tp.importance.FPGMImportance()
    if name == "taylor":
        return tp.importance.TaylorImportance()
    if name == "obdc":
        return tp.importance.OBDCImportance(num_classes=num_classes)
    if name == "random":
        return tp.importance.RandomImportance()
    if name == "lamp":
        # Layer-Adaptive Magnitude-based Pruning, Lee et al. 2021. Data-free.
        return tp.importance.LAMPImportance(p=2)
    if name == "fisher":
        # Fisher Pruning (Theis et al. 2018), hors PruningBench. Recette Taylor.
        FisherImportance = _build_fisher_importance_class()
        return FisherImportance()
    if name == "group_lasso":
        # Importance used for the pruning that FOLLOWS sparsity learning: group L2
        # norm. The regularisation itself is applied by GroupNormPruner.regularize().
        return tp.importance.GroupMagnitudeImportance(p=2)
    if name == "hrank":
        # HRank (Lin et al. 2020), data-driven activation-based.
        HRankImportance = _build_hrank_importance_class()
        return HRankImportance()
    raise ValueError(f"Importance inconnue : {name}. Disponibles : {ALL_IMPORTANCES}")


def get_ignored_layers(model, name):
    """Return the layers that must not be pruned, in practice the final classifier.

    Pruning the classifier would change the number of output classes, so its output
    dimension is protected. Its input dimension still shrinks with the last feature
    layer, which is what makes the classifier follow the network it sits on.
    """
    ignored = []
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        ignored.append(model.fc)
    elif hasattr(model, "linear") and isinstance(model.linear, nn.Linear):
        ignored.append(model.linear)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            for layer in model.classifier:
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    ignored.append(layer)
        elif isinstance(model.classifier, (nn.Linear, nn.Conv2d)):
            ignored.append(model.classifier)
    if not ignored:
        last_layer = None
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                last_layer = m
        if last_layer:
            ignored.append(last_layer)
    return ignored


def has_batchnorm(model):
    """Report whether a model contains at least one BatchNorm layer.

    Gate for the bn_scale criterion, which ranks channels by BatchNorm gamma and is
    therefore meaningless without it. squeezenet1_1 is the case in this lineup.
    """
    return any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
               for m in model.modules())


# ─────────────────────────────────────────────
# Sparsity learning phase (bn_scale and group_lasso; PruningBench `slim` / `group_sl`)
# ─────────────────────────────────────────────
def sparsity_learning_phase(name, sparsity_method, args, device, models_dir):
    """Produce (and cache) the sparsity-trained model that bn_scale requires.

    On a normally trained network, all BatchNorm gammas sit in the same range and
    ranking channels by their magnitude discriminates almost nothing. This phase
    trains for 100 epochs with pruner.regularize(model) called after each backward
    pass, which adds an L1 penalty on the gammas and drives a subset of them towards
    zero, creating the separation the ranking needs.

    Args:
        sparsity_method: "bn_scale" for the L1-on-gammas variant (PruningBench slim),
            "group_lasso" for group L2 on convolution weights (PruningBench group_sl).

    The result is cached as models/<name>_sparse.pt and reused across every target of
    that model, since it depends only on the model: the phase costs 5-6 h on an
    RTX A6000 and would otherwise be repeated nine times per architecture.

    Returns:
        Path to the sparse model, or None when the model has no BatchNorm and the
        criterion therefore does not apply.
    """
    import torch_pruning as tp

    cache_suffix = "sparse" if sparsity_method == "bn_scale" else f"{sparsity_method}_sparse"
    sparse_path = models_dir / f"{name}_{cache_suffix}.pt"
    if sparse_path.exists() and not args.force:
        print(f"  [sparsity] {sparse_path.name} already present, reusing it.")
        return sparse_path

    # Load the baseline and, where the criterion needs it, check for BatchNorm
    model = torch.load(str(models_dir / f"{name}.pt"),
                       map_location="cpu", weights_only=False)
    if sparsity_method == "bn_scale" and not has_batchnorm(model):
        print(f"  [sparsity] {name} n'a pas de BatchNorm → bn_scale incompatible, skip.")
        return None
    model = model.to(device)

    ignored_layers = get_ignored_layers(model, name)
    input_size = MODEL_SIZES[name]
    example_input = torch.randn(1, 3, input_size, input_size).to(device)

    if sparsity_method == "bn_scale":
        importance = tp.importance.BNScaleImportance()
        pruner = tp.pruner.BNScalePruner(
            model=model, example_inputs=example_input, importance=importance,
            reg=PRUNING_PROTOCOL["bn_scale_reg"],
            pruning_ratio=0.5,
            global_pruning=PRUNING_PROTOCOL["global_pruning"],
            max_pruning_ratio=PRUNING_PROTOCOL["max_pruning_ratio"],
            iterative_steps=PRUNING_PROTOCOL["iterative_steps"],
            ignored_layers=ignored_layers,
        )
    elif sparsity_method == "group_lasso":
        importance = tp.importance.GroupMagnitudeImportance(p=2)
        pruner = tp.pruner.GroupNormPruner(
            model=model, example_inputs=example_input, importance=importance,
            reg=PRUNING_PROTOCOL["bn_scale_reg"],  # same regularisation coefficient
            pruning_ratio=0.5,
            global_pruning=PRUNING_PROTOCOL["global_pruning"],
            max_pruning_ratio=PRUNING_PROTOCOL["max_pruning_ratio"],
            iterative_steps=PRUNING_PROTOCOL["iterative_steps"],
            ignored_layers=ignored_layers,
        )
    else:
        raise ValueError(f"sparsity_method inconnu : {sparsity_method}")

    # Boucle de sparsity learning
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size, args.seed)
    # --sparsity_epochs can shorten this, which is what smoke tests use.
    epochs = args.sparsity_epochs if args.sparsity_epochs is not None else SPARSITY_RECIPE["epochs"]
    optimizer = optim.SGD(model.parameters(),
                          lr=SPARSITY_RECIPE["lr"],
                          momentum=SPARSITY_RECIPE["momentum"],
                          weight_decay=SPARSITY_RECIPE["weight_decay"])
    ms = [m for m in SPARSITY_RECIPE["lr_decay_epochs"] if m < epochs]
    if not ms:
        ms = [max(1, epochs - 1)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms,
                                               gamma=SPARSITY_RECIPE["lr_decay_rate"])
    criterion = nn.CrossEntropyLoss()

    print(f"  [sparsity={sparsity_method}] {name} — {epochs} ep, SGD lr={SPARSITY_RECIPE['lr']} "
          f"reg={PRUNING_PROTOCOL['bn_scale_reg']}, MultiStep {ms} γ={SPARSITY_RECIPE['lr_decay_rate']}",
          flush=True)

    t_start = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            pruner.regularize(model)  # ajoute L1 sur BN gammas dans .grad
            optimizer.step()
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"    [sparsity {name}] Ep {epoch+1}/{epochs}  val={val_acc:.2f}%  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  ({time.time()-t0:.0f}s)",
                  flush=True)

    torch.save(model, str(sparse_path))
    dt_min = (time.time() - t_start) / 60
    print(f"  [sparsity] {name} → {sparse_path.name} "
          f"({sparse_path.stat().st_size/1024/1024:.1f} MB, {dt_min:.1f} min)", flush=True)
    return sparse_path


# ─────────────────────────────────────────────
# Progressive pruning (PruningBench)
# ─────────────────────────────────────────────
def progressive_pruning_to_target(model, pruner, target_pct,
                                  train_loader, importance, importance_name,
                                  device, dd_batches=10):
    """Prune in small steps until the target parameter reduction is reached.

    Each pruner.step() removes about 1/iterative_steps of the budget, with no
    fine-tuning in between. Taking many small steps rather than one large one keeps
    the importance ranking approximately valid throughout: a single big step would
    rank channels using a network that ceases to exist after the first removals.

    For the data-driven criteria (taylor, obdc, fisher, hrank), a forward and
    backward pass over `dd_batches` batches runs before every step, so the gradients
    reflect the current network rather than the original one.

    The loop stops on the achieved parameter reduction, not on a step count, because
    global pruning removes whole dependency groups whose sizes vary; the achieved
    rate is therefore never exactly the requested one, and it is the achieved rate
    that gets recorded and used downstream.
    """
    import torch_pruning as tp

    initial_params = count_params(model)
    target_params = initial_params * (1 - target_pct / 100)

    n_steps = 0
    while count_params(model) > target_params:
        if pruner.current_step >= pruner.iterative_steps:
            break

        if importance_name == "obdc":
            # Empirical Fisher information: the label is sampled from the model's
            # softmax, which is what distinguishes it from a plain squared gradient.
            model.zero_grad(set_to_none=True)
            importance._prepare_model(model, pruner)
            for k, (imgs, lbls) in enumerate(train_loader):
                if k >= dd_batches:
                    break
                imgs = imgs.to(device, non_blocking=True)
                output = model(imgs)
                # Sample the label from the model's own softmax rather than using
                # the dataset label: that is what makes the accumulated quantity the
                # empirical Fisher information instead of a plain gradient product.
                with torch.no_grad():
                    probs = F.softmax(output.detach().cpu(), dim=1)
                    sampled_y = torch.multinomial(probs, 1).squeeze().to(device)
                loss = F.cross_entropy(output, sampled_y)
                loss.backward()
                importance.step()
            pruner.step()
            importance._rm_hooks(model)
            importance._clear_buffer()

        elif importance_name in ("taylor", "fisher"):
            # First-order, gradient-based: forward and backward over dd_batches.
            # Taylor lit (w*dw).abs() ; Fisher lit dw² (cf. FisherImportance.__call__).
            model.zero_grad(set_to_none=True)
            for k, (imgs, lbls) in enumerate(train_loader):
                if k >= dd_batches:
                    break
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                loss = F.cross_entropy(model(imgs), lbls)
                loss.backward()
            pruner.step()

        elif importance_name == "hrank":
            # HRank: recompute the ranks at EVERY pruning step. torch_pruning
            # rebases channel indices after each removal, so scores cached once
            # would be read through indices that no longer denote the same
            # channels; _importance[idxs] then maps onto the wrong channels and
            # the pruning becomes effectively random. Observed in a smoke test
            # (val=1.00 %, i.e. chance level on 100 classes).
            # one-shot gave 1.00 %, recomputing gave 49.84 %).
            # Expensive on CPU (an SVD per batch); acceptable on a cluster GPU.
            # _hrank_sum=None resets the accumulators between pruning steps.
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m._hrank_sum = None
                    m._hrank_n = 0
            with importance.compute_importance(model):
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    for k, (imgs, _) in enumerate(train_loader):
                        if k >= dd_batches:
                            break
                        imgs = imgs.to(device, non_blocking=True)
                        _ = model(imgs)
                if was_training:
                    model.train()
            pruner.step()

        else:
            # magnitude_l1, magnitude_l2, bn_scale, fpgm, random, lamp, group_lasso :
            # data-free: the score comes from the weights, or from the sparsity
            # that the regularisation phase already produced.
            pruner.step()

        n_steps += 1

    actual_params = count_params(model)
    actual_pct = 100 * (1 - actual_params / initial_params)
    return n_steps, actual_pct, actual_params


# ─────────────────────────────────────────────
# A single run: (model, importance, target percentage)
# ─────────────────────────────────────────────
def run_one(name, imp_name, checkpoint_pct, args, device,
            models_dir, pruned_dir, log_dir):
    """Run one independent (model, importance, target) triple end to end.

    Loads a fresh baseline copy, builds the pruner, prunes progressively to the
    target, fine-tunes for recovery, evaluates with confidence intervals, and writes
    the checkpoint and its JSON log.

    bn_scale takes the sparsity-trained checkpoint and a BNScalePruner(reg=1e-5);
    every other criterion takes the plain baseline and a MetaPruner.

    Starting from the baseline every time, rather than from the previous target, is
    what keeps the criteria comparable: with an iterative schedule the 30 % model
    would depend on the path taken through 10 % and 20 %, and part of the measured
    difference between criteria would be that path.
    """
    import torch_pruning as tp

    # Seed suffix: seed 42 (the default) keeps the original naming, so the
    # published filenames stay valid. Any other seed appends `_seed{N}`.
    seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"

    out_name = f"{name}_pruned{checkpoint_pct}pct_{imp_name}{seed_suffix}"
    out_path = pruned_dir / f"{out_name}.pt"
    log_path = log_dir / f"{name}_{imp_name}_{checkpoint_pct}pct{seed_suffix}.json"

    if out_path.exists() and not args.force:
        print(f"  [{out_name}] Already present, skipping.")
        return None

    print(f"\n{'━' * 70}")
    print(f"  {name.upper()} — {imp_name.upper()} — {checkpoint_pct}%  (seed={args.seed})")
    print(f"{'━' * 70}", flush=True)

    # Criteria that need a sparsity-trained model first:
    # bn_scale (cache _sparse.pt) et group_lasso (cache _group_lasso_sparse.pt).
    if imp_name in SPARSITY_WEIGHT_BASED:
        sparse_path = sparsity_learning_phase(name, imp_name, args, device, models_dir)
        if sparse_path is None:
            print(f"  [{out_name}] {imp_name} does not apply to {name} (no BatchNorm), skipping.",
                  flush=True)
            return None
        model_load_path = sparse_path
    else:
        model_load_path = models_dir / f"{name}.pt"

    seed_everything(args.seed)
    model = torch.load(str(model_load_path),
                       map_location="cpu", weights_only=False)
    model = model.to(device)

    # Reference point, measured before any pruning
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size, args.seed)
    ref_acc = evaluate(model, val_loader, device)
    pre_params = count_params(model)
    input_size = MODEL_SIZES[name]
    pre_macs = count_macs(model, input_size)
    print(f"  Pre-pruning: {ref_acc:.2f}% acc, {pre_params:,} params, {pre_macs:,.0f} MACs "
          f"(loaded from {model_load_path.name})", flush=True)

    # Build the pruner: BNScalePruner for bn_scale, MetaPruner for everything else
    importance = get_importance(imp_name, args.num_classes)
    ignored_layers = get_ignored_layers(model, name)
    example_input = torch.randn(1, 3, input_size, input_size).to(device)

    pruner_common_kwargs = dict(
        model=model, example_inputs=example_input,
        importance=importance,
        pruning_ratio=checkpoint_pct / 100.0,
        iterative_steps=PRUNING_PROTOCOL["iterative_steps"],
        global_pruning=PRUNING_PROTOCOL["global_pruning"],
        max_pruning_ratio=PRUNING_PROTOCOL["max_pruning_ratio"],
        ignored_layers=ignored_layers,
    )
    if imp_name == "bn_scale":
        pruner = tp.pruner.BNScalePruner(
            reg=PRUNING_PROTOCOL["bn_scale_reg"],
            **pruner_common_kwargs,
        )
    elif imp_name == "group_lasso":
        pruner = tp.pruner.GroupNormPruner(
            reg=PRUNING_PROTOCOL["bn_scale_reg"],
            **pruner_common_kwargs,
        )
    else:
        pruner = tp.pruner.MetaPruner(**pruner_common_kwargs)

    # Progressive pruning
    print(f"  [Pruning] target={checkpoint_pct}%, iterative_steps={PRUNING_PROTOCOL['iterative_steps']}, "
          f"global={PRUNING_PROTOCOL['global_pruning']}, "
          f"max_per_layer={PRUNING_PROTOCOL['max_pruning_ratio']}", flush=True)
    t_prune_start = time.time()
    dd_batches = (args.dd_batches if args.dd_batches is not None
                  else PRUNING_PROTOCOL["data_driven_batches"])
    n_steps, actual_pct, post_params = progressive_pruning_to_target(
        model, pruner, checkpoint_pct, train_loader, importance, imp_name, device,
        dd_batches=dd_batches,
    )
    t_prune = time.time() - t_prune_start
    post_macs = count_macs(model, input_size)
    val_acc_post = evaluate(model, val_loader, device)
    print(f"  [Pruning] {n_steps} steps en {t_prune:.0f}s — {post_params:,} params "
          f"({actual_pct:.1f}% pruned), val={val_acc_post:.2f}%", flush=True)

    # Final fine-tune
    final_epochs = args.final_epochs if args.final_epochs is not None else RECIPE_FT["epochs"]
    t_ft_start = time.time()
    final_acc, ft_history = finetune_pruningbench(
        model, train_loader, val_loader, device, final_epochs,
        label=f"Final FT @{checkpoint_pct}%",
    )
    t_ft = time.time() - t_ft_start

    # Save the pruned model
    torch.save(model, str(out_path))

    # -- Final FP32 evaluation (top-1, top-5, bootstrap 95 % CI) --------------
    # Costs 5-10 s on a GPU for the 10k CIFAR-100 test images, negligible
    # compared with the fine-tuning that precedes it, and yields the
    # against the fine-tuning it follows, and it yields publishable figures.
    # val_loader already points at the CIFAR-100 test split (train=False).
    print(f"  [FP32 test] Bootstrap 95 % CI over the 10k CIFAR-100 test images...",
          flush=True)
    fp32_acc = evaluate_with_topk_ci(model, val_loader, device)
    print(f"  [FP32 test] Top-1 = {fp32_acc['top1_pct']:.2f}% "
          f"[{fp32_acc['top1_ci95_lo']:.2f}, {fp32_acc['top1_ci95_hi']:.2f}]  "
          f"Top-5 = {fp32_acc['top5_pct']:.2f}%  (n={fp32_acc['n_eval']})",
          flush=True)

    # -- Layer structure of the pruned model ----------------------------------
    # {layer_name, kind, out_channels} for every Conv2d/Linear/BatchNorm. The
    # aggregator diffs this against the baseline to get per-layer sparsity, which
    # is what explains a latency result rather than merely correlating with it.
    layer_structure = extract_layer_structure(model)
    n_conv = sum(1 for x in layer_structure if x["kind"] == "Conv2d")
    n_lin = sum(1 for x in layer_structure if x["kind"] == "Linear")
    print(f"  [structure]  {n_conv} Conv2d + {n_lin} Linear couches post-pruning",
          flush=True)

    result = {
        "model": name, "importance": imp_name, "checkpoint_pct": checkpoint_pct,
        "ref_acc": ref_acc, "post_prune_acc": val_acc_post, "final_acc": final_acc,
        "acc_delta": final_acc - ref_acc,
        "pre_params": pre_params, "post_params": post_params,
        "param_reduction_pct": 100 * (1 - post_params / pre_params),
        "actual_pct": actual_pct,
        "pre_macs": pre_macs, "post_macs": post_macs,
        "macs_reduction_pct": 100 * (1 - post_macs / pre_macs) if pre_macs > 0 else 0,
        "n_pruning_steps": n_steps,
        "duration_s": {"pruning": t_prune, "finetune": t_ft, "total": t_prune + t_ft},
        "output_file": str(out_path),
        # -- Accuracy with CIs, and the per-layer surviving structure ----------
        "fp32_test_top1_pct":     fp32_acc["top1_pct"],
        "fp32_test_top5_pct":     fp32_acc["top5_pct"],
        "fp32_test_top1_ci95_lo": fp32_acc["top1_ci95_lo"],
        "fp32_test_top1_ci95_hi": fp32_acc["top1_ci95_hi"],
        "fp32_test_top5_ci95_lo": fp32_acc["top5_ci95_lo"],
        "fp32_test_top5_ci95_hi": fp32_acc["top5_ci95_hi"],
        "n_eval_test":            fp32_acc["n_eval"],
        "layer_structure_post":   layer_structure,
    }

    # Per-run JSON log
    full_log = {
        **result,
        "recipe_ft": RECIPE_FT,
        "pruning_protocol": PRUNING_PROTOCOL,
        "seed": args.seed, "device": str(device),
        "ft_history": ft_history,
    }
    with open(log_path, "w") as f:
        json.dump(full_log, f, indent=2, default=str)

    print(f"  == DONE ==")
    print(f"   Accuracy : {ref_acc:.2f}% -> {final_acc:.2f}% (delta={final_acc - ref_acc:+.2f})")
    print(f"   Params   : {pre_params:,} -> {post_params:,} ({result['param_reduction_pct']:.1f}% removed)")
    if pre_macs > 0:
        print(f"   MACs     : {pre_macs:,.0f} -> {post_macs:,.0f} ({result['macs_reduction_pct']:.1f}% removed)")
    print(f"   Duration : pruning={t_prune:.0f}s, fine-tune={t_ft:.0f}s")
    print(f"  ║ → {out_path.name} ({out_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"  ║ → {log_path.name}")
    print(f"  ╚{'═' * 50}", flush=True)

    return result


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    """Parse arguments and run every requested (model, importance, target) triple.

    Failures of a single triple are caught and logged rather than allowed to abort
    the sweep, because several criterion/architecture pairs are known to be
    unsupported by torch_pruning and are expected to fail (see the module docstring).
    """
    parser = argparse.ArgumentParser(
        description="Structured pruning of the CIFAR-100 baselines (PruningBench protocol)")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the `cifar-100-python/` subfolder "
                             "(Krizhevsky's original binary pickle format)")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--checkpoints", nargs="+", type=int, required=True,
                        help="Pruning targets, in %% of parameters removed (e.g. 10 20 30 40 50 60). "
                             "Each target is an independent run starting from the baseline.")
    parser.add_argument("--batch_size", type=int, required=True)

    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODEL_SIZES.keys()),
                        help="Subset of models to process (default: every model found in models/)")
    parser.add_argument("--importance", nargs="+", default=None,
                        choices=ALL_IMPORTANCES,
                        help=f"Subset of importance criteria (default: all of {ALL_IMPORTANCES})")

    parser.add_argument("--final_epochs", type=int, default=None,
                        help="Override the recovery fine-tuning epochs (default 100, PruningBench)")
    parser.add_argument("--sparsity_epochs", type=int, default=None,
                        help="Override le nb d'epochs de sparsity learning pour bn_scale "
                             "(default 100, PruningBench). Lower it for smoke tests.")
    parser.add_argument("--dd_batches", type=int, default=None,
                        help="Override the number of gradient batches used by the data-driven "
                             "criteria (taylor, obdc, fisher, hrank). Default 10, per PruningBench. "
                             "Lower it for CPU smoke tests: fisher and hrank are expensive without a GPU.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Recompute even when the output file already exists")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Directory holding models/ and receiving pytorch_pruned/ "
                             "and pruning_logs/. Defaults to this script's own "
                             "directory, which is what the published runs used. Set it "
                             "to keep several gigabytes of checkpoints out of a clone "
                             "of this repository.")
    args = parser.parse_args()

    # Default to the script's directory, preserving the layout every published
    # result was produced with; --work_dir relocates the whole set together, so
    # inputs and outputs never end up split across two trees.
    BASE_DIR = Path(args.work_dir) if args.work_dir else Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    PRUNED_DIR = BASE_DIR / "pytorch_pruned"
    LOG_DIR = BASE_DIR / "pruning_logs"
    PRUNED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    importance_list = args.importance if args.importance else ALL_IMPORTANCES
    checkpoints = sorted(args.checkpoints)

    available = [f.stem for f in sorted(MODELS_DIR.glob("*.pt"))]
    if not available:
        print(f"\n[ERROR] No baseline found in {MODELS_DIR}/ "
              f"(run 00_prepare_baselines.py first, or pass --work_dir)")
        return

    model_list = args.models if args.models else available
    model_list = [m for m in model_list if m in available]
    if not model_list:
        print(f"\n[ERROR] None of {args.models} found among {available}")
        return

    n_runs = len(model_list) * len(importance_list) * len(checkpoints)

    print("=" * 70)
    print("STRUCTURED PRUNING (CIFAR-100, PruningBench protocol)")
    print("=" * 70)
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Classes      : {args.num_classes}")
    print(f"  Models       : {model_list}")
    print(f"  Importances  : {importance_list}")
    print(f"  Targets      : {checkpoints}%")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Seed         : {args.seed}")
    print(f"  Total runs   : {n_runs}  ({len(model_list)} × {len(importance_list)} × {len(checkpoints)})")
    if args.final_epochs is not None:
        print(f"  ⚠ Override final_epochs={args.final_epochs}")
    print("=" * 70, flush=True)

    all_results = []
    t_start = time.time()

    for name in model_list:
        for imp_name in importance_list:
            for cp in checkpoints:
                try:
                    r = run_one(name, imp_name, cp, args, device,
                                MODELS_DIR, PRUNED_DIR, LOG_DIR)
                    if r is not None:
                        all_results.append(r)
                except Exception as e:
                    print(f"\n  [{name} {imp_name} @{cp}%] [FAILED] {type(e).__name__}: {e}",
                          flush=True)
                    import traceback; traceback.print_exc()

    total_min = (time.time() - t_start) / 60

    # Overall summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY ({total_min:.1f} min total, {len(all_results)} runs)")
    print(f"{'=' * 70}")
    for r in all_results:
        print(f"  {r['model']:<14s} [{r['importance']:<14s}] @{r['checkpoint_pct']:>2d}%  "
              f"{r['ref_acc']:.2f}% → {r['final_acc']:.2f}%  "
              f"({r['param_reduction_pct']:.1f}% params, "
              f"Δacc={r['acc_delta']:+.2f})")
    if all_results:
        # Strip ft_history pour le summary global (volume)
        log_path = LOG_DIR / "pruning_summary.json"
        existing = []
        if log_path.exists():
            with open(log_path) as f:
                existing = json.load(f)
        # Ne pas dupliquer : index par (model, imp, cp)
        seen = {(r["model"], r["importance"], r["checkpoint_pct"]) for r in all_results}
        existing = [r for r in existing
                    if (r["model"], r["importance"], r["checkpoint_pct"]) not in seen]
        merged = existing + all_results
        with open(log_path, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"\n-> {log_path}  ({len(merged)} runs accumulated)")
    else:
        print("\nNo new run computed.")


if __name__ == "__main__":
    main()

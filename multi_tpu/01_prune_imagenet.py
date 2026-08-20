#!/usr/bin/env python3
"""
01_prune_imagenet.py — size-targeted structured pruning of ImageNet models.

WHAT THIS PRODUCES
    pytorch_pruned_imagenet/<model>_pruned<P>pct_<importance>.pt
    pruning_logs_imagenet/<model>_<importance>_<run_id>.json, containing
        ref_top1         accuracy of the pretrained model, before pruning
        post_prune_top1  after pruning, before any fine-tuning
        final_top1       after recovery fine-tuning
        ft_history       per-epoch train loss/acc, val top-1/top-5, lr, duration
        layer_structure_post   surviving channels per layer

    With --prune_only it stops after pruning and writes a PREFT checkpoint plus a
    .meta.json sidecar; with --ft_only --resume_from it skips pruning and
    fine-tunes an existing PREFT. That split is what lets pipeline_full.py prune,
    compile, and measure the result before committing to the expensive
    fine-tuning run.

HOW THIS DIFFERS FROM THE SINGLE-TPU AXIS, AND WHY
    mono_tpu/01_prune.py sweeps a uniform grid of nine pruning targets to compare
    criteria against each other. This script does something else: it prunes each
    model to ONE target, chosen so the result fits the hardware.

    The reason is cost. Training the CIFAR-100 grid from scratch was affordable;
    doing the equivalent on ImageNet was estimated at roughly 4000 GPU-hours and
    would have added a resolution artefact on top, since these architectures are
    not designed for upscaled 32x32 input. Starting from published ImageNet
    weights and pruning to a hardware-derived target costs a small fraction of
    that and asks a question the grid cannot: what does it take to make a given
    model fit a given number of accelerators.

THE SIZE TARGET
    target_pct = 100 * (1 - target_params / current_params), with
    target_params ~ target_mb * 1e6, from the INT8 approximation of one byte per
    weight. A target of 8 MB corresponds to one Edge TPU's SRAM; 16, 32 and 64 MB
    correspond to pooling the SRAM of 2, 4 and 8 accelerators through
    edgetpu_compiler --num_segments.

    A model already under the target (GoogLeNet, at 6.6 M parameters) is not
    pruned at all; it is simply evaluated to record its baseline.

    IMPORTANT: this parameter-count target is only a first guess, and it is
    systematically optimistic. The compiler reports substantially more bytes than
    the weight count implies, because of a per-architecture fixed overhead that
    does not shrink with pruning (fused batch-norm constants, int32 biases,
    per-tensor alignment padding). Measured by regression over the guided loop:

        model                  slope (MiB per MiB of params)   fixed overhead
        ResNet-101             0.962 +/- 0.013                 2.29 +/- 0.36 MiB
        Inception-V4           1.027 +/- 0.011                 2.86 +/- 0.32 MiB
        Inception-ResNet-V2    0.970 +/- 0.009                 5.24 +/- 0.28 MiB

    The slope near 1.0 confirms the one-byte-per-weight assumption. It is the
    constant that the naive prediction ignores, and it is worse for architectures
    with many branches and 1x1 convolutions, which have many small tensors and
    therefore many per-channel constants for few weights. This is why
    pipeline_full.py measures the compiled result and re-prunes rather than
    trusting the arithmetic: the achieved rate runs 6 to 35 points deeper than
    predicted.

RECOVERY FINE-TUNING RECIPE
    Deliberately NOT the PruningBench recipe, since we start from pretrained
    weights rather than from scratch. Aligned instead with the ImageNet
    post-pruning literature (Li et al. 2016; Renda et al. 2020):

        SGD, momentum 0.9, weight decay 1e-4
        peak LR 0.01
        optional linear warmup over --warmup_epochs, then CosineAnnealingLR to 0,
            stepped per batch rather than per epoch
        batch 128, AMP fp16 (about 2.3x faster on an A6000)
        RandomResizedCrop + horizontal flip; validation on the full 50k set every
            epoch, keeping the best state

    The warmup matters at deep pruning rates: without it the first epoch sees
    very large gradients on a network that has just lost most of its channels,
    and the run can take several epochs to recover ground it need not have lost.

    Epoch budget scales with the ACHIEVED rate, not the requested one; see
    FT_BUDGET_BANDS in pipeline_full.py. Fixing the budget from the predicted
    rate was a real defect in an earlier campaign: the loop converges deeper than
    predicted, so several runs were under-trained, and because the shortfall
    correlated with architecture it biased the comparison between families.

IMPORTANCE CRITERIA
    magnitude_l2 (data-free) and taylor (data-driven, gradients over
    --taylor_batches batches before each pruning step). Taylor is markedly better
    at deep rates: on ResNet-50 at an 8 MB target, magnitude_l2 with 30 epochs
    reached 58.55 % top-1 where taylor with 60 epochs and warmup reached 73.80 %,
    against a 76.13 % baseline. Part of that gap is the longer budget, so the two
    are not cleanly separated by that comparison alone.

USAGE
    python 01_prune_imagenet.py --data_dir /datasets/Imagenet_1k \\
        --model resnet50 --target_mb 8 --ft_epochs 60 --warmup_epochs 3 \\
        --importance taylor

    # prune only, for the guided loop
    python 01_prune_imagenet.py ... --prune_only --preft_output out_PREFT.pt

    # fine-tune an existing PREFT
    python 01_prune_imagenet.py ... --ft_only --resume_from out_PREFT.pt

    # local smoke test
    python 01_prune_imagenet.py --data_dir ... --model resnet50 \\
        --target_mb 8 --ft_epochs 1 --batch_size 8 --device cpu

NOTE ON THE DATASET
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES is set at import. ImageNet contains a few
    truncated JPEGs, and without it the DataLoader raises part-way through an
    epoch, hours into a run.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_pruning as tp
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ImageNet contains a few truncated JPEGs. Without this, the DataLoader raises
# part-way through an epoch, hours into a run.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from model_zoo import (ALL_MODEL_NAMES, build_model,
                       get_input_size, get_preprocessing)


# ─────────────────────────────────────────────
# Recettes
# ─────────────────────────────────────────────
RECIPE_FT = {
    "optimizer": "SGD", "momentum": 0.9, "weight_decay": 1e-4,
    "lr": 0.01, "scheduler": "cosine_to_zero",
    "batch_size": 128,
    "augmentation": "RandomResizedCrop + HFlip",
    "ref": "Standard ImageNet pruning recovery — cf. Li 2016, He 2018, "
           "Renda 2020, Torch-Pruning examples/imagenet",
}

PRUNING_PROTOCOL = {
    "iterative_steps": 400,
    "global_pruning": True,
    "max_pruning_ratio": 0.95,    # 0.95, not the 0.9 used on the CIFAR axis:
                                  # inception_resnet_v2 and resnet152 need 86-87 %
                                  # to reach an 8 MB target, and at 0.9 the pruner
                                  # saturates against its own bound instead.
    "importance": "magnitude_l2",
    "ref": "Magnitude pruning Li et al. 2016, via Torch-Pruning MetaPruner",
}


# ─────────────────────────────────────────────
# Reproducibility and DataLoader workers
# ─────────────────────────────────────────────
def seed_everything(seed: int):
    """Seed random, numpy and torch (CPU and CUDA) from a single integer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_workers():
    """Choose a DataLoader worker count, honouring the SLURM allocation when present.

    Matters more here than on CIFAR: ImageNet decoding is JPEG-bound, so too few
    workers leaves the GPU starved and inflates every epoch-time estimate.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return max(1, int(slurm) - 1)
    try:
        return max(1, len(os.sched_getaffinity(0)) - 1)
    except AttributeError:
        return 4


# ─────────────────────────────────────────────
# ImageNet transforms. Preprocessing is per model, read from model_zoo, because
# three incompatible conventions coexist in this lineup (see model_zoo.py).
# ─────────────────────────────────────────────
class _BGR255:
    """Channel-order and range transform for BN-Inception.

    BN-Inception expects BGR in [0, 255], not RGB in [0, 1] like every other model
    here. Applying the wrong convention does not raise, it just quietly costs
    accuracy, which is easy to misread as a pruning effect.
    """
    def __call__(self, x):
        """Swap RGB to BGR and rescale to [0, 255]."""
        return x[[2, 1, 0], :, :] * 255.0


class _BGR:
    """Channel-order transform: RGB to BGR, leaving the range untouched."""
    def __call__(self, x):
        """Swap the channel order from RGB to BGR."""
        return x[[2, 1, 0], :, :]


def _build_transforms(input_size: int, pp: dict, training: bool):
    """Build the train or validation transform pipeline for one model.

    Train gets RandomResizedCrop and a horizontal flip; validation gets Resize to
    256/224 of the input size then a centre crop, which is the standard ImageNet
    evaluation protocol. Normalisation comes from model_zoo.get_preprocessing(), so
    each architecture gets its own convention.
    """
    ops = []
    if training:
        ops.append(transforms.RandomResizedCrop(
            input_size, interpolation=transforms.InterpolationMode.BICUBIC))
        ops.append(transforms.RandomHorizontalFlip())
    else:
        scale = int(round(input_size * 256 / 224))
        ops.append(transforms.Resize(scale,
                                     interpolation=transforms.InterpolationMode.BICUBIC))
        ops.append(transforms.CenterCrop(input_size))
    ops.append(transforms.ToTensor())
    if pp["input_range_255"] and pp["bgr"]:
        ops.append(_BGR255())
    elif pp["bgr"]:
        ops.append(_BGR())
    ops.append(transforms.Normalize(mean=list(pp["mean"]), std=list(pp["std"])))
    return transforms.Compose(ops)


def get_dataloaders(data_dir, batch_size, seed, model_name):
    """Build the ImageNet train and validation loaders for one model's resolution."""
    input_size = get_input_size(model_name)
    pp = get_preprocessing(model_name)
    train_tf = _build_transforms(input_size, pp, training=True)
    val_tf = _build_transforms(input_size, pp, training=False)

    train_root = Path(data_dir) / "train"
    val_root = Path(data_dir) / "val"
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError(
            f"Expected ImageNet at {data_dir}/{{train,val}}/ (ImageFolder layout).")
    train_ds = datasets.ImageFolder(str(train_root), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_root), transform=val_tf)

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
# Evaluate. Some architectures return a tuple when aux logits are present.
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate_topk(model, val_loader, device):
    """Top-1 and top-5 accuracy over the full 50 000-image ImageNet validation set."""
    model.eval()
    correct1, correct5, total = 0, 0, 0
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(images)
        if isinstance(out, tuple):
            out = out[0]
        top5 = out.topk(5, dim=1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)
    return (100.0 * correct1 / total, 100.0 * correct5 / total, total)


# ─────────────────────────────────────────────
# Train epoch + recovery FT cosine
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp):
    """Run one training epoch, stepping the scheduler once per batch.

    The scheduler is stepped per batch, not per epoch: over a 30-to-90-epoch run the
    cosine then descends over hundreds of thousands of steps instead of a few dozen,
    which is much smoother and matters on short fine-tuning schedules.

    AMP wraps the forward in torch.amp.autocast and the backward in a GradScaler.
    On an A6000 this is worth roughly 1.8-2.5x in wall-clock time for these
    convolutional models, with no measurable accuracy cost.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.float16, enabled=use_amp):
            out = model(images)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        running_loss += loss.item() * images.size(0)
        _, pred = out.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total


def recovery_finetune(model, train_loader, val_loader, device, n_epochs, label,
                      use_amp, warmup_epochs=0):
    """Recovery fine-tuning: cosine schedule, full validation each epoch, best state kept.

    With warmup_epochs > 0 the learning rate follows a composed schedule: a linear
    ramp from 0 to the peak over the warmup epochs, then a cosine decay to 0 over the
    remainder, both stepped per batch.

    The warmup is not decoration. At deep pruning rates the first epoch faces very
    large gradients on a network that has just lost most of its channels, and without
    the ramp the run can spend several epochs recovering ground it never needed to
    lose.

    Validation runs on the full 50k set every epoch and the best state is kept, rather
    than the last one, because the tail of a cosine schedule can drift slightly down.
    """
    if n_epochs <= 0:
        return None, []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=RECIPE_FT["lr"],
                          momentum=RECIPE_FT["momentum"],
                          weight_decay=RECIPE_FT["weight_decay"])

    steps_per_epoch = len(train_loader)
    warmup_epochs = max(0, min(warmup_epochs, n_epochs - 1))  # au moins 1 ep de cosine
    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * steps_per_epoch
        cosine_steps = (n_epochs - warmup_epochs) * steps_per_epoch
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=0.0)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps])
    else:
        T_max = n_epochs * steps_per_epoch
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=0.0)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    amp_str = f"AMP=fp16" if use_amp else "AMP=off"
    wu_str = f"warmup={warmup_epochs}ep + " if warmup_epochs > 0 else ""
    print(f"  [{label}] {n_epochs} ep, {wu_str}SGD lr={RECIPE_FT['lr']} cosine→0, "
          f"wd={RECIPE_FT['weight_decay']}, batch={train_loader.batch_size}, "
          f"{amp_str}", flush=True)

    best_top1, best_top5, best_state, best_epoch = 0.0, 0.0, None, -1
    history = []

    for epoch in range(n_epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, scheduler, scaler,
                                            device, use_amp)
        t_train = time.time() - t0
        t1 = time.time()
        val_top1, val_top5, n_val = evaluate_topk(model, val_loader, device)
        t_val = time.time() - t1

        improved = val_top1 > best_top1
        if improved:
            best_top1, best_top5, best_epoch = val_top1, val_top5, epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc_pct": train_acc,
            "val_top1_pct": val_top1, "val_top5_pct": val_top5, "n_val": n_val,
            "lr": optimizer.param_groups[0]["lr"],
            "duration_train_s": t_train, "duration_val_s": t_val,
        })
        star = " *best" if improved else ""
        print(f"    Ep {epoch+1:2d}/{n_epochs}  loss={train_loss:.4f}  "
              f"train={train_acc:5.2f}%  val_top1={val_top1:5.2f}%  "
              f"val_top5={val_top5:5.2f}%  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"({t_train:.0f}s + {t_val:.0f}s){star}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    print(f"  [{label}] Best : val_top1={best_top1:.2f}% / top5={best_top5:.2f}% "
          f"@ ep {best_epoch}/{n_epochs}", flush=True)
    return {"best_top1": best_top1, "best_top5": best_top5,
            "best_epoch": best_epoch}, history


# ─────────────────────────────────────────────
# Pruning utilities, mirroring those of the single-TPU axis
# ─────────────────────────────────────────────
def count_params(model):
    """Total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_macs(model, input_size):
    """Estimate multiply-accumulate operations for one forward pass at the model's
    native input resolution.
    """
    try:
        example = torch.randn(1, 3, input_size, input_size)
        macs, _ = tp.utils.count_ops_and_params(model, example)
        return macs
    except Exception:
        return 0


def get_ignored_layers(model):
    """Return the classifier layers that must be protected from pruning.

    Pruning the classifier would change the number of output classes. The five
    architecture families in this lineup name their head differently, and a name that
    fails to match here would silently leave the head prunable:
        torchvision googlenet, resnet*   m.fc
        timm inception_v3                m.fc
        pretrainedmodels bninception     m.last_linear
        timm inception_v4                m.last_linear
        timm inception_resnet_v2         m.classif
    """
    ignored = []
    for attr in ("fc", "last_linear", "classif", "classifier", "linear", "head"):
        if not hasattr(model, attr):
            continue
        m = getattr(model, attr)
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            ignored.append(m)
        elif isinstance(m, nn.Sequential):
            for layer in m:
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    ignored.append(layer)
    if not ignored:
        last = None
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                last = m
        if last:
            ignored.append(last)
    return ignored


def extract_layer_structure(model):
    """Record the output width of every Conv2d, Linear and BatchNorm2d, in order.

    Diffed against the unpruned model, this gives the per-layer pruning rate, which
    is what explains a latency result rather than merely correlating with it.
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


def progressive_pruning_to_target(model, pruner, target_pct, importance_name,
                                  train_loader=None, device=None,
                                  taylor_batches=10):
    """Prune in small steps until the target parameter reduction is reached.

    magnitude_l2 is data-free and steps directly. taylor needs gradients that reflect
    the CURRENT network, so a forward and backward pass over `taylor_batches` batches
    runs before every pruning step; reusing gradients from the original network would
    rank channels that no longer exist.

    The loop stops on the achieved reduction rather than a step count, because global
    pruning removes whole dependency groups of varying size. The achieved rate is
    therefore never exactly the requested one, and it is the achieved rate that is
    recorded and that decides the fine-tuning budget.
    """
    initial = count_params(model)
    target_params = initial * (1 - target_pct / 100)
    n_steps = 0

    def _taylor_prime():
        """Run forward and backward over a few batches to populate gradients for Taylor."""
        model.zero_grad(set_to_none=True)
        for k, (imgs, lbls) in enumerate(train_loader):
            if k >= taylor_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            loss = nn.functional.cross_entropy(model(imgs), lbls)
            loss.backward()

    while count_params(model) > target_params:
        if pruner.current_step >= pruner.iterative_steps:
            break
        if importance_name == "taylor":
            _taylor_prime()
        pruner.step()
        n_steps += 1

    actual = count_params(model)
    return n_steps, 100 * (1 - actual / initial), actual


# ─────────────────────────────────────────────
# Helpers for the split prune / fine-tune mode used by pipeline_full.py
# ─────────────────────────────────────────────
def _preft_meta_path(preft_pt_path):
    """Path of the JSON sidecar for a PREFT checkpoint (same name, .meta.json).

    Note that PREFT filenames contain a decimal point
    (..._4.47mb_iter1_PREFT.pt). The write side splits on the '.pt' suffix and
    the read side uses with_suffix('.meta.json'); the two agree, and this has
    been verified against real filenames. Do not "fix" either one.
    """
    return Path(str(preft_pt_path).rsplit(".pt", 1)[0] + ".meta.json")


def _save_preft(args, model, name, seed_suffix, pruned_dir,
                pre_params, pre_macs, pre_size_mb_est,
                actual_pct, post_params, post_size_mb_est, post_macs,
                ref_top1, ref_top5, post_top1, post_top5,
                n_steps, target_pct, t_prune, input_size):
    """Write the post-pruning, pre-fine-tuning checkpoint and its metadata sidecar.

    This is the artefact the guided loop in pipeline_full.py iterates on: it can be
    converted and compiled to see whether the model actually fits the accelerator,
    before committing to a fine-tuning run that costs tens of GPU-hours. The sidecar
    carries ref_top1, actual_pct and post_prune_top1, so the fine-tuning stage can
    resume with full context.
    """
    if args.preft_output:
        preft_out = Path(args.preft_output)
    else:
        preft_out = pruned_dir / (
            f"{name}_pruned{int(round(actual_pct))}pct_{args.importance}"
            f"_target{args.target_mb}mb{seed_suffix}_PREFT.pt")
    torch.save(model, str(preft_out))

    meta = {
        "model": name,
        "importance": args.importance,
        "target_mb": args.target_mb,
        "target_pct_computed": target_pct,
        "actual_pct": actual_pct,
        "n_pruning_steps": n_steps,
        "pre_params": pre_params,
        "post_params": post_params,
        "pre_size_mb_est": pre_size_mb_est,
        "post_size_mb_est": post_size_mb_est,
        "pre_macs": pre_macs,
        "post_macs": post_macs,
        "ref_top1_pct": ref_top1,
        "ref_top5_pct": ref_top5,
        "post_prune_top1_pct": post_top1,
        "post_prune_top5_pct": post_top5,
        "input_size": input_size,
        "seed": args.seed,
        "duration_prune_s": t_prune,
        "pruning_protocol": PRUNING_PROTOCOL,
    }
    meta_out = _preft_meta_path(preft_out)
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n  ╔══ PREFT SAVED (--prune_only) ══")
    print(f"  ║ Top-1 : {ref_top1:.2f}% → post-prune {post_top1:.2f}%")
    print(f"  ║ Params: {pre_params:,} → {post_params:,} "
          f"({actual_pct:.1f}% pruned, ~{post_size_mb_est:.1f} MB)")
    print(f"   Duration: pruning={t_prune:.0f}s")
    print(f"  ║ → {preft_out.name}")
    print(f"  ║ → {meta_out.name}")
    print(f"  ╚{'═' * 50}", flush=True)


def _load_preft_for_ft(args, device, name):
    """Load a PREFT checkpoint and its sidecar to resume in --ft_only mode.

    Returns everything the fine-tuning and logging stages need. A missing sidecar is
    not fatal: placeholders are filled in and a warning is printed, so a checkpoint
    can still be fine-tuned, with the caveat that its log will lack the reference
    accuracies.
    """
    print(f"  [--ft_only] Loading PREFT from {args.resume_from}", flush=True)
    t0 = time.time()
    model = torch.load(args.resume_from, weights_only=False, map_location=device)
    model.to(device)
    input_size = get_input_size(name)
    train_loader, val_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.seed, name)
    print(f"    loaded in {time.time()-t0:.1f}s, "
          f"params={count_params(model):,}", flush=True)

    meta_path = _preft_meta_path(args.resume_from)
    if meta_path.exists():
        with open(meta_path) as f:
            m = json.load(f)
        print(f"  [PREFT meta loaded] ref={m.get('ref_top1_pct', -1):.2f}% "
              f"post-prune={m.get('post_prune_top1_pct', -1):.2f}% "
              f"actual_pct={m.get('actual_pct', 0):.1f}%", flush=True)
        pre_params = m.get("pre_params", 0)
        pre_macs = m.get("pre_macs", 0)
        pre_size_mb_est = m.get("pre_size_mb_est", 0)
        actual_pct = m.get("actual_pct", 0)
        post_params = m.get("post_params", count_params(model))
        post_size_mb_est = m.get("post_size_mb_est", post_params / 1e6)
        post_macs = m.get("post_macs", 0)
        ref_top1 = m.get("ref_top1_pct", -1.0)
        ref_top5 = m.get("ref_top5_pct", -1.0)
        post_top1 = m.get("post_prune_top1_pct", -1.0)
        post_top5 = m.get("post_prune_top5_pct", -1.0)
        n_steps = m.get("n_pruning_steps", 0)
        target_pct = m.get("target_pct_computed", 100.0)
        t_prune = m.get("duration_prune_s", 0.0)
    else:
        print(f"  [WARN] no meta sidecar at {meta_path} — placeholders",
              flush=True)
        params = count_params(model)
        pre_params = post_params = params
        pre_size_mb_est = post_size_mb_est = params / 1e6
        pre_macs = post_macs = 0
        actual_pct = 0.0
        ref_top1 = ref_top5 = -1.0
        post_top1 = post_top5 = -1.0
        n_steps = 0
        target_pct = 100.0  # non-zero so the fine-tuning stage still runs
        t_prune = 0.0

    return (model, input_size, train_loader, val_loader,
            pre_params, pre_macs, pre_size_mb_est,
            actual_pct, post_params, post_size_mb_est, post_macs,
            ref_top1, ref_top5, post_top1, post_top5,
            n_steps, target_pct, t_prune)


# ─────────────────────────────────────────────
# One run: a model pruned to whatever rate its size target implies
# ─────────────────────────────────────────────
def run_one(args, device, pruned_dir, log_dir):
    """Run one model end to end: prune to the size target, fine-tune, evaluate, log.

    Honours --prune_only (stop after pruning, write PREFT) and --ft_only (skip
    pruning, load a PREFT and fine-tune it), which is how pipeline_full.py splits the
    work around its compile-and-measure loop.
    """
    name = args.model
    seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    imp_tag = args.importance
    run_id = args.run_tag if args.run_tag else f"target{args.target_mb}mb"
    log_path = log_dir / f"{name}_{imp_tag}_{run_id}{seed_suffix}.json"

    print(f"\n{'━' * 72}")
    print(f"  {name.upper()}  |  {imp_tag}  |  target {args.target_mb} MB  "
          f"(seed={args.seed})")
    print(f"{'━' * 72}", flush=True)

    seed_everything(args.seed)

    if args.ft_only:
        # === Mode FT-only : charger PREFT, skip 1-5 ==================
        (model, input_size, train_loader, val_loader,
         pre_params, pre_macs, pre_size_mb_est,
         actual_pct, post_params, post_size_mb_est, post_macs,
         ref_top1, ref_top5, post_top1, post_top5,
         n_steps, target_pct, t_prune) = _load_preft_for_ft(args, device, name)
    else:
        # ── 1. Charger le pretrained ImageNet ────────────────────────
        print(f"  Loading {name} pretrained ImageNet (1000 classes)…", flush=True)
        t0 = time.time()
        model = build_model(name, num_classes=1000, pretrained=True).to(device)
        pre_params = count_params(model)
        input_size = get_input_size(name)
        pre_macs = count_macs(model.cpu(), input_size); model = model.to(device)
        pre_size_mb_est = pre_params / 1e6  # int8 ≈ 1 byte/param
        print(f"    loaded in {time.time()-t0:.1f}s, params={pre_params:,} "
              f"({pre_params/1e6:.1f}M ≈ {pre_size_mb_est:.1f} MB int8), "
              f"MACs={pre_macs:,.0f}", flush=True)

        # -- 2. Work out the pruning rate the size target implies -----
        target_params = args.target_mb * 1_000_000
        if pre_params <= target_params:
            target_pct = 0.0
            print(f"  -> Model is already under the target ({pre_params/1e6:.1f}M <= "
                  f"{args.target_mb}M params). Pas de pruning, juste eval baseline.",
                  flush=True)
        else:
            target_pct = 100 * (1 - target_params / pre_params)
            print(f"  -> Target rate : {target_pct:.1f}%  "
                  f"({pre_params:,} -> {int(target_params):,} params)",
                  flush=True)

        # ── 3. Dataloaders ImageNet ──────────────────────────────────
        train_loader, val_loader = get_dataloaders(
            args.data_dir, args.batch_size, args.seed, name)

        # ── 4. Ref accuracy (pretrained, avant pruning) ──────────────
        print(f"  Eval baseline pretrained (val 50k)…", flush=True)
        t0 = time.time()
        ref_top1, ref_top5, n_ref = evaluate_topk(model, val_loader, device)
        print(f"    ref : top1={ref_top1:.2f}% top5={ref_top5:.2f}% "
              f"(n={n_ref}, {time.time()-t0:.0f}s)", flush=True)

        # -- 5. Pruning (skipped when the model is already under target) ----
        if target_pct > 0:
            if args.importance == "taylor":
                importance = tp.importance.TaylorImportance()
            else:
                importance = tp.importance.MagnitudeImportance(p=2)
            ignored_layers = get_ignored_layers(model)
            example_input = torch.randn(1, 3, input_size, input_size).to(device)
            pruner = tp.pruner.MetaPruner(
                model=model, example_inputs=example_input,
                importance=importance,
                pruning_ratio=target_pct / 100.0,
                iterative_steps=PRUNING_PROTOCOL["iterative_steps"],
                global_pruning=PRUNING_PROTOCOL["global_pruning"],
                max_pruning_ratio=PRUNING_PROTOCOL["max_pruning_ratio"],
                ignored_layers=ignored_layers,
            )
            driven = "data-driven" if args.importance == "taylor" else "data-free"
            print(f"  [Pruning:{args.importance}] {driven}, target={target_pct:.1f}%, "
                  f"iterative_steps={PRUNING_PROTOCOL['iterative_steps']}, "
                  f"max_per_layer={PRUNING_PROTOCOL['max_pruning_ratio']}"
                  + (f", taylor_batches={args.taylor_batches}" if args.importance == "taylor" else ""),
                  flush=True)
            t_p = time.time()
            n_steps, actual_pct, post_params = progressive_pruning_to_target(
                model, pruner, target_pct, args.importance,
                train_loader=train_loader, device=device,
                taylor_batches=args.taylor_batches)
            t_prune = time.time() - t_p
            post_macs = count_macs(model.cpu(), input_size); model = model.to(device)
            post_size_mb_est = post_params / 1e6
            print(f"  [Pruning] {n_steps} steps en {t_prune:.0f}s — "
                  f"{post_params:,} params ({actual_pct:.1f}% pruned, "
                  f"~{post_size_mb_est:.1f} MB int8)", flush=True)

            # Evaluate before fine-tuning. The drop is expected to be large;
            # what it measures is the quality of the mask itself, separately
            # from how much of it fine-tuning can recover.
            t0 = time.time()
            post_top1, post_top5, _ = evaluate_topk(model, val_loader, device)
            print(f"    post-prune : top1={post_top1:.2f}% top5={post_top5:.2f}% "
                  f"(drop {ref_top1-post_top1:+.2f} pts, {time.time()-t0:.0f}s)",
                  flush=True)
        else:
            n_steps, actual_pct, post_params = 0, 0.0, pre_params
            post_macs = pre_macs
            post_size_mb_est = pre_size_mb_est
            t_prune = 0.0
            post_top1, post_top5 = ref_top1, ref_top5

        # ── 5b. Mode --prune_only : sauvegarde PREFT + exit ──────────
        if args.prune_only:
            _save_preft(args, model, name, seed_suffix, pruned_dir,
                        pre_params, pre_macs, pre_size_mb_est,
                        actual_pct, post_params, post_size_mb_est, post_macs,
                        ref_top1, ref_top5, post_top1, post_top5,
                        n_steps, target_pct, t_prune, input_size)
            return

    # ── 6. Recovery FT cosine ────────────────────────────────────────────
    if args.ft_epochs > 0 and target_pct > 0:
        # AMP is on by default when CUDA is available: about 2x faster on an
        # A6000 with no measurable accuracy cost. --no_amp disables it.
        use_amp = (device.type == "cuda") and (not args.no_amp)
        t_ft = time.time()
        best_summary, ft_history = recovery_finetune(
            model, train_loader, val_loader, device, args.ft_epochs,
            label=f"Recovery FT @{actual_pct:.1f}%",
            use_amp=use_amp, warmup_epochs=args.warmup_epochs)
        t_ft = time.time() - t_ft
        final_top1 = best_summary["best_top1"]
        final_top5 = best_summary["best_top5"]
        best_epoch = best_summary["best_epoch"]
    else:
        ft_history = []; t_ft = 0.0
        final_top1, final_top5, best_epoch = ref_top1, ref_top5, 0

    # -- 7. Save the model and write the log ----------------------------
    if target_pct > 0:
        out_path = pruned_dir / (
            f"{name}_pruned{int(round(actual_pct))}pct_{args.importance}"
            f"{seed_suffix}.pt")
        torch.save(model, str(out_path))
        out_file = str(out_path)
    else:
        out_file = None  # pas de save pour googlenet "baseline only"

    layer_structure = extract_layer_structure(model)

    full_log = {
        "model": name, "importance": args.importance,
        "warmup_epochs": args.warmup_epochs,
        "ft_epochs": args.ft_epochs,
        "target_mb": args.target_mb, "target_pct_computed": target_pct,
        "actual_pct": actual_pct, "n_pruning_steps": n_steps,
        "pre_params": pre_params, "post_params": post_params,
        "pre_size_mb_est": pre_size_mb_est, "post_size_mb_est": post_size_mb_est,
        "pre_macs": pre_macs, "post_macs": post_macs,
        "ref_top1_pct": ref_top1, "ref_top5_pct": ref_top5,
        "post_prune_top1_pct": post_top1, "post_prune_top5_pct": post_top5,
        "final_top1_pct": final_top1, "final_top5_pct": final_top5,
        "best_epoch": best_epoch,
        "acc_delta_top1": final_top1 - ref_top1,
        "input_size": input_size, "seed": args.seed,
        "device": str(device),
        "duration_s": {"pruning": t_prune, "finetune": t_ft},
        "recipe_ft": RECIPE_FT, "pruning_protocol": PRUNING_PROTOCOL,
        "output_file": out_file,
        "layer_structure_post": layer_structure,
        "ft_history": ft_history,
    }
    with open(log_path, "w") as f:
        json.dump(full_log, f, indent=2, default=str)

    print(f"\n  == DONE ==")
    print(f"  ║ Top-1 : {ref_top1:.2f}% → {final_top1:.2f}% "
          f"(Δ={final_top1-ref_top1:+.2f})")
    print(f"  ║ Params: {pre_params:,} → {post_params:,} "
          f"({actual_pct:.1f}% pruned, ~{post_size_mb_est:.1f} MB)")
    if pre_macs > 0:
        macs_red = 100 * (1 - post_macs / pre_macs)
        print(f"   MACs : {pre_macs:,.0f} -> {post_macs:,.0f} ({macs_red:.1f}% removed)")
    print(f"   Duration: pruning={t_prune:.0f}s, fine-tune={t_ft/60:.1f}min")
    if out_file:
        print(f"  ║ → {Path(out_file).name}")
    print(f"  ║ → {log_path.name}")
    print(f"  ╚{'═' * 50}", flush=True)


def main():
    """Parse arguments, resolve the device, and run the requested model."""
    parser = argparse.ArgumentParser(
        description="multi_tpu/models — pruning ImageNet size-targeted (magnitude_l2)")
    parser.add_argument("--data_dir", required=True,
                        help="Root ImageNet : contient train/ et val/ en layout ImageFolder")
    parser.add_argument("--model", required=True, choices=ALL_MODEL_NAMES)
    parser.add_argument("--target_mb", type=float, default=8.0,
                        help="INT8 size target in MB (roughly params/1e6). Default 8, one Edge TPU's SRAM.")
    parser.add_argument("--importance", type=str, default="magnitude_l2",
                        choices=["magnitude_l2", "taylor"],
                        help="Importance criterion for structured pruning.")
    parser.add_argument("--taylor_batches", type=int, default=10,
                        help="Nombre de batches fwd+bwd avant chaque pruner.step() "
                             "for Taylor, which is data-driven. Ignored for magnitude_l2.")
    parser.add_argument("--ft_epochs", type=int, default=30,
                        help="Recovery fine-tuning epochs (default 30, cosine to 0)")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Linear warmup epochs from 0 to the peak LR before the cosine decay "
                             "(default 0; use 3-5 for runs of 60+ epochs at rates above 60%%, "
                             "where the first epoch otherwise faces very large gradients)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pruned_dir", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP/fp16. Useful as a control, or if numerical instability "
                             "is suspected.")
    parser.add_argument("--prune_only", action="store_true",
                        help="Stop after pruning and the post-prune evaluation, saving a "
                             "checkpoint PREFT.pt et exit avant le FT. "
                             "Used by pipeline_full.py for its prune -> convert -> compile loop, "
                             "which is guided by the compiler's off-chip figure.")
    parser.add_argument("--ft_only", action="store_true",
                        help="Skip pruning and load an existing PREFT checkpoint via "
                             "--resume_from et lance directement le FT recovery.")
    parser.add_argument("--resume_from", default=None,
                        help="Path of the PREFT checkpoint to load for --ft_only.")
    parser.add_argument("--preft_output", default=None,
                        help="Explicit path for the PREFT checkpoint "
                             "in --prune_only mode (default: derived automatically).")
    parser.add_argument("--run_tag", default=None,
                        help="Run identifier substituted for 'target<X>mb' in the log filename "
                             "(e.g. 'N6'). Required when several segment counts share the "
                             "same initial target_mb, which would otherwise collide. "
                             "— cf. sweep_multiN.slurm.")
    args = parser.parse_args()

    if args.prune_only and args.ft_only:
        parser.error("--prune_only and --ft_only are mutually exclusive")
    if args.ft_only and not args.resume_from:
        parser.error("--ft_only requires --resume_from PATH")

    BASE_DIR = Path(__file__).parent
    pruned_dir = Path(args.pruned_dir) if args.pruned_dir else BASE_DIR / "pytorch_pruned_imagenet"
    log_dir = Path(args.log_dir) if args.log_dir else BASE_DIR / "pruning_logs_imagenet"
    pruned_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("multi_tpu/models — PRUNING IMAGENET (size-targeted, magnitude_l2)")
    print("=" * 72)
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Model      : {args.model}")
    print(f"  Target     : {args.target_mb} MB (about {args.target_mb}M params at 1 byte each)")
    print(f"  FT epochs  : {args.ft_epochs} (SGD lr={RECIPE_FT['lr']}, cosine to 0)")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Seed       : {args.seed}")
    print(f"  Out pruned : {pruned_dir}")
    print(f"  Out logs   : {log_dir}")
    print("=" * 72, flush=True)

    try:
        run_one(args, device, pruned_dir, log_dir)
    except Exception as e:
        print(f"\n[FAILED] {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
pipeline_full.py — prune until the model actually fits N Edge TPUs, then recover.

WHAT THIS PRODUCES
    For one (model, size target):
        pruning_logs_imagenet/<model>_<run_id>_pipeline_summary.json
            every loop iteration, the achieved rate, the winning PREFT, the
            fine-tuning budget that was derived, and the final result
        pruning_logs_imagenet/<model>_<importance>_<run_id>.json
            the fine-tuning log, with its full per-epoch history
        plus the intermediate PREFT checkpoints, INT8 models and compiled
        segments written by the three scripts it drives

WHY A LOOP AND NOT A SINGLE PASS
    Pruning to a parameter count does not guarantee the result fits the
    accelerator. The compiler consistently reports more bytes than the weight
    count implies, and the gap grows as the model shrinks: a per-architecture
    fixed overhead (fused batch-norm constants, int32 biases, per-tensor
    alignment) that does not shrink with pruning. Regressed over the loop
    iterations of the full sweep:

        model                  slope (MiB/MiB)      fixed overhead
        ResNet-101             0.962 +/- 0.013      2.29 +/- 0.36 MiB
        Inception-V4           1.027 +/- 0.011      2.86 +/- 0.32 MiB
        Inception-ResNet-V2    0.970 +/- 0.009      5.24 +/- 0.28 MiB

    The slope confirms one byte per weight; the constant is what a naive
    prediction misses. For Inception-ResNet-V2 at one segment, of roughly 6.1 MiB
    that fit in SRAM, 3.9 MiB are that overhead and only about 2.2 MiB are left
    for weights, which is why it converges at 95 % pruned rather than the 86 %
    the arithmetic suggests. Across the sweep the loop lands 6 to 35 points
    deeper than predicted.

    So the loop measures instead of predicting: prune, quantize, compile, read
    the compiler's off-chip figure, and if anything is still streaming, prune by
    that much again plus a margin.

THE ALGORITHM
    target_mb = target_mb_max - 0.1        # small initial margin
    for iteration in 1..max_iters:
        [1] 01_prune_imagenet.py --prune_only --target_mb=$target_mb
              -> PREFT.pt + .meta.json
        [2] 02_convert_pruned.py            -> INT8 .tflite
        [3] 03_compile_edgetpu_segments.py --num_segments N
              -> compile_report.json
        if compilation failed, or produced the wrong number of segments:
            stop, NOGO
        if totals.off_chip_used_mb == 0.0:
            the model fits; break
        target_mb -= off_chip_used_mb + 0.5    # 0.5 MiB of margin
        if target_mb < 1.0 or iteration > 8:
            stop, NOGO (safety net)
    [3b] read actual_pct from the winning PREFT's sidecar and derive the
         fine-tuning budget from it
    [4] 01_prune_imagenet.py --ft_only --resume_from <winning PREFT>

    The fit criterion is off_chip == 0 exactly, not "small". Anything above zero
    is re-streamed on every single inference, which is precisely the regime this
    work exists to avoid.

    Convergence takes 2 to 3 iterations, roughly 15-20 minutes each. The
    fine-tuning that follows takes 18 to 90 hours.

WHY THE FINE-TUNING BUDGET IS DERIVED AT STEP [3b] AND NOT UP FRONT
    This is the reason the script has a step [3b] at all. In an earlier campaign
    the epoch budget was fixed in advance from the PREDICTED rate. Since the loop
    converges much deeper than predicted, several runs were badly under-trained.
    Worse, the shortfall was not random: it correlated with architecture, hitting
    the Inception models in two cases out of three, which is exactly the direction
    that flatters the ResNets and contaminates any claim comparing the two
    families' robustness to pruning.

    Step [3b] is the first moment the achieved rate is known, so that is where
    the budget is decided, by reading actual_pct from the winning PREFT's sidecar
    and looking it up in FT_BUDGET_BANDS.

A PROPERTY WORTH KNOWING: off_chip IS NOT MONOTONIC IN N
    More segments does not always mean a better fit. A ResNet-101 checkpoint that
    fits at 6 segments overflows by 2.02 MiB at 7 and fits again at 8. The
    compiler's segmentation heuristic balances on some other criterion and does
    not optimise for fitting, so "more segments can never be worse" is false.

USAGE
    python pipeline_full.py \\
        --model resnet50 --target_mb 16 --importance taylor \\
        --data_dir /datasets/Imagenet_1k \\
        --pruned_dir pytorch_pruned_imagenet \\
        --int8_dir tflite_int8_pruned \\
        --edgetpu_dir edgetpu_compiled_pruned \\
        --epochs_from_actual --run_tag N2

    --epochs_from_actual enables step [3b]; without it, --ft_epochs and
    --warmup_epochs are used as given, which is the older behaviour.
    --run_tag is required when several segment counts share one initial
    target_mb, since their filenames would otherwise collide.

REQUIREMENTS
    edgetpu_compiler must be on PATH. On a cluster without root it is typically
    unpacked into $HOME/local, and the job script must put that on PATH itself,
    since it resets the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Default mapping from size target to segment count: each Edge TPU holds about
# 8 MB in SRAM, so a model targeting N x 8 MB needs N segments to be held on-chip.
DEFAULT_SEG_MAP = {8: 1, 16: 2, 24: 3, 32: 4, 40: 5, 48: 6, 56: 7, 64: 8}

# Recovery fine-tuning budget as a function of the ACHIEVED pruning rate, never
# the requested one. Each entry is (inclusive lower bound in %, epochs, warmup
# epochs); the last band whose bound is at or below the achieved rate wins.
#
# Deeper cuts need longer budgets because they destroy more of the learned
# function: post-prune top-1 falls from around 6 % at a 45 % rate to the 0.1 %
# chance level beyond 80 %. The warmup grows alongside, absorbing the large
# gradients of the first epoch after a deep cut.
FT_BUDGET_BANDS = [
    (0.0,  15, 1),
    (10.0, 20, 2),
    (30.0, 40, 2),
    (45.0, 60, 3),
    (70.0, 75, 4),
    (85.0, 90, 5),
]


def ft_budget_for(actual_pct: float) -> tuple[int, int]:
    """Return (ft_epochs, warmup_epochs) for an ACHIEVED pruning rate.

    Picks the last band whose lower bound is at or below actual_pct. Deeper cuts get
    longer budgets because they destroy more of the learned function: post-prune
    top-1 falls from around 6 % at a 45 % rate to the 0.1 % chance level beyond 80 %,
    and recovering from that takes more epochs. The warmup grows with it, to absorb
    the large gradients of the first epoch after a deep cut.

    This must be called with the achieved rate, never the requested one. Using the
    requested rate under-trains the deepest runs, and because the error correlates
    with architecture it biases comparisons between model families.
    """
    ep, wu = FT_BUDGET_BANDS[0][1], FT_BUDGET_BANDS[0][2]
    for lo, e, w in FT_BUDGET_BANDS:
        if actual_pct >= lo:
            ep, wu = e, w
    return ep, wu


def infer_num_segments(target_mb: float) -> int:
    """Derive the segment count from the size target, at about 8 MB of SRAM per TPU.

    A target of N x 8 MB needs N segments to be held on-chip across N devices.
    """
    key = int(round(target_mb))
    if key in DEFAULT_SEG_MAP:
        return DEFAULT_SEG_MAP[key]
    # Fallback : approx ceil(target_mb / 8)
    import math
    return max(1, math.ceil(target_mb / 8))


def run_step(cmd, label):
    """Run one pipeline stage as a subprocess; return (exit_code, duration_s).

    Stages run as separate processes because they need different things: pruning
    wants a GPU and torch, compiling wants the Edge TPU binary, and a crash in
    one must not take the orchestrator with it.
    """
    print(f"\n{'━' * 72}")
    print(f"  [{label}] {' '.join(str(c) for c in cmd)}")
    print(f"{'━' * 72}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd)
    dur = time.time() - t0
    print(f"  [{label}] exit={res.returncode}, {dur:.0f}s", flush=True)
    return res.returncode, dur


def load_report(report_path: Path) -> dict:
    """Load a compiler report produced by 03_compile_edgetpu_segments.py."""
    with open(report_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    """Run the guided loop for one (model, target), then launch the recovery fine-tune.

    Steps [1] to [3] iterate until the compiler reports zero off-chip bytes; [3b]
    derives the fine-tuning budget from the achieved rate; [4] fine-tunes the
    winning checkpoint. Every iteration is recorded in the pipeline summary JSON, so
    the convergence path is auditable after the fact.
    """
    p = argparse.ArgumentParser(
        description="Orchestrateur prune → convert → compile (boucle off_chip) → FT")
    # Model and data selection
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", required=True,
                   help="Root ImageNet (contient train/ et val/)")
    # Cible + segments
    p.add_argument("--target_mb", type=float, required=True,
                   help="Upper INT8 size target in MB (8/16/32/64), one per multi-TPU configuration")
    p.add_argument("--num_segments", type=int, default=None,
                   help="Edge TPU segment count (default derived from the target: 8->1, 16->2, 32->4, 64->8)")
    p.add_argument("--target_start_offset", type=float, default=0.1,
                   help="Initial margin below target_mb (default 0.1 MB) "
                        "against the compiler overhead")
    p.add_argument("--refine_margin_mb", type=float, default=0.5,
                   help="Extra MB subtracted alongside off_chip when tightening the target "
                        "at each iteration (default 0.5)")
    # Safety
    p.add_argument("--max_iters", type=int, default=8)
    p.add_argument("--min_target_mb", type=float, default=1.0)
    # Prune / FT hypers
    p.add_argument("--importance", default="taylor", choices=["magnitude_l2", "taylor"])
    p.add_argument("--taylor_batches", type=int, default=10)
    p.add_argument("--ft_epochs", type=int, default=60,
                   help="Fixed fine-tuning budget. Ignored when --epochs_from_actual is set.")
    p.add_argument("--warmup_epochs", type=int, default=3,
                   help="Fixed warmup. Ignored when --epochs_from_actual is set.")
    p.add_argument("--epochs_from_actual", action="store_true",
                   help="Calcule ft_epochs/warmup_epochs depuis actual_pct du "
                        "PREFT gagnant (grille FT_BUDGET_BANDS) au lieu de les "
                        "instead of the values passed in. The achieved rate is only known once "
                        "the guided loop has converged, which is why it is decided there.")
    p.add_argument("--run_tag", default=None,
                   help="Run identifier (e.g. 'N6') used in filenames in place of "
                        "'target<X>mb'. Required when two segment counts share the "
                        "same initial target_mb, whose filenames would collide.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")
    # Chemins
    p.add_argument("--pruned_dir", default=None)
    p.add_argument("--int8_dir", default=None)
    p.add_argument("--edgetpu_dir", default=None)
    p.add_argument("--log_dir", default=None)
    p.add_argument("--staging_dir", default=None)
    # Python interpreter used for the subprocess stages
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter for the subprocess stages (default: the one running "
                        "pipeline_full.py itself)")
    args = p.parse_args()

    pruned_dir = Path(args.pruned_dir) if args.pruned_dir else BASE_DIR / "pytorch_pruned_imagenet"
    int8_dir = Path(args.int8_dir) if args.int8_dir else BASE_DIR / "tflite_int8_pruned"
    edgetpu_dir = Path(args.edgetpu_dir) if args.edgetpu_dir else BASE_DIR / "edgetpu_compiled_pruned"
    log_dir = Path(args.log_dir) if args.log_dir else BASE_DIR / "pruning_logs_imagenet"
    staging_dir = Path(args.staging_dir) if args.staging_dir else int8_dir / "_staging"
    for d in (pruned_dir, int8_dir, edgetpu_dir, log_dir, staging_dir):
        d.mkdir(parents=True, exist_ok=True)

    num_segments = args.num_segments or infer_num_segments(args.target_mb)
    run_id = args.run_tag if args.run_tag else f"target{args.target_mb}mb"

    # Session summary, written whatever the outcome
    session_summary = {
        "model": args.model,
        "run_tag": args.run_tag,
        "target_mb_max": args.target_mb,
        "num_segments": num_segments,
        "importance": args.importance,
        "epochs_from_actual": args.epochs_from_actual,
        "ft_epochs": args.ft_epochs,
        "warmup_epochs": args.warmup_epochs,
        "iterations": [],
        "final_preft": None,
        "fit_success": False,
        "reason": "",
    }
    session_path = log_dir / f"{args.model}_{run_id}_pipeline_summary.json"

    def save_session():
        """Write the session summary to disk.

        Called after every loop iteration, so an interrupted run still records how far
        the guided loop got and why it stopped.
        """
        with open(session_path, "w") as f:
            json.dump(session_summary, f, indent=2, default=str)

    print(f"\n{'═' * 72}")
    print(f"  pipeline_full — {args.model.upper()}")
    print(f"  target_mb_max = {args.target_mb} MB  →  {num_segments} segment(s)")
    budget_mode = ("derived from the achieved rate via FT_BUDGET_BANDS" if args.epochs_from_actual
                   else f"fixe {args.ft_epochs} ep / warmup {args.warmup_epochs}")
    print(f"  importance = {args.importance}, budget FT = {budget_mode}")
    print(f"  data_dir = {args.data_dir}")
    print(f"  summary → {session_path}")
    print(f"{'═' * 72}", flush=True)
    save_session()

    # --- The guided prune / convert / compile loop ----------------------
    current_tm = args.target_mb - args.target_start_offset
    winning_preft = None
    for it in range(1, args.max_iters + 1):
        if current_tm < args.min_target_mb:
            session_summary["reason"] = (
                f"target_mb {current_tm:.2f} < min_target_mb "
                f"{args.min_target_mb} — abandon.")
            break

        print(f"\n{'#' * 72}")
        print(f"  ITERATION {it}/{args.max_iters}  --  target_mb = {current_tm:.2f}")
        print(f"{'#' * 72}", flush=True)

        # Name each attempt with an iterN suffix, so the loop's successive candidates
        # remain distinguishable after the fact
        iter_tag = f"iter{it}"
        preft_path = pruned_dir / (
            f"{args.model}_{args.importance}_{run_id}_{current_tm:.2f}mb"
            f"_{iter_tag}_PREFT.pt")

        # ─── [1] Pruning ────────────────────────────────────────────
        cmd1 = [
            args.python, str(BASE_DIR / "01_pruning_imagenet.py"),
            "--data_dir", args.data_dir,
            "--model", args.model,
            "--target_mb", f"{current_tm:.4f}",
            "--importance", args.importance,
            "--taylor_batches", str(args.taylor_batches),
            "--batch_size", str(args.batch_size),
            "--seed", str(args.seed),
            "--pruned_dir", str(pruned_dir),
            "--log_dir", str(log_dir),
            "--prune_only",
            "--preft_output", str(preft_path),
        ]
        if args.run_tag:
            cmd1 += ["--run_tag", args.run_tag]
        if args.no_amp:
            cmd1.append("--no_amp")
        rc, dur = run_step(cmd1, f"prune iter{it}")
        if rc != 0:
            session_summary["reason"] = f"pruning failed (iter {it}, exit {rc})"
            break

        # ─── [2] Convert TFLite int8 ─────────────────────────────────
        int8_path = int8_dir / f"{preft_path.stem}_int8.tflite"
        cmd2 = [
            args.python, str(BASE_DIR / "02_convert_pruned.py"),
            "--input", str(preft_path),
            "--model", args.model,
            "--data_dir", args.data_dir,
            "--output_dir", str(int8_dir),
            "--num_calib", "100",
            "--seed", str(args.seed),
            "--staging_dir", str(staging_dir),
        ]
        rc, dur = run_step(cmd2, f"convert iter{it}")
        if rc != 0:
            session_summary["reason"] = f"conversion failed (iter {it}, exit {rc})"
            break

        # ─── [3] Compile Edge TPU ────────────────────────────────────
        report_path = edgetpu_dir / f"{preft_path.stem}_compile_report.json"
        seg_out_dir = edgetpu_dir / f"{preft_path.stem}_segments"
        seg_out_dir.mkdir(parents=True, exist_ok=True)
        cmd3 = [
            args.python, str(BASE_DIR / "03_compile_edgetpu_multitpu.py"),
            "--input", str(int8_path),
            "--num_segments", str(num_segments),
            "--output_dir", str(seg_out_dir),
            "--report_out", str(report_path),
            "--timeout_sec", "600",
        ]
        rc, dur = run_step(cmd3, f"compile iter{it}")
        # A non-zero return code means either "does not fit" or "compile failed";
        # both are handled below by reading the report rather than by the code alone.
        if not report_path.exists():
            session_summary["reason"] = (
                f"compile did not produce report (iter {it}, exit {rc})")
            break

        report = load_report(report_path)
        off_chip = report["totals"]["off_chip_used_mb"]
        on_chip = report["totals"]["on_chip_used_mb"]
        segs_ok = (report["num_segments_produced"] == num_segments)
        compile_ok = report["compile_success"]

        iter_record = {
            "iter": it, "target_mb": current_tm,
            "preft": str(preft_path),
            "int8_tflite": str(int8_path),
            "compile_report": str(report_path),
            "compile_success": compile_ok,
            "segments_produced": report["num_segments_produced"],
            "on_chip_mb": on_chip,
            "off_chip_mb": off_chip,
            "duration_s_prune": dur,  # approximate: the last subprocess's own measurement
        }
        session_summary["iterations"].append(iter_record)
        save_session()

        print(f"\n  [iter {it} verdict] compile_ok={compile_ok}, "
              f"segs={report['num_segments_produced']}/{num_segments}, "
              f"on_chip={on_chip:.2f} MB, off_chip={off_chip:.3f} MB", flush=True)

        # Analyse verdict
        if not compile_ok or not segs_ok:
            session_summary["reason"] = (
                f"compile failed unexpectedly (iter {it}, "
                f"success={compile_ok}, segs={report['num_segments_produced']})")
            break

        if off_chip == 0.0:
            # The model fits: nothing is streamed from off-chip.
            winning_preft = preft_path
            session_summary["fit_success"] = True
            session_summary["reason"] = f"fit at iteration {it}, target={current_tm:.2f} MB"
            session_summary["final_preft"] = str(winning_preft)
            save_session()
            print(f"\n  FIT FOUND: off_chip == 0 at target={current_tm:.2f} MB. "
                  f"Passe au FT.", flush=True)
            break

        # Otherwise, tighten the target by what overflowed, plus a margin
        reduction = off_chip + args.refine_margin_mb
        next_tm = current_tm - reduction
        print(f"\n  off_chip={off_chip:.3f} MB, tightening the target by "
              f"{reduction:.3f} MB : {current_tm:.2f} → {next_tm:.2f}",
              flush=True)
        current_tm = next_tm

    else:
        # Iteration budget exhausted without a fit
        session_summary["reason"] = (
            f"max_iters {args.max_iters} reached without a fit")

    save_session()

    if not session_summary["fit_success"]:
        print(f"\n[pipeline_full] NOGO — {session_summary['reason']}", flush=True)
        sys.exit(2)

    # --- [3b] Fine-tuning budget, derived from the ACHIEVED rate ---------
    # This is the first point at which the achieved rate is known: the loop
    # determines it by reading the compiler's off_chip figure, and it exceeds the
    # naive "bytes ~ parameters" prediction every time, by 6 to 35 points across
    # the sweep. Fixing the budget from the target produced under-trained runs,
    # and since the error correlates with architecture it biased the comparison
    # between model families. So the budget is decided here, after the fact.
    ft_epochs, warmup_epochs = args.ft_epochs, args.warmup_epochs
    if args.epochs_from_actual:
        meta_path = winning_preft.with_suffix(".meta.json")
        try:
            with open(meta_path) as f:
                actual_pct = float(json.load(f)["actual_pct"])
        except Exception as exc:
            print(f"\n[pipeline_full] NOGO — sidecar illisible ({meta_path}) : "
                  f"{exc}. Cannot derive the fine-tuning budget.", flush=True)
            session_summary["reason"] = f"meta sidecar unreadable: {exc}"
            save_session()
            sys.exit(3)
        ft_epochs, warmup_epochs = ft_budget_for(actual_pct)
        session_summary["actual_pct"] = actual_pct
        session_summary["ft_epochs"] = ft_epochs
        session_summary["warmup_epochs"] = warmup_epochs
        save_session()
        print(f"\n  [budget FT] actual_pct = {actual_pct:.2f}%  →  "
              f"{ft_epochs} epochs, warmup {warmup_epochs}", flush=True)

    # --- [4] Recovery fine-tuning, from the PREFT that won the loop ------
    print(f"\n{'═' * 72}")
    print(f"  RECOVERY FINE-TUNING from {winning_preft.name}")
    print(f"{'═' * 72}", flush=True)

    cmd4 = [
        args.python, str(BASE_DIR / "01_pruning_imagenet.py"),
        "--data_dir", args.data_dir,
        "--model", args.model,
        "--target_mb", f"{args.target_mb}",   # the initial target, recorded for reference only
        "--importance", args.importance,
        "--ft_epochs", str(ft_epochs),
        "--warmup_epochs", str(warmup_epochs),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--pruned_dir", str(pruned_dir),
        "--log_dir", str(log_dir),
        "--ft_only",
        "--resume_from", str(winning_preft),
    ]
    if args.run_tag:
        cmd4 += ["--run_tag", args.run_tag]
    if args.no_amp:
        cmd4.append("--no_amp")
    rc, dur = run_step(cmd4, "recovery FT")
    session_summary["ft_exit_code"] = rc
    session_summary["ft_duration_s"] = dur
    save_session()

    if rc != 0:
        print(f"\n[pipeline_full] fine-tuning failed (exit {rc})", flush=True)
        sys.exit(3)

    print(f"\n[pipeline_full] DONE. Session → {session_path}", flush=True)


if __name__ == "__main__":
    main()

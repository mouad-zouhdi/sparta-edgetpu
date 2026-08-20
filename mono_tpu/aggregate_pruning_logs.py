#!/usr/bin/env python3
"""
aggregate_pruning_logs.py — consolidate the per-run pruning logs.

WHAT THIS PRODUCES
    fp32_accuracy.json   FP32 top-1/top-5 with bootstrap 95 % intervals, per model
    layer_sparsity.json  the per-layer channel structure of every baseline and
                         every pruned model

    Both are read by the benchmark and the analysis scripts. fp32_accuracy.json
    in particular is what lets 04_benchmark.py skip recomputing FP32 accuracy on
    the Raspberry Pi, where the full grid would take about five days of CPU time
    and produce the same numbers as the GPU already did.

WHY layer_sparsity.json MATTERS
    It holds the surviving channel count of every layer, for the baselines and
    for each pruned model. Diffing the two gives the per-layer pruning rate, and
    that is what turns a latency observation into an explanation: two criteria at
    equal overall sparsity can distribute it very differently across layers, and
    that distribution is what changes the memory regime, hence the latency.
    Without this file, the relationship between criterion and speedup stays a
    correlation.

HOW IT WORKS, AND WHY IT IS FAST
    Everything needed is already written into the per-run JSON logs by 01_prune.py
    at the time of the run. This script therefore loads NO pruned checkpoint: it
    reads the logs, and opens only the baselines, to record their reference
    structure. Seconds instead of hours, and no GPU needed.

    Per-run logs carrying fp32_test_top1_pct and layer_structure_post are merged
    in. Older logs written before those fields existed are skipped silently, and
    the number skipped is reported at the end.

USAGE
    python aggregate_pruning_logs.py

    python aggregate_pruning_logs.py \\
        --logs_dir ./pruning_logs --models_dir ./models \\
        --fp32_output ./fp32_accuracy.json --layer_output ./layer_sparsity.json

    # merge into existing JSON files rather than rewriting them, which is what
    # you want when adding a new batch of runs to an already-published set
    python aggregate_pruning_logs.py --merge
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Imports pour torch.load(weights_only=False) sur les baselines custom
sys.path.insert(0, str(Path(__file__).parent))
import cifar_resnet  # noqa: F401
import cifar_vgg     # noqa: F401
import wrn           # noqa: F401


# Naming convention, inherited from 01_prune.py:
#   baseline  models/<name>.pt        -> key `<name>_finetuned`
#   pruned    <name>_pruned<P>pct_<imp>[_seed<N>] -> key equal to the .pt stem
# These are the same keys 04_benchmark.py uses as its model name, which is what
# lets the two files be joined without a translation table.


def extract_layer_structure(model):
    """List every Conv2d, Linear and BatchNorm2d with its output width, in order.

    This must stay byte-for-byte equivalent to the function of the same name in
    01_prune.py. The baseline structures are produced here and the pruned structures
    there, and they are only comparable if both were enumerated by identical logic:
    a divergence in traversal order or in which module types are included would
    silently misalign the diff and yield wrong per-layer sparsity.
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


def load_baseline_structure(models_dir):
    """Return {base_name: layer_structure} for every baseline found.

    Skips the .pt files that are not baselines: the sparsity-learning caches and the
    pruned checkpoints, which would otherwise be taken as their own reference.
    """
    out = {}
    for pt_path in sorted(models_dir.glob("*.pt")):
        stem = pt_path.stem
        if any(s in stem for s in ("_sparse", "_pruned")):
            continue
        try:
            # weights_only=False car les .pt contiennent l'objet model complet
            # (cf. 01_pruning.py:torch.save(model, ...)). Les classes custom
            # cifar_resnet, cifar_vgg and wrn must be importable; see the imports above.
            m = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            out[stem] = extract_layer_structure(m)
            del m
            print(f"  [baseline] {stem}: {len(out[stem])} couches Conv2d/Linear/BN extraites",
                  flush=True)
        except Exception as e:
            print(f"  [baseline] {stem}: FAILED {type(e).__name__}: {e}", flush=True)
    return out


def baseline_lookup(layers, layer_name):
    """Cherche le nb d'out_channels d'une couche dans une liste de couches."""
    for entry in layers:
        if entry["layer"] == layer_name:
            return entry["out_channels"]
    return None


def reconstruct_pruned_name(log):
    """Rebuild the pruned model's filename from the log contents.

    Follows the naming convention of 01_prune.py:run_one():
        <model>_pruned<pct>pct_<importance>[_seed<N>]
    The seed suffix appears only for seeds other than 42, which keeps every published
    filename unchanged.
    """
    base = log["model"]
    pct = log["checkpoint_pct"]
    imp = log["importance"]
    seed = log.get("seed", 42)
    seed_suffix = "" if seed == 42 else f"_seed{seed}"
    return f"{base}_pruned{pct}pct_{imp}{seed_suffix}"


def main():
    """Parse arguments, walk the per-run logs, and write the two consolidated JSON files."""
    parser = argparse.ArgumentParser(
        description="Consolidate the per-run pruning logs into two global JSON files")
    parser.add_argument("--logs_dir", default="./pruning_logs",
                        help="Directory of per-run logs (default ./pruning_logs)")
    parser.add_argument("--models_dir", default="./models",
                        help="Directory of baseline .pt files (default ./models). Opened only to "
                             "record their reference layer structure.")
    parser.add_argument("--fp32_output", default="./fp32_accuracy.json",
                        help="Chemin du JSON fp32_accuracy de sortie")
    parser.add_argument("--layer_output", default="./layer_sparsity.json",
                        help="Chemin du JSON layer_sparsity de sortie")
    parser.add_argument("--merge", action="store_true",
                        help="Keep the entries of any existing output JSON and add to them, "
                             "rather than rewriting. Use this when adding a new batch of "
                             "runs to an already-published set. "
                             "archive/compute_fp32_accuracy.py).")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    models_dir = Path(args.models_dir)
    fp32_path = Path(args.fp32_output)
    layer_path = Path(args.layer_output)

    print("=" * 70)
    print("Aggregating per-run pruning logs")
    print("=" * 70)
    print(f"  Logs dir     : {logs_dir}")
    print(f"  Models dir   : {models_dir}")
    print(f"  FP32 output  : {fp32_path}")
    print(f"  Layer output : {layer_path}")
    print(f"  Merge mode   : {args.merge}")
    print("=" * 70, flush=True)

    # ── 1. Charger les JSON existants si --merge ──────────────────────────
    existing_fp32 = {}
    existing_pruned_layers = {}
    existing_baseline_layers = {}
    if args.merge:
        if fp32_path.exists():
            try:
                existing_fp32 = json.load(open(fp32_path)).get("models", {})
                print(f"  [merge] existing fp32_accuracy.json: {len(existing_fp32)} entries",
                      flush=True)
            except Exception as e:
                print(f"  [merge] could not read fp32_accuracy.json: {e}", flush=True)
        if layer_path.exists():
            try:
                d = json.load(open(layer_path))
                existing_pruned_layers = d.get("pruned", {})
                existing_baseline_layers = d.get("baselines", {})
                print(f"  [merge] layer_sparsity.json existant : "
                      f"{len(existing_baseline_layers)} baselines + "
                      f"{len(existing_pruned_layers)} pruned", flush=True)
            except Exception as e:
                print(f"  [merge] could not read layer_sparsity.json: {e}", flush=True)

    # ── 2. Structure des baselines (chargement des .pt) ───────────────────
    print(f"\n[1/3] Extraction structure baselines depuis {models_dir}...")
    fresh_baselines = load_baseline_structure(models_dir)
    # Prefer the freshly read structure; fall back to the stored one when the
    # baseline .pt is no longer where it was.
    baselines_struct = dict(existing_baseline_layers)
    baselines_struct.update(fresh_baselines)

    # ── 3. Per-run logs ───────────────────────────────────────────────────
    print(f"\n[2/3] Lecture des per-run logs dans {logs_dir}...")
    fp32_models = dict(existing_fp32)
    pruned_layers = dict(existing_pruned_layers)
    n_added, n_skipped_old, n_failed = 0, 0, 0

    for log_path in sorted(logs_dir.glob("*.json")):
        if log_path.name == "pruning_summary.json":
            continue  # the global summary file, not a per-run log
        try:
            with open(log_path) as f:
                log = json.load(f)
        except Exception as e:
            print(f"  [WARN] {log_path.name}: {e}")
            n_failed += 1
            continue

        # Only logs that carry fp32_test_top1_pct have the fields this needs
        if "fp32_test_top1_pct" not in log:
            n_skipped_old += 1
            continue

        out_name = reconstruct_pruned_name(log)
        base = log["model"]

        # FP32 accuracy
        fp32_models[out_name] = {
            "top1_pct":     log["fp32_test_top1_pct"],
            "top5_pct":     log["fp32_test_top5_pct"],
            "top1_ci95_lo": log["fp32_test_top1_ci95_lo"],
            "top1_ci95_hi": log["fp32_test_top1_ci95_hi"],
            "top5_ci95_lo": log["fp32_test_top5_ci95_lo"],
            "top5_ci95_hi": log["fp32_test_top5_ci95_hi"],
            "n_eval":       log["n_eval_test"],
        }

        # Per-layer sparsity: diff the pruned structure against the baseline
        ref_layers = baselines_struct.get(base, [])
        pruned_struct = log["layer_structure_post"]
        per_layer = {}
        for entry in pruned_struct:
            ref_n = baseline_lookup(ref_layers, entry["layer"])
            post_n = entry["out_channels"]
            if ref_n is None:
                # A layer present in the pruned model but not in the baseline.
                # This should not happen with structural pruning, so it is never
                # silently treated as 0 % pruned: it is recorded as None, which
                # keeps the anomaly visible instead of averaging it away.
                sparsity = None
            elif ref_n == 0:
                sparsity = 0.0
            else:
                sparsity = 100.0 * (1.0 - post_n / ref_n)
            per_layer[entry["layer"]] = {
                "kind": entry["kind"],
                "ref": ref_n if ref_n is not None else post_n,
                "post": post_n,
                "sparsity_pct": sparsity,
            }
        pruned_layers[out_name] = per_layer
        n_added += 1

    print(f"  -> {n_added} per-run logs merged "
          f"({n_skipped_old} pre-fusion logs skipped, {n_failed} unreadable)", flush=True)

    # -- 4. Write the consolidated files ---------------------------------
    print(f"\n[3/3] Writing the consolidated JSON files...")
    fp32_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fp32_path, "w") as f:
        json.dump({
            "_meta": {
                "source": "aggregate_pruning_logs.py",
                "n_models": len(fp32_models),
                "logs_dir": str(logs_dir),
            },
            "models": fp32_models,
        }, f, indent=2)
    print(f"  -> {fp32_path} ({len(fp32_models)} models)")

    layer_path.parent.mkdir(parents=True, exist_ok=True)
    with open(layer_path, "w") as f:
        json.dump({
            "_meta": {
                "source": "aggregate_pruning_logs.py",
                "n_baselines": len(baselines_struct),
                "n_pruned": len(pruned_layers),
            },
            "baselines": baselines_struct,
            "pruned": pruned_layers,
        }, f, indent=2)
    print(f"  → {layer_path} ({len(baselines_struct)} baselines + "
          f"{len(pruned_layers)} pruned)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
fetch_models.py — download the measured artefacts from the Hugging Face Hub.

WHAT THIS DOES
    Downloads all or part of huggingface.co/mouad-zouhdi/sparta-edgetpu-models
    into a local directory. The full collection is 37.38 GiB across 7479 files, so
    downloading a subset is usually what you want; --list shows what is on offer
    and how big each part is.

USAGE
    python setup/fetch_models.py --list
    python setup/fetch_models.py --set axis1-edgetpu --out models/
    python setup/fetch_models.py --set axis1-edgetpu measurements --out models/
    python setup/fetch_models.py --all --out models/
    python setup/fetch_models.py --patterns 'axis1_cifar100/edgetpu/resnet18_*'

    Downloads resume: a re-run skips files already present with a matching hash,
    so an interrupted transfer costs nothing to restart.

NOTE ON READING WHAT YOU DOWNLOAD
    The .tflite files are INT8. Read them with `ai_edge_litert`, NOT with
    `tflite_runtime` 2.5: that version misreads the quantization produced by
    ai-edge-quantizer, collapsing outputs onto the zero point so that accuracy
    reads as chance, silently.

    The PyTorch checkpoints are whole-model pickles, not state dicts, because
    structured pruning changes the architecture. Load them with
    weights_only=False, and with mono_tpu/ on sys.path so that `cifar_resnet`,
    `cifar_vgg` and `wrn` resolve.
"""
from __future__ import annotations

import argparse
import sys

REPO = "mouad-zouhdi/sparta-edgetpu-models"

# Named subsets, so nobody has to remember the directory layout. Sizes are
# approximate and meant for deciding what to pull, not for accounting.
SETS: dict[str, tuple[str, list[str]]] = {
    "axis1-edgetpu": (
        "3.2 GiB, 410 files. CIFAR-100 models compiled for the Edge TPU: the "
        "binaries the single-TPU benchmarks actually ran.",
        ["axis1_cifar100/edgetpu/*"],
    ),
    "axis1-tflite": (
        "2.97 GiB, 411 files. The same models quantized but not yet compiled, "
        "for recompiling with your own edgetpu_compiler.",
        ["axis1_cifar100/tflite_int8/*"],
    ),
    "axis1-pytorch": (
        "11.75 GiB, 412 files. Baselines and pruned checkpoints, PyTorch. Only "
        "needed to re-derive the TFLite models or to inspect the masks.",
        ["axis1_cifar100/baselines/*", "axis1_cifar100/pruned_pytorch/*"],
    ),
    "axis1-logs": (
        "12 MiB, 410 files. Per-run logs: accuracies with confidence intervals, "
        "achieved pruning rates, per-layer surviving structure.",
        ["axis1_cifar100/logs/*"],
    ),
    "axis2-edgetpu": (
        "8.0 GiB, 1548 files. ImageNet models compiled across 1 to 8 segments.",
        ["axis2_imagenet/edgetpu/*"],
    ),
    "axis2-pytorch": (
        "1.2 GiB, 38 files. Final pruned ImageNet models, plus the PREFT "
        "checkpoints that won their guided loop.",
        ["axis2_imagenet/pruned_pytorch/*"],
    ),
    "axis2-logs": (
        "4 MiB, 392 files. Training logs, pipeline summaries with every guided "
        "loop iteration, and compiler reports.",
        ["axis2_imagenet/logs/*"],
    ),
    "synthetic-edgetpu": (
        "2.2 GiB, 656 files. Synthetic corpus compiled at N = 1 to 8.",
        ["synthetic/edgetpu/*"],
    ),
    "synthetic-tflite": (
        "8.0 GiB, 307 files. Synthetic corpus quantized, before compilation.",
        ["synthetic/tflite_int8/*"],
    ),
    "synthetic-meta": (
        "14 MiB, 2872 files. Structural metadata and per (model, N) compiler "
        "reports, including the 109 configurations that failed to build.",
        ["synthetic/metadata/*", "synthetic/compile_reports/*"],
    ),
    "measurements": (
        "0.12 GiB, 21 files. Every benchmark CSV. Start here: this is what the "
        "results are computed from, and it is small.",
        ["measurements/*"],
    ),
}


def print_sets() -> None:
    """Print the available subsets with their sizes and what they are for."""
    print(f"Subsets of {REPO}:\n")
    for name, (desc, _) in SETS.items():
        print(f"  {name}")
        for line in _wrap(desc, 72):
            print(f"      {line}")
        print()
    print("  --all      everything, 37.38 GiB across 7479 files")


def _wrap(text: str, width: int) -> list[str]:
    """Wrap text to a width, for the --list output."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main() -> int:
    """Resolve the requested subsets into glob patterns and download them."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--list", action="store_true", help="Show the subsets and exit")
    ap.add_argument("--set", nargs="+", metavar="NAME", help="Named subsets to fetch")
    ap.add_argument("--patterns", nargs="+", help="Explicit glob patterns instead")
    ap.add_argument("--all", action="store_true", help="Fetch everything (37.38 GiB)")
    ap.add_argument("--out", default="models", help="Destination directory")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    if args.list or not (args.set or args.patterns or args.all):
        print_sets()
        return 0

    patterns: list[str] | None = None
    if not args.all:
        patterns = list(args.patterns or [])
        for name in args.set or []:
            if name not in SETS:
                print(f"ERROR: unknown subset {name!r}. Try --list.", file=sys.stderr)
                return 2
            patterns.extend(SETS[name][1])

    from huggingface_hub import snapshot_download

    print(f"Downloading from {args.repo} into {args.out}")
    print(f"  {'everything' if patterns is None else patterns}\n")
    path = snapshot_download(
        repo_id=args.repo,
        repo_type="model",
        local_dir=args.out,
        allow_patterns=patterns,
    )
    print(f"\nDone: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
stage_models.py — assemble the measured artefacts into the Hugging Face layout.

WHAT THIS PRODUCES
    A directory tree mirroring the structure published at
    huggingface.co/mouad-zouhdi/sparta-edgetpu-models, built from the research
    directories where the artefacts were originally produced.

WHY HARD LINKS
    The collection is about 41 GB. Copying it would need 41 GB of free space on
    top of the originals, and would take as long as the upload itself. Hard links
    cost nothing: the staged tree and the original file are the same bytes on
    disk, and Hugging Face's uploader cannot tell the difference.

    The catch is that hard links only work within one filesystem. If a source
    lives on another mount, this script falls back to copying that file and says
    so, rather than silently producing a broken tree.

    Because the links share inodes with the originals, DELETING the staged tree
    is safe (it only drops one link), but EDITING a staged file would edit the
    original too. Nothing here edits them.

LAYOUT PRODUCED
    axis1_cifar100/{baselines,pruned_pytorch,tflite_int8,edgetpu,logs}/
    axis2_imagenet/{pruned_pytorch,tflite_int8,edgetpu,logs}/
    synthetic/{tflite_int8,edgetpu,metadata}/
    measurements/

USAGE
    python setup/stage_models.py --config setup/model_sources.json --out /path/to/stage
    python setup/stage_models.py ... --dry-run     # report sizes, touch nothing

    The source paths live in a JSON config rather than in this file, because they
    are specific to the machine the research ran on. Anyone reproducing this work
    will have their own layout; edit the config, not the script.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from glob import glob
from pathlib import Path


def iter_sources(patterns: list[str], root: Path) -> list[Path]:
    """Expand glob patterns into a sorted list of existing files.

    Patterns are resolved against `root` unless absolute, so a config can mix
    repository-relative and absolute paths without ambiguity.
    """
    out: list[Path] = []
    for pat in patterns:
        p = pat if os.path.isabs(pat) else str(root / pat)
        out.extend(Path(f) for f in glob(p) if os.path.isfile(f))
    return sorted(set(out))


def link_or_copy(src: Path, dst: Path, stats: dict) -> None:
    """Hard-link src to dst, falling back to a copy across filesystems.

    A pre-existing destination with the same size is left alone, which makes the
    whole staging step idempotent and cheap to re-run after adding one group.
    """
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            stats["skipped"] += 1
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        stats["linked"] += 1
    except OSError:
        # Different filesystem, or a filesystem without hard links.
        shutil.copy2(src, dst)
        stats["copied"] += 1


def stage_group(name: str, spec: dict, root: Path, out: Path,
                dry_run: bool, stats: dict) -> tuple[int, int]:
    """Stage one named group of files; return (file count, total bytes)."""
    files = iter_sources(spec["patterns"], root)
    dest_dir = out / spec["dest"]
    total = sum(f.stat().st_size for f in files)

    print(f"  {name:44s} {len(files):6d} files  {total / 2**30:7.2f} GiB  -> {spec['dest']}")
    if not files:
        print(f"    WARNING: no file matched. Check the paths in the config.")
        return 0, 0

    if not dry_run:
        for f in files:
            # Preserve one level of grouping for the segmented Edge TPU models,
            # whose per-model directories are what distinguish the segments of
            # one model from those of another.
            if spec.get("keep_parent"):
                link_or_copy(f, dest_dir / f.parent.name / f.name, stats)
            else:
                link_or_copy(f, dest_dir / f.name, stats)
    return len(files), total


def main() -> int:
    """Read the source config, stage every group, and report what was staged."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", required=True, help="JSON describing the source paths")
    ap.add_argument("--out", required=True, help="Directory to build the staged tree in")
    ap.add_argument("--root", default=".", help="Base for relative source patterns")
    ap.add_argument("--only", nargs="+", help="Stage only these group names")
    ap.add_argument("--dry-run", action="store_true", help="Report sizes, write nothing")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    stats = {"linked": 0, "copied": 0, "skipped": 0}

    groups = cfg["groups"]
    if args.only:
        missing = [g for g in args.only if g not in groups]
        if missing:
            print(f"ERROR: unknown group(s): {missing}", file=sys.stderr)
            return 2
        groups = {k: v for k, v in groups.items() if k in args.only}

    print(f"Staging into {out}{'  (dry run)' if args.dry_run else ''}\n")
    n_tot = b_tot = 0
    for name, spec in groups.items():
        n, b = stage_group(name, spec, root, out, args.dry_run, stats)
        n_tot += n
        b_tot += b

    print(f"\n  {'TOTAL':44s} {n_tot:6d} files  {b_tot / 2**30:7.2f} GiB")
    if not args.dry_run:
        print(f"  hard-linked {stats['linked']}, copied {stats['copied']}, "
              f"already present {stats['skipped']}")
        if stats["copied"]:
            print(f"  NOTE: {stats['copied']} file(s) crossed a filesystem boundary and "
                  f"were copied, which used real disk space.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

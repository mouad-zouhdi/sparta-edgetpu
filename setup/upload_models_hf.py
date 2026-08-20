#!/usr/bin/env python3
"""
upload_models_hf.py — push the staged artefact tree to the Hugging Face Hub.

WHAT THIS DOES
    Uploads a directory produced by setup/stage_models.py to a Hub repository,
    using the uploader meant for this shape of problem: tens of gigabytes across
    thousands of files, over a link that may drop.

RESUMING
    The uploader commits incrementally and keeps progress state under
    `.cache/huggingface/` inside the folder, so an interrupted run is resumed by
    re-running the same command: files already sent are skipped rather than
    re-hashed and re-uploaded. Over a domestic uplink with a 37 GiB tree, that is
    the difference between a recoverable interruption and starting over.

    `upload_folder` absorbed the large-folder path in huggingface_hub 1.x, where
    `upload_large_folder` still works but is deprecated. This script prefers the
    former and falls back to the latter on older versions, so it works either
    way without emitting a deprecation warning on current releases.

NOTE ON HARD LINKS
    stage_models.py builds the tree with hard links, so the staged files share
    inodes with the originals. Uploading reads them; nothing is modified. Do not
    add a step that rewrites a staged file in place, since that would rewrite the
    original too.

USAGE
    python setup/upload_models_hf.py --folder /path/to/stage \\
        --repo mouad-zouhdi/sparta-edgetpu-models [--private] [--workers 8]

    Authentication comes from the usual Hugging Face token resolution
    (`huggingface-cli login`, or the HF_TOKEN environment variable). The token
    needs write access.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Create the repository if needed, then upload the folder with resume."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--folder", required=True, help="Staged tree to upload")
    ap.add_argument("--repo", required=True, help="Target repo, as owner/name")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--private", action="store_true", help="Create the repo private")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel upload workers (default 8)")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"ERROR: {folder} is not a directory", file=sys.stderr)
        return 2

    api = HfApi()
    api.create_repo(args.repo, repo_type=args.repo_type,
                    private=args.private, exist_ok=True)
    print(f"Uploading {folder} -> {args.repo} ({args.repo_type})")
    print("Re-run this exact command to resume after an interruption.\n")

    common = dict(repo_id=args.repo, repo_type=args.repo_type,
                  folder_path=str(folder))
    try:
        # huggingface_hub >= 1.0: upload_folder handles large trees itself.
        api.upload_folder(**common, num_workers=args.workers, print_report=True)
    except TypeError:
        # Older releases: upload_folder has no num_workers, and the large-folder
        # path lives in its own function.
        api.upload_large_folder(**common, num_workers=args.workers,
                                print_report=True)
    print(f"\nDone: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

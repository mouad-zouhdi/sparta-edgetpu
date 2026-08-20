"""
bench_pipeline.py — pipeline benchmarks over N Edge TPUs (phases A, B and C).

WHAT THIS PRODUCES
    outputs/bench_full/N1_baseline.csv         phase A
    outputs/bench_full/pipeline_canonical.csv  phase B
    outputs/bench_full/pipeline_spread.csv     phase C

    All three share one schema:
        tag, N, perm, kind, perm_idx, throughput_fps,
        lat_ms_{mean,std,median,p95,p99,min,max},
        cold_first_lat_ms, warmup, reps, timestamp
    with kind in {baseline, canonical, random}.

THE THREE PHASES
    A  single accelerator at N=1, every model. The reference every speedup is
       measured against.
    B  the canonical TPU order at N in 2..8, every model. Phases A and B together
       already give pipeline_speedup(model, N), which is the main result.
    C  a stratified sample of about 40 models, 10 random TPU orders each. This
       was meant to quantify how much the assignment of segments to accelerators
       matters.

    They run separately rather than as one sweep because their budgets and
    purposes differ, and because phase B is worth analysing before committing the
    six hours phase C costs.

WHAT PHASE C ACTUALLY SHOWED
    Not what it was designed to show. The dispersion across TPU orderings is not
    an effect of the ordering: interleaving 990 fixed-order measurements with 990
    random-order ones in a single session gave F = 0.905, p = 0.94 on the
    variances and p = 0.45 on the means. Repeating one assignment 990 times
    disperses as much as using 990 different assignments.

    So there is no optimal assignment to search for, and the 8! permutations do
    not need exploring. What the number does measure is run-to-run repeatability:
    about 1.6 % of throughput per measurement, which means two configurations
    measured once each are only distinguishable beyond roughly 2 %.

    Two methodological notes worth keeping, since the original mistaken reading
    came from them: never compare a range against a standard deviation, and never
    estimate a within-group variance on fewer than about 20 degrees of freedom.
    Repeatability also varies between sessions (0.64 % on one day, 1.55 % on
    another), so only compare measurements taken in the same session.

PARAMETERS AND WHY
    k_perm = 10 in phase C, not the 1000 used in the single-model pilot: the
        question here is a pattern across about 40 models, not a deep
        characterisation of one.
    warmup 20, steady 200: validated in the pilot at 0.63 % drift over 3 h.
    seeds: 123 for the input, 42 for permutations and for phase B's ordering.
    No thermal cooldown by default; the pilot's drift was below measurement
        noise. Use --cooldown-sec if a short run shows more than 2 % drift.

USAGE
    python bench_pipeline.py --phase A [--filter <substr>]
    python bench_pipeline.py --phase B [--n-list 2,3,4]
    python bench_pipeline.py --phase C [--k-perm 10]
    python bench_pipeline.py --phase all

    # worker form, spawned by the orchestrator
    python bench_pipeline.py --single-cfg <tag> <N> <perm_csv>

NOTE
    Run only one instance at a time. Two concurrent processes contend for the
    accelerators and crash; check with `pgrep -af bench_pipeline` before
    launching.
"""
from __future__ import annotations

import argparse
import os
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench_utils import (  # noqa: E402
    bench_pipeline_permutation,
    bench_single_tpu,
    csv_append,
    discover_segments,
    list_available_configs,
    run_worker_subprocess,
    select_stratified_subset,
)

# ============================================================
# Defaults (tweakable via CLI)
# ============================================================
# Root of the benchmark working directory on the 8x Edge TPU host: it holds the
# compiled models, their compile reports and the output CSVs. Override with
# BENCH_ROOT rather than editing this line, or pass the individual --*-root
# options. The default is this script's grandparent, which is correct when the
# repository itself is the working directory.
DEFAULT_ROOT = Path(os.environ.get(
    "BENCH_ROOT", Path(__file__).resolve().parent.parent.parent))
DEFAULT_MODELS_ROOT = DEFAULT_ROOT / "outputs_pipeline"
DEFAULT_REPORTS_DIR = DEFAULT_MODELS_ROOT / "reports"
DEFAULT_METADATA_ROOT = DEFAULT_ROOT / "metadata"   # rsynced from a131

DEFAULT_OUT_ROOT = DEFAULT_ROOT / "outputs" / "bench_full"

WARMUP = 20
STEADY = 200
MIN_WALL_SEC = 2.0

K_PERM_C = 10           # random perms per (tag, N) in Phase C
# Per subprocess (batches all k_perm for one (tag, N) inside a single process).
# Sized for the worst case: slowest tag at N=2 (~7 fps → ~32 s per perm × 11
# perms ≈ 350 s). 1200 s gives generous headroom (up to ~5 s/perm×20 wall each).
SUBPROC_TIMEOUT = 1200

CSV_COLUMNS = [
    "tag", "N", "perm", "kind", "perm_idx",
    "throughput_fps",
    "lat_ms_mean", "lat_ms_std", "lat_ms_median",
    "lat_ms_p95", "lat_ms_p99", "lat_ms_min", "lat_ms_max",
    "cold_first_lat_ms",
    "warmup", "reps", "timestamp",
]


# ============================================================
# Worker mode — invoked as subprocess
# ============================================================
def worker_mode(tag: str, N: int, perms: list[list[int]], models_root: Path) -> None:
    """Run k measurements for (tag, N) with the given permutations.

    Each perm produces one JSON line on stdout. The last non-empty line is
    what the parent parses (so put all measurements first, then a NOTHING
    line? No — parent parses only the LAST line by design — so we print
    one JSON *array* on the last line).
    """
    segs = discover_segments(models_root, tag, N)
    results = []
    for i, perm in enumerate(perms):
        if N == 1:
            r = bench_single_tpu(segs[0], tpu_id=perm[0],
                                 warmup=WARMUP, reps=STEADY,
                                 min_wall_sec=MIN_WALL_SEC)
            r["cold_first_lat_ms"] = None
        else:
            assert len(perm) == N and sorted(perm) == list(range(N)), \
                f"invalid perm {perm} for N={N}"
            r = bench_pipeline_permutation(segs, perm, WARMUP, STEADY,
                                            min_wall_sec=MIN_WALL_SEC)
        r["perm"] = ",".join(map(str, perm))
        r["perm_idx"] = i
        results.append(r)
    print(json.dumps(results))


# ============================================================
# Orchestrator
# ============================================================
def _worker_call(tag: str, N: int, perms: list[list[int]],
                 crash_dir: Path, models_root: Path) -> list[dict]:
    """Spawn worker for (tag, N, perms). Returns list of result dicts."""
    perms_arg = ";".join(",".join(str(x) for x in p) for p in perms)
    args = ["--single-cfg", tag, str(N), perms_arg,
            "--models-root", str(models_root)]
    ret = run_worker_subprocess(
        Path(__file__), args, crash_dir, f"{tag}_N{N}",
        timeout=SUBPROC_TIMEOUT,
    )
    if isinstance(ret, dict) and ret.get("crashed"):
        # Return a stub row per requested perm so we track the failure
        return [{"crashed": True, "reason": ret.get("reason"),
                 "perm": ",".join(map(str, p)), "perm_idx": i}
                for i, p in enumerate(perms)]
    if isinstance(ret, list):
        return ret
    # Backward compat if worker returned a bare dict for 1 perm
    return [ret]


def _row_from_result(tag: str, N: int, kind: str, r: dict) -> dict:
    """Flatten one measurement result into the shared CSV row schema."""
    ts = time.time()
    base = {"tag": tag, "N": N, "kind": kind, "timestamp": ts}
    if r.get("crashed"):
        base.update({"perm": r.get("perm", ""),
                     "perm_idx": r.get("perm_idx", 0),
                     "throughput_fps": None,
                     "warmup": WARMUP, "reps": STEADY})
        return base
    base.update(r)
    return base


def phase_A(models_root: Path, reports_dir: Path, out_root: Path,
            filter_substr: str | None = None) -> None:
    """N=1 baseline on TPU 0 for every tag with a successful N=1 compile."""
    reports = list_available_configs(reports_dir)
    tags = [t for t, r in reports.items()
            if r.get(1) == "success"
            and (filter_substr is None or filter_substr in t)]
    tags.sort()
    print(f"[Phase A] {len(tags)} tags at N=1")

    csv_path = out_root / "N1_baseline.csv"
    crash_dir = out_root / "crash_logs"

    for i, tag in enumerate(tags, 1):
        t0 = time.time()
        results = _worker_call(tag, 1, [[0]], crash_dir, models_root)
        elapsed = time.time() - t0
        for r in results:
            row = _row_from_result(tag, 1, "baseline", r)
            csv_append(csv_path, row, CSV_COLUMNS)
        thr = results[0].get("throughput_fps") if results else None
        thr_s = f"{thr:8.2f} fps" if thr else "  CRASHED  "
        print(f"  [{i:3d}/{len(tags)}] {tag:45s}  {thr_s}  ({elapsed:.1f}s)",
              flush=True)


def phase_B(models_root: Path, reports_dir: Path, out_root: Path,
            n_list: list[int], filter_substr: str | None = None,
            shuffle_seed: int = 42) -> None:
    """Canonical [0..N-1] perm for every (tag, N∈n_list) with success compile."""
    reports = list_available_configs(reports_dir)
    csv_path = out_root / "pipeline_canonical.csv"
    crash_dir = out_root / "crash_logs"

    for N in n_list:
        tags = [t for t, r in reports.items()
                if r.get(N) == "success"
                and (filter_substr is None or filter_substr in t)]
        rng = random.Random(shuffle_seed + N)
        rng.shuffle(tags)
        print(f"[Phase B N={N}] {len(tags)} tags")
        canonical = list(range(N))
        for i, tag in enumerate(tags, 1):
            t0 = time.time()
            results = _worker_call(tag, N, [canonical], crash_dir, models_root)
            elapsed = time.time() - t0
            for r in results:
                row = _row_from_result(tag, N, "canonical", r)
                csv_append(csv_path, row, CSV_COLUMNS)
            thr = results[0].get("throughput_fps") if results else None
            thr_s = f"{thr:8.2f} fps" if thr else "  CRASHED  "
            print(f"  N={N} [{i:3d}/{len(tags)}] {tag:45s}  {thr_s}  ({elapsed:.1f}s)",
                  flush=True)


def phase_C(models_root: Path, reports_dir: Path, metadata_root: Path,
            out_root: Path, n_list: list[int], k_perm: int = K_PERM_C,
            seed: int = 42) -> None:
    """Stratified spread sampling: ~40 tags × N∈{2..8} × (k_perm random + 1 canonical)."""
    tags = select_stratified_subset(metadata_root, reports_dir)
    print(f"[Phase C] stratified subset: {len(tags)} tags")
    for t in tags:
        print(f"    {t}")

    csv_path = out_root / "pipeline_spread.csv"
    crash_dir = out_root / "crash_logs"
    reports = list_available_configs(reports_dir)

    for N in n_list:
        rng = random.Random(seed + N)
        canonical = list(range(N))
        for tag_i, tag in enumerate(tags, 1):
            if reports.get(tag, {}).get(N) != "success":
                print(f"  N={N} skip {tag} (compile not success)")
                continue
            # Build the perm list: canonical first, then k_perm random
            perms: list[list[int]] = [canonical]
            for _ in range(k_perm):
                p = list(range(N))
                rng.shuffle(p)
                perms.append(p)
            t0 = time.time()
            results = _worker_call(tag, N, perms, crash_dir, models_root)
            elapsed = time.time() - t0
            for j, r in enumerate(results):
                kind = "canonical" if j == 0 else "random"
                row = _row_from_result(tag, N, kind, r)
                csv_append(csv_path, row, CSV_COLUMNS)
            thrs = [r.get("throughput_fps") for r in results
                    if r.get("throughput_fps") is not None]
            if thrs:
                spread = 100 * (max(thrs) - min(thrs)) / max(thrs)
                print(f"  N={N} [{tag_i:3d}/{len(tags)}] {tag:45s}  "
                      f"spread={spread:5.2f}%  ({elapsed:.1f}s)", flush=True)
            else:
                print(f"  N={N} [{tag_i:3d}/{len(tags)}] {tag:45s}  ALL CRASHED",
                      flush=True)


# ============================================================
# CLI
# ============================================================
def main() -> int:
    """Dispatch to a phase, or to the worker path when --single-cfg is given."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    ap.add_argument("--reports-dir", type=Path, default=None,
                    help="default: <models-root>/reports")
    ap.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT,
                    help="dir containing <tag>.json metadata (Phase C)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--filter", type=str, default=None,
                    help="substring filter on tag (Phase A/B)")
    ap.add_argument("--n-list", type=str, default="2,3,4,5,6,7,8",
                    help="comma-separated N values (Phase B/C)")
    ap.add_argument("--k-perm", type=int, default=K_PERM_C,
                    help="random perms per (tag,N) in Phase C")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--single-cfg", nargs=3, metavar=("TAG", "N", "PERMS"),
                    default=None,
                    help="Worker mode. PERMS = semi-colon-separated CSV of perms, "
                         "e.g. '0,1,2;2,0,1'.")
    args = ap.parse_args()

    reports_dir = args.reports_dir or (args.models_root / "reports")

    if args.single_cfg is not None:
        tag, N_str, perms_str = args.single_cfg
        N = int(N_str)
        perms = [[int(x) for x in p.split(",")] for p in perms_str.split(";")]
        worker_mode(tag, N, perms, args.models_root)
        return 0

    n_list = [int(x) for x in args.n_list.split(",")]
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.phase in ("A", "all"):
        phase_A(args.models_root, reports_dir, args.out_root, args.filter)
    if args.phase in ("B", "all"):
        phase_B(args.models_root, reports_dir, args.out_root, n_list,
                args.filter, args.seed)
    if args.phase in ("C", "all"):
        phase_C(args.models_root, reports_dir, args.metadata_root,
                args.out_root, n_list, args.k_perm, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

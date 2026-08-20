"""
Compile the regenerated 291 synthetic tflite corpus at num_segments N ∈ {1..8}.

Input:
    regen_a131/outputs/tflite/*.tflite   (291 int8 tflite, path 'onnx2tf flatbuffer_direct')

Output tree (matches pilot_incv4 layout so `rsync -av` to dio is trivial):
    outputs_pipeline/
        N1/<tag>_edgetpu.tflite + .log
        N2/<tag>_segment_{0,1}_of_2_edgetpu.tflite + .log
        ...
        N8/<tag>_segment_{0..7}_of_8_edgetpu.tflite + .log
        reports/<tag>__N<N>.json      (per-(tag,N) rich report)
        sweep_summary.json            (rebuilt each pass)

Idempotent: skip a (tag, N) if its report already exists with
compile_status ∈ {success, refused}. Rerun failed/timeout with --retry-failed.

Intermediate `_of_N.tflite` files (uncompiled segments) are deleted after
metrics extraction to save disk (they're only useful during compile itself).

Run on a machine with `edgetpu_compiler` in PATH (v16 recommended). No coral
runtime needed — this only invokes the compiler and parses stdout/logs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

UNIT_TO_MIB = {"B": 1 / (1024 * 1024), "KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}


def parse_size(val: str, unit: str) -> float:
    """Normalise a compiler size to MiB; the compiler switches units with magnitude."""
    return float(val) * UNIT_TO_MIB.get(unit.strip(), 1.0)


def parse_log_ops(log_path: Path) -> tuple[int, int]:
    """Return (ops_tpu, ops_cpu) parsed from a single _edgetpu.log file."""
    if not log_path.exists():
        return 0, 0
    ops_tpu, ops_cpu = 0, 0
    with open(log_path) as f:
        for line in f:
            m = re.match(r"(\S+)\s+(\d+)\s+(.*)", line.strip())
            if not m or m.group(1) == "Operator":
                continue
            count = int(m.group(2))
            if "Mapped to Edge TPU" in m.group(3):
                ops_tpu += count
            else:
                ops_cpu += count
    return ops_tpu, ops_cpu


BLOCK_SPLIT_RE = re.compile(r"Started a compilation timeout timer of \d+ seconds\.")


def parse_stdout_blocks(text: str) -> list[dict]:
    """Split stdout into per-segment blocks and extract memory + timing.

    edgetpu_compiler restarts its 'Started a compilation timeout timer of X
    seconds.' banner for each segment, so we split on that marker and parse
    each chunk independently.
    """
    chunks = BLOCK_SPLIT_RE.split(text)
    # first chunk before any marker (banner/env) — drop
    blocks: list[dict] = []
    for chunk in chunks[1:]:
        b: dict = {}
        m = re.search(r"Model compiled successfully in\s+(\d+)\s*ms", chunk)
        if m:
            b["compile_ms"] = int(m.group(1))
        m = re.search(
            r"On-chip memory used for caching model parameters:\s*([\d.]+)\s*(\S+)",
            chunk,
        )
        if m:
            b["on_chip_mb"] = round(parse_size(m.group(1), m.group(2)), 3)
        m = re.search(
            r"Off-chip memory used for streaming uncached model parameters:\s*([\d.]+)\s*(\S+)",
            chunk,
        )
        if m:
            b["off_chip_mb"] = round(parse_size(m.group(1), m.group(2)), 3)
        m = re.search(r"Number of Edge TPU subgraphs:\s*(\d+)", chunk)
        if m:
            b["num_subgraphs"] = int(m.group(1))
        m = re.search(r"Total number of operations:\s*(\d+)", chunk)
        if m:
            b["total_ops"] = int(m.group(1))
        # segment_i_of_N marker in the block's Input/Output model paths
        m = re.search(r"segment_(\d+)_of_(\d+)", chunk)
        if m:
            b["segment_index"] = int(m.group(1))
            b["num_segments_stdout"] = int(m.group(2))
        if b:
            blocks.append(b)
    return blocks


REFUSED_MARKERS = (
    "recommend",         # "we recommend using N=X" style messages
    "cannot be split",
    "too small",
    "cannot subdivide",
    "cannot be segmented",
)


def is_refused(stdout_stderr: str) -> bool:
    """Detect a graceful refusal, as opposed to a failure.

    The compiler declines to split a model that is too small for the requested
    segment count, recommending a lower N. That is an expected outcome across the
    grid, not an error, and is recorded separately so the success rate is not
    understated.
    """
    low = stdout_stderr.lower()
    return any(m in low for m in REFUSED_MARKERS)


def compile_one(
    tflite_path: Path,
    out_dir: Path,
    num_segments: int,
    subproc_timeout_s: int,
    compiler_timeout_sec: int,
) -> dict:
    """Compile a single tflite at a given num_segments, return a rich dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tflite_path.stem

    cmd = ["edgetpu_compiler", "-s", "--out_dir", str(out_dir)]
    if num_segments >= 2:
        cmd += ["--num_segments", str(num_segments)]
    cmd += [f"--timeout_sec={compiler_timeout_sec}", str(tflite_path)]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=subproc_timeout_s
        )
    except FileNotFoundError:
        return {
            "compile_status": "compiler_unavailable",
            "compile_error": "edgetpu_compiler binary not found in PATH",
            "wall_seconds": None,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "compile_status": "timeout",
            "compile_error": f"subprocess timeout after {subproc_timeout_s}s "
                             f"(compiler internal timer was {compiler_timeout_sec}s)",
            "wall_seconds": round(time.time() - t0, 2),
        }

    wall = round(time.time() - t0, 2)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    # Expected segment files
    if num_segments == 1:
        expected = [out_dir / f"{stem}_edgetpu.tflite"]
    else:
        expected = [
            out_dir / f"{stem}_segment_{i}_of_{num_segments}_edgetpu.tflite"
            for i in range(num_segments)
        ]

    missing = [p for p in expected if not p.exists()]

    # Detect graceful refusal (model too small / recommends smaller N).
    # Refused = compiler explicitly declined to segment, not a crash.
    if num_segments >= 2 and missing == expected and is_refused(combined):
        return {
            "compile_status": "refused",
            "compile_error": combined[-1000:].strip(),
            "wall_seconds": wall,
        }

    if proc.returncode != 0 or missing:
        return {
            "compile_status": "failed",
            "compile_error": combined[-1500:].strip(),
            "wall_seconds": wall,
            "returncode": proc.returncode,
            "missing_files": [str(p.name) for p in missing],
        }

    # Success — collect per-segment metrics from stdout blocks + log ops.
    stdout_blocks = parse_stdout_blocks(combined)
    # Index stdout blocks by segment_index if available (only for N>=2 blocks
    # that got past the split filter and had an 'segment_i_of_N' marker).
    by_seg: dict[int, dict] = {}
    for b in stdout_blocks:
        i = b.get("segment_index")
        if i is not None:
            by_seg[i] = b

    segments = []
    for i, p in enumerate(expected):
        size_mb = round(p.stat().st_size / (1024 * 1024), 3)
        log_path = p.with_suffix(".log")
        ops_tpu, ops_cpu = parse_log_ops(log_path)
        # For N=1 there's a single block with no segment_i_of_N marker → pick
        # the last block that has memory info (should be the only one).
        if num_segments == 1:
            block = stdout_blocks[-1] if stdout_blocks else {}
        else:
            block = by_seg.get(i, {})
        segments.append({
            "i": i,
            "size_mb": size_mb,
            "on_chip_mb": block.get("on_chip_mb"),
            "off_chip_mb": block.get("off_chip_mb"),
            "num_subgraphs": block.get("num_subgraphs"),
            "num_ops_tpu": ops_tpu,
            "num_ops_cpu": ops_cpu,
            "compile_ms": block.get("compile_ms"),
            "total_ops_stdout": block.get("total_ops"),
        })

    # Totals
    totals = {
        "on_chip_mb": round(sum((s["on_chip_mb"] or 0.0) for s in segments), 3),
        "off_chip_mb": round(sum((s["off_chip_mb"] or 0.0) for s in segments), 3),
        "size_mb": round(sum(s["size_mb"] for s in segments), 3),
        "num_ops_tpu": sum(s["num_ops_tpu"] for s in segments),
        "num_ops_cpu": sum(s["num_ops_cpu"] for s in segments),
        "compile_ms": sum((s["compile_ms"] or 0) for s in segments),
    }

    return {
        "compile_status": "success",
        "wall_seconds": wall,
        "segments": segments,
        "totals": totals,
    }


def cleanup_intermediates(out_dir: Path, stem: str, num_segments: int) -> int:
    """Remove uncompiled `_of_N.tflite` intermediates (kept per-segment inputs).

    Returns number of files deleted. Only invoked on success.
    """
    if num_segments < 2:
        return 0
    n = 0
    for i in range(num_segments):
        p = out_dir / f"{stem}_segment_{i}_of_{num_segments}.tflite"
        if p.exists():
            p.unlink()
            n += 1
    return n


def load_existing_report(reports_dir: Path, tag: str, n: int) -> dict | None:
    """Load a previous report for a (tag, N), for idempotent reruns."""
    p = reports_dir / f"{tag}__N{n}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def save_report(reports_dir: Path, tag: str, n: int, report: dict) -> None:
    """Write the JSON report for one (tag, N)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    p = reports_dir / f"{tag}__N{n}.json"
    p.write_text(json.dumps(report, indent=2))


def rebuild_summary(out_root: Path) -> dict:
    """Rebuild the global summary by rescanning every per-config report."""
    reports_dir = out_root / "reports"
    summary_by_n = {n: {} for n in range(1, 9)}
    all_reports = []
    for p in sorted(reports_dir.glob("*__N*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        n = r.get("num_segments")
        st = r.get("compile_status", "unknown")
        if n in summary_by_n:
            summary_by_n[n][st] = summary_by_n[n].get(st, 0) + 1
        all_reports.append(r)
    grand = {}
    for counts in summary_by_n.values():
        for k, v in counts.items():
            grand[k] = grand.get(k, 0) + v
    summary = {
        "total_reports": len(all_reports),
        "grand_status_counts": grand,
        "by_num_segments": summary_by_n,
    }
    (out_root / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    """Sweep every (tag, N) combination, skipping those already compiled."""
    p = argparse.ArgumentParser(description="Compile 291 synth tflite × N ∈ {1..8}")
    p.add_argument("--tflite-dir", default="regen_a131/outputs/tflite",
                   help="Input dir with the 291 int8 tflite files")
    p.add_argument("--out-root", default="outputs_pipeline",
                   help="Output root (creates N1..N8/, reports/, sweep_summary.json)")
    p.add_argument("--n-list", default="1,2,3,4,5,6,7,8",
                   help="Comma-separated N values to run (default all 1..8)")
    p.add_argument("--filter", default=None,
                   help="Substring filter on tflite stem (e.g. 'sequential_d4')")
    p.add_argument("--tags", default=None,
                   help="Comma-separated explicit list of tags (overrides --filter)")
    p.add_argument("--compiler-timeout-sec", type=int, default=600,
                   help="Pass --timeout_sec to edgetpu_compiler (default 600s; "
                        "large models can push past the default 180s)")
    p.add_argument("--subproc-timeout-s", type=int, default=None,
                   help="Python subprocess timeout (default: compiler-timeout-sec + 120)")
    p.add_argument("--retry-failed", action="store_true",
                   help="Rerun reports whose compile_status is failed/timeout")
    p.add_argument("--force", action="store_true",
                   help="Rerun everything even if a success report exists")
    p.add_argument("--keep-intermediates", action="store_true",
                   help="Keep uncompiled `_of_N.tflite` files (default: deleted after success)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned (tag, N) pairs and exit")
    args = p.parse_args()

    tflite_dir = Path(args.tflite_dir).resolve()
    out_root = Path(args.out_root).resolve()
    reports_dir = out_root / "reports"
    if not tflite_dir.exists():
        print(f"[ERROR] tflite dir not found: {tflite_dir}")
        return 1
    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    for n in n_list:
        if not (1 <= n <= 8):
            print(f"[ERROR] --n-list contains invalid N={n} (allowed 1..8)")
            return 1
    subproc_timeout = args.subproc_timeout_s or (args.compiler_timeout_sec + 120)

    files = sorted(tflite_dir.glob("*.tflite"))
    if args.tags:
        wanted = {t.strip() for t in args.tags.split(",") if t.strip()}
        files = [f for f in files if f.stem in wanted]
    elif args.filter:
        files = [f for f in files if args.filter in f.stem]
    if not files:
        print(f"[ERROR] No tflite matched (dir={tflite_dir}, filter={args.filter}, tags={args.tags})")
        return 1

    # Build plan
    plan: list[tuple[Path, int]] = []
    for f in files:
        for n in n_list:
            existing = load_existing_report(reports_dir, f.stem, n)
            if existing and not args.force:
                st = existing.get("compile_status")
                if st in ("success", "refused"):
                    continue
                if not args.retry_failed and st in ("failed", "timeout"):
                    continue
            plan.append((f, n))

    total_all = len(files) * len(n_list)
    print(f"[plan] tflite files matched: {len(files)}  "
          f"N values: {n_list}  total (tag,N): {total_all}  "
          f"to run: {len(plan)}")
    if args.dry_run:
        for f, n in plan[:20]:
            print(f"  {f.stem}  N={n}")
        if len(plan) > 20:
            print(f"  ... ({len(plan) - 20} more)")
        return 0
    if not plan:
        print("[plan] nothing to do (all already reported)")
        rebuild_summary(out_root)
        return 0

    ok = failed = refused = timeout = other = 0
    for idx, (tflite, n) in enumerate(plan, 1):
        n_dir = out_root / f"N{n}"
        tag = tflite.stem
        print(f"[{idx}/{len(plan)}] {tag}  N={n} ...", flush=True)
        result = compile_one(
            tflite, n_dir, n,
            subproc_timeout_s=subproc_timeout,
            compiler_timeout_sec=args.compiler_timeout_sec,
        )
        report = {
            "tag": tag,
            "num_segments": n,
            "tflite_input": str(tflite.relative_to(tflite_dir.parent))
                             if tflite.is_relative_to(tflite_dir.parent)
                             else str(tflite),
            **result,
        }
        save_report(reports_dir, tag, n, report)

        st = result["compile_status"]
        if st == "success":
            ok += 1
            if not args.keep_intermediates:
                deleted = cleanup_intermediates(n_dir, tag, n)
                if deleted:
                    print(f"    → OK  wall={result['wall_seconds']}s  "
                          f"on_chip={result['totals']['on_chip_mb']}MB  "
                          f"off_chip={result['totals']['off_chip_mb']}MB  "
                          f"cleaned {deleted} intermediates")
                    continue
            print(f"    → OK  wall={result['wall_seconds']}s  "
                  f"on_chip={result['totals']['on_chip_mb']}MB  "
                  f"off_chip={result['totals']['off_chip_mb']}MB")
        elif st == "refused":
            refused += 1
            print(f"    → REFUSED (compiler declined N={n} for this model)")
        elif st == "timeout":
            timeout += 1
            print(f"    → TIMEOUT after {result.get('wall_seconds')}s")
        elif st == "failed":
            failed += 1
            err = (result.get("compile_error") or "")[:200].replace("\n", " ")
            print(f"    → FAILED  {err}")
        elif st == "compiler_unavailable":
            print("    → COMPILER NOT FOUND — aborting sweep")
            return 2
        else:
            other += 1
            print(f"    → {st}")

    summary = rebuild_summary(out_root)
    print("\n[summary] this session:")
    print(f"  ok={ok}  refused={refused}  failed={failed}  timeout={timeout}  other={other}")
    print(f"[summary] cumulative reports across all sessions:")
    print(f"  grand_status_counts={summary['grand_status_counts']}")
    print(f"[summary] written to {out_root/'sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

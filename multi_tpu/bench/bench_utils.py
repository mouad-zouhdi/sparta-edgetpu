"""
bench_utils.py — shared measurement primitives for the multi-TPU benchmarks.

WHAT THIS PROVIDES
    Segment discovery, stratified corpus sampling, deterministic input
    construction, the two measurement primitives (single-accelerator and
    pipelined), CSV appending, and the subprocess runner that every measurement
    goes through.

DESIGN DECISIONS THAT MATTER

  SUBPROCESS ISOLATION IS MANDATORY, NOT DEFENSIVE
    Beyond roughly 1.5 GB of simultaneous mappings, the apex kernel driver fails
    with "Could not map pages" and then aborts at the C level. That abort SIGKILLs
    the entire Python process and cannot be caught by try/except. Measured in
    practice: 220 MB across 4 accelerators is fine, 220 MB across 8 is not.
    Running each measurement in a subprocess means such a crash costs one
    measurement instead of the whole campaign.

  ONE SUBPROCESS PER (tag, N), NOT PER PERMUTATION
    Crashes correlate with the model and its segment count, since those determine
    the total mapping, not with any particular TPU ordering. Isolating per
    permutation would add spawn overhead for no additional protection.

  ROW-BY-ROW CSV APPEND
    Results are written as they are produced, so an interrupted run keeps
    everything measured up to that point.

  DETERMINISTIC INPUT AT SEED 123
    Distinct from the calibration seed of 42, and reused across measurements.
    Latency on this hardware does not depend on input values.

RUNTIME
    coral-env, on the host holding the 8x Edge TPU card.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# Model discovery
# ============================================================
SEGMENT_RE = re.compile(r"(.+)_segment_(\d+)_of_(\d+)_edgetpu\.tflite$")


def discover_segments(models_root: Path, tag: str, N: int) -> list[Path]:
    """Return the ordered list of edgetpu tflite paths for (tag, N).

    N=1  → 1 path: `<tag>_edgetpu.tflite`
    N≥2  → N paths: `<tag>_segment_0_of_N_edgetpu.tflite` ... `_(N-1)_of_N_...`

    Raises FileNotFoundError if any expected file is missing.
    """
    d = models_root / f"N{N}"
    if N == 1:
        p = d / f"{tag}_edgetpu.tflite"
        if not p.exists():
            raise FileNotFoundError(p)
        return [p]
    segs = []
    for i in range(N):
        p = d / f"{tag}_segment_{i}_of_{N}_edgetpu.tflite"
        if not p.exists():
            raise FileNotFoundError(p)
        segs.append(p)
    return segs


def list_available_configs(reports_dir: Path) -> dict[str, dict[int, str]]:
    """Scan reports/ and return {tag: {N: status}} where status ∈ {success, refused, failed, timeout}."""
    out: dict[str, dict[int, str]] = {}
    for p in sorted(reports_dir.glob("*__N*.json")):
        m = re.match(r"(.+)__N(\d+)\.json$", p.name)
        if not m:
            continue
        tag, N = m.group(1), int(m.group(2))
        try:
            data = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        status = data.get("compile_status") or data.get("status") or "unknown"
        out.setdefault(tag, {})[N] = status
    return out


# ============================================================
# Stratified subset selection (Phase C)
# ============================================================
def _load_metadata(metadata_root: Path) -> dict[str, dict]:
    """Load all <tag>.json metadata files as {tag: meta}."""
    metas = {}
    for p in metadata_root.glob("*.json"):
        try:
            d = json.loads(p.read_text())
            metas[p.stem] = d
        except Exception:  # noqa: BLE001
            continue
    return metas


def select_stratified_subset(
    metadata_root: Path,
    reports_dir: Path,
    min_N_success: int = 7,
    families: tuple[str, ...] = ("sequential", "residual", "dense",
                                  "branched_2way", "branched_4way"),
    resolutions: tuple[int, ...] = (96, 160, 224, 384),
) -> list[str]:
    """Pick a stratified sample of models across family, resolution and size.

    Sampling at random would over-represent whichever regimes happen to have the
    most models, which is exactly the bias that would distort a curve fitted across
    regimes. Stratifying guarantees at least one point per bucket of the factorial
    design.
    """
    metas = _load_metadata(metadata_root)
    reports = list_available_configs(reports_dir)

    def n_success(tag: str) -> int:
        """Count how many segment counts compiled successfully for a tag."""
        r = reports.get(tag, {})
        return sum(1 for v in r.values() if v in ("success",))

    candidates = [
        tag for tag, m in metas.items()
        if m.get("build_status") == "success"
        and m.get("aeq_status") == "success"
        and m.get("family") in families
        and m.get("resolution") in resolutions
        and n_success(tag) >= min_N_success
    ]

    # Group by (family, resolution), sort by num_params, pick min+max.
    selected: list[str] = []
    for fam in families:
        for res in resolutions:
            bucket = [t for t in candidates
                      if metas[t]["family"] == fam
                      and metas[t]["resolution"] == res]
            if not bucket:
                continue
            bucket.sort(key=lambda t: metas[t].get("num_params") or 0)
            if len(bucket) == 1:
                selected.append(bucket[0])
            else:
                selected.append(bucket[0])   # smallest
                if bucket[-1] != bucket[0]:
                    selected.append(bucket[-1])  # largest
    return selected


# ============================================================
# Deterministic input
# ============================================================
def make_input_buf(shape: tuple[int, ...], dtype, n: int, seed: int = 123) -> np.ndarray:
    """Deterministic input batch of shape (n, *shape[1:]) matching dtype."""
    rng = np.random.default_rng(seed)
    # int8 range covers all signed 8-bit quantized inputs
    raw = rng.integers(-128, 128, size=(n, *shape[1:]), dtype=np.int8)
    if np.dtype(dtype) == np.int8:
        return raw
    return raw.astype(dtype)


# ============================================================
# Single-TPU bench (Phase A)
# ============================================================
def make_interpreter(model_path: Path, tpu_id: int):
    """Interpreter bound to a specific Edge TPU (via ':<id>' delegate)."""
    import tflite_runtime.interpreter as tflite
    delegate = tflite.load_delegate(
        "libedgetpu.so.1", options={"device": f":{tpu_id}"}
    )
    interp = tflite.Interpreter(
        model_path=str(model_path),
        experimental_delegates=[delegate],
    )
    interp.allocate_tensors()
    return interp


def bench_single_tpu(model_path: Path, tpu_id: int, warmup: int, reps: int,
                     min_wall_sec: float = 2.0, seed: int = 123) -> dict:
    """Measure throughput and latency of one model on one accelerator.

    The repetition count is extended until the run covers at least min_wall_sec. On a
    fast model, a fixed 200 repetitions would finish in a third of a second and the
    measurement would be dominated by clock granularity rather than by the model.
    """
    interp = make_interpreter(model_path, tpu_id)
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    shape = tuple(int(x) for x in inp["shape"])

    buf = make_input_buf(shape, inp["dtype"], warmup + reps, seed=seed)

    # Warmup
    for i in range(warmup):
        interp.set_tensor(inp["index"], buf[i:i + 1])
        interp.invoke()

    per_rep_ns = np.empty(reps, dtype=np.int64)
    t_wall0 = time.perf_counter_ns()
    for i in range(reps):
        t0 = time.perf_counter_ns()
        interp.set_tensor(inp["index"], buf[warmup + i:warmup + i + 1])
        interp.invoke()
        _ = interp.get_tensor(out["index"])
        per_rep_ns[i] = time.perf_counter_ns() - t0
    t_wall1 = time.perf_counter_ns()

    elapsed_sec = (t_wall1 - t_wall0) / 1e9
    # Adaptive extension if wall too short
    while elapsed_sec < min_wall_sec:
        # Extend by the same reps count until we clear min_wall_sec
        extra_ns = np.empty(reps, dtype=np.int64)
        t0 = time.perf_counter_ns()
        for i in range(reps):
            t_a = time.perf_counter_ns()
            interp.set_tensor(inp["index"], buf[(i % (warmup + reps)):(i % (warmup + reps)) + 1])
            interp.invoke()
            _ = interp.get_tensor(out["index"])
            extra_ns[i] = time.perf_counter_ns() - t_a
        t1 = time.perf_counter_ns()
        per_rep_ns = np.concatenate([per_rep_ns, extra_ns])
        elapsed_sec += (t1 - t0) / 1e9

    lat_ms = per_rep_ns / 1e6
    throughput = len(per_rep_ns) / elapsed_sec
    return _summarize_latency(lat_ms, throughput, warmup, len(per_rep_ns))


# ============================================================
# Pipelined bench (Phase B + C)
# ============================================================
def bench_pipeline_permutation(segment_paths: list[Path], tpu_order: list[int],
                                warmup: int, reps: int,
                                min_wall_sec: float = 2.0,
                                seed: int = 123) -> dict:
    """Measure a pipelined model spread over N accelerators in a given TPU order.

    Runs pycoral's PipelinedModelRunner with a producer and a consumer thread, so
    the segments overlap the way a real pipeline would.

    The TPU order is a parameter because it was initially suspected of mattering. It
    does not: interleaving 990 measurements at a fixed order with 990 at random
    orders in one session gave F = 0.905, p = 0.94 on the variances and p = 0.45 on
    the means. Repeating one assignment 990 times disperses as much as using 990
    different ones. What the spread actually measures is run-to-run repeatability,
    about 1.6 % of throughput per measurement, so two configurations measured once
    each are only distinguishable beyond roughly 2 %.
    """
    from pycoral.pipeline.pipelined_model_runner import PipelinedModelRunner
    import tflite_runtime.interpreter as tflite
    import threading

    assert len(segment_paths) == len(tpu_order), \
        f"segments={len(segment_paths)} != tpu_order={len(tpu_order)}"

    delegates_keep = []
    interps = []
    for seg_path, tpu in zip(segment_paths, tpu_order):
        dele = tflite.load_delegate("libedgetpu.so.1",
                                     options={"device": f":{tpu}"})
        delegates_keep.append(dele)
        interp = tflite.Interpreter(model_path=str(seg_path),
                                     experimental_delegates=[dele])
        interp.allocate_tensors()
        interps.append(interp)

    runner = PipelinedModelRunner(interps)

    first_inp = interps[0].get_input_details()[0]
    input_name = first_inp["name"]
    input_dtype = first_inp["dtype"]
    input_shape = tuple(int(x) for x in first_inp["shape"])

    # Pre-generate all inputs (adaptive extension may need extra later)
    n_total = warmup + reps
    buf = make_input_buf(input_shape, input_dtype, n_total, seed=seed)

    push_ns = np.empty(n_total, dtype=np.int64)
    pop_ns = np.empty(n_total, dtype=np.int64)
    pop_count = [0]

    def producer():
        """Feed inputs into the pipeline runner, timestamping each push."""
        for i in range(n_total):
            push_ns[i] = time.perf_counter_ns()
            runner.push({input_name: buf[i:i + 1]})
        runner.push({})  # EOF sentinel

    def consumer():
        """Drain the runner, timestamping each output as it arrives.

        A RuntimeError from PipelinedModelRunner.__del__ after the sentinel is harmless
        noise from pycoral and does not affect the measurements.
        """
        while True:
            out = runner.pop()
            if not out:
                break
            pop_ns[pop_count[0]] = time.perf_counter_ns()
            pop_count[0] += 1

    t_p = threading.Thread(target=producer, daemon=True)
    t_c = threading.Thread(target=consumer, daemon=True)
    t_p.start(); t_c.start()
    t_p.join(); t_c.join()

    if pop_count[0] != n_total:
        raise RuntimeError(f"consumer got {pop_count[0]} outputs, expected {n_total}")

    # Note: for adaptive extension in pipeline mode we'd have to restart —
    # skip it, warmup+steady=20+200 is comfortable at N>=2 (pipeline slows
    # per-input latency by ~N so wall grows accordingly).

    steady_push = push_ns[warmup:]
    steady_pop = pop_ns[warmup:]
    throughput = reps / ((steady_pop[-1] - steady_pop[0]) / 1e9)
    lat_ms = (steady_pop - steady_push) / 1e6
    cold_first_lat_ms = float((pop_ns[0] - push_ns[0]) / 1e6)

    result = _summarize_latency(lat_ms, throughput, warmup, reps)
    result["cold_first_lat_ms"] = cold_first_lat_ms
    return result


# ============================================================
# Stats helper
# ============================================================
def _summarize_latency(lat_ms: np.ndarray, throughput_fps: float,
                       warmup: int, reps: int) -> dict:
    """Summarise a latency array as mean, std, median, p95, p99, min and max."""
    return {
        "warmup": warmup,
        "reps": reps,
        "throughput_fps": float(throughput_fps),
        "lat_ms_mean": float(np.mean(lat_ms)),
        "lat_ms_std": float(np.std(lat_ms)),
        "lat_ms_median": float(np.median(lat_ms)),
        "lat_ms_p95": float(np.percentile(lat_ms, 95)),
        "lat_ms_p99": float(np.percentile(lat_ms, 99)),
        "lat_ms_min": float(np.min(lat_ms)),
        "lat_ms_max": float(np.max(lat_ms)),
    }


# ============================================================
# CSV append
# ============================================================
def csv_append(csv_path: Path, row: dict, columns: list[str]) -> None:
    """Append one row to a CSV, writing the header if the file is new.

    Rows are appended as they are produced rather than at the end, so an interrupted
    campaign keeps everything measured so far.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ============================================================
# Subprocess dispatch
# ============================================================
def run_worker_subprocess(worker_script: Path, args: list,
                          crash_log_dir: Path, tag: str,
                          timeout: int = 300) -> dict:
    """Run one measurement in a separate process and parse its result from stdout.

    Subprocess isolation is not defensive style here, it is required. When the apex
    driver exceeds its mapping limit it aborts at the C level, which SIGKILLs the
    whole Python process and cannot be caught by try/except. In-process, one such
    crash ends the entire sweep.

    One subprocess covers all permutations of a single (tag, N) rather than one per
    permutation: crashes correlate with the model and segment count, not with a
    particular permutation, so finer isolation would only add spawn overhead.
    """
    cmd = [sys.executable, str(worker_script)] + [str(a) for a in args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        crash_log_dir.mkdir(parents=True, exist_ok=True)
        (crash_log_dir / f"{tag}_timeout.log").write_text(f"timeout after {timeout}s\n")
        return {"crashed": True, "reason": "timeout"}

    if r.returncode != 0:
        crash_log_dir.mkdir(parents=True, exist_ok=True)
        (crash_log_dir / f"{tag}_rc{r.returncode}.log").write_text(
            f"cmd:\n{' '.join(cmd)}\n\nstdout:\n{r.stdout}\n\nstderr:\n{r.stderr}"
        )
        return {"crashed": True, "reason": f"rc={r.returncode}",
                "stderr_tail": r.stderr[-500:]}

    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if not lines:
        return {"crashed": True, "reason": "no stdout"}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        return {"crashed": True, "reason": f"bad_json: {e}",
                "last_line": lines[-1][:200]}

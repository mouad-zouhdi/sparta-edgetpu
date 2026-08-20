"""
bench_parallel.py — N independent copies of one model on N accelerators.

WHAT THIS PRODUCES
    A cumulative CSV, one row per (mode, tag, n_tpu, tpu_idx, rep), joined with
    each model's structural and compiler metadata (family, resolution,
    edgetpu_size_mb, on_chip_mb, off_chip_mb).

THE REGIME THIS MEASURES, AND HOW IT DIFFERS FROM PIPELINING
    bench_pipeline.py splits ONE model across N accelerators, so a single
    inference is faster and the SRAM of several devices is pooled. This script
    does the opposite: it runs N independent COPIES of the same model, each on
    its own accelerator, each on its own stream of inferences.

    Together they answer the practical question of when to pipeline and when to
    parallelise. Pipelining buys throughput at the cost of latency, and it is the
    only option when a model does not fit one accelerator; parallelism buys
    throughput more cheaply, but every copy must fit on its own device.

    The interesting quantity is the slowdown against the single-accelerator
    baseline: 1.0 means perfect scaling, above 1.0 means contention on the shared
    PCIe link, the driver's locks, or SRAM re-initialisation.

TWO MEASUREMENT MODES
    steady  20 warmup repetitions then N timed ones, all on the same
            interpreter, with a threading.Barrier BEFORE EACH timed repetition
            so every accelerator starts together. That barrier is what makes
            this a worst-case contention measurement rather than an average one:
            without it the copies drift apart and stop competing.
    cold    a FRESH interpreter per repetition on every thread, then one timed
            inference after a barrier. This forces SRAM and streaming
            re-initialisation and primes nothing, mirroring the cold-start
            measurement of the single-TPU axis.

    The two disagree sharply, which is the point. At 8 accelerators the median
    cold slowdown is 1.60x while the steady-state median is 1.00x: contention is
    real while weights are being moved, and essentially absent once they are
    resident.

    A counter-intuitive finding from that data: cold slowdown is INVERSELY
    correlated with model size. Small models (under 2 MB) show 15-25x, large ones
    (over 100 MB) only 1.1-1.3x. The fixed per-inference overhead dominates on
    small models and is contended for; on large models the per-device PCIe
    transfer dominates and scales cleanly across devices.

SAFEGUARDS
    --max-total-map-mb 1500  refuses configurations whose total mapping would
        exceed the apex driver's limit, beyond which it aborts at the C level and
        SIGKILLs the process.
    --orchestrate            runs one subprocess per (mode, tag), so a driver
        crash costs one model rather than the campaign.
    --resume                 skips any (tag, n_tpu) already in the CSV.

    Models with CPU fallback operations are skipped by default: they mix CPU and
    TPU work, which adds a second source of contention and muddies the analysis.
    --include-cpu-fallback keeps them.

USAGE (coral-env, on the 8x Edge TPU host)
    python bench_parallel.py                        # steady, 100 reps
    python bench_parallel.py --mode cold --reps 30
    python bench_parallel.py --mode both
    python bench_parallel.py --filter dense_d4

NOTE
    Run this AFTER the pipeline phases, never alongside them: two processes
    competing for the accelerators crash each other.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter

DEFAULT_REPS_STEADY = 100
DEFAULT_REPS_COLD = 30
DEFAULT_WARMUP = 20
DEFAULT_NTPU = [1, 2, 4, 8]

METADATA_COLS = (
    "family", "depth", "base_width", "resolution",
    "num_params", "tflite_size_mb", "edgetpu_size_mb",
    "on_chip_mb", "off_chip_mb",
    "num_ops_tpu", "num_ops_cpu_fallback", "num_subgraphs",
)


# ---------------------------------------------------------------------------
# Steady-state worker (from_pc-compatible protocol)
# ---------------------------------------------------------------------------

def steady_worker(pci_idx, model_path, n_reps, n_warmup,
                  warmup_barrier, rep_barrier, out, idx):
    """Thread body for steady-state mode: warm up, then time reps on one accelerator.

    Waits on the barrier before EVERY timed repetition, so all accelerators start
    each one together. Without that synchronisation the copies drift apart and stop
    contending, which would turn a worst-case contention measurement into an
    average-case one.
    """
    device_str = f"pci:{pci_idx}"
    try:
        interp = make_interpreter(model_path, device=device_str)
        interp.allocate_tensors()
    except Exception as e:  # noqa: BLE001
        out[idx] = {"error": str(e), "latencies_ms": []}
        warmup_barrier.wait()
        for _ in range(n_reps):
            rep_barrier.wait()
        return

    inp = interp.get_input_details()[0]
    dummy = np.zeros(inp["shape"], dtype=inp["dtype"])

    for _ in range(n_warmup):
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()

    warmup_barrier.wait()

    latencies = []
    for _ in range(n_reps):
        rep_barrier.wait()
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    out[idx] = {"device": device_str, "latencies_ms": latencies}


# ---------------------------------------------------------------------------
# Cold-start worker
# ---------------------------------------------------------------------------

def cold_worker(pci_idx, model_path, n_reps, rep_barrier, out, idx):
    """Thread body for cold mode: a fresh interpreter per repetition, one timed call.

    Recreating the interpreter forces SRAM and streaming re-initialisation, so each
    repetition genuinely pays the weight transfer instead of reusing resident
    weights.
    """
    device_str = f"pci:{pci_idx}"
    latencies = []
    error = None
    for _ in range(n_reps):
        try:
            interp = make_interpreter(model_path, device=device_str)
            interp.allocate_tensors()
            inp = interp.get_input_details()[0]
            dummy = np.zeros(inp["shape"], dtype=inp["dtype"])
            interp.set_tensor(inp["index"], dummy)
        except Exception as e:  # noqa: BLE001
            error = str(e)
            # Still hit the remaining barriers so peers don't deadlock.
            rep_barrier.wait()
            for _ in range(n_reps - len(latencies) - 1):
                rep_barrier.wait()
            break

        rep_barrier.wait()
        t0 = time.perf_counter()
        interp.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        # Drop the interpreter for the next rep to be truly cold.
        del interp
        gc.collect()

    if error:
        out[idx] = {"error": error, "latencies_ms": latencies}
    else:
        out[idx] = {"device": device_str, "latencies_ms": latencies}


# ---------------------------------------------------------------------------
# Single benchmark dispatch
# ---------------------------------------------------------------------------

def run_one(mode, model_path, pci_indices, reps, warmup):
    """Measure one model at one accelerator count, in the requested mode."""
    n = len(pci_indices)
    results = [None] * n
    if mode == "steady":
        warmup_barrier = threading.Barrier(n)
        rep_barrier = threading.Barrier(n)
        threads = [
            threading.Thread(
                target=steady_worker,
                args=(p, model_path, reps, warmup,
                      warmup_barrier, rep_barrier, results, i),
                daemon=True,
            )
            for i, p in enumerate(pci_indices)
        ]
    elif mode == "cold":
        rep_barrier = threading.Barrier(n)
        threads = [
            threading.Thread(
                target=cold_worker,
                args=(p, model_path, reps, rep_barrier, results, i),
                daemon=True,
            )
            for i, p in enumerate(pci_indices)
        ]
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_models(edgetpu_dir, metadata_dir,
                    include_cpu_fallback=False,
                    max_mb=None,
                    filter_substr=None):
    """Find the compiled models and attach their metadata.

    Skips models with CPU fallback operations unless asked otherwise: they mix CPU
    and TPU work and add a second source of contention to the analysis.
    """
    edgetpu_dir = Path(edgetpu_dir)
    metadata_dir = Path(metadata_dir)
    if not edgetpu_dir.exists():
        sys.exit(f"[ERROR] {edgetpu_dir} does not exist.")

    models = []
    skipped_cpu = skipped_size = skipped_meta = 0
    for f in sorted(edgetpu_dir.glob("*_edgetpu.tflite")):
        tag = f.stem.replace("_edgetpu", "")
        if filter_substr and filter_substr not in tag:
            continue
        meta_path = metadata_dir / f"{tag}.json"
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
            except Exception:  # noqa: BLE001
                skipped_meta += 1

        if not include_cpu_fallback:
            cpu_ops = meta.get("num_ops_cpu_fallback")
            if cpu_ops is not None and cpu_ops > 0:
                skipped_cpu += 1
                continue
        if max_mb is not None:
            size = meta.get("edgetpu_size_mb") or meta.get("tflite_size_mb")
            if size is not None and size > max_mb:
                skipped_size += 1
                continue
        models.append({"tag": tag, "path": str(f), "meta": meta})

    print(f"Discovered {len(models)} models in {edgetpu_dir}")
    if skipped_cpu:  print(f"  - {skipped_cpu} skipped (CPU fallback present)")
    if skipped_size: print(f"  - {skipped_size} skipped (size > {max_mb} MB)")
    if skipped_meta: print(f"  - {skipped_meta} had unreadable metadata (used filename only)")
    return models


# ---------------------------------------------------------------------------
# CSV management (cumulative, resume-friendly)
# ---------------------------------------------------------------------------

def already_measured_keys(csv_path: Path, mode: str) -> set:
    """Return set of (tag, n_tpu) tuples already in the CSV for this mode."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["tag", "mode", "n_tpu"])
        sub = df[df["mode"] == mode]
        return set(zip(sub["tag"], sub["n_tpu"]))
    except Exception:  # noqa: BLE001
        return set()


def append_rows(csv_path: Path, rows: list[dict]):
    """Append measurement rows to the CSV, writing the header if the file is new."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep(mode, models, n_tpu_list, reps, warmup, csv_path, resume_keys,
          max_total_map_mb=None):
    """
    Pool of n_tpu workers running on the same model. Saves rows to CSV after
    EACH (model, n_tpu) pair (not at end-of-model) so a driver-level crash
    doesn't lose successful partial measurements.

    `resume_keys` : set of (tag, n_tpu) tuples already in the CSV — those are
    skipped.

    `max_total_map_mb` : if given, skip a (model, n_tpu) pair when
    `edgetpu_size_mb * n_tpu > max_total_map_mb`. Use this to dodge the
    apex driver's mmap limit on big models at high parallelism — without it
    the whole Python process gets SIGKILLed at the C level.
    """
    total = len(models) * len(n_tpu_list)
    done = 0
    for model in models:
        size_mb = (model["meta"].get("edgetpu_size_mb")
                   or model["meta"].get("tflite_size_mb"))
        for n in n_tpu_list:
            done += 1
            if (model["tag"], n) in resume_keys:
                print(f"[{done}/{total}] [resume-skip] {model['tag']} n_tpu={n}")
                continue
            if (max_total_map_mb is not None
                    and size_mb is not None
                    and size_mb * n > max_total_map_mb):
                print(f"[{done}/{total}] [size-skip] {model['tag']} n_tpu={n}  "
                      f"size×n={size_mb*n:.0f} MB > {max_total_map_mb} MB limit")
                continue
            pci = list(range(n))
            print(
                f"[{done}/{total}] mode={mode}  {model['tag']}  "
                f"n_tpu={n} ({', '.join(f'pci:{i}' for i in pci)})",
                flush=True,
            )
            try:
                results = run_one(mode, model["path"], pci, reps, warmup)
            except Exception as e:  # noqa: BLE001
                print(f"  [ERROR] run_one crashed: {e}", flush=True)
                continue

            rows_for_this_n = []
            for tpu_idx, res in enumerate(results):
                if res is None or "error" in res:
                    err = res["error"] if res else "no result"
                    print(f"  TPU {tpu_idx} ERROR: {err}", flush=True)
                    if not res or not res.get("latencies_ms"):
                        continue
                lats = np.array(res["latencies_ms"])
                if lats.size == 0:
                    continue
                print(
                    f"  TPU {tpu_idx}: "
                    f"p50={np.percentile(lats,50):.2f}ms  "
                    f"p95={np.percentile(lats,95):.2f}ms  "
                    f"p99={np.percentile(lats,99):.2f}ms  "
                    f"mean={lats.mean():.2f}ms  (n={lats.size})",
                    flush=True,
                )
                m = model["meta"]
                for rep_idx, lat in enumerate(lats):
                    row = {
                        "mode": mode,
                        "tag": model["tag"],
                        "n_tpu": n,
                        "tpu_idx": tpu_idx,
                        "rep": rep_idx,
                        "latency_ms": float(lat),
                    }
                    for col in METADATA_COLS:
                        row[col] = m.get(col)
                    rows_for_this_n.append(row)

            # Save after each (model, n_tpu) — driver-level crash on the
            # next iteration won't lose what we just measured.
            append_rows(csv_path, rows_for_this_n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def orchestrate(args, models, modes, n_tpu_list, csv_path, out_dir):
    """Run the sweep with one subprocess per (mode, tag).

    The apex driver aborts at the C level when its mapping limit is exceeded, which
    SIGKILLs the whole process and cannot be caught in Python. Isolating each model
    in its own process turns that from a campaign-ending failure into one logged
    line in the crash summary.
    """
    import subprocess

    crash_dir = out_dir / "crash_logs"
    crash_dir.mkdir(parents=True, exist_ok=True)
    crash_summary = out_dir / "crash_summary.csv"
    if not crash_summary.exists():
        with open(crash_summary, "w") as f:
            f.write("tag,mode,returncode,elapsed_s,timestamp\n")

    total = len(models) * len(modes)
    done = 0
    n_ok = n_crash = n_skip = 0
    models_dir_abs = str(Path(args.models_dir).resolve())
    metadata_dir_abs = str(Path(args.metadata_dir).resolve())
    out_dir_abs = str(Path(args.output_dir).resolve())
    csv_path_abs = str(Path(csv_path).resolve())

    for mode in modes:
        if args.resume:
            measured = already_measured_keys(csv_path, mode)
        else:
            measured = set()
        for model in models:
            done += 1
            tag = model["tag"]

            # If --resume, skip the child entirely when every n_tpu for this
            # (mode, tag) is already in the CSV — saves a TF import cost.
            if args.resume:
                missing = [n for n in n_tpu_list if (tag, n) not in measured]
                if not missing:
                    print(f"[{done}/{total}] [resume-skip] {tag} mode={mode}  (fully done)")
                    n_skip += 1
                    continue

            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--single-model", tag,
                "--mode", mode,
                "--models-dir", models_dir_abs,
                "--metadata-dir", metadata_dir_abs,
                "--output-dir", out_dir_abs,
                "--csv", csv_path_abs,
                "--warmup", str(args.warmup),
            ]
            if args.reps is not None:
                cmd += ["--reps", str(args.reps)]
            if args.n_tpu is not None:
                cmd += ["--n-tpu"] + [str(n) for n in args.n_tpu]
            if args.resume:
                cmd += ["--resume"]
            if args.include_cpu_fallback:
                cmd += ["--include-cpu-fallback"]
            if args.max_mb is not None:
                cmd += ["--max-mb", str(args.max_mb)]
            if args.max_total_map_mb > 0:
                cmd += ["--max-total-map-mb", str(args.max_total_map_mb)]

            print(f"[{done}/{total}] [child] {tag} mode={mode} starting…", flush=True)
            t0 = time.perf_counter()
            try:
                # stdout/stderr inherited so user sees live progress.
                # On crash we re-run a quick capture for the log if useful.
                proc = subprocess.run(
                    cmd, timeout=args.child_timeout_sec,
                )
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                returncode = -1
                print(f"  [TIMEOUT] {tag} mode={mode} exceeded "
                      f"{args.child_timeout_sec}s wall", flush=True)
            elapsed = time.perf_counter() - t0

            if returncode == 0:
                n_ok += 1
                print(f"[{done}/{total}] [child] {tag} mode={mode} OK in {elapsed:.1f}s",
                      flush=True)
            else:
                n_crash += 1
                ts = datetime.now().isoformat(timespec="seconds")
                log_path = crash_dir / f"{tag}_{mode}.log"
                with open(log_path, "w") as f:
                    f.write(f"tag: {tag}\nmode: {mode}\n")
                    f.write(f"returncode: {returncode}\nelapsed_s: {elapsed:.1f}\n")
                    f.write(f"timestamp: {ts}\n")
                    f.write(f"cmd: {' '.join(cmd)}\n")
                    f.write(f"\nNote: stdout/stderr were inherited to the parent "
                            f"terminal during the run; this file only records the "
                            f"exit metadata. Re-run the command above manually to "
                            f"reproduce and capture full output if needed.\n")
                with open(crash_summary, "a") as f:
                    f.write(f"{tag},{mode},{returncode},{elapsed:.1f},{ts}\n")
                print(f"[{done}/{total}] [child] {tag} mode={mode} CRASHED "
                      f"(rc={returncode}, {elapsed:.1f}s) — see {log_path}",
                      flush=True)

    print(f"\n=== Orchestration done ===")
    print(f"  ok       = {n_ok}")
    print(f"  crashed  = {n_crash}")
    print(f"  skipped  = {n_skip}")
    print(f"  CSV      = {csv_path}")
    print(f"  crashes  = {crash_summary}")
    return 0


def main():
    """Parse arguments and run the sweep, orchestrated or in-process."""
    p = argparse.ArgumentParser(description="Multi-TPU benchmark for synthetic_models")
    p.add_argument("--mode", choices=["steady", "cold", "both"], default="steady",
                   help="steady-state (warm), cold-start, or both in sequence")
    p.add_argument("--models-dir", default="outputs/edgetpu",
                   help="dir containing *_edgetpu.tflite")
    p.add_argument("--metadata-dir", default="outputs/metadata",
                   help="dir containing per-model metadata JSON")
    p.add_argument("--output-dir", default="outputs/bench",
                   help="dir for benchmark results CSV")
    p.add_argument("--reps", type=int, default=None,
                   help=f"reps per (model, n_tpu). Default: {DEFAULT_REPS_STEADY} steady, "
                        f"{DEFAULT_REPS_COLD} cold.")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                   help=f"warmup reps for steady mode (default: {DEFAULT_WARMUP})")
    p.add_argument("--n-tpu", nargs="+", type=int, default=None,
                   help=f"levels of parallelism (default: {DEFAULT_NTPU})")
    p.add_argument("--filter", default=None,
                   help="substring filter on tag (e.g. 'dense_d4')")
    p.add_argument("--include-cpu-fallback", action="store_true",
                   help="keep models with num_ops_cpu_fallback > 0 (off by default)")
    p.add_argument("--max-mb", type=float, default=None,
                   help="skip models whose edgetpu_size_mb exceeds this")
    p.add_argument("--max-total-map-mb", type=float, default=0.0,
                   help="If > 0: predictive skip when edgetpu_size_mb*n_tpu > this. "
                        "Default 0 (no predictive skip — rely on subprocess "
                        "isolation to survive driver crashes).")
    p.add_argument("--resume", action="store_true",
                   help="skip (tag, n_tpu) pairs already present in the output "
                        "CSV for the active mode")
    p.add_argument("--csv", default=None,
                   help="explicit CSV path (default: <output-dir>/bench_results.csv)")
    p.add_argument("--orchestrate", action="store_true",
                   help="Spawn one CHILD subprocess per (mode, model). A driver "
                        "crash on one model kills only the child; the parent "
                        "logs it and moves on. Required for big models that "
                        "trip the apex mmap limit. Recommended for full sweeps.")
    p.add_argument("--single-model", default=None,
                   help="Used internally by --orchestrate to run a single tag. "
                        "Exact match on tag, no substring.")
    p.add_argument("--child-timeout-sec", type=int, default=3600,
                   help="Wall timeout per child subprocess in --orchestrate mode "
                        "(default: 3600).")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else out_dir / "bench_results.csv"

    n_available = len(list_edge_tpus())
    print(f"Available TPUs: {n_available}  (pci:0 … pci:{n_available-1})")

    n_tpu_list = args.n_tpu or [k for k in DEFAULT_NTPU if k <= n_available]
    print(f"n_tpu levels: {n_tpu_list}")

    models = discover_models(
        args.models_dir, args.metadata_dir,
        include_cpu_fallback=args.include_cpu_fallback,
        max_mb=args.max_mb,
        filter_substr=args.filter,
    )
    if args.single_model is not None:
        models = [m for m in models if m["tag"] == args.single_model]
        if not models:
            sys.exit(f"--single-model {args.single_model!r} did not match any tag.")
    if not models:
        sys.exit("No models to bench.")

    modes = ["steady", "cold"] if args.mode == "both" else [args.mode]

    if args.orchestrate and args.single_model is None:
        return orchestrate(args, models, modes, n_tpu_list, csv_path, out_dir)

    for mode in modes:
        reps = args.reps
        if reps is None:
            reps = DEFAULT_REPS_COLD if mode == "cold" else DEFAULT_REPS_STEADY
        resume_keys = already_measured_keys(csv_path, mode) if args.resume else set()
        if resume_keys:
            print(f"\n[resume] {len(resume_keys)} (tag, n_tpu) pairs already measured in mode={mode}")
        print(f"\n=== MODE = {mode}  ({reps} reps × {len(models)} models × {len(n_tpu_list)} n_tpu levels) ===\n")
        max_map = args.max_total_map_mb if args.max_total_map_mb > 0 else None
        if max_map:
            print(f"[guard] skip when edgetpu_size_mb * n_tpu > {max_map:.0f} MB "
                  f"(apex driver mmap protection)")
        t0 = time.perf_counter()
        sweep(mode, models, n_tpu_list, reps, args.warmup, csv_path, resume_keys,
              max_total_map_mb=max_map)
        print(f"\nMode {mode} done in {(time.perf_counter()-t0)/60:.1f} min")
        print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

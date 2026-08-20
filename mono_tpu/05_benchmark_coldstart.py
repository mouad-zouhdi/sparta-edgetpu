#!/usr/bin/env python3
"""
05_benchmark_coldstart.py — first-inference cost and the warm-up curve.

WHAT THIS PRODUCES
    cold_start_results.json / .csv, and for every (model, device):
        samples_matrix_ms  K lists of N timings, the raw data
        per_position       N entries, each aggregating one position across the
                           K passes (mean, std, median, p95, p99, min, max)
        cold               statistics for position 1, the very first inference
        steady             statistics for positions STEADY_START..N, pooled

WHY THIS MEASUREMENT EXISTS, AND WHAT IT REVEALS
    04_benchmark.py deliberately discards the first inferences and reports the
    steady state. What it discards is not noise: it is the cost of moving the
    model's weights across the host bus into the accelerator.

    Comparing the two measurements separates two things that a single latency
    figure fuses together:

        steady-state latency  = C + k * E
        first-inference cost  = C + k * E + k * I + c

    where C is compute time, E the parameter volume streamed from off-chip on
    every inference, I the volume cached in SRAM once, k the transfer cost per
    MiB, and c a fixed overhead. The internal part I is paid once and then kept;
    the external part E is paid again on every single inference.

    The evidence that this decomposition is real, and not just a fitted form:
    the four USB models that stream all pay nearly the same first-inference
    surcharge (25.1 to 25.5 ms, agreeing to within 1.6 %) despite their total
    sizes ranging from 10.8 to 35.1 MiB. What they have in common is not their
    size but their INTERNAL volume, 7.58 to 7.66 MiB.

HOW THE MEASUREMENT IS STRUCTURED, AND WHY
    One pass = N timed inferences on ONE interpreter with no warmup, then move on
    to the next model. Position 1 is the genuinely cold inference; positions
    2..N trace the climb towards the steady state, which is what makes it
    possible to see how many inferences the warm-up actually takes rather than
    assuming a number.

    Model order is reshuffled at every pass (--no-shuffle to disable). Without
    it, any effect that drifts over the run, thermal throttling above all, would
    be absorbed into the per-model result: the models measured last would look
    uniformly slower, and that bias would be invisible in the output.

    The output schema is cumulative: re-running appends passes to the existing
    matrix rather than replacing it, so K can be raised incrementally (10 passes
    tonight, 20 more tomorrow) without discarding what was already collected.

USAGE
    python 05_benchmark_coldstart.py \\
        --platform_dir /home/raspberrypi/data \\
        --results_dir /home/raspberrypi/results \\
        --device both --passes 30 --inferences_per_pass 10
"""

import argparse
import csv
import gc
import importlib.util
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
SCHEMA_VERSION = "cold_v2"
DEFAULT_PASSES = 30
DEFAULT_INFERENCES_PER_PASS = 10
# Positions treated as steady state in the `steady` summary (0-indexed).
# With N=10 this keeps positions 5..9, i.e. inferences 6 to 10, discarding the
# five that still show the warm-up climb.
STEADY_START_POSITION = 5
INTER_MODEL_SLEEP_S = 0.05


# ─────────────────────────────────────────────
# 04_benchmark.py is imported dynamically: its filename starts with a digit and
# is therefore not a valid module identifier.
# ─────────────────────────────────────────────

def _import_bench():
    """Import 04_benchmark.py dynamically, to reuse its model discovery and paths.

    A plain import statement will not do: the filename starts with a digit, so it is
    not a valid Python identifier.
    """
    spec = importlib.util.spec_from_file_location(
        "_bench04", BASE_DIR / "04_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bench = _import_bench()


# ─────────────────────────────────────────────
# Paths, resolved in main()
# ─────────────────────────────────────────────

TFLITE_DIR = BASE_DIR / "tflite_int8"
TFLITE_F32_DIR = BASE_DIR / "tflite_float32"
EDGETPU_DIR = BASE_DIR / "edgetpu_compiled"
RESULTS_CSV = BASE_DIR / "cold_start_results.csv"
RESULTS_JSON = BASE_DIR / "cold_start_results.json"


def resolve_paths(platform_dir, results_dir):
    """Resolve model and result paths, delegating to 04_benchmark.py's own resolver."""
    global TFLITE_DIR, TFLITE_F32_DIR, EDGETPU_DIR
    global RESULTS_CSV, RESULTS_JSON

    platform_dir = Path(platform_dir)
    results_dir = Path(results_dir)

    models_root = platform_dir / "models"
    if not models_root.is_dir():
        models_root = platform_dir

    TFLITE_DIR = models_root / "tflite_int8"
    TFLITE_F32_DIR = models_root / "tflite_float32"
    EDGETPU_DIR = models_root / "edgetpu_compiled"

    _bench.TFLITE_DIR = TFLITE_DIR
    _bench.TFLITE_F32_DIR = TFLITE_F32_DIR
    _bench.EDGETPU_DIR = EDGETPU_DIR

    results_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV = results_dir / "cold_start_results.csv"
    RESULTS_JSON = results_dir / "cold_start_results.json"


# ─────────────────────────────────────────────
# Input construction, dtype-aware, matching 04_benchmark.measure_lat
# ─────────────────────────────────────────────

def _make_input(input_details, seed=0):
    """Build one random input tensor of the interpreter's dtype and shape.

    Latency on this hardware does not depend on input values, so random data keeps
    the measurement independent of the dataset.
    """
    shape, dtype = input_details["shape"], input_details["dtype"]
    rng = np.random.RandomState(seed)
    if dtype == np.uint8:
        return rng.randint(0, 255, shape, dtype=np.uint8)
    elif dtype == np.int8:
        return rng.randint(-128, 127, shape, dtype=np.int8)
    return rng.randn(*shape).astype(np.float32)


def _peek_input_details(model_path, use_tpu=False):
    """Read a model's input details without keeping the interpreter alive.

    The interpreter must be discarded: leaving one open would leave the weights
    resident, and the next measurement would no longer be cold.
    """
    if use_tpu:
        from pycoral.utils.edgetpu import make_interpreter as make_tpu
        interp = make_tpu(model_path)
    else:
        import tflite_runtime.interpreter as tfl
        interp = tfl.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    details = interp.get_input_details()[0]
    details = {
        "index": int(details["index"]),
        "shape": tuple(int(x) for x in details["shape"]),
        "dtype": details["dtype"],
    }
    del interp
    gc.collect()
    return details


# ─────────────────────────────────────────────
# Measurement: one pass = one mode, one model, N inferences on one interpreter
# ─────────────────────────────────────────────

def measure_one_pass_one_mode(model_path, n_inferences, use_tpu=False):
    """Run one pass: n_inferences timings on a single freshly created interpreter.

    No warmup, by design. Position 0 is the genuinely cold first inference, the one
    that pays the weight transfer; the following positions trace the climb to the
    steady state.

    The interpreter is created inside this function and destroyed on return, so each
    pass starts from the same state and the cold measurement stays cold.
    """
    in_details = _peek_input_details(model_path, use_tpu=use_tpu)
    data = _make_input(in_details, seed=0)

    if use_tpu:
        from pycoral.utils.edgetpu import make_interpreter as make_tpu
        interp = make_tpu(model_path)
    else:
        import tflite_runtime.interpreter as tfl
        interp = tfl.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    interp.set_tensor(in_details["index"], data)

    times = []
    for _ in range(n_inferences):
        t0 = time.perf_counter()
        interp.invoke()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    del interp
    gc.collect()
    time.sleep(INTER_MODEL_SLEEP_S)
    return times


# ─────────────────────────────────────────────
# Aggregate statistics
# ─────────────────────────────────────────────

def _stats(vals):
    """Summarise a list of timings as mean, std, median, p95, p99, min, max."""
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0,
                "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0, "cv": 0.0, "n": 0}
    mean = float(np.mean(a))
    std = float(np.std(a))
    return {
        "mean": mean,
        "std": std,
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)) if n >= 2 else mean,
        "p99": float(np.percentile(a, 99)) if n >= 2 else mean,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "cv": std / mean if mean > 0 else 0.0,
        "n": n,
    }


def _per_position_stats(matrix):
    """Aggregate a K x N matrix down each column, one summary per position.

    Input is K passes of N timings. The result answers "how long does the i-th
    inference after load take", pooled across passes, which is what the warm-up curve
    is made of.
    """
    if not matrix:
        return []
    arr = np.array(matrix, dtype=np.float64)  # (K, n_inf)
    return [_stats(arr[:, i]) for i in range(arr.shape[1])]


def _summary(matrix, steady_start=STEADY_START_POSITION):
    """Produce the two headline summaries: cold and steady.

    cold pools column 0 across the K passes: the first inference, weight transfer
    included. steady pools columns steady_start..N-1 across all passes, that is, the
    positions where warm-up is over. Their difference is the quantity this script
    exists to measure.
    """
    if not matrix:
        return {"cold": _stats([]), "steady": _stats([])}
    arr = np.array(matrix, dtype=np.float64)
    return {
        "cold": _stats(arr[:, 0]),
        "steady": _stats(arr[:, steady_start:].flatten()),
    }


def _recompute_all_stats(models_dict):
    """Recompute per_position, cold and steady for every entry from the raw matrices.

    Called after each pass so that an interrupted run still leaves consistent
    summaries next to whatever raw data was collected.
    """
    for r in models_dict.values():
        for mode_data in r.get("modes", {}).values():
            matrix = mode_data.get("samples_matrix_ms", [])
            if not matrix:
                continue
            mode_data["n_passes"] = len(matrix)
            mode_data["per_position"] = _per_position_stats(matrix)
            s = _summary(matrix)
            mode_data["cold"] = s["cold"]
            mode_data["steady"] = s["steady"]


# ─────────────────────────────────────────────
# Persistence (cumulatif)
# ─────────────────────────────────────────────

def load_existing():
    """Load previous results so new passes append to the matrix instead of replacing it."""
    if not RESULTS_JSON.exists():
        return {}
    try:
        with open(RESULTS_JSON) as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict) or data.get("_meta", {}).get("schema_version") != SCHEMA_VERSION:
        print(f"[WARN] {RESULTS_JSON.name} has an incompatible schema "
              f"(expected {SCHEMA_VERSION}), ignoring it.")
        return {}
    return data.get("models", {})


def save_results(models_dict, devices_in_run, args, passes_done_this_run):
    """Write the cumulative JSON, merging this run's passes into any earlier ones."""
    meta = {
        "schema_version": SCHEMA_VERSION,
        "last_run": datetime.now().isoformat(timespec="seconds"),
        "devices_measured_this_run": devices_in_run,
        "passes_added_this_run": passes_done_this_run,
        "inferences_per_pass": args.inferences_per_pass,
        "shuffle_models": args.shuffle,
        "steady_start_position": STEADY_START_POSITION,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump({"_meta": meta, "models": models_dict}, f, indent=2,
                  default=lambda x: int(x) if isinstance(x, np.integer)
                  else float(x) if isinstance(x, np.floating) else str(x))
    print(f"→ {RESULTS_JSON}")


# CSV: one scalar summary row per (model, mode), holding cold and steady.
# La matrice K × N et per_position restent dans le JSON pour analyse fine.
SUMMARY_KEYS = ["mean", "std", "median", "p95", "p99", "min", "max", "cv", "n"]
MODES = ["cpu_int8", "cpu_f32", "tpu_int8"]
SECTIONS = ["cold", "steady"]

CSV_FIELDS = ["model", "tag", "importance", "prune_pct"] + [
    f"{mode}_n_passes" for mode in MODES
] + [
    f"{mode}_{section}_{k}"
    for mode in MODES
    for section in SECTIONS
    for k in SUMMARY_KEYS
]


def write_csv(models_dict):
    """Flatten the per-model cold and steady summaries into a CSV."""
    rows = sorted(
        models_dict.values(),
        key=lambda r: (
            _bench.get_base_name(r.get("model", "")),
            {"baseline": 0, "finetuned": 1, "pruned": 2}.get(r.get("tag", ""), 9),
            r.get("prune_pct") or 0,
            r.get("importance") or "",
        ))
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in ["model", "tag", "importance", "prune_pct"]}
            for mode in MODES:
                md = r.get("modes", {}).get(mode, {})
                row[f"{mode}_n_passes"] = md.get("n_passes", 0)
                for section in SECTIONS:
                    s = md.get(section, {})
                    for k in SUMMARY_KEYS:
                        row[f"{mode}_{section}_{k}"] = s.get(k, "")
            w.writerow(row)
    print(f"→ {RESULTS_CSV}")


def _identifiers(cfg):
    """Pull the identifying fields out of a discovery config, for the result row."""
    return {
        "model": cfg["name"],
        "tag": cfg["tag"],
        "importance": cfg["importance"],
        "prune_pct": cfg["prune_pct"],
    }


def _max_passes_in(models_dict):
    """Return the highest pass count already recorded, across all models and modes.

    Used to pick shuffle seeds that continue the sequence rather than restarting it,
    so that resuming a campaign does not replay the same model orders and reintroduce
    the ordering bias the shuffling exists to remove.
    """
    m = 0
    for r in models_dict.values():
        for mode_data in r.get("modes", {}).values():
            m = max(m, len(mode_data.get("samples_matrix_ms", [])))
    return m


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    """Parse arguments, then run K passes over the discovered models, saving as it goes."""
    parser = argparse.ArgumentParser(
        description="Cold-start benchmark: warm-up curve over K passes of N inferences")
    parser.add_argument("--platform_dir", type=str, default=str(BASE_DIR),
                        help="Input root. "
                             f"Default: {BASE_DIR}")
    parser.add_argument("--results_dir", type=str, default=str(BASE_DIR),
                        help="Dossier de sortie pour cold_start_results.{json,csv}. "
                             f"Default: {BASE_DIR}")
    parser.add_argument("--device", choices=["cpu", "tpu", "both"], default="both",
                        help="Quel(s) device(s) mesurer ce run. Cumulatif via le JSON.")
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES,
                        help=f"Full passes over the model list (default {DEFAULT_PASSES}). The JSON is "
                             f"cumulative: re-running appends passes to the existing models.")
    parser.add_argument("--inferences_per_pass", type=int, default=DEFAULT_INFERENCES_PER_PASS,
                        help=f"Timed inferences per freshly created interpreter "
                             f"(default {DEFAULT_INFERENCES_PER_PASS}).")
    parser.add_argument("--shuffle", dest="shuffle", action="store_true", default=True,
                        help="Reshuffle the model order at every pass (the default; removes drift bias).")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                        help="Keep a deterministic order; useful for debugging, biased for measurement.")
    parser.add_argument("--print_curve", action="store_true", default=False,
                        help="Print all N timings for each (model, mode, pass) "
                             "au lieu du seul (cold, last). Utile pour comparer visuellement "
                             "les courbes de chauffe entre passes (smoke test).")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Substring filter on the model name (default: all).")
    args = parser.parse_args()

    resolve_paths(args.platform_dir, args.results_dir)
    devices_to_run = ["cpu", "tpu"] if args.device == "both" else [args.device]
    require_tpu = "tpu" in devices_to_run

    print("=" * 72)
    print("SPARTA — COLD-START BENCHMARK (courbe warm-up)")
    print(f"  Device(s)        : {', '.join(devices_to_run)}")
    print(f"  Platform dir     : {args.platform_dir}")
    print(f"  Results dir      : {args.results_dir}")
    print(f"  Models in        : {TFLITE_DIR}")
    print(f"  Passes (ce run)  : {args.passes}")
    print(f"  Inferences/pass  : {args.inferences_per_pass}")
    print(f"  Shuffle          : {args.shuffle}")
    print("=" * 72)

    discovered = _bench.discover_models(require_tpu_compiled=require_tpu)
    if args.models:
        discovered = [m for m in discovered if any(p in m["name"] for p in args.models)]
    if not discovered:
        print("No model found."); return

    models_dict = load_existing()
    existing_passes = _max_passes_in(models_dict)
    if existing_passes > 0:
        print(f"[INFO] Existing JSON has up to {existing_passes} passes; appending. "
              f"Shuffle seeds continue from that index so orders are not replayed.")

    print(f"\n{len(discovered)} model(s); {args.passes} passes x "
          f"{args.inferences_per_pass} inferences per mode.")

    for pass_local in range(args.passes):
        pass_global = existing_passes + pass_local
        order = list(discovered)
        if args.shuffle:
            random.Random(pass_global).shuffle(order)

        print(f"\n{'═' * 72}")
        print(f" PASSE {pass_local + 1}/{args.passes}  (index global #{pass_global})")
        print(f"{'═' * 72}")

        for idx, cfg in enumerate(order):
            name = cfg["name"]
            imp = f" ({cfg['importance']})" if cfg["importance"] else ""
            r = models_dict.setdefault(name, {})
            r.update(_identifiers(cfg))
            r.setdefault("modes", {})

            modes_to_measure = []
            if "cpu" in devices_to_run:
                modes_to_measure.append(("cpu_int8", cfg["int8_path"], False))
                if cfg["f32_path"] and os.path.exists(cfg["f32_path"]):
                    modes_to_measure.append(("cpu_f32", cfg["f32_path"], False))
            if "tpu" in devices_to_run and cfg["edgetpu_path"]:
                modes_to_measure.append(("tpu_int8", cfg["edgetpu_path"], True))

            print(f"  [{idx+1:3d}/{len(order)}] {name}{imp}")
            for mode_label, path, use_tpu in modes_to_measure:
                times = measure_one_pass_one_mode(
                    path, args.inferences_per_pass, use_tpu=use_tpu)
                mode_data = r["modes"].setdefault(mode_label, {"samples_matrix_ms": []})
                mode_data["samples_matrix_ms"].append(times)
                if args.print_curve:
                    # Full curve: the N timings in order, so the warm-up slope is
                    # readable directly from the log.
                    curve = "  ".join(f"{t:6.2f}" for t in times)
                    print(f"      [{mode_label:8}] {curve}  ms")
                else:
                    t_cold, t_last = times[0], times[-1]
                    ratio = t_cold / t_last if t_last > 0 else 0.0
                    print(f"      [{mode_label:8}] cold(1): {t_cold:7.2f} ms  |  "
                          f"last({len(times)}): {t_last:7.2f} ms  |  "
                          f"ratio: {ratio:.2f}×")

        # Save after every pass: if the machine dies mid-campaign, every
        # completed pass survives.
        _recompute_all_stats(models_dict)
        save_results(models_dict, devices_to_run, args, passes_done_this_run=pass_local + 1)
        write_csv(models_dict)
        print(f"  [PASS {pass_local + 1}/{args.passes} OK, saved]")

    print("\n" + "=" * 72)
    print(f"[OK] {args.passes} passes completed over {len(discovered)} models.")
    print(f"     JSON + CSV dans {RESULTS_JSON.parent}")
    print("=" * 72)


if __name__ == "__main__":
    main()

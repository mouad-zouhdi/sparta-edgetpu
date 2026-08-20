#!/usr/bin/env python3
"""
bench_coldstart_pcie.py — first-inference latency of the single-TPU-axis models,
measured on ONE accelerator of the PCIe multi-TPU card.

WHAT THIS PRODUCES
    A CSV with one row per (pass, model, position):
        pass, tag, position, lat_ms, timestamp
    30 passes x 410 models x 10 positions = 123 000 rows for the full campaign.

WHY IT EXISTS: SEPARATING THE BUS FROM THE CORPUS
    The latency model in this work is calibrated on two different corpora on two
    different buses: the CIFAR-100 models over USB, the synthetic models over
    PCIe. Bus and corpus are therefore confounded, and no comparison between the
    two calibrations can attribute a difference to either one.

    This script removes the confound by running the SAME binaries that were
    measured on the Coral USB accelerator. That gives per-model paired ratios
    between the two buses, hence k_PCIe / k_USB directly, with no intermediate
    model, and the compute time C of one binary on both platforms.

    What it found: on the 241 models that fit entirely in SRAM and therefore
    transfer no weights at all, USB remains 1.63x slower than PCIe. The overhead
    is proportional rather than fixed (the coefficient of variation of the ratio
    is 3.3 %, against 44 % for the absolute difference), and it barely correlates
    with operation count (+0.14 over a 4x range), which rules out a per-layer
    transaction cost and is consistent with a lower clock on the USB device.

PROTOCOL, IDENTICAL TO THE OTHER TWO COLD-START SCRIPTS
    Same runtime, same seeds, same timing boundary. Any difference in the
    software stack would contaminate exactly the comparison this measurement
    exists to make.
      - a FRESH interpreter for every repetition;
      - N consecutive inferences, position 1 being the genuinely cold one;
      - K passes, with the model order reshuffled between passes so that thermal
        drift is not absorbed into the per-model result;
      - only invoke() inside the timed region.

    One pass per process: a driver crash costs one pass, not the campaign. The
    runner skips any pass already present in the CSV, so relaunching resumes.

USAGE
    bench_coldstart_pcie.py --pass-idx K [--positions 10] [--tpu 0]
"""
from __future__ import annotations
import argparse, csv, random, sys, time
from pathlib import Path

import numpy as np

ROOT = Path("/home/mzouhdi/Bureau/multi_tpu_benchmark_pipeline")
MODEL_DIR = ROOT / "axis1_edgetpu"
OUT = ROOT / "outputs" / "bench_full" / "coldstart_axis1.csv"
COLS = ["pass", "tag", "position", "lat_ms", "timestamp"]


def discover():
    """List the Edge TPU binaries to measure, sorted for a deterministic base order."""
    out = []
    for f in sorted(MODEL_DIR.glob("*_edgetpu.tflite")):
        tag = f.name[: -len("_edgetpu.tflite")]
        if "_segment_" in tag:          # ne garder que les modeles non segmentes
            continue
        out.append((tag, f))
    return out


def make_interpreter(path: Path, tpu_id: int):
    """Build an Edge TPU interpreter bound to one accelerator."""
    import tflite_runtime.interpreter as tflite
    delegate = tflite.load_delegate("libedgetpu.so.1",
                                    options={"device": f":{tpu_id}"})
    interp = tflite.Interpreter(model_path=str(path),
                                experimental_delegates=[delegate])
    interp.allocate_tensors()
    return interp


def one_model(path: Path, tpu_id: int, positions: int, seed: int = 123):
    """Measure one model: a fresh interpreter, then `positions` consecutive inferences.

    Position 1 is the cold inference, which pays the weight transfer; the rest trace
    the climb to the steady state. The interpreter is created here and dropped on
    return, so the next call starts cold again.
    """
    interp = make_interpreter(path, tpu_id)
    inp = interp.get_input_details()[0]
    rng = np.random.default_rng(seed)
    if np.issubdtype(inp["dtype"], np.integer):
        info = np.iinfo(inp["dtype"])
        x = rng.integers(info.min, info.max + 1, size=inp["shape"], dtype=inp["dtype"])
    else:
        x = rng.standard_normal(inp["shape"]).astype(inp["dtype"])
    lat = []
    for _ in range(positions):
        interp.set_tensor(inp["index"], x)
        t0 = time.perf_counter()
        interp.invoke()
        lat.append((time.perf_counter() - t0) * 1000.0)
    del interp
    return lat


def main():
    """Run one pass over every model, in shuffled order, appending rows to the CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-idx", type=int, required=True)
    ap.add_argument("--positions", type=int, default=10)
    ap.add_argument("--tpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="ne mesurer que les N premiers modeles (smoke test)")
    a = ap.parse_args()

    models = discover()
    if not models:
        print(f"aucun modele dans {MODEL_DIR}", file=sys.stderr)
        return 1
    random.Random(42 + a.pass_idx).shuffle(models)   # ordre propre a la passe
    if a.limit:
        models = models[: a.limit]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with OUT.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLS)
        ok = fail = 0
        for tag, path in models:
            try:
                lat = one_model(path, a.tpu, a.positions)
            except Exception as e:
                print(f"  [ECHEC] {tag}: {type(e).__name__}: {e}", flush=True)
                fail += 1
                continue
            ts = time.time()
            for i, ms in enumerate(lat, start=1):
                w.writerow([a.pass_idx, tag, i, f"{ms:.6f}", f"{ts:.3f}"])
            fh.flush()
            ok += 1
    print(f"passe {a.pass_idx} : {ok} modeles mesures, {fail} echecs", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

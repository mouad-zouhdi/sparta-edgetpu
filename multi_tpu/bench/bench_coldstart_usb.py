#!/usr/bin/env python3
"""
bench_coldstart_usb.py — first-inference latency on a Coral USB Accelerator.

WHAT THIS PRODUCES
    The same CSV schema as the PCIe and synthetic cold-start scripts:
        pass, tag, position, lat_ms, timestamp

WHY IT EXISTS: SEPARATING THE HOST FROM THE ACCELERATOR
    Completes a 2x2 design over host architecture and accelerator:

                     | Coral USB              | PCIe card
        -------------+------------------------+---------------------------
        ARM host     | Raspberry Pi 4         | (not possible)
        x86 host     | this script            | bench_coldstart_pcie.py

    The 241 single-TPU-axis models that fit entirely in internal memory transfer
    no weights, so their latency is pure compute. If the 1.54x gap seen between
    Pi4/USB and PCIe also appears between x86/USB and PCIe, it cannot be
    attributed to the host, which leaves the accelerator itself.

PROTOCOL
    Identical to the other cold-start scripts: same runtime, seeds 42 + pass
    index for shuffling and 123 for the input, and the timing taken around
    invoke() alone.

PITFALLS SPECIFIC TO THE USB ACCELERATOR, ALL ENCOUNTERED IN PRACTICE
    - Before its firmware loads, the device enumerates as "Global Unichip" on the
      480 Mb/s bus; after the first inference it re-enumerates as "Google Inc."
      at 5000 Mb/s. Do not conclude the port is USB 2.0 without running an
      inference first.
    - load_delegate fails intermittently, roughly once in five, when interpreters
      are created back to back: the device needs a moment between them.
      make_interpreter() retries up to eight times with increasing waits. The
      wait precedes the timed region and does not contaminate it. Without this
      the campaign is unusable.
    - Creating an interpreter costs about 2.7 s on the USB accelerator, against
      roughly 0.06 s over PCIe, which makes it about 97 % of the campaign's total
      runtime. Budget in passes, not in models: 30 passes over 410 models is
      9.4 h, and 10 passes are usually enough, since the dispersion that matters
      is between models rather than between repetitions.
    - That 2.7 s figure is 2.68 s on x86 and 2.73 s on the Pi, a 2 % difference,
      which places the cost on the accelerator rather than on the host.

USAGE
    bench_coldstart_usb.py --pass-idx K --model-dir DIR --out CSV
                           [--positions 10] [--tags-file F]
"""
from __future__ import annotations
import argparse, csv, random, sys, time
from pathlib import Path

import numpy as np


def discover(model_dir: Path, tags_file: Path | None):
    """List the models to measure, optionally restricted by a tags file."""
    keep = None
    if tags_file:
        keep = {l.strip() for l in tags_file.read_text().split("\n") if l.strip()}
    out = []
    for f in sorted(model_dir.glob("*_edgetpu.tflite")):
        tag = f.name[: -len("_edgetpu.tflite")]
        if "_segment_" in tag:
            continue
        if keep is not None and tag not in keep:
            continue
        out.append((tag, f))
    return out


def make_interpreter(path: Path, retries: int = 8):
    """Build a USB Edge TPU interpreter, retrying while the device settles.

    load_delegate fails intermittently when interpreters are created back to back,
    roughly once in five. Retries use increasing waits and happen entirely before
    the timed region, so they do not contaminate the measurement. Without this the
    campaign aborts part-way through.
    """
    import tflite_runtime.interpreter as tflite
    last = None
    for attempt in range(retries):
        try:
            delegate = tflite.load_delegate("libedgetpu.so.1")
            interp = tflite.Interpreter(model_path=str(path),
                                        experimental_delegates=[delegate])
            interp.allocate_tensors()
            return interp
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(0.05 * (attempt + 1))
    raise RuntimeError(f"delegate indisponible apres {retries} tentatives : {last}")


def one_model(path: Path, positions: int, seed: int = 123):
    """Interpreteur neuf, puis `positions` inferences chronometrees."""
    interp = make_interpreter(path)
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
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--positions", type=int, default=10)
    ap.add_argument("--tags-file", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    models = discover(a.model_dir, a.tags_file)
    if not models:
        print(f"aucun modele dans {a.model_dir}", file=sys.stderr)
        return 1
    random.Random(42 + a.pass_idx).shuffle(models)
    if a.limit:
        models = models[: a.limit]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    with a.out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["pass", "tag", "position", "lat_ms", "timestamp"])
        ok = fail = 0
        for tag, path in models:
            try:
                lat = one_model(path, a.positions)
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

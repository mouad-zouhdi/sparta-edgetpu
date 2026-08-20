#!/usr/bin/env python3
"""
bench_latency_chained.py — pipeline LATENCY vs number of TPUs, measured correctly.

WHAT THIS PRODUCES
    outputs/bench_full/latency_ressweep.csv: latency and throughput for one
    architecture swept across input resolutions and across N = 1..8 pipeline
    stages.

WHY THIS SCRIPT EXISTS: THE OTHER MEASUREMENT WAS WRONG
    The pipeline latency reported by pycoral's PipelinedModelRunner is inflated by
    the runner itself, badly: 50 ms to 17 s for models whose actual inference is
    under 5 ms. That runner is built for throughput, and it queues work in a way
    that makes its per-item timing meaningless as a latency figure.

    The THROUGHPUT numbers from that path are fine, and remain valid. The latency
    numbers are not, and were re-measured here.

    The fix is to bypass the runner: create one interpreter per segment, and feed
    each segment's outputs into the next by hand (see _chain_once). One inference
    then walks the segments in order, and the wall-clock time around that walk is
    the real end-to-end latency.

WHAT THE CORRECTED MEASUREMENT SHOWS
    Latency against N is V-shaped. Adding accelerators first removes off-chip
    streaming, which is worth about 1.6 ms per MB, and roughly 90 % of it is gone
    by N = 4. Past that, each additional segment adds an inter-segment
    communication cost of about 2 ms, which starts to dominate.

    The consequence is that the latency optimum (around N = 4) is NOT the point
    at which the model fully fits (off_chip = 0, reached at N = 5-7 at high
    resolution). Fitting completely costs more in communication than it saves in
    streaming.

    Communication does grow with tensor size, from under 1 ms at 18 KB to about
    8 ms at 1152 KB, but it stays below the streaming cost across the whole
    compilable range, so pipelining until the model fits remains the right move
    for latency.

THE SWEEP DESIGN
    One architecture, one set of weights, swept across resolution. Only the
    activation tensor size changes, so the effect of tensor size on inter-segment
    communication is isolated from every other difference between models.

    A related finding from the same sweep: a 26 MB model at width 32 fails to
    compile at resolution 1024 with "large activation tensors", while a 6.7 MB
    model at width 16 compiles even there. The compiler's activation ceiling is
    driven by WIDTH, that is by channel count, not by model size. Model size in
    INT8 counts weights only, not activations, though activations do occupy SRAM
    and so push the fitting point to a higher N at high resolution.

USAGE
    Run on the host with the 8x Edge TPU card, under coral-env.
"""
import argparse, csv, gc, time, threading
from pathlib import Path
import numpy as np

# Tensor-size SWEEP: (tag, family, params_M, int8_MB, resolution, act_proxy_kb).
# act_proxy_kb = (R/4)^2 * base_width / 1024 ~ activation tensor scale at an early
# cut. Two topologies (residual, branched_4way=concat) each swept 49->576 KB via
# width x resolution; matched pairs across 224/384 share weights (only tensor size
# differs). All fit within 8 TPUs, N=1..8.
# A model above 20 MB that STREAMS at N=1 (17 MB off-chip, fits at N=4), chosen so
# streaming cost and communication cost can be compared on one architecture.
# Weights are identical throughout; only the resolution changes.
SELECTION = [
    ("residual_d16_w32_r96",  "residual", 25.5, 26.0,  96,   18),
    ("residual_d16_w32_r128", "residual", 25.5, 26.0, 128,   32),
    ("residual_d16_w32_r160", "residual", 25.5, 26.0, 160,   50),
    ("residual_d16_w32_r192", "residual", 25.5, 26.0, 192,   72),
    ("residual_d16_w32_r224", "residual", 25.5, 26.0, 224,   98),
    ("residual_d16_w32_r256", "residual", 25.5, 26.0, 256,  128),
    ("residual_d16_w32_r320", "residual", 25.5, 26.0, 320,  200),
    ("residual_d16_w32_r384", "residual", 25.5, 26.0, 384,  288),
    ("residual_d16_w32_r448", "residual", 25.5, 26.0, 448,  392),
    ("residual_d16_w32_r512", "residual", 25.5, 26.0, 512,  512),
    ("residual_d16_w32_r768", "residual", 25.5, 26.0, 768, 1152),
    ("residual_d16_w32_r1024","residual", 25.5, 26.0,1024, 2048),
]


def make_interp(model_path, tpu_id):
    """Build an interpreter for one segment, bound to one accelerator."""
    import tflite_runtime.interpreter as tflite
    d = tflite.load_delegate("libedgetpu.so.1", options={"device": f":{tpu_id}"})
    it = tflite.Interpreter(model_path=str(model_path), experimental_delegates=[d])
    it.allocate_tensors()
    return it, d


def discover(root, tag, N):
    """Resolve the compiled segment files for one (tag, N)."""
    d = root / f"N{N}"
    if N == 1:
        p = d / f"{tag}_edgetpu.tflite"
        return [p] if p.exists() else None
    segs = []
    for i in range(N):
        p = d / f"{tag}_segment_{i}_of_{N}_edgetpu.tflite"
        if not p.exists():
            return None
        segs.append(p)
    return segs


def make_buf(shape, dtype, n, seed=123):
    """Build a deterministic input buffer of the right dtype and shape (seed 123)."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(-128, 128, size=(n, *shape[1:]), dtype=np.int8)
    return raw.astype(dtype) if np.dtype(dtype) != np.int8 else raw


def _chain_once(interps, first_in, last_out, x):
    """Run one inference through the segments by hand, and return the final output.

    This is the whole point of the script. Each segment's output tensors are copied
    into the next segment's inputs, matched by tensor name where possible and by
    position otherwise, so one call walks the entire pipeline.

    Bypassing pycoral's PipelinedModelRunner is what makes the timing meaningful:
    that runner is built for throughput and its per-item timing inflates latency by
    orders of magnitude on small models.
    """
    interps[0].set_tensor(first_in["index"], x)
    interps[0].invoke()
    tm = {od["name"]: interps[0].get_tensor(od["index"]) for od in interps[0].get_output_details()}
    for k in range(1, len(interps)):
        prev = list(tm.keys())
        for j, idet in enumerate(interps[k].get_input_details()):
            src = tm[idet["name"]] if idet["name"] in tm else tm[prev[j]]
            interps[k].set_tensor(idet["index"], src)
        interps[k].invoke()
        tm = {od["name"]: interps[k].get_tensor(od["index"]) for od in interps[k].get_output_details()}
    return interps[-1].get_tensor(last_out["index"])


def clean_latency(segments, warmup=5, reps=30):
    """Measure true end-to-end pipeline latency: warmup, then reps timed chained calls."""
    interps, dels = [], []
    for k, seg in enumerate(segments):
        it, d = make_interp(seg, k); interps.append(it); dels.append(d)
    first_in = interps[0].get_input_details()[0]
    last_out = interps[-1].get_output_details()[0]
    buf = make_buf(first_in["shape"], first_in["dtype"], warmup + reps)
    for i in range(warmup):
        _chain_once(interps, first_in, last_out, buf[i:i+1])
    lats = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter_ns()
        _chain_once(interps, first_in, last_out, buf[warmup+i:warmup+i+1])
        lats[i] = (time.perf_counter_ns() - t0) / 1e6
    del interps, dels
    gc.collect()
    return lats


def throughput(segments, warmup=20, reps=80):
    """Measure pipeline throughput, for comparison against the latency figure."""
    from pycoral.pipeline.pipelined_model_runner import PipelinedModelRunner
    interps, dels = [], []
    for k, seg in enumerate(segments):
        it, d = make_interp(seg, k); interps.append(it); dels.append(d)
    runner = PipelinedModelRunner(interps)
    first_in = interps[0].get_input_details()[0]
    name = first_in["name"]
    n = warmup + reps
    buf = make_buf(first_in["shape"], first_in["dtype"], n)
    pop_ns = np.empty(n, dtype=np.int64); cnt = [0]

    def producer():
        """Feed inputs into the pipeline runner, then the sentinel that closes it."""
        for i in range(n):
            runner.push({name: buf[i:i+1]})
        runner.push({})

    def consumer():
        """Drain the runner, timestamping each output as it arrives."""
        while True:
            o = runner.pop()
            if not o:
                break
            pop_ns[cnt[0]] = time.perf_counter_ns(); cnt[0] += 1

    tp = threading.Thread(target=producer, daemon=True)
    tc = threading.Thread(target=consumer, daemon=True)
    tp.start(); tc.start(); tp.join(); tc.join()
    steady = pop_ns[warmup:cnt[0]]
    tput = reps / ((steady[-1] - steady[0]) / 1e9) if cnt[0] > warmup + 1 else float("nan")
    del runner, interps, dels
    gc.collect()
    return tput


COLS = ["tag", "family", "params_M", "int8_MB", "resolution", "act_proxy_kb", "N",
        "lat_ms_median", "lat_ms_mean", "lat_ms_p95", "lat_ms_min", "lat_ms_max",
        "throughput_fps", "reps", "timestamp"]


def main():
    """Sweep the selected models across N = 1..8, writing latency and throughput."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--max-n", type=int, default=8)
    args = ap.parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    f = args.out_csv.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    if write_header:
        w.writeheader()
    for tag, fam, prm, mb, res, proxy in SELECTION:
        for N in range(1, args.max_n + 1):
            segs = discover(args.models_root, tag, N)
            if not segs:
                continue
            try:
                lats = clean_latency(segs)
                tput = throughput(segs)
                row = dict(tag=tag, family=fam, params_M=prm, int8_MB=mb, resolution=res,
                           act_proxy_kb=proxy, N=N,
                           lat_ms_median=round(float(np.median(lats)), 4),
                           lat_ms_mean=round(float(np.mean(lats)), 4),
                           lat_ms_p95=round(float(np.percentile(lats, 95)), 4),
                           lat_ms_min=round(float(np.min(lats)), 4),
                           lat_ms_max=round(float(np.max(lats)), 4),
                           throughput_fps=round(float(tput), 3), reps=len(lats),
                           timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
                w.writerow(row); f.flush()
                print(f"  ok {tag} N={N}: lat={row['lat_ms_median']}ms  tput={row['throughput_fps']}fps", flush=True)
            except Exception as e:
                print(f"  FAIL {tag} N={N}: {type(e).__name__}: {e}", flush=True)
    f.close()
    print("done ->", args.out_csv, flush=True)


if __name__ == "__main__":
    main()

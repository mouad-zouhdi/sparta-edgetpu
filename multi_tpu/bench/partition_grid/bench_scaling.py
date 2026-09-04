#!/usr/bin/env python3
"""
bench_scaling.py — Mesures pour graphes A (speedup vs N pipeline) et
B (pipeline vs parallel).

Pour chaque BenchPoint discoverable :
  - Si N==1 : parallel k∈{1,2,4,8} + single-TPU (déjà couvert par k=1)
  - Si N in {2,4,8} : pipeline single (1 pipeline sur N TPUs)

Sortie : outputs/scaling.csv
"""
from __future__ import annotations
import argparse, csv, sys, threading, time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from wave_configs_v2 import build_all_bench_points, discover_segments


def make_interp(model_path: Path, tpu_id: int):
    import tflite_runtime.interpreter as tflite
    d = tflite.load_delegate("libedgetpu.so.1", options={"device": f":{tpu_id}"})
    interp = tflite.Interpreter(model_path=str(model_path), experimental_delegates=[d])
    interp.allocate_tensors()
    return interp, d


def make_input_buf(shape, dtype, n, seed=123):
    rng = np.random.default_rng(seed)
    raw = rng.integers(-128, 128, size=(n, *shape[1:]), dtype=np.int8)
    if np.dtype(dtype) != np.int8:
        raw = raw.astype(dtype)
    return raw


def summarize(lat_ms, tput_total, tput_per, warmup, reps):
    return dict(
        warmup=warmup, reps=reps,
        throughput_fps_total=float(tput_total),
        throughput_fps_per_unit=float(tput_per),
        lat_ms_mean=float(np.mean(lat_ms)),
        lat_ms_median=float(np.median(lat_ms)),
        lat_ms_p10=float(np.percentile(lat_ms, 10)),
        lat_ms_p25=float(np.percentile(lat_ms, 25)),
        lat_ms_p75=float(np.percentile(lat_ms, 75)),
        lat_ms_p90=float(np.percentile(lat_ms, 90)),
        lat_ms_p95=float(np.percentile(lat_ms, 95)),
    )


def bench_parallel(seg: Path, k_tpu: int, warmup: int, reps: int) -> dict:
    interps, dels = [], []
    for i in range(k_tpu):
        interp, d = make_interp(seg, i)
        interps.append(interp); dels.append(d)
    inp = interps[0].get_input_details()[0]
    out = interps[0].get_output_details()[0]
    buf = make_input_buf(inp["shape"], inp["dtype"], warmup + reps)
    for i in range(warmup):
        for it in interps:
            it.set_tensor(inp["index"], buf[i:i+1]); it.invoke()
    barrier = threading.Barrier(k_tpu)
    lats_ns = [np.empty(reps, dtype=np.int64) for _ in range(k_tpu)]
    def worker(tid):
        it = interps[tid]
        for r in range(reps):
            barrier.wait()
            t0 = time.perf_counter_ns()
            it.set_tensor(inp["index"], buf[warmup+r:warmup+r+1]); it.invoke()
            _ = it.get_tensor(out["index"])
            lats_ns[tid][r] = time.perf_counter_ns() - t0
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(k_tpu)]
    t0 = time.perf_counter_ns()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = (time.perf_counter_ns() - t0) / 1e9
    all_lat = np.concatenate(lats_ns) / 1e6
    tput_total = (k_tpu * reps) / elapsed
    return summarize(all_lat, tput_total, tput_total / k_tpu, warmup, reps)


def bench_pipeline(segments: list[Path], warmup: int, reps: int) -> dict:
    """Bench pipeline en 2 phases:
      1. Throughput (pycoral PipelinedModelRunner en continuous push) -> throughput_fps_total
      2. Solo latency (manual segment chaining, contourne pycoral runner overhead)
    """
    from pycoral.pipeline.pipelined_model_runner import PipelinedModelRunner
    import tflite_runtime.interpreter as tflite

    N = len(segments)
    # ── Phase 1 : throughput via pycoral runner ──
    dels_thr, interps_thr = [], []
    for i, seg in enumerate(segments):
        interp, d = make_interp(seg, i)
        interps_thr.append(interp); dels_thr.append(d)
    runner = PipelinedModelRunner(interps_thr)
    first_inp = interps_thr[0].get_input_details()[0]
    input_name = first_inp["name"]
    n_total = warmup + reps
    buf = make_input_buf(first_inp["shape"], first_inp["dtype"], n_total)
    push_ns = np.empty(n_total, dtype=np.int64)
    pop_ns = np.empty(n_total, dtype=np.int64)
    pop_cnt = [0]
    def producer():
        for i in range(n_total):
            push_ns[i] = time.perf_counter_ns()
            runner.push({input_name: buf[i:i+1]})
        runner.push({})
    def consumer():
        while True:
            o = runner.pop()
            if not o: break
            pop_ns[pop_cnt[0]] = time.perf_counter_ns(); pop_cnt[0] += 1
    tp = threading.Thread(target=producer, daemon=True)
    tc = threading.Thread(target=consumer, daemon=True)
    tp.start(); tc.start(); tp.join(); tc.join()
    steady_pop = pop_ns[warmup:]
    tput = reps / ((steady_pop[-1] - steady_pop[0]) / 1e9)

    # ── Phase 2 : solo latency via manual chaining (bypass pycoral runner) ──
    # Chaque segment sur son TPU, chaînage synchrone (identique à bench_accuracy)
    dels_lat, interps_lat = [], []
    for i, seg in enumerate(segments):
        interp, d = make_interp(seg, i)
        interps_lat.append(interp); dels_lat.append(d)
    first_inp = interps_lat[0].get_input_details()[0]
    last_out = interps_lat[-1].get_output_details()[0]
    buf_lat = make_input_buf(first_inp["shape"], first_inp["dtype"], 20)

    # Warmup manuel : 5 reps pour éviter le cold-start
    for _ in range(5):
        tensor_map = {first_inp["name"]: buf_lat[0:1]}
        interps_lat[0].set_tensor(first_inp["index"], tensor_map[first_inp["name"]])
        interps_lat[0].invoke()
        tm = {od["name"]: interps_lat[0].get_tensor(od["index"])
              for od in interps_lat[0].get_output_details()}
        for k in range(1, N):
            for id_ in interps_lat[k].get_input_details():
                interps_lat[k].set_tensor(id_["index"], tm[id_["name"]])
            interps_lat[k].invoke()
            for od in interps_lat[k].get_output_details():
                tm[od["name"]] = interps_lat[k].get_tensor(od["index"])

    # Mesures solo — 30 reps
    solo_reps = 30
    solo_lats = np.empty(solo_reps, dtype=np.int64)
    for r in range(solo_reps):
        x = buf_lat[r % 20:r % 20 + 1]
        t0 = time.perf_counter_ns()
        interps_lat[0].set_tensor(first_inp["index"], x)
        interps_lat[0].invoke()
        tm = {od["name"]: interps_lat[0].get_tensor(od["index"])
              for od in interps_lat[0].get_output_details()}
        for k in range(1, N):
            for id_ in interps_lat[k].get_input_details():
                interps_lat[k].set_tensor(id_["index"], tm[id_["name"]])
            interps_lat[k].invoke()
            for od in interps_lat[k].get_output_details():
                tm[od["name"]] = interps_lat[k].get_tensor(od["index"])
        _ = interps_lat[-1].get_tensor(last_out["index"])
        solo_lats[r] = time.perf_counter_ns() - t0

    lat_ms = solo_lats / 1e6
    return summarize(lat_ms, tput, tput, warmup, len(solo_lats))


CSV_COLS = ["timestamp", "kind", "model", "pct", "target_mb", "point_tag",
            "regime", "n_units", "warmup", "reps",
            "throughput_fps_total", "throughput_fps_per_unit",
            "lat_ms_mean", "lat_ms_median",
            "lat_ms_p10", "lat_ms_p25", "lat_ms_p75", "lat_ms_p90", "lat_ms_p95"]


def csv_append(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if write_header: w.writeheader()
        w.writerow(row)


def csv_read_keys(path: Path) -> set[tuple]:
    if not path.exists(): return set()
    seen = set()
    with path.open() as f:
        for row in csv.DictReader(f):
            seen.add((row["point_tag"], row["regime"], int(row["n_units"])))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruned-root", required=True, type=Path)
    ap.add_argument("--baseline-root", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    pts = build_all_bench_points()
    ok = []
    for p in pts:
        root = args.pruned_root if p.kind == "wave" else args.baseline_root
        if discover_segments(p, root):
            ok.append(p)
    print(f"[configs] {len(ok)}/{len(pts)} discoverable")

    seen = csv_read_keys(args.out_csv) if args.resume else set()

    for i, pt in enumerate(ok, 1):
        print(f"\n[{i}/{len(ok)}] {pt.tag} (N={pt.N})")
        common = dict(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            kind=pt.kind, model=pt.model, pct=pt.pct, target_mb=pt.target_mb,
            point_tag=pt.tag,
        )
        if pt.N == 1:
            # parallel k∈{1,2,4,8}
            for k in [1, 2, 3, 4, 5, 6, 7, 8]:
                if (pt.tag, "parallel", k) in seen: continue
                try:
                    r = bench_parallel(pt.segments[0], k_tpu=k,
                                        warmup=args.warmup, reps=args.reps)
                    row = dict(common, regime="parallel", n_units=k, **r)
                    csv_append(args.out_csv, row)
                    print(f"  parallel k={k}: {r['throughput_fps_total']:.1f} fps")
                except Exception as e:
                    print(f"  parallel k={k} FAILED: {e}")
        else:
            # pipeline @ pt.N (single pipeline)
            if (pt.tag, "pipeline", pt.N) in seen: continue
            try:
                r = bench_pipeline(pt.segments, warmup=args.warmup, reps=args.reps)
                row = dict(common, regime="pipeline", n_units=pt.N, **r)
                csv_append(args.out_csv, row)
                print(f"  pipeline N={pt.N}: {r['throughput_fps_total']:.1f} fps")
            except Exception as e:
                print(f"  pipeline N={pt.N} FAILED: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
verify_tpu.py — check that a compiled Edge TPU model agrees with its CPU original.

WHAT THIS PRODUCES
    tpu_verification_report.json, per model:
        top1_match_rate        fraction of inputs where CPU and TPU agree on top-1
        top5_overlap_mean      mean size of the intersection of the two top-5 sets
        cos_cpu_tpu_mean/min   cosine similarity between the dequantised logits
        max_abs_diff_int8_*    largest difference on the RAW int8 outputs, before
                               dequantisation: a direct measure of how far the two
                               implementations diverge
        latency_cpu_int8, latency_tpu_edgetpu   warmup plus N runs, full statistics
        tpu_speedup_vs_cpu_int8

WHY THIS CHECK IS WORTH RUNNING
    Compilation for the Edge TPU is not a repackaging: the compiler substitutes
    its own kernels, and the result is not bit-identical to the CPU reference. It
    is normally very close, but "normally" is doing real work in that sentence,
    and a compilation that has gone wrong produces a model that still runs and
    still returns plausible-looking logits.

    Both interpreters are fed the SAME deterministic inputs, drawn at seed 123 to
    keep them distinct from the calibration images at seed 42; reusing the
    calibration images would flatter the quantization.

READING THE RESULT
    cos >= 0.999 with max |delta int8| <= 10 indicates a healthy compilation.
    Beyond that, something is worth investigating, though not automatically
    wrong: Inception-ResNet-V2, with its very long residual path, reaches about
    0.978 with |delta| around 65 and is still consistent in practice.

REQUIREMENTS
    Both the INT8 model in tflite_int8/ and the compiled binary in
    edgetpu_compiled/. Compile with 03_compile_edgetpu_segments.py or the
    single-TPU compiler script.

USAGE (coral-env)
    python verify_tpu.py
    python verify_tpu.py --models resnet50 inception_v3 --runs 200
"""
from __future__ import annotations

import argparse
import json
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
# ai_edge_litert (LiteRT moderne) — supporte les ops produites par ai-edge-quantizer.
# The tflite_runtime 2.5 shipped with pycoral is too old for models produced by
# ai-edge-quantizer: it saturates their output at the zero point silently, giving
# a constant prediction rather than an error. ai_edge_litert reads them correctly
# and also supports the Edge TPU delegate through load_delegate.
from ai_edge_litert.interpreter import Interpreter, load_delegate


BASE_DIR = Path(__file__).parent
INT8_DIR = BASE_DIR / "tflite_int8"
EDGETPU_DIR = BASE_DIR / "edgetpu_compiled"
DATA_DIR = BASE_DIR / "calib_data"
REPORT_PATH = BASE_DIR / "tpu_verification_report.json"

# Model sources and preprocessing, duplicated from the conversion script so this
# stays self-contained under coral-env, which has no torch.
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)

MODELS = {
    "inception_v1_googlenet": dict(input_size=224, mean=IMAGENET_RGB_MEAN, std=IMAGENET_RGB_STD, bgr=False),
    "inception_v2_bninception": dict(input_size=224, mean=(104.0, 117.0, 128.0), std=(1.0, 1.0, 1.0), bgr=True, input_range_255=True),
    "inception_v3":             dict(input_size=299, mean=INCEPTION_MEAN, std=INCEPTION_STD, bgr=False),
    "inception_v4":             dict(input_size=299, mean=INCEPTION_MEAN, std=INCEPTION_STD, bgr=False),
    "resnet50":                 dict(input_size=224, mean=IMAGENET_RGB_MEAN, std=IMAGENET_RGB_STD, bgr=False),
    "resnet101":                dict(input_size=224, mean=IMAGENET_RGB_MEAN, std=IMAGENET_RGB_STD, bgr=False),
    "resnet152":                dict(input_size=224, mean=IMAGENET_RGB_MEAN, std=IMAGENET_RGB_STD, bgr=False),
    "inception_resnet_v2":      dict(input_size=299, mean=INCEPTION_MEAN, std=INCEPTION_STD, bgr=False),
}


# ─────────────────────────────────────────────
# CIFAR-100 read from the raw pickle: torchvision is not available here.
# ─────────────────────────────────────────────
_CIFAR_CACHE = {"arr": None}
_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def _cifar100_train_uint8() -> np.ndarray:
    """Load the CIFAR-100 training images, downloading them if needed."""
    if _CIFAR_CACHE["arr"] is not None:
        return _CIFAR_CACHE["arr"]
    extracted = DATA_DIR / "cifar-100-python" / "train"
    if not extracted.exists():
        tar_path = DATA_DIR / "cifar-100-python.tar.gz"
        if not tar_path.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            print(f"  [data] Downloading CIFAR-100 -> {tar_path}...")
            urllib.request.urlretrieve(_CIFAR_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(DATA_DIR)
    with open(extracted, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    # CIFAR-100 train pickle : 'data' = (50000, 3072) uint8 row-major (R then G then B)
    arr = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # → NHWC RGB
    _CIFAR_CACHE["arr"] = arr
    return arr


def _resize_uint8(img_uint8: np.ndarray, target: int) -> np.ndarray:
    """Resize a uint8 image to the model's input size."""
    if img_uint8.shape[0] == target and img_uint8.shape[1] == target:
        return img_uint8
    pil = Image.fromarray(img_uint8)
    pil = pil.resize((target, target), Image.BICUBIC)
    return np.array(pil, dtype=np.uint8)


def _sample_uint8(n: int, size: int, seed: int) -> np.ndarray:
    """Draw n images at the given seed.

    Called with seed 123, deliberately different from the calibration seed 42:
    verifying on the calibration images would flatter the quantization.
    """
    arr = _cifar100_train_uint8()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
    out = np.empty((len(idx), size, size, 3), dtype=np.uint8)
    for k, i in enumerate(idx):
        out[k] = _resize_uint8(arr[i], size)
    return out


def preprocess(img_uint8: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply one model's preprocessing convention to a uint8 image."""
    img = img_uint8.astype(np.float32)
    if not cfg.get("input_range_255", False):
        img = img / 255.0
    if cfg["bgr"]:
        img = img[..., ::-1]
    mean = np.array(cfg["mean"], dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)
    img = (img - mean) / std
    return np.ascontiguousarray(img)


# ─────────────────────────────────────────────
# CPU INT8 and Edge TPU interpreters
# ─────────────────────────────────────────────
def make_cpu_interp(path: str) -> Interpreter:
    """Build a CPU interpreter for an INT8 model.

    Uses ai_edge_litert rather than tflite_runtime: version 2.5 of the latter
    saturates ai-edge-quantizer output at the zero point without raising.
    """
    interp = Interpreter(model_path=path, num_threads=1)
    interp.allocate_tensors()
    return interp


def make_tpu_interp(path: str) -> Interpreter:
    """Build an Edge TPU interpreter, loading the libedgetpu delegate."""
    interp = Interpreter(
        model_path=path,
        experimental_delegates=[load_delegate("libedgetpu.so.1")],
        num_threads=1,
    )
    interp.allocate_tensors()
    return interp


def quantize_input(x_nhwc_f32: np.ndarray, inp_details: dict) -> np.ndarray:
    """Map a float input onto the interpreter's own input scale and zero point."""
    if inp_details["dtype"] in (np.int8, np.uint8):
        scale, zero = inp_details["quantization"]
        x = np.round(x_nhwc_f32 / scale + zero)
        info = np.iinfo(inp_details["dtype"])
        x = np.clip(x, info.min, info.max).astype(inp_details["dtype"])
        return x
    return x_nhwc_f32.astype(np.float32)


def dequant_output(y_raw: np.ndarray, out_details: dict) -> np.ndarray:
    """Convert raw int8 output back to float using the tensor's quantization parameters."""
    if out_details["dtype"] in (np.int8, np.uint8):
        scale, zero = out_details["quantization"]
        return (y_raw.astype(np.float32) - zero) * scale
    return y_raw.astype(np.float32)


def infer(interp: Interpreter, x_nhwc_f32: np.ndarray):
    """Run one input and return both the raw int8 output and its dequantised form.

    Both are needed: the dequantised logits give cosine similarity, while the raw
    int8 difference measures implementation divergence without the dequantisation
    step smoothing it over.
    """
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x = quantize_input(x_nhwc_f32, inp)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    y_raw = interp.get_tensor(out["index"]).copy()
    return y_raw, dequant_output(y_raw, out)


def time_inferences(interp: Interpreter, x_nhwc_f32: np.ndarray,
                    warmup: int, runs: int) -> dict:
    """Measure latency over warmup plus N runs; return the full statistics."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x = quantize_input(x_nhwc_f32, inp)
    # warmup
    for _ in range(warmup):
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        _ = interp.get_tensor(out["index"])
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        _ = interp.get_tensor(out["index"])
        ts.append((time.perf_counter() - t0) * 1000.0)  # ms
    ts = np.array(ts)
    return {
        "mean_ms": float(ts.mean()),
        "std_ms": float(ts.std()),
        "median_ms": float(np.median(ts)),
        "p95_ms": float(np.percentile(ts, 95)),
        "p99_ms": float(np.percentile(ts, 99)),
        "min_ms": float(ts.min()),
        "max_ms": float(ts.max()),
    }


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def verify_model(name: str, cfg: dict, n_inputs: int, runs: int, warmup: int) -> dict:
    """Compare one model's CPU and Edge TPU outputs on identical inputs, and time both."""
    cpu_path = str(INT8_DIR / f"{name}_int8.tflite")
    tpu_path = str(EDGETPU_DIR / f"{name}_int8_edgetpu.tflite")
    res = {"name": name, "cpu_path": cpu_path, "tpu_path": tpu_path,
           "status": "pending", "error": None}

    if not Path(cpu_path).exists():
        res["status"] = "skipped"; res["error"] = f"missing {cpu_path}"; return res
    if not Path(tpu_path).exists():
        res["status"] = "skipped"; res["error"] = f"missing {tpu_path}"; return res

    try:
        cpu = make_cpu_interp(cpu_path)
        tpu = make_tpu_interp(tpu_path)
    except Exception as e:
        res["status"] = "failed"; res["error"] = f"load: {type(e).__name__}: {e}"; return res

    # Deterministic inputs at seed 123, deliberately different from the seed 42
    # used for calibration: verifying on the calibration images would flatter the
    # quantization.
    size = cfg["input_size"]
    raw = _sample_uint8(n_inputs, size, seed=123)

    per_input = []
    raw_diffs_int8 = []  # |y_raw_cpu - y_raw_tpu| (post-quant)
    for i in range(n_inputs):
        x = np.expand_dims(preprocess(raw[i], cfg), 0)  # (1,H,W,3) float32
        y_cpu_raw, y_cpu = infer(cpu, x)
        y_tpu_raw, y_tpu = infer(tpu, x)
        y_cpu_f = y_cpu.reshape(-1)
        y_tpu_f = y_tpu.reshape(-1)
        top5_cpu = np.argsort(y_cpu_f)[-5:][::-1]
        top5_tpu = np.argsort(y_tpu_f)[-5:][::-1]
        per_input.append({
            "input_idx": i,
            "top1_cpu": int(top5_cpu[0]),
            "top1_tpu": int(top5_tpu[0]),
            "top1_match": int(top5_cpu[0] == top5_tpu[0]),
            "top5_overlap": int(len(set(top5_cpu.tolist()) & set(top5_tpu.tolist()))),
            "cos_cpu_tpu": cos_sim(y_cpu_f, y_tpu_f),
            "max_abs_diff_int8": int(np.abs(y_cpu_raw.astype(int) - y_tpu_raw.astype(int)).max()),
        })
        raw_diffs_int8.append(np.abs(y_cpu_raw.astype(int) - y_tpu_raw.astype(int)).max())

    # Latence (1 input fixe, warmup + runs)
    x_lat = np.expand_dims(preprocess(raw[0], cfg), 0)
    lat_cpu = time_inferences(cpu, x_lat, warmup, runs)
    lat_tpu = time_inferences(tpu, x_lat, warmup, runs)
    speedup = lat_cpu["mean_ms"] / lat_tpu["mean_ms"] if lat_tpu["mean_ms"] > 0 else None

    res.update({
        "status": "ok",
        "n_inputs_checked": n_inputs,
        "top1_match_rate": float(np.mean([p["top1_match"] for p in per_input])),
        "top5_overlap_mean": float(np.mean([p["top5_overlap"] for p in per_input])),
        "cos_cpu_tpu_mean": float(np.mean([p["cos_cpu_tpu"] for p in per_input])),
        "cos_cpu_tpu_min": float(np.min([p["cos_cpu_tpu"] for p in per_input])),
        "max_abs_diff_int8_overall": int(max(raw_diffs_int8)),
        "max_abs_diff_int8_mean": float(np.mean(raw_diffs_int8)),
        "per_input": per_input,
        "latency_cpu_int8": lat_cpu,
        "latency_tpu_edgetpu": lat_tpu,
        "tpu_speedup_vs_cpu_int8": speedup,
    })
    return res


def main():
    """Verify each requested model and write the JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--n_inputs", type=int, default=20,
                        help="Number of deterministic inputs used to compare outputs (default 20)")
    parser.add_argument("--runs", type=int, default=100,
                        help="Timed inferences for the latency measurement (default 100)")
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    names = list(MODELS.keys()) if args.models is None else args.models
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"ERROR: unknown models: {unknown}")
        return

    print("=" * 78)
    print("EDGE TPU VERIFICATION")
    print(f"  Models     : {names}")
    print(f"  N inputs   : {args.n_inputs} (output consistency)")
    print(f"  Latence    : warmup={args.warmup}, runs={args.runs}")
    print("=" * 78)

    report = {"models": {}, "summary": {}}
    for name in names:
        print(f"\n─── {name} ───")
        try:
            r = verify_model(name, MODELS[name], args.n_inputs, args.runs, args.warmup)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"name": name, "status": "exception", "error": f"{type(e).__name__}: {e}"}
        report["models"][name] = r
        if r["status"] == "ok":
            print(f"  top1 match  : {r['top1_match_rate']*100:5.1f}% ({int(r['top1_match_rate']*args.n_inputs)}/{args.n_inputs})")
            print(f"  top5 overlap: {r['top5_overlap_mean']:.2f}/5 (moyenne sur {args.n_inputs} inputs)")
            print(f"  cos(CPU,TPU): mean={r['cos_cpu_tpu_mean']:.6f}  min={r['cos_cpu_tpu_min']:.6f}")
            print(f"  max |Δ int8|: overall={r['max_abs_diff_int8_overall']}  mean={r['max_abs_diff_int8_mean']:.2f}")
            print(f"  latence CPU : {r['latency_cpu_int8']['mean_ms']:7.2f} ms  "
                  f"(med {r['latency_cpu_int8']['median_ms']:.2f}, p95 {r['latency_cpu_int8']['p95_ms']:.2f})")
            print(f"  latence TPU : {r['latency_tpu_edgetpu']['mean_ms']:7.2f} ms  "
                  f"(med {r['latency_tpu_edgetpu']['median_ms']:.2f}, p95 {r['latency_tpu_edgetpu']['p95_ms']:.2f})")
            print(f"  speedup     : {r['tpu_speedup_vs_cpu_int8']:.2f}×")
        else:
            print(f"  [{r['status']}] {r['error']}")

    ok = [n for n, r in report["models"].items() if r["status"] == "ok"]
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"  OK : {len(ok)}/{len(names)}  {ok}")
    print(f"{'=' * 78}")

    report["summary"] = {
        "total": len(report["models"]),
        "success": len(ok),
        "n_inputs": args.n_inputs,
        "runs": args.runs,
        "warmup": args.warmup,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Rapport JSON : {REPORT_PATH}")


if __name__ == "__main__":
    main()

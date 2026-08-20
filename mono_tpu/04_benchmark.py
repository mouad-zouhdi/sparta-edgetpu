#!/usr/bin/env python3
"""
04_benchmark.py — steady-state latency and accuracy on the Edge TPU (single-TPU axis).

WHAT THIS PRODUCES
    benchmark_results.json   {_meta, models: {<name>: {...}}}, cumulative
    benchmark_results.csv    the same content flattened into one row per model

    Every column of the CSV is documented in the repository README, under
    "Benchmark metrics". The short version is below.

WHAT IS MEASURED, PER DEVICE
    --device cpu   INT8 CPU latency, FP32 CPU latency, FP32 CPU accuracy
    --device tpu   INT8 Edge TPU latency, INT8 accuracy (the reference top-1)
    --device both  all of the above

    Accuracy is evaluated on the full 10 000-image CIFAR-100 test set. Latency is
    the steady-state figure: `warmup` inferences are discarded, then `runs`
    inferences are timed on ONE interpreter that stays alive throughout. That
    number therefore excludes model loading and the first-inference weight
    transfer; 05_benchmark_coldstart.py measures those separately, and the two
    together are what separate compute time from transfer time.

WHY THE OUTPUT FILE IS CUMULATIVE
    The two devices are not always available on the same machine: the Edge TPU
    runs on a Raspberry Pi, the FP32 CPU reference is more convenient elsewhere.
    Running with --device cpu now and --device tpu later fills in the missing
    fields of the same JSON rather than overwriting it, and the cross-device
    metrics are recomputed at the end of every pass from whatever is present.
    Anything not yet computable is written as null, never as zero, so that a
    missing measurement is never mistaken for a measured zero.

THE METRICS, AND WHAT EACH ONE IS FOR
    Latency, per (device, precision), in milliseconds:
        lat_{cpu,tpu}_{f32,int8}_ms_{mean,std,median,p95,p99}
        Both mean and median are kept because they disagree when the
        distribution is skewed by scheduling noise; p95 and p99 are what tell
        you whether that skew is a tail or a shift.

    Accuracy, in percent:
        top1_cpu_f32_pct / top5_cpu_f32_pct   FP32 reference
        top1_int8_pct / top5_int8_pct         after quantization

    Accuracy deltas (all signed, negative meaning a loss):
        quant_drop_top{1,5}     INT8 minus FP32, same weights. Isolates the cost
                                of quantization alone. This is what makes
                                "quantization-friendliness" comparable across
                                pruning criteria.
        prune_drop_top{1,5}_f32 pruned minus baseline, both in FP32. Isolates the
                                cost of pruning alone.
        combined_drop_top{1,5}  pruned INT8 minus baseline FP32: what a user
                                actually gives up end to end.

    Speedups (ratios, higher is faster):
        tpu_speedup_int8         CPU INT8 latency / TPU INT8 latency
        quant_speedup_cpu        CPU FP32 / CPU INT8
        prune_speedup_{cpu,tpu}  baseline latency / pruned latency
        theoretical_speedup_macs baseline MACs / pruned MACs, the arithmetic
                                 prediction
        tpu_realization_efficiency
                                 prune_speedup_tpu / theoretical_speedup_macs.
                                 This is the central quantity of the whole study:
                                 it says what fraction of the arithmetic saving
                                 the hardware actually delivers. Values well
                                 below 1 mean the model is limited by weight
                                 transfer rather than by computation, and that is
                                 exactly where two criteria at equal accuracy
                                 stop being interchangeable.

    Memory regime, taken from the compiler report:
        tpu_on_chip_mib / tpu_off_chip_mib   parameters cached vs streamed
        tpu_streaming_ratio                  off-chip / total
        tpu_sram_util_pct                    SRAM occupancy
        tpu_ops_coverage_pct                 share of operations on the TPU
        A model with tpu_off_chip_mib == 0 is read from SRAM once; a model above
        zero re-streams the excess on every inference. That boundary explains
        most of the variance in tpu_realization_efficiency.

    Size and compression:
        size_int8_mib, size_reduction_pct, compression_ratio,
        param_reduction_pct, macs_reduction_pct

    Note: param_reduction_pct is the ACHIEVED reduction, and it is what should be
    used on every axis. prune_pct is only the requested target; global pruning
    removes whole dependency groups of varying size, so the two differ, and
    plotting against the target quietly misplaces every point.

USAGE
    # development machine, everything in the current directory
    python 04_benchmark.py --device both --warmup 30 --runs 200 --num_images 0

    # Raspberry Pi, deployed layout
    python 04_benchmark.py \\
        --platform_dir /path/to/deployed/data \\
        --results_dir /path/to/results \\
        --device tpu --warmup 30 --runs 200 --num_images 0

    --num_images 0 means the full 10k test set. Models, compiler metrics and the
    CIFAR-100 data are read from --platform_dir; results are written to
    --results_dir. Both default to the script's own directory.

RUNTIME NOTE
    Run this under coral-env for TPU measurements. Models produced by
    ai-edge-quantizer must be read through ai_edge_litert, not through
    tflite_runtime 2.5, which saturates their output at the zero point without
    raising: the symptom is every prediction landing on the same class and an
    accuracy near 1 %.
"""

import argparse
import csv
import json
import os
import pickle
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).parent

# Effective paths, resolved in main() from --platform_dir and --results_dir.
# Initialised to BASE_DIR so that a reference before main() runs still resolves
# to something sensible.
TFLITE_DIR = BASE_DIR / "tflite_int8"
TFLITE_F32_DIR = BASE_DIR / "tflite_float32"
EDGETPU_DIR = BASE_DIR / "edgetpu_compiled"
METRICS_FILE = BASE_DIR / "compiler_metrics.json"
FP32_ACC_FILE = BASE_DIR / "fp32_accuracy.json"
RESULTS_CSV = BASE_DIR / "benchmark_results.csv"
RESULTS_JSON = BASE_DIR / "benchmark_results.json"


def resolve_paths(platform_dir, results_dir):
    """Point the module-level path globals at the chosen directory layout.

    Two layouts are supported. In the development layout (platform_dir is the
    script's own directory) tflite_int8/, tflite_float32/, edgetpu_compiled/ and
    compiler_metrics.json sit side by side with the script. In the deployed layout
    (platform_dir is a data directory mirrored onto the Raspberry Pi) they sit under
    platform_dir/models/.

    Results are written to results_dir, which is separate so the model directory can
    be read-only on the deployment target.
    """
    global TFLITE_DIR, TFLITE_F32_DIR, EDGETPU_DIR, METRICS_FILE, FP32_ACC_FILE
    global RESULTS_CSV, RESULTS_JSON

    platform_dir = Path(platform_dir)
    results_dir = Path(results_dir)

    models_root = platform_dir / "models"
    if not models_root.is_dir():
        # Fallback: the artefacts sit directly inside platform_dir
        models_root = platform_dir

    TFLITE_DIR = models_root / "tflite_int8"
    TFLITE_F32_DIR = models_root / "tflite_float32"
    EDGETPU_DIR = models_root / "edgetpu_compiled"
    METRICS_FILE = models_root / "compiler_metrics.json"
    FP32_ACC_FILE = models_root / "fp32_accuracy.json"

    results_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV = results_dir / "benchmark_results.csv"
    RESULTS_JSON = results_dir / "benchmark_results.json"


def resolve_data_dir(data_dir_arg, platform_dir):
    """Locate the directory holding cifar-100-python/.

    When --data_dir is not given, the usual layouts are tried in order:
        {platform_dir}/dataset/cifar100   deployed layout
        {platform_dir}/data               development layout
        {platform_dir}                    cifar-100-python directly inside it
    """
    if data_dir_arg:
        return Path(data_dir_arg)
    platform_dir = Path(platform_dir)
    for candidate in (
        platform_dir / "dataset" / "cifar100",
        platform_dir / "data",
        platform_dir,
    ):
        if (candidate / "cifar-100-python").is_dir():
            return candidate
    return platform_dir / "dataset" / "cifar100"  # fallback: fails with a clear message below

CIFAR_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
CIFAR_STD = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)

SCHEMA_VERSION = "2"
TPU_SRAM_MIB = 8.0  # Coral Edge TPU on-chip SRAM cache


# ─────────────────────────────────────────────
# CIFAR-100 (lecture pickle binaire d'origine)
# ─────────────────────────────────────────────

def load_cifar100_test(data_dir):
    """Load the 10 000-image CIFAR-100 test split from the original binary pickle.

    Returns (data, labels) with data as NHWC uint8. The original pickle format is
    read directly rather than through torchvision, because this script runs in
    coral-env, which deliberately has no torch.
    """
    test_path = Path(data_dir) / "cifar-100-python" / "test"
    if not test_path.exists():
        raise FileNotFoundError(
            f"CIFAR-100 introuvable : {test_path}\n"
            f"Run 00_prepare_baselines.py first; it downloads the dataset.")
    with open(test_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    data = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC uint8
    labels = np.array(d["fine_labels"], dtype=np.int64)
    return data, labels


def prepare_val_samples(data_dir, target_size, max_n=None):
    """Build the evaluation sample list, resizing only if the model needs it.

    With max_n set, a fixed-seed subset is drawn so that a shortened run still
    evaluates the same images for every model and stays comparable.
    """
    data, labels = load_cifar100_test(data_dir)
    if max_n and len(data) > max_n:
        idx = np.random.RandomState(42).choice(len(data), max_n, replace=False)
        data = data[idx]
        labels = labels[idx]
    samples = []
    for img, lbl in zip(data, labels):
        if target_size != 32:
            pil = Image.fromarray(img).resize((target_size, target_size), Image.BILINEAR)
            img = np.array(pil, dtype=np.uint8)
        samples.append((img, int(lbl)))
    return samples


# ─────────────────────────────────────────────
# Input preparation
# ─────────────────────────────────────────────

def prepare_input_int8(img_uint8, inp_details):
    """Normalise an image and quantize it into the interpreter's input tensor.

    Applies the CIFAR-100 normalisation used in training, then maps to the input
    tensor's own scale and zero point, read from the model rather than assumed:
    quantization parameters differ per model, and hard-coding them would
    misrepresent the input to every model but one.
    """
    dtype = inp_details["dtype"]
    qp = inp_details.get("quantization_parameters", {})
    scales = qp.get("scales", [])
    zero_points = qp.get("zero_points", [])
    arr = img_uint8.astype(np.float32) / 255.0
    arr = (arr - CIFAR_MEAN) / CIFAR_STD
    scale = scales[0] if len(scales) > 0 else 1.0
    zp = zero_points[0] if len(zero_points) > 0 else 0
    if dtype == np.uint8:
        return np.expand_dims(np.clip(arr / scale + zp, 0, 255).astype(np.uint8), 0)
    elif dtype == np.int8:
        return np.expand_dims(np.clip(arr / scale + zp, -128, 127).astype(np.int8), 0)
    return np.expand_dims(arr.astype(np.float32), 0)


def prepare_input_float32(img_uint8):
    """Normalise an image into a float32 NCHW-free batch of one, for the FP32 path."""
    arr = img_uint8.astype(np.float32) / 255.0
    arr = (arr - CIFAR_MEAN) / CIFAR_STD
    return np.expand_dims(arr.astype(np.float32), 0)


# ─────────────────────────────────────────────
# Accuracy (single pass over the evaluation set)
# ─────────────────────────────────────────────

def _bootstrap_ci95(correct_flags, n_resamples=1000, seed=0):
    """Bootstrap percentile 95 % confidence interval over per-image correctness flags.

    Resamples the 0/1 flags with replacement `n_resamples` times, takes the mean of
    each resample, and returns the 2.5 % and 97.5 % percentiles.

    This is what makes an accuracy difference interpretable: without it, a 0.3-point
    gap between two pruning criteria cannot be told apart from evaluation noise. It
    costs under half a second for 1000 resamples over 10k images and needs no
    retraining, so there is no reason to omit it.
    """
    arr = np.asarray(correct_flags, dtype=np.uint8)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        means[k] = arr[idx].mean()
    lo = 100.0 * np.percentile(means, 2.5)
    hi = 100.0 * np.percentile(means, 97.5)
    return (float(lo), float(hi))


def _accuracy_with_ci(correct_top1, correct_top5):
    """Return top-1 and top-5 accuracy with their bootstrap 95 % interval bounds."""
    n = len(correct_top1)
    if n == 0:
        return {"top1_pct": 0.0, "top5_pct": 0.0,
                "top1_ci95_lo": 0.0, "top1_ci95_hi": 0.0,
                "top5_ci95_lo": 0.0, "top5_ci95_hi": 0.0, "n_eval": 0}
    t1_lo, t1_hi = _bootstrap_ci95(correct_top1)
    t5_lo, t5_hi = _bootstrap_ci95(correct_top5)
    return {
        "top1_pct": 100.0 * sum(correct_top1) / n,
        "top5_pct": 100.0 * sum(correct_top5) / n,
        "top1_ci95_lo": t1_lo, "top1_ci95_hi": t1_hi,
        "top5_ci95_lo": t5_lo, "top5_ci95_hi": t5_hi,
        "n_eval": n,
    }


def eval_int8_interp(interp, samples):
    """Evaluate top-1 and top-5 on an INT8 interpreter, CPU or Edge TPU alike, with
    bootstrap 95 % confidence intervals.
    """
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    correct1, correct5 = [], []
    for img, lbl in samples:
        interp.set_tensor(inp["index"], prepare_input_int8(img, inp))
        interp.invoke()
        o = interp.get_tensor(out["index"]).flatten()
        t5 = np.argsort(o)[-5:][::-1]
        correct1.append(int(t5[0] == lbl))
        correct5.append(int(lbl in t5))
    return _accuracy_with_ci(correct1, correct5)


def eval_f32_tflite(tflite_path, samples):
    """Evaluate top-1 and top-5 on the float32 TFLite model, the accuracy reference."""
    import tflite_runtime.interpreter as tfl
    interp = tfl.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    correct1, correct5 = [], []
    for img, lbl in samples:
        interp.set_tensor(inp["index"], prepare_input_float32(img))
        interp.invoke()
        o = interp.get_tensor(out["index"]).flatten()
        t5 = np.argsort(o)[-5:][::-1]
        correct1.append(int(t5[0] == lbl))
        correct5.append(int(lbl in t5))
    del interp
    return _accuracy_with_ci(correct1, correct5)


# ─────────────────────────────────────────────
# Latence
# ─────────────────────────────────────────────

def measure_lat(interp, warmup, runs, seed=0):
    """Measure steady-state inference latency; return mean, std, median, p95, p99, cv.

    Runs `warmup` inferences first and discards them, then times `runs` inferences on
    the SAME interpreter. Keeping one interpreter alive is what makes this the
    steady-state figure: model loading and the first-inference weight transfer are
    excluded, and for a model that fits in SRAM the weights stay resident throughout.

    Only invoke() is inside the timed region. Input preparation is deliberately
    outside it, since it is host-side work that would otherwise be attributed to the
    accelerator.

    The input is random data of the correct dtype and shape: latency on this hardware
    does not depend on input values, and using random data avoids pulling the dataset
    into a pure timing measurement.

    What this measurement CANNOT see is the cost of the first inference; that is what
    05_benchmark_coldstart.py exists for, and the difference between the two is what
    separates compute time from weight-transfer time.
    """
    inp = interp.get_input_details()[0]
    shape, dtype = inp["shape"], inp["dtype"]
    rng = np.random.RandomState(seed)
    if dtype == np.uint8:
        d = rng.randint(0, 255, shape, dtype=np.uint8)
    elif dtype == np.int8:
        d = rng.randint(-128, 127, shape, dtype=np.int8)
    else:
        d = rng.randn(*shape).astype(np.float32)
    for _ in range(warmup):
        interp.set_tensor(inp["index"], d); interp.invoke()
    lats = []
    for _ in range(runs):
        interp.set_tensor(inp["index"], d)
        t0 = time.perf_counter()
        interp.invoke()
        lats.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(lats, dtype=np.float64)
    mean = float(np.mean(a)); std = float(np.std(a))
    return {
        "mean": mean, "std": std,
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "cv": float(std / mean) if mean > 0 else 0.0,
    }


# ─────────────────────────────────────────────
# Static statistics read from the .tflite itself
# ─────────────────────────────────────────────

def file_size_mib(p):
    """File size in MiB."""
    return os.path.getsize(p) / (1024 * 1024)


def count_params_int8(p):
    """Count parameters by walking the tensors of the .tflite flatbuffer."""
    try:
        from tflite.Model import Model
        with open(p, "rb") as f: buf = f.read()
        m = Model.GetRootAs(buf, 0); sg = m.Subgraphs(0)
        skip = set(sg.InputsAsNumpy().tolist()) | set(sg.OutputsAsNumpy().tolist())
        t = 0
        for i in range(sg.TensorsLength()):
            if i in skip: continue
            te = sg.Tensors(i)
            if m.Buffers(te.Buffer()).DataLength() > 0:
                t += int(np.prod([te.Shape(j) for j in range(te.ShapeLength())]))
        return t
    except Exception:
        return 0


def estimate_macs(p):
    """Estimate multiply-accumulate operations from the .tflite graph.

    This is the arithmetic prediction that theoretical_speedup_macs is built on, and
    the quantity that measured latency is compared against to expose how much of a
    pruning gain the hardware actually realises.
    """
    try:
        from tflite.Model import Model
        with open(p, "rb") as f: buf = f.read()
        m = Model.GetRootAs(buf, 0); sg = m.Subgraphs(0); t = 0
        for i in range(sg.OperatorsLength()):
            op = sg.Operators(i)
            bc = m.OperatorCodes(op.OpcodeIndex()).DeprecatedBuiltinCode()
            if op.InputsLength() < 2 or op.OutputsLength() < 1: continue
            ot = sg.Tensors(op.Outputs(0))
            os_ = [ot.Shape(j) for j in range(ot.ShapeLength())]
            kt = sg.Tensors(op.Inputs(1))
            ks = [kt.Shape(j) for j in range(kt.ShapeLength())]
            if bc == 3 and len(ks) == 4 and len(os_) == 4:        # CONV_2D
                t += os_[1]*os_[2]*ks[0]*ks[1]*ks[2]*ks[3]
            elif bc == 4 and len(ks) == 4 and len(os_) == 4:      # DEPTHWISE_CONV_2D
                t += os_[1]*os_[2]*ks[3]*ks[1]*ks[2]
            elif bc == 9 and len(ks) == 2:                        # FULLY_CONNECTED
                t += ks[0]*ks[1]
        return t
    except Exception:
        return 0


def load_compiler_metrics():
    """Load the memory-regime metrics written by 03_compile_edgetpu.py."""
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f: return json.load(f)
    return {}


def load_fp32_accuracy():
    """Load pre-computed FP32 reference accuracies, when they are available.

    FP32 accuracy is a property of the weights, not of the hardware: it varies by
    well under 0.1 point between machines. It is therefore computed once on the GPU
    cluster during pruning and imported here, rather than recomputed on the Pi, where
    running 276 models over 10 000 images in FP32 on the CPU would take roughly five
    days and add nothing.

    Expected format: {"_meta": {...}, "models": {<name>: {top1_pct, top5_pct,
    top1_ci95_lo, top1_ci95_hi, top5_ci95_lo, top5_ci95_hi, n_eval}}}.
    Returns the "models" sub-dict, or {} when the file is absent.
    """
    if FP32_ACC_FILE.exists():
        try:
            with open(FP32_ACC_FILE) as f:
                d = json.load(f)
            return d.get("models", {})
        except Exception as e:
            print(f"[WARN] could not read {FP32_ACC_FILE.name}: {e}")
    return {}


# ─────────────────────────────────────────────
# Helpers de nommage
# ─────────────────────────────────────────────

def get_base_name(name):
    """Strip the pruning suffixes to recover the base architecture name.

    Used to pair every pruned model with its own baseline when computing the
    cross-model metrics; a mismatch here would silently compare against the wrong
    reference.
    """
    for suffix in ["_pruned", "_finetuned"]:
        if suffix in name:
            return name.split(suffix)[0]
    return name


def get_model_tag(name):
    """Classify a model name as 'finetuned' (the baseline) or 'pruned'."""
    if "_pruned" in name: return "pruned"
    elif "_finetuned" in name: return "finetuned"
    else: return "baseline"


def get_importance_from_name(name):
    """Recover the pruning criterion from the filename, or None for a baseline."""
    if "_pruned" not in name: return None
    import re
    m = re.search(r"_pruned\d+pct_(.+)$", name)
    return m.group(1) if m else None


def get_prune_pct_from_name(name):
    """Recover the REQUESTED pruning percentage from the filename.

    This is the target, not the achieved reduction. Use param_reduction_pct for any
    axis or correlation: global pruning removes whole dependency groups, so the
    achieved rate differs from the target, model by model.
    """
    import re
    m = re.search(r"_pruned(\d+)pct_", name)
    return int(m.group(1)) if m else None


# ─────────────────────────────────────────────
# Model discovery
# ─────────────────────────────────────────────

def discover_models(require_tpu_compiled):
    """List the models available to benchmark.

    With require_tpu_compiled set, only models that have an Edge TPU binary are
    returned, which is the right filter when the run will measure the accelerator.
    """
    models = []
    for int8_file in sorted(TFLITE_DIR.glob("*_int8.tflite")):
        stem = int8_file.stem
        name = stem.replace("_int8", "")
        edgetpu_file = EDGETPU_DIR / f"{stem}_edgetpu.tflite"
        if require_tpu_compiled and not edgetpu_file.exists():
            continue
        f32_path = TFLITE_F32_DIR / f"{name}_float32.tflite"
        models.append({
            "name": name,
            "int8_path": str(int8_file),
            "edgetpu_path": str(edgetpu_file) if edgetpu_file.exists() else None,
            "f32_path": str(f32_path) if f32_path.exists() else None,
            "tag": get_model_tag(name),
            "importance": get_importance_from_name(name),
            "prune_pct": get_prune_pct_from_name(name),
        })
    return models


# ─────────────────────────────────────────────
# Mesures par device
# ─────────────────────────────────────────────

def measure_static(cfg, compiler_metrics):
    """Collect the size, parameter, MAC and memory-regime figures for one model.

    Read from the .tflite itself and from compiler_metrics.json, so no inference
    is needed and these are gathered even when no accelerator is attached.
    """
    cp = cfg["int8_path"]
    name = cfg["name"]
    out = {
        "model": name,
        "tag": cfg["tag"],
        "importance": cfg["importance"],
        "prune_pct": cfg["prune_pct"],
        "size_int8_mib": file_size_mib(cp),
        "params_int8": count_params_int8(cp),
        "macs_int8": estimate_macs(cp),
    }
    cm = compiler_metrics.get(name, {})
    on_chip = cm.get("on_chip_mib", 0.0)
    off_chip = cm.get("off_chip_mib", 0.0)
    ops_tpu = cm.get("ops_tpu", 0)
    ops_cpu = cm.get("ops_cpu", 0)
    total_ops = ops_tpu + ops_cpu
    out.update({
        "tpu_on_chip_mib": on_chip,
        "tpu_off_chip_mib": off_chip,
        "tpu_streaming_ratio": (off_chip / on_chip) if on_chip > 0 else None,
        "tpu_sram_util_pct": (100.0 * on_chip / TPU_SRAM_MIB) if on_chip > 0 else 0.0,
        "tpu_subgraphs": cm.get("num_subgraphs", 0),
        "tpu_ops_count": ops_tpu,
        "cpu_ops_count": ops_cpu,
        "tpu_ops_coverage_pct": (100.0 * ops_tpu / total_ops) if total_ops else None,
    })
    return out


def measure_cpu(cfg, samples_for_acc, warmup, runs, fp32_acc=None):
    """Measure the CPU side: INT8 latency, FP32 latency, and FP32 accuracy.

    `fp32_acc` is the optional dict from load_fp32_accuracy(). When the current model
    appears in it, FP32 accuracy is imported instead of recomputed, since it does not
    depend on the machine. FP32 LATENCY is always measured locally: unlike accuracy,
    it is a property of the hardware being characterised.
    """
    import tflite_runtime.interpreter as tfl
    out = {}

    print("  [CPU INT8] Latence...")
    ci = tfl.Interpreter(model_path=cfg["int8_path"])
    ci.allocate_tensors()
    cl = measure_lat(ci, warmup, runs)
    out["lat_cpu_int8_ms_mean"] = cl["mean"]
    out["lat_cpu_int8_ms_std"] = cl["std"]
    out["lat_cpu_int8_ms_median"] = cl["median"]
    out["lat_cpu_int8_ms_p95"] = cl["p95"]
    out["lat_cpu_int8_ms_p99"] = cl["p99"]
    out["lat_cpu_int8_cv"] = cl["cv"]
    out["throughput_cpu_int8_fps"] = (1000.0 / cl["mean"]) if cl["mean"] > 0 else None
    print(f"  [CPU INT8] {cl['mean']:.2f} ± {cl['std']:.2f} ms "
          f"(med {cl['median']:.2f}, p95 {cl['p95']:.2f}, p99 {cl['p99']:.2f}, "
          f"cv {cl['cv']*100:.1f}%)")
    del ci

    if cfg["f32_path"] and os.path.exists(cfg["f32_path"]):
        # FP32 CPU accuracy: when fp32_accuracy.json has an entry for this model,
        # import it; otherwise run the evaluation locally, which is slow on a Pi.
        precomputed = (fp32_acc or {}).get(cfg["name"])
        if precomputed and "_error" not in precomputed:
            out["top1_cpu_f32_pct"] = precomputed["top1_pct"]
            out["top5_cpu_f32_pct"] = precomputed["top5_pct"]
            out["top1_cpu_f32_ci95_lo"] = precomputed["top1_ci95_lo"]
            out["top1_cpu_f32_ci95_hi"] = precomputed["top1_ci95_hi"]
            out["top5_cpu_f32_ci95_lo"] = precomputed["top5_ci95_lo"]
            out["top5_cpu_f32_ci95_hi"] = precomputed["top5_ci95_hi"]
            out["n_eval_acc"] = precomputed["n_eval"]
            print(f"  [CPU FP32] Accuracy (imported from fp32_accuracy.json) - "
                  f"Top-1: {precomputed['top1_pct']:.2f}% "
                  f"[{precomputed['top1_ci95_lo']:.2f}, {precomputed['top1_ci95_hi']:.2f}]  "
                  f"Top-5: {precomputed['top5_pct']:.2f}%  (n={precomputed['n_eval']})")
        else:
            print("  [CPU FP32] Accuracy...")
            acc = eval_f32_tflite(cfg["f32_path"], samples_for_acc)
            out["top1_cpu_f32_pct"] = acc["top1_pct"]
            out["top5_cpu_f32_pct"] = acc["top5_pct"]
            out["top1_cpu_f32_ci95_lo"] = acc["top1_ci95_lo"]
            out["top1_cpu_f32_ci95_hi"] = acc["top1_ci95_hi"]
            out["top5_cpu_f32_ci95_lo"] = acc["top5_ci95_lo"]
            out["top5_cpu_f32_ci95_hi"] = acc["top5_ci95_hi"]
            out["n_eval_acc"] = acc["n_eval"]
            print(f"  [CPU FP32] Top-1: {acc['top1_pct']:.2f}% "
                  f"[{acc['top1_ci95_lo']:.2f}, {acc['top1_ci95_hi']:.2f}]  "
                  f"Top-5: {acc['top5_pct']:.2f}%")

        print("  [CPU FP32] Latence...")
        cf = tfl.Interpreter(model_path=cfg["f32_path"])
        cf.allocate_tensors()
        fl = measure_lat(cf, warmup, runs)
        out["lat_cpu_f32_ms_mean"] = fl["mean"]
        out["lat_cpu_f32_ms_std"] = fl["std"]
        out["lat_cpu_f32_ms_median"] = fl["median"]
        out["lat_cpu_f32_ms_p95"] = fl["p95"]
        out["lat_cpu_f32_ms_p99"] = fl["p99"]
        out["lat_cpu_f32_cv"] = fl["cv"]
        out["throughput_cpu_f32_fps"] = (1000.0 / fl["mean"]) if fl["mean"] > 0 else None
        print(f"  [CPU FP32] {fl['mean']:.2f} ± {fl['std']:.2f} ms")
        del cf
    else:
        print("  [CPU FP32] No float32 model, skipping.")

    return out


def measure_tpu(cfg, samples_for_acc, warmup, runs):
    """Measure Edge TPU latency and INT8 accuracy for one model.

    The TPU is the reference for top1_int8_pct rather than the INT8 CPU path: the two
    can differ slightly because the compiler's implementation of an operation is not
    bit-identical to the CPU kernel's, and the number that should be reported is the
    one the target hardware actually produces.
    """
    from pycoral.utils.edgetpu import make_interpreter as make_tpu
    out = {}

    if not cfg["edgetpu_path"]:
        print("  [TPU] No compiled Edge TPU model, skipping.")
        return out

    ti = make_tpu(cfg["edgetpu_path"])
    ti.allocate_tensors()

    print("  [TPU INT8] Accuracy...")
    acc = eval_int8_interp(ti, samples_for_acc)
    out["top1_int8_pct"] = acc["top1_pct"]
    out["top5_int8_pct"] = acc["top5_pct"]
    out["top1_int8_ci95_lo"] = acc["top1_ci95_lo"]
    out["top1_int8_ci95_hi"] = acc["top1_ci95_hi"]
    out["top5_int8_ci95_lo"] = acc["top5_ci95_lo"]
    out["top5_int8_ci95_hi"] = acc["top5_ci95_hi"]
    out["n_eval_acc"] = acc["n_eval"]
    print(f"  [TPU INT8] Top-1: {acc['top1_pct']:.2f}% "
          f"[{acc['top1_ci95_lo']:.2f}, {acc['top1_ci95_hi']:.2f}]  "
          f"Top-5: {acc['top5_pct']:.2f}%")

    print("  [TPU INT8] Latence...")
    tl = measure_lat(ti, warmup, runs)
    out["lat_tpu_int8_ms_mean"] = tl["mean"]
    out["lat_tpu_int8_ms_std"] = tl["std"]
    out["lat_tpu_int8_ms_median"] = tl["median"]
    out["lat_tpu_int8_ms_p95"] = tl["p95"]
    out["lat_tpu_int8_ms_p99"] = tl["p99"]
    out["lat_tpu_int8_cv"] = tl["cv"]
    out["throughput_tpu_int8_fps"] = (1000.0 / tl["mean"]) if tl["mean"] > 0 else None
    print(f"  [TPU INT8] {tl['mean']:.2f} ± {tl['std']:.2f} ms "
          f"(med {tl['median']:.2f}, p95 {tl['p95']:.2f}, p99 {tl['p99']:.2f}, "
          f"cv {tl['cv']*100:.1f}%)")
    del ti
    return out


# ─────────────────────────────────────────────
# Cross-device and cross-model metrics
# ─────────────────────────────────────────────

CROSS_FIELDS = [
    "tpu_speedup_int8", "quant_speedup_cpu",
    "quant_drop_top1", "quant_drop_top5",
    "combined_drop_top1", "combined_drop_top5",
    "prune_drop_top1_f32", "prune_drop_top5_f32",
    "prune_drop_top1_i8", "prune_drop_top5_i8",
    "prune_speedup_tpu", "prune_speedup_cpu",
    "param_reduction_pct", "size_reduction_pct", "macs_reduction_pct",
    "compression_ratio", "theoretical_speedup_macs",
    "tpu_realization_efficiency",
]


def _div(num, den):
    """Divide, returning None rather than raising when either side is missing or zero."""
    if num is None or den is None or den == 0:
        return None
    return num / den


def _sub(a, b):
    """Subtract, returning None when either side is missing."""
    if a is None or b is None: return None
    return a - b


def compute_cross(models_dict):
    """Recompute every cross-device and cross-model metric from what is present.

    Run at the end of each pass. Each field is reset to None first and only filled in
    when its inputs exist, so a metric is never left stale from an earlier pass, and
    a missing measurement stays visibly null rather than becoming a plausible-looking
    zero.

    Cross-device metrics (within one model) compare CPU against TPU and FP32 against
    INT8. Cross-model metrics compare a pruned model against the baseline of the same
    architecture, which is looked up through get_base_name().

    The last one computed, tpu_realization_efficiency, is the point of the exercise:
    it divides the measured speedup by the arithmetic one, and so states what
    fraction of the theoretical gain the hardware delivered.
    """

    by_base = OrderedDict()
    for name, r in models_dict.items():
        by_base.setdefault(get_base_name(name), {})[r["tag"] +
            ("_" + r["importance"] if r.get("importance") else "")] = r

    for name, r in models_dict.items():
        for f in CROSS_FIELDS:
            r[f] = None

        # -- Cross-device, within one model --
        r["tpu_speedup_int8"] = _div(r.get("lat_cpu_int8_ms_mean"),
                                     r.get("lat_tpu_int8_ms_mean"))
        r["quant_speedup_cpu"] = _div(r.get("lat_cpu_f32_ms_mean"),
                                      r.get("lat_cpu_int8_ms_mean"))

        # Quantization drop: same model, same weights, FP32 versus INT8. This
        # isolates the cost of quantization from the cost of pruning, which is
        # what makes "quantization-friendliness" comparable across criteria.
        # Normally negative.
        r["quant_drop_top1"] = _sub(r.get("top1_int8_pct"), r.get("top1_cpu_f32_pct"))
        r["quant_drop_top5"] = _sub(r.get("top5_int8_pct"), r.get("top5_cpu_f32_pct"))

        # -- Cross-model: pruned against the baseline of the same architecture --
        if r["tag"] != "pruned":
            continue
        ref = by_base.get(get_base_name(name), {}).get("finetuned")
        if not ref:
            continue

        # Drops
        r["combined_drop_top1"] = _sub(r.get("top1_int8_pct"), ref.get("top1_cpu_f32_pct"))
        r["combined_drop_top5"] = _sub(r.get("top5_int8_pct"), ref.get("top5_cpu_f32_pct"))
        r["prune_drop_top1_f32"] = _sub(r.get("top1_cpu_f32_pct"), ref.get("top1_cpu_f32_pct"))
        r["prune_drop_top5_f32"] = _sub(r.get("top5_cpu_f32_pct"), ref.get("top5_cpu_f32_pct"))
        r["prune_drop_top1_i8"] = _sub(r.get("top1_int8_pct"), ref.get("top1_int8_pct"))
        r["prune_drop_top5_i8"] = _sub(r.get("top5_int8_pct"), ref.get("top5_int8_pct"))

        # Speedups
        r["prune_speedup_tpu"] = _div(ref.get("lat_tpu_int8_ms_mean"),
                                      r.get("lat_tpu_int8_ms_mean"))
        r["prune_speedup_cpu"] = _div(ref.get("lat_cpu_int8_ms_mean"),
                                      r.get("lat_cpu_int8_ms_mean"))

        # Reductions and ratios
        if ref.get("params_int8"):
            r["param_reduction_pct"] = 100.0 * (1 - r["params_int8"] / ref["params_int8"])
        if ref.get("size_int8_mib"):
            r["size_reduction_pct"] = 100.0 * (1 - r["size_int8_mib"] / ref["size_int8_mib"])
            r["compression_ratio"] = ref["size_int8_mib"] / r["size_int8_mib"]
        if ref.get("macs_int8"):
            r["macs_reduction_pct"] = 100.0 * (1 - r["macs_int8"] / ref["macs_int8"])
            r["theoretical_speedup_macs"] = ref["macs_int8"] / r["macs_int8"]

        # How much of the arithmetic saving the hardware actually delivered.
        # Well below 1 means the model is limited by weight transfer rather than
        # by computation, which is where two criteria at equal accuracy stop
        # being interchangeable.
        r["tpu_realization_efficiency"] = _div(r.get("prune_speedup_tpu"),
                                               r.get("theoretical_speedup_macs"))


# ─────────────────────────────────────────────
# Persistence (merge avec JSON existant)
# ─────────────────────────────────────────────

def load_existing():
    """Load the previous results JSON so this pass can merge into it."""
    if not RESULTS_JSON.exists():
        return {}
    try:
        with open(RESULTS_JSON) as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict) or data.get("_meta", {}).get("schema_version") != SCHEMA_VERSION:
        print(f"[WARN] {RESULTS_JSON.name} has an incompatible schema (expected v{SCHEMA_VERSION}), ignoring it.")
        return {}
    return data.get("models", {})


def save_results(models_dict, devices_in_run, args):
    """Merge this pass into the existing JSON, recompute cross metrics, write JSON+CSV."""
    meta = {
        "schema_version": SCHEMA_VERSION,
        "last_run": datetime.now().isoformat(timespec="seconds"),
        "devices_measured_this_run": devices_in_run,
        "warmup": args.warmup,
        "runs": args.runs,
        "num_images": args.num_images if args.num_images > 0 else 10000,
        "data_dir": str(args.data_dir),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump({"_meta": meta, "models": models_dict}, f, indent=2,
                  default=lambda x: int(x) if isinstance(x, np.integer)
                  else float(x) if isinstance(x, np.floating) else str(x))
    print(f"\n→ {RESULTS_JSON}")


# ─────────────────────────────────────────────
# Sortie CSV + console
# ─────────────────────────────────────────────

CSV_FIELDS = [
    "model", "tag", "importance", "prune_pct",
    "size_int8_mib", "params_int8", "macs_int8",
    "tpu_on_chip_mib", "tpu_off_chip_mib", "tpu_streaming_ratio",
    "tpu_sram_util_pct", "tpu_subgraphs", "tpu_ops_count", "cpu_ops_count",
    "tpu_ops_coverage_pct",
    "top1_cpu_f32_pct", "top5_cpu_f32_pct",
    "top1_cpu_f32_ci95_lo", "top1_cpu_f32_ci95_hi",
    "top5_cpu_f32_ci95_lo", "top5_cpu_f32_ci95_hi",
    "top1_int8_pct", "top5_int8_pct",
    "top1_int8_ci95_lo", "top1_int8_ci95_hi",
    "top5_int8_ci95_lo", "top5_int8_ci95_hi",
    "n_eval_acc",
    "lat_cpu_f32_ms_mean", "lat_cpu_f32_ms_std", "lat_cpu_f32_ms_median",
    "lat_cpu_f32_ms_p95", "lat_cpu_f32_ms_p99", "lat_cpu_f32_cv",
    "lat_cpu_int8_ms_mean", "lat_cpu_int8_ms_std", "lat_cpu_int8_ms_median",
    "lat_cpu_int8_ms_p95", "lat_cpu_int8_ms_p99", "lat_cpu_int8_cv",
    "lat_tpu_int8_ms_mean", "lat_tpu_int8_ms_std", "lat_tpu_int8_ms_median",
    "lat_tpu_int8_ms_p95", "lat_tpu_int8_ms_p99", "lat_tpu_int8_cv",
    "throughput_cpu_f32_fps", "throughput_cpu_int8_fps", "throughput_tpu_int8_fps",
    "tpu_speedup_int8", "quant_speedup_cpu",
    "quant_drop_top1", "quant_drop_top5",
    "combined_drop_top1", "combined_drop_top5",
    "prune_drop_top1_f32", "prune_drop_top5_f32",
    "prune_drop_top1_i8", "prune_drop_top5_i8",
    "prune_speedup_tpu", "prune_speedup_cpu",
    "param_reduction_pct", "size_reduction_pct", "macs_reduction_pct",
    "compression_ratio", "theoretical_speedup_macs",
    "tpu_realization_efficiency",
]


def write_csv(models_dict):
    """Flatten the results into one CSV row per model, for analysis elsewhere."""
    rows = sorted(models_dict.values(),
                  key=lambda r: (get_base_name(r["model"]),
                                 {"baseline": 0, "finetuned": 1, "pruned": 2}.get(r["tag"], 9),
                                 r.get("prune_pct") or 0,
                                 r.get("importance") or ""))
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"→ {RESULTS_CSV}")


def fmt(v, f=".1f", na="  -"):
    """Format a possibly-missing number for the console table, showing '-' for None."""
    return na if v is None else f"{v:{f}}"


def print_table(models_dict):
    """Print the human-readable summary table at the end of a run."""
    groups = OrderedDict()
    for r in models_dict.values():
        groups.setdefault(get_base_name(r["model"]), []).append(r)
    tag_order = {"baseline": 0, "finetuned": 1, "pruned": 2}
    for g in groups.values():
        g.sort(key=lambda r: (tag_order.get(r["tag"], 9),
                              r.get("prune_pct") or 0,
                              r.get("importance") or ""))

    hdr = (f"  {'Model':<55s}"
           f"{'Tag':>10s} {'Imp':>14s} "
           f"{'F32T1':>6s} {'I8T1':>6s} {'QDrop':>6s} {'CDrop':>6s} "
           f"{'CPUf32':>7s} {'CPUi8':>7s} {'TPUi8':>7s} "
           f"{'QSp':>5s} {'TpuSp':>6s} {'PruSp':>6s} "
           f"{'SRAM%':>6s} {'StrR':>5s} {'OpsT%':>6s} "
           f"{'MACs':>10s} {'MiB':>6s} {'ΔMiB':>6s}")
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    print("RESULTS (grouped by architecture)")
    print(f"{'=' * len(hdr)}")
    for base, group in groups.items():
        print(f"\n  ┌── {base.upper()} ──")
        print(hdr); print(sep)
        for r in group:
            mc = f"{r['macs_int8']/1e6:.0f}M" if r.get("macs_int8") else "  -"
            print(f"  {r['model']:<55s}"
                  f"{r['tag']:>10s} {(r.get('importance') or '-'):>14s} "
                  f"{fmt(r.get('top1_cpu_f32_pct')):>6s} "
                  f"{fmt(r.get('top1_int8_pct')):>6s} "
                  f"{fmt(r.get('quant_drop_top1'), '+.1f'):>6s} "
                  f"{fmt(r.get('combined_drop_top1'), '+.1f'):>6s} "
                  f"{fmt(r.get('lat_cpu_f32_ms_mean'), '.2f'):>7s} "
                  f"{fmt(r.get('lat_cpu_int8_ms_mean'), '.2f'):>7s} "
                  f"{fmt(r.get('lat_tpu_int8_ms_mean'), '.2f'):>7s} "
                  f"{fmt(r.get('quant_speedup_cpu'), '.1f'):>5s} "
                  f"{fmt(r.get('tpu_speedup_int8'), '.1f'):>6s} "
                  f"{fmt(r.get('prune_speedup_tpu'), '.2f'):>6s} "
                  f"{fmt(r.get('tpu_sram_util_pct'), '.0f'):>6s} "
                  f"{fmt(r.get('tpu_streaming_ratio'), '.2f'):>5s} "
                  f"{fmt(r.get('tpu_ops_coverage_pct'), '.0f'):>6s} "
                  f"{mc:>10s} "
                  f"{fmt(r.get('size_int8_mib'), '.1f'):>6s} "
                  f"{fmt(r.get('size_reduction_pct'), '.0f'):>6s}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    """Parse arguments, discover models, measure each on the requested devices, save."""
    parser = argparse.ArgumentParser(
        description="SPARTA — Benchmark CIFAR-100 (CPU/TPU configurable, JSON cumulatif)")
    parser.add_argument("--platform_dir", type=str, default=str(BASE_DIR),
                        help="Input root. Looks for "
                             "models/{tflite_int8,tflite_float32,edgetpu_compiled,"
                             "compiler_metrics.json} et dataset/cifar100/cifar-100-python/. "
                             f"Default: {BASE_DIR}")
    parser.add_argument("--results_dir", type=str, default=str(BASE_DIR),
                        help="Output directory for benchmark_results.{json,csv}. "
                             f"Default: {BASE_DIR}")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing cifar-100-python/. When omitted, it is "
                             "derived from --platform_dir when omitted.")
    parser.add_argument("--device", choices=["cpu", "tpu", "both"], default="both",
                        help="Which device(s) to measure in this run. The output JSON is cumulative: "
                             "running repeatedly with different --device values fills in the missing fields.")
    parser.add_argument("--warmup", type=int, required=True,
                        help="Discarded warmup inferences. 30 is usually enough, 50 for sub-millisecond models")
    parser.add_argument("--runs", type=int, required=True,
                        help="Timed inferences. 200 gives a stable mean and usable p95/p99")
    parser.add_argument("--num_images", type=int, required=True,
                        help="Evaluation images (0 = the full 10 000-image test set, which is what published results use)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Substring filter on the model name (default: all)")
    args = parser.parse_args()

    # Resolve paths; this rebinds the TFLITE_DIR, EDGETPU_DIR, ... globals
    resolve_paths(args.platform_dir, args.results_dir)
    args.data_dir = str(resolve_data_dir(args.data_dir, args.platform_dir))

    num_images = args.num_images if args.num_images > 0 else None
    devices_to_run = ["cpu", "tpu"] if args.device == "both" else [args.device]
    require_tpu = "tpu" in devices_to_run

    print("=" * 72)
    print("SPARTA — BENCHMARK (CIFAR-100)")
    print(f"  Device(s)    : {', '.join(devices_to_run)}")
    print(f"  Platform dir : {args.platform_dir}")
    print(f"  Results dir  : {args.results_dir}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Models in    : {TFLITE_DIR}")
    print(f"  Warmup       : {args.warmup}")
    print(f"  Runs         : {args.runs}")
    print(f"  Images       : {num_images or 10000}")
    print("=" * 72)

    compiler_metrics = load_compiler_metrics()
    if not compiler_metrics:
        print("[WARN] compiler_metrics.json not found; the TPU memory-regime metrics will be missing.")

    fp32_acc = load_fp32_accuracy()
    if fp32_acc:
        print(f"[INFO] fp32_accuracy.json found ({len(fp32_acc)} models); FP32 CPU accuracy "
              f"will be imported from it rather than recomputed, which on a Pi is "
              f"about a thousand times faster and gives the same numbers.")
    else:
        print(f"[INFO] no fp32_accuracy.json in {FP32_ACC_FILE.parent.name}/; "
              f"FP32 CPU accuracy will be measured locally, which is slow on a Pi. "
              f"To skip it, produce that file first with aggregate_pruning_logs.py "
              f"on the machine that ran the pruning.")

    discovered = discover_models(require_tpu_compiled=require_tpu)
    if args.models:
        discovered = [m for m in discovered if any(p in m["name"] for p in args.models)]
    if not discovered:
        print("No model found."); return

    # Load the existing JSON so this pass merges into it
    models_dict = load_existing()
    if models_dict:
        print(f"[INFO] {len(models_dict)} model(s) already in {RESULTS_JSON.name}; "
              f"this pass will fill in their missing fields.")

    print(f"\n{len(discovered)} model(s) to measure:")
    for m in discovered:
        imp = f" ({m['importance']})" if m['importance'] else ""
        print(f"  {m['name']:55s} [{m['tag']}{imp}]")

    for cfg in discovered:
        name = cfg["name"]
        imp = f" ({cfg['importance']})" if cfg["importance"] else ""
        print(f"\n{'─' * 72}\n  {name.upper()}{imp}\n{'─' * 72}")

        # Static stats: cheap, so always refreshed
        static = measure_static(cfg, compiler_metrics)
        models_dict.setdefault(name, {}).update(static)
        r = models_dict[name]

        print(f"  Size: {r['size_int8_mib']:.2f} MiB  Params: {r['params_int8']:,}  "
              f"MACs: {r['macs_int8']:,}")
        print(f"  TPU SRAM: {r['tpu_on_chip_mib']:.2f} MiB ({r['tpu_sram_util_pct']:.0f}% / 8 MiB)  "
              f"Stream: {r['tpu_off_chip_mib']:.2f} MiB  "
              f"StrR: {fmt(r.get('tpu_streaming_ratio'), '.2f')}  "
              f"Ops: {r['tpu_ops_count']} TPU / {r['cpu_ops_count']} CPU")

        # Load the evaluation samples once per model; input size is model-dependent
        import tflite_runtime.interpreter as tfl
        tmp = tfl.Interpreter(model_path=cfg["int8_path"]); tmp.allocate_tensors()
        in_size = tmp.get_input_details()[0]["shape"][1]
        del tmp
        print(f"  Chargement val ({in_size}×{in_size})...")
        samples = prepare_val_samples(args.data_dir, in_size, num_images)
        print(f"  {len(samples)} images")

        # Mesures par device
        #
        # top1_int8_pct is deliberately NOT taken from the CPU path. The Coral
        # stack's tflite_runtime 2.5 (frozen since 2021) misreads the
        # quantization ai-edge-quantizer produces: the output collapses onto the
        # zero point, every prediction lands on the same class, and top-1 comes
        # out at chance level. The failure is silent, which is what makes it
        # dangerous. The Edge TPU path, through libedgetpu, is the only source
        # for top1_int8_pct.
        # CPU INT8 LATENCY is still measured and still valid: timing does not
        # depend on the output being correct.
        if "cpu" in devices_to_run:
            r.update(measure_cpu(cfg, samples, args.warmup, args.runs, fp32_acc=fp32_acc))

        if "tpu" in devices_to_run:
            r.update(measure_tpu(cfg, samples, args.warmup, args.runs))

        # -- Save after every model, not at the end -------------------------
        # Costs 50-200 ms per model to rewrite the JSON and CSV, negligible against
        # the 30-300 s each model takes to measure. If the Pi dies mid-sweep
        # (thermal throttling, power loss, a killed job), everything measured so
        # far survives instead of being lost. On a re-run, load_existing()
        # restores those models, and an existing measurement is only overwritten
        # by a fresh measurement of the same device.
        compute_cross(models_dict)
        save_results(models_dict, devices_to_run, args)
        write_csv(models_dict)

    # Final summary. The JSON and CSV are already current from the last
    # iteration; only the table has not been printed yet.
    print_table(models_dict)


if __name__ == "__main__":
    main()

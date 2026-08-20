#!/usr/bin/env python3
"""
02_convert_tflite_int8.py — PyTorch to INT8 TFLite, for the single-TPU axis.

WHAT THIS PRODUCES
    For every baseline in models/ and every pruned model in pytorch_pruned/:
        onnx/<name>.onnx                    intermediate, kept for debugging
        tflite_float32/<name>_float32.tflite intermediate, NHWC
        tflite_int8/<name>_int8.tflite       the artefact that gets compiled

    The INT8 file has INT8 input and output tensors, not float ones with
    quantise/dequantise wrappers, because the Edge TPU compiler maps a float
    boundary onto the CPU and that shows up as CPU fallback operations in the
    latency measurement.

WHY THE CONVERSION GOES THROUGH ONNX AND onnx2tf
    PyTorch is NCHW, TFLite and the Edge TPU are NHWC. There is no direct
    exporter that both handles the layout change and produces a graph the Edge
    TPU compiler accepts, so the path is:

        PyTorch -> ONNX (NCHW) -> onnx2tf -> TFLite float32 (NHWC)
                -> ai-edge-quantizer -> TFLite INT8

WHY POST-TRAINING QUANTIZATION AND NOT QUANTIZATION-AWARE TRAINING
    Three reasons, in order of weight:
      1. Comparability. The PruningBench leaderboard this work aligns with does
         not use QAT, so introducing it would break the comparison.
      2. Isolation. The object of study is the effect of the pruning criterion.
         QAT would let quantisation recover accuracy by adapting the weights,
         which entangles the two effects and makes the pruning result harder to
         attribute.
      3. It keeps a measurement available: with PTQ, the accuracy lost between
         FP32 and INT8 is a property of the pruned model, so it can be plotted
         per criterion. That plot (how quantisation-friendly each criterion's
         output is) only exists because quantisation is not allowed to adapt.

    The cost is that PTQ accuracy is a floor, not a ceiling. Reported INT8
    accuracies would be higher with QAT; the ranking between criteria is what
    matters here, and it is measured under identical conditions.

CALIBRATION
    100 images drawn from the CIFAR-100 training split at a fixed seed,
    normalised with the same CIFAR-100 statistics used in training. Drawing from
    train and not test matters: calibrating on the test set would leak it into
    the quantisation parameters and inflate every accuracy reported downstream.

TWO PITFALLS THIS SCRIPT WORKS AROUND
    - The legacy TorchScript ONNX exporter is used deliberately
      (dynamo=False). PyTorch 2.9's newer exporter emits
      _native_batch_norm_legit_no_training, which edgetpu_compiler rejects with
      "non-broadcastable operands" once the graph has been moved to NHWC. The
      legacy exporter folds batch norm into a standard ONNX BatchNormalization,
      which survives both quantisation and compilation.
    - Temporary directories are suffixed with the process ID. Parallel SLURM
      tasks convert the same baseline concurrently (it is shared by every task of
      a given model), and without the suffix they overwrite each other's
      intermediates and produce truncated files.

USAGE (pytorch-env, after setup/apply_patches.py has been run)
    python 02_convert_tflite_int8.py --data_dir ./data --num_calib 100
    python 02_convert_tflite_int8.py --data_dir ./data --models resnet18
"""

import argparse
import sys
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

# Imports requis pour torch.load(weights_only=False)
import cifar_resnet  # noqa: F401
import cifar_vgg  # noqa: F401
import wrn  # noqa: F401


# Working directory: holds models/ and pytorch_pruned/ as inputs, and receives
# onnx_models/, tflite_float32/ and tflite_int8/. Defaults to this script's own
# directory, which is the layout every published result was produced with;
# --work_dir (resolved in main) relocates the whole set together.
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
PRUNED_DIR = BASE_DIR / "pytorch_pruned"
ONNX_DIR = BASE_DIR / "onnx_models"
FLOAT_DIR = BASE_DIR / "tflite_float32"
INT8_DIR = BASE_DIR / "tflite_int8"


def _set_work_dir(root: Path) -> None:
    """Repoint every derived directory at `root`.

    Rebinds the module-level globals rather than threading a path through every
    function, which keeps the change additive: callers that never pass
    --work_dir see exactly the previous behaviour.
    """
    global BASE_DIR, MODELS_DIR, PRUNED_DIR, ONNX_DIR, FLOAT_DIR, INT8_DIR
    BASE_DIR = root
    MODELS_DIR = root / "models"
    PRUNED_DIR = root / "pytorch_pruned"
    ONNX_DIR = root / "onnx_models"
    FLOAT_DIR = root / "tflite_float32"
    INT8_DIR = root / "tflite_int8"
    _make_output_dirs()


def _make_output_dirs() -> None:
    """Create the three output directories, which the conversion writes into.

    Called at import for the default layout and again from _set_work_dir(), since
    rebinding the paths does not create the directories they now point at. That
    omission is not loud: the ONNX export fails per model with a missing-file
    error rather than a missing-directory one, and the script still exits 0.
    """
    for d in (ONNX_DIR, FLOAT_DIR, INT8_DIR):
        d.mkdir(parents=True, exist_ok=True)


_make_output_dirs()

MODEL_SIZES = {
    "resnet18":      32,
    "resnet50":      32,
    "vgg19":         32,
    "mobilenetv2":   32,
    "mnasnet1_0":    32,  # archived: the .pt still loads, but it is out of the lineup
    "googlenet":     32,
    "squeezenet1_1": 32,
    "wrn_28_10":     32,
}

CIFAR_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
CIFAR_STD = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)


# ─────────────────────────────────────────────
# Dataset de calibration CIFAR-100 (NHWC, float32)
# ─────────────────────────────────────────────
def _cifar_train_arrays(data_dir):
    """Load the CIFAR-100 training images as a (N, 32, 32, 3) uint8 array."""
    ds = datasets.CIFAR100(root=str(data_dir), train=True, download=True, transform=None)
    return ds.data  # numpy uint8 (50000, 32, 32, 3)


def _resize_nhwc(img_uint8, target_size):
    """Resize bilinear NHWC uint8 → NHWC uint8."""
    if target_size == 32:
        return img_uint8
    from PIL import Image
    pil = Image.fromarray(img_uint8).resize((target_size, target_size), Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def load_calib_nhwc(data_dir, input_size, input_name, num, calib_seed=42):
    """Load the calibration sample as [{input_name: ndarray (1, H, W, 3) float32}].

    Images come from the CIFAR-100 TRAINING split, never the test split:
    calibrating on test would leak it into the quantization scales and inflate
    every accuracy reported downstream.

    `calib_seed` selects which images are drawn. Varying it and re-quantizing is
    how the sensitivity of the INT8 result to the calibration draw is measured,
    which is a real source of variance in post-training quantization and is
    otherwise invisible.
    """
    arr = _cifar_train_arrays(data_dir)
    rng = np.random.RandomState(calib_seed)
    idx = rng.choice(len(arr), size=min(num, len(arr)), replace=False)

    samples = []
    for i in idx:
        img = _resize_nhwc(arr[i], input_size).astype(np.float32) / 255.0
        img = (img - CIFAR_MEAN) / CIFAR_STD
        samples.append({input_name: np.expand_dims(img.astype(np.float32), 0)})
    print(f"    [calib seed={calib_seed}] {len(samples)} images CIFAR-100 "
          f"({input_size}×{input_size}, NHWC)")
    return samples


def representative_dataset_factory(data_dir, input_size, num=200, calib_seed=42):
    """Build the calibration sample generator for post-training quantization.

    Draws `num_calib` images from the CIFAR-100 TRAINING split at a fixed seed and
    normalises them exactly as during training. Quantization reads this sample to
    choose the scale and zero point of every activation tensor, so the distribution
    it sees must match what the model will see at inference.

    Sampling from train, never test, is deliberate: calibrating on the test set
    would leak it into the quantization parameters and inflate every accuracy
    reported downstream. The seed is exposed so the sensitivity of the result to the
    calibration draw can itself be measured.
    """
    arr = _cifar_train_arrays(data_dir)
    rng = np.random.RandomState(calib_seed)
    idx = rng.choice(len(arr), size=min(num, len(arr)), replace=False)

    def gen():
        """Yield one calibration sample at a time, in the layout the quantizer expects."""
        for i in idx:
            img = _resize_nhwc(arr[i], input_size).astype(np.float32) / 255.0
            img = (img - CIFAR_MEAN) / CIFAR_STD
            yield [np.expand_dims(img.astype(np.float32), 0)]
    return gen


# ---------------------------------------------
# Stage 1: PyTorch -> ONNX
# ─────────────────────────────────────────────
def export_to_onnx(model, input_size, onnx_path):
    """Export a PyTorch model to ONNX at opset 18 via the legacy TorchScript exporter.

    dynamo=False is not a leftover. PyTorch 2.9's newer exporter emits
    _native_batch_norm_legit_no_training, which edgetpu_compiler later rejects with
    "non-broadcastable operands" once onnx2tf has moved the graph to NHWC. The legacy
    exporter folds batch norm into a standard ONNX BatchNormalization, which survives
    quantization and compilation, and as a side benefit keeps the weights inline
    instead of writing an external .data file.

    Simplification with onnxsim is attempted and treated as optional: it shrinks the
    graph when it works, and a failure is logged and ignored rather than fatal.
    """
    model = model.eval().cpu()
    # DenseNet safety: memory_efficient checkpointing is not traceable to ONNX.
    for mod in model.modules():
        if hasattr(mod, "memory_efficient"):
            mod.memory_efficient = False
    dummy = torch.randn(1, 3, input_size, input_size)

    print(f"    [onnx] Exporting (opset 18, legacy TorchScript exporter)...")
    # Legacy exporter (dynamo=False) on purpose: PyTorch 2.9's newer exporter
    # emits _native_batch_norm_legit_no_training, which edgetpu_compiler rejects
    # with "non-broadcastable operands" after the NHWC conversion. The legacy
    # exporter folds batch norm into a standard ONNX BatchNormalization, which
    # quantises and compiles cleanly, and keeps weights inline (no .data file).
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )

    try:
        import onnxsim, onnx
        m = onnx.load(onnx_path)
        m_simp, ok = onnxsim.simplify(m)
        if ok:
            onnx.save(m_simp, onnx_path)
            print(f"    [onnx] Simplified OK")
        else:
            print(f"    [onnx] Simplification failed, keeping the original")
    except Exception as e:
        print(f"    [onnx] Simplification skipped: {e}")

    size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"    → {onnx_path} ({size:.1f} MiB)")


# ---------------------------------------------
# Stage 2: ONNX -> float32 TFLite (NHWC)
# ─────────────────────────────────────────────
def convert_onnx_to_tflite(onnx_path, float_path, model_name):
    """Convert ONNX (NCHW) to float32 TFLite (NHWC) with onnx2tf.

    The layout change is the point: the Edge TPU is NHWC, so a converter that only
    rewraps the graph would leave transposes on every convolution.

    The temporary directory carries the process ID because parallel SLURM tasks
    convert the same baseline at the same time (it is shared across all tasks of one
    model); without the suffix they overwrite each other and yield truncated files.
    """
    print(f"    [onnx2tf] Converting -> float32 TFLite (NHWC)...")
    # The tmp dir carries the PID: parallel SLURM tasks convert the same
    # baseline concurrently (it is shared by every task of a given model), and
    # without the suffix they overwrite each other's intermediates.
    tmp_dir = FLOAT_DIR / f"_tmp_{model_name}_pid{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    import onnx2tf
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(tmp_dir),
        non_verbose=True,
        copy_onnx_input_output_names_to_tflite=True,
    )

    candidates = list(tmp_dir.glob("*.tflite"))
    if not candidates:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise FileNotFoundError(f"onnx2tf produced no .tflite for {model_name}")

    src = None
    for c in candidates:
        if "float32" in c.name:
            src = c; break
    if src is None:
        src = max(candidates, key=lambda p: p.stat().st_size)

    shutil.copy2(str(src), float_path)
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    size = os.path.getsize(float_path) / (1024 * 1024)
    print(f"    → {float_path} ({size:.1f} MiB)")


# ---------------------------------------------
# Stage 3: float32 TFLite -> INT8 TFLite
# ─────────────────────────────────────────────
def quantize_to_int8(float_path, int8_path, input_size, data_dir, num_calib, calib_seed=42):
    """Quantize float32 TFLite to INT8, preferring ai-edge-quantizer.

    Falls back to the TFLiteConverter path if ai-edge-quantizer raises, which happens
    on a few architectures whose graphs it cannot handle. The fallback produces a
    valid INT8 model but through a different code path, so a model that took it is
    worth noting when comparing results.
    """
    try:
        _quantize_ai_edge(float_path, int8_path, input_size, data_dir, num_calib, calib_seed)
    except Exception as e:
        print(f"    [quant] ai-edge-quantizer failed: {type(e).__name__}: {e}")
        print(f"    [quant] Falling back to TFLiteConverter int8...")
        _quantize_via_onnx2tf(int8_path, input_size, data_dir, calib_seed)


def _quantize_ai_edge(float_path, int8_path, input_size, data_dir, num_calib, calib_seed=42):
    """Quantize with ai-edge-quantizer: INT8 weights, activations, input and output.

    Input and output are forced to INT8 rather than left as float. A float boundary
    would make the compiler insert quantise and dequantise operations, which it maps
    onto the CPU; those show up as CPU fallback operations and add host-side latency
    to every inference, contaminating the measurement this pipeline exists to make.

    Note that models produced here must be read with ai_edge_litert, not with
    tflite_runtime 2.5, which saturates their output at the zero point silently.
    """
    from ai_edge_quantizer import quantizer, qtyping
    import ai_edge_litert.interpreter as tfl

    print(f"    [quant] Quantizing -> int8 (ai-edge-quantizer)...")
    interp = tfl.Interpreter(model_path=float_path)
    interp.allocate_tensors()

    sig = interp.get_signature_list()
    if sig:
        sig_key = list(sig.keys())[0]
        input_name = sig[sig_key]["inputs"][0]
    else:
        inp_details = interp.get_input_details()[0]
        input_name = inp_details["name"]
        sig_key = "serving_default"

    inp_details = interp.get_input_details()[0]
    print(f"    [quant] Input : shape={inp_details['shape']}, name={input_name}")
    del interp

    q = quantizer.Quantizer(float_path)
    q.add_static_config(regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=8, weight_num_bits=8)
    q.add_static_config(regex=".*",
        operation_name=qtyping.TFLOperationName.INPUT,
        activation_num_bits=8, weight_num_bits=8)
    q.add_static_config(regex=".*",
        operation_name=qtyping.TFLOperationName.OUTPUT,
        activation_num_bits=8, weight_num_bits=8)

    calib_samples = load_calib_nhwc(data_dir, input_size, input_name, num_calib, calib_seed)
    calib_result = q.calibrate({sig_key: calib_samples})
    result = q.quantize(calib_result)

    with open(int8_path, "wb") as f:
        f.write(result.quantized_model)

    size = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"    → {int8_path} ({size:.1f} MiB)")


def _quantize_via_onnx2tf(int8_path, input_size, data_dir, calib_seed=42):
    """Fallback quantization through TFLiteConverter, used when ai-edge-quantizer fails."""
    import tensorflow as tf

    model_name = Path(int8_path).stem.replace("_int8", "")
    onnx_path = str(ONNX_DIR / f"{model_name}.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX introuvable : {onnx_path}")

    # PID-suffixed tmp dir, same race as in convert_onnx_to_tflite()
    tmp_dir = str(FLOAT_DIR / f"_tmp_sm_{model_name}_pid{os.getpid()}")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"    [fallback] onnx2tf → SavedModel...")
    import onnx2tf
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tmp_dir,
        non_verbose=True,
        copy_onnx_input_output_names_to_tflite=True,
        tflite_backend="tf_converter",
    )

    print(f"    [fallback] TFLiteConverter → int8...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_factory(data_dir, input_size, 200, calib_seed)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(int8_path, "wb") as f:
        f.write(tflite_model)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    size = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"    → {int8_path} ({size:.1f} MiB) [TFLiteConverter int8]")


# ---------------------------------------------
# Per-model pipeline
# ─────────────────────────────────────────────
def convert_model(name, pt_path, input_size, data_dir, num_calib, force=False, calib_seed=42):
    """Run the three conversion stages for one model and report which ones succeeded.

    Each stage is caught separately so that a failure is attributed to the stage that
    produced it (ONNX export, onnx2tf, or quantization) rather than surfacing as one
    opaque error at the end. Already-converted models are skipped unless --force.
    """
    # A non-default calibration seed gets its own filename suffix, so several
    # calibration draws of one model can coexist. seed=42 keeps the plain name,
    # which is what every published artefact uses.
    cseed_suffix = "" if calib_seed == 42 else f"_cseed{calib_seed}"
    onnx_path = str(ONNX_DIR / f"{name}.onnx")
    float_path = str(FLOAT_DIR / f"{name}_float32.tflite")
    int8_path = str(INT8_DIR / f"{name}_int8{cseed_suffix}.tflite")

    if os.path.exists(int8_path) and os.path.getsize(int8_path) > 0 and not force:
        print(f"    Already converted: {name}_int8.tflite, skipping.")
        return True

    print(f"    Chargement {pt_path}...")
    model = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    model.eval()
    if hasattr(model, "aux_logits"):
        model.aux_logits = False

    try:
        if not os.path.exists(onnx_path) or force:
            export_to_onnx(model, input_size, onnx_path)
        else:
            print(f"    [onnx] Existe, skip.")
    except Exception as e:
        print(f"    [FAILED] ONNX export: {e}")
        import traceback; traceback.print_exc()
        return False

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        if not os.path.exists(float_path) or force:
            convert_onnx_to_tflite(onnx_path, float_path, name)
        else:
            print(f"    [onnx2tf] Float32 existe, skip.")
    except Exception as e:
        print(f"    [FAILED] onnx2tf: {e}")
        import traceback; traceback.print_exc()
        return False

    try:
        quantize_to_int8(float_path, int8_path, input_size, data_dir, num_calib, calib_seed)
    except Exception as e:
        print(f"    [FAILED] Quantization: {e}")
        import traceback; traceback.print_exc()
        return False

    return True


# ---------------------------------------------
# Model discovery
# ─────────────────────────────────────────────
def discover_models(args):
    """Find every .pt to convert: the baselines and all pruned variants.

    Baselines are written out with a `_finetuned` suffix. The suffix is semantically
    redundant (they are simply the unpruned models) but is kept because the whole
    downstream chain, benchmark and analysis alike, identifies the baseline row by
    that name.
    """
    to_convert = []

    if MODELS_DIR.exists():
        for pt_file in sorted(MODELS_DIR.glob("*.pt")):
            stem = pt_file.stem
            if args.models and stem not in args.models:
                continue
            size = MODEL_SIZES.get(stem)
            if size is None:
                continue
            # The `_finetuned` suffix is semantically redundant (this is simply
            # the unpruned baseline) but the whole downstream chain, benchmark and
            # analysis alike, identifies the baseline row by that name.
            name = f"{stem}_finetuned"
            to_convert.append({"name": name, "pt_path": pt_file,
                               "input_size": size, "category": "finetuned"})

    for pt_file in sorted(PRUNED_DIR.glob("*.pt")):
        stem = pt_file.stem
        base_name = stem.split("_pruned")[0]
        if args.models and base_name not in args.models:
            continue
        size = MODEL_SIZES.get(base_name)
        if size is None:
            print(f"  [WARN] Unknown input size for {base_name}, skipping.")
            continue
        to_convert.append({"name": stem, "pt_path": pt_file,
                           "input_size": size, "category": "pruned"})

    return to_convert


def main():
    """Parse arguments, discover the models to convert, and convert each in turn."""
    parser = argparse.ArgumentParser(
        description="Convert CIFAR-100 models from PyTorch to INT8 TFLite")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the `cifar-100-python/` subfolder "
                             "(Krizhevsky's original binary pickle format)")
    parser.add_argument("--num_calib", type=int, required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--calibration_seed", type=int, default=42,
                        help="Seed for drawing the calibration sample. Vary it and re-run to "
                             "measure how sensitive the INT8 result is to the calibration "
                             "draw. seed=42 keeps the plain filename; any other seed "
                             "appends _cseed{N}.")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Directory holding models/ and pytorch_pruned/, and "
                             "receiving onnx_models/, tflite_float32/ and tflite_int8/. "
                             "Defaults to this script's own directory, which is the "
                             "layout the published runs used.")
    args = parser.parse_args()

    if args.work_dir:
        _set_work_dir(Path(args.work_dir))

    print("=" * 60)
    print("PYTORCH -> INT8 TFLITE CONVERSION (CIFAR-100)")
    print("  Pipeline : ONNX -> onnx2tf (NHWC) -> ai-edge-quantizer")
    print(f"  Work dir : {BASE_DIR}")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Calib    : {args.num_calib} images (seed={args.calibration_seed})")
    print("=" * 60)

    to_convert = discover_models(args)
    print(f"\n{len(to_convert)} models to convert.")

    success, failed = [], []
    for entry in to_convert:
        print(f"\n{'─' * 60}")
        print(f"  {entry['name'].upper()} ({entry['input_size']}×{entry['input_size']}, "
              f"{entry['category']})")
        print(f"{'─' * 60}")
        ok = convert_model(name=entry["name"], pt_path=entry["pt_path"],
                           input_size=entry["input_size"],
                           data_dir=args.data_dir, num_calib=args.num_calib,
                           force=args.force, calib_seed=args.calibration_seed)
        (success if ok else failed).append(entry["name"])

    print(f"\n{'=' * 60}")
    print("CONVERSION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Succeeded: {len(success)} {success}")
    print(f"  Failed   : {len(failed)} {failed}")

    if INT8_DIR.exists():
        print(f"\nINT8 models:")
        for f in sorted(INT8_DIR.iterdir()):
            if f.suffix == ".tflite":
                print(f"  {f.name:55s} {f.stat().st_size/1024/1024:8.1f} MiB")

    # Exit non-zero when nothing converted. Individual failures are expected and
    # must not abort the batch, but a run where EVERY model failed is a setup
    # problem (a missing directory, a broken environment), and returning 0 makes
    # a scripted pipeline carry on as if it had produced models.
    return 1 if (failed and not success) else 0


if __name__ == "__main__":
    sys.exit(main())

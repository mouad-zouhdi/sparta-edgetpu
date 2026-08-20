#!/usr/bin/env python3
"""
02_convert_pruned.py — convert a pruned ImageNet checkpoint to INT8 TFLite.

WHAT THIS PRODUCES
    <output_dir>/<input_basename>_int8.tflite

    One model per invocation, unlike the single-TPU converter which sweeps a
    directory. That is because it sits inside pipeline_full.py's guided loop,
    which converts exactly one candidate per iteration and needs the result
    before deciding what to do next.

PIPELINE
    PyTorch (.pt, full architecture) -> ONNX opset 18 -> onnx2tf (NHWC)
    -> ai_edge_quantizer INT8, calibrated on 100 ImageNet training images at
    seed 42.

WHY THE CHECKPOINT IS LOADED WITH weights_only=False
    Structured pruning changes the architecture, so these checkpoints store the
    whole model rather than a state dict: there is no unpruned class definition
    that could rebuild them. Unpickling therefore has to execute the stored class
    references, which is safe here because we produced the files ourselves.

CALIBRATION USES IMAGENET, NOT CIFAR
    The models are ImageNet models and will be evaluated on ImageNet, so the
    quantization scales must be chosen from the distribution they will actually
    see. Preprocessing (mean, std, BGR ordering, input range) comes from
    model_zoo.get_preprocessing(), since three incompatible conventions coexist
    in this lineup and applying the wrong one degrades accuracy silently.

USAGE (pytorch-env)
    python 02_convert_pruned.py \\
        --input pytorch_pruned_imagenet/resnet50_..._PREFT.pt \\
        --model resnet50 \\
        --data_dir /datasets/Imagenet_1k \\
        --output_dir tflite_int8_pruned/ \\
        --num_calib 100
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# model_zoo is the single source of truth for per-model preprocessing.
sys.path.insert(0, str(Path(__file__).parent))
from model_zoo import get_preprocessing, get_input_size, ALL_MODEL_NAMES


# ─────────────────────────────────────────────
# Calibration ImageNet
# ─────────────────────────────────────────────
def _list_imagenet_train_images(data_dir: str, n: int, seed: int = 42):
    """Draw n image paths from the ImageNet training split at a fixed seed.

    Training split, never validation: calibrating on the evaluation data would
    leak it into the quantization scales and inflate every accuracy reported
    afterwards.
    """
    train_root = Path(data_dir) / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"{train_root} introuvable")
    # Take the first .JPEG of each class until N images are collected. This
    # avoids a full directory scan and still spreads the sample across classes,
    # which is what the calibration needs.
    all_files = []
    for wnid_dir in sorted(train_root.iterdir()):
        if not wnid_dir.is_dir():
            continue
        for f in wnid_dir.iterdir():
            if f.suffix.upper() in {".JPEG", ".JPG", ".PNG"}:
                all_files.append(str(f))
                break  # 1 par classe
        if len(all_files) >= 2 * n:
            break
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(all_files), size=min(n, len(all_files)), replace=False)
    return [all_files[i] for i in idx]


def _load_and_preprocess(path: str, cfg: dict) -> np.ndarray:
    """Charge une image, resize + preprocess selon `cfg` → NHWC float32."""
    size = cfg["input_size"]
    with Image.open(path) as img:
        img = img.convert("RGB")
        scale = int(round(size * 256 / 224))
        img = img.resize((scale, scale), Image.BICUBIC)
        # CenterCrop → size×size
        left = (scale - size) // 2
        top = (scale - size) // 2
        img = img.crop((left, top, left + size, top + size))
        arr = np.array(img, dtype=np.float32)  # HWC uint8→float32 direct
    if cfg.get("input_range_255", False):
        pass  # BN-Inception : pas de /255
    else:
        arr = arr / 255.0
    if cfg["bgr"]:
        arr = arr[..., ::-1]
    mean = np.array(cfg["mean"], dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)
    arr = (arr - mean) / std
    return np.ascontiguousarray(arr)


def make_imagenet_calib_dataset(data_dir: str, name: str, n: int, seed: int = 42):
    """Build the calibration sample the quantizer reads to choose its scales.

    Returns a list of float32 (1, H, W, 3) arrays, already preprocessed with this
    model's own convention.
    """
    cfg = get_preprocessing(name)
    paths = _list_imagenet_train_images(data_dir, n, seed)
    samples = []
    for p in paths:
        arr = _load_and_preprocess(p, cfg)
        samples.append(np.expand_dims(arr, 0))
    return samples


# ─────────────────────────────────────────────
# PyTorch (.pt) → ONNX
# ─────────────────────────────────────────────
def export_to_onnx(model: torch.nn.Module, input_size: int, onnx_path: str):
    """Export the pruned model to ONNX at opset 18, via the legacy TorchScript exporter.

    dynamo=False is deliberate: PyTorch's newer exporter emits a batch-norm operation
    that edgetpu_compiler rejects once the graph has been moved to NHWC.
    """
    model = model.eval().cpu()
    for mod in model.modules():
        if hasattr(mod, "memory_efficient"):
            mod.memory_efficient = False
    if hasattr(model, "aux_logits"):
        model.aux_logits = False
    if hasattr(model, "AuxLogits"):
        model.AuxLogits = None

    dummy = torch.randn(1, 3, input_size, input_size)
    print(f"    [onnx] Export (opset 18)…", flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["input"], output_names=["output"],
            dynamic_axes=None, opset_version=18,
            do_constant_folding=True, dynamo=False,
        )
    try:
        import onnxsim, onnx
        m = onnx.load(onnx_path)
        m_simp, ok = onnxsim.simplify(m)
        if ok:
            onnx.save(m_simp, onnx_path)
            print(f"    [onnx] Simplified OK")
    except Exception as e:
        print(f"    [onnx] Simplification skipped: {e}")
    size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"    → {onnx_path} ({size:.1f} MiB)")


# ─────────────────────────────────────────────
# ONNX → TFLite float32 (NHWC)
# ─────────────────────────────────────────────
def convert_onnx_to_tflite(onnx_path: str, float_path: str, name: str,
                           staging_root: Path):
    """Convert ONNX (NCHW) to float32 TFLite (NHWC) with onnx2tf."""
    print(f"    [onnx2tf] Conversion → TFLite float32 (NHWC)…", flush=True)
    tmp_dir = staging_root / f"_tmp_{name}_pid{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        import onnx2tf
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(tmp_dir),
            non_verbose=True,
            copy_onnx_input_output_names_to_tflite=True,
        )
        candidates = list(tmp_dir.glob("*.tflite"))
        if not candidates:
            raise FileNotFoundError(f"onnx2tf n'a rien produit pour {name}")
        src = next((c for c in candidates if "float32" in c.name), None)
        if src is None:
            src = max(candidates, key=lambda p: p.stat().st_size)
        shutil.copy2(str(src), float_path)
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
    size = os.path.getsize(float_path) / (1024 * 1024)
    print(f"    → {float_path} ({size:.1f} MiB)")


# ─────────────────────────────────────────────
# TFLite float32 → TFLite int8 (ai_edge_quantizer)
# ─────────────────────────────────────────────
def quantize_to_int8(float_path: str, int8_path: str, data_dir: str,
                     name: str, num_calib: int, seed: int = 42):
    """Quantize float32 TFLite to INT8 with ai_edge_quantizer, input and output included.

    Forcing INT8 boundaries avoids the quantise/dequantise pair the compiler would
    otherwise map onto the CPU, which would show up as CPU fallback operations and
    add host-side latency to every inference.
    """
    from ai_edge_quantizer import quantizer, qtyping
    import ai_edge_litert.interpreter as tfl

    print(f"    [quant] Quantification → int8…", flush=True)
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

    calib_arrays = make_imagenet_calib_dataset(data_dir, name, num_calib, seed)
    calib_samples = [{input_name: a} for a in calib_arrays]
    cfg = get_preprocessing(name)
    print(f"    [calib] {len(calib_samples)} images ImageNet train "
          f"({cfg['input_size']}×{cfg['input_size']}, NHWC)")
    calib_result = q.calibrate({sig_key: calib_samples})
    result = q.quantize(calib_result)

    with open(int8_path, "wb") as f:
        f.write(result.quantized_model)
    size = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"    → {int8_path} ({size:.1f} MiB)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    """Convert one pruned checkpoint end to end: ONNX, float32 TFLite, INT8 TFLite."""
    p = argparse.ArgumentParser(
        description="Convert a pruned PyTorch checkpoint to INT8 TFLite "
                    "avec calibration ImageNet.")
    p.add_argument("--input", required=True,
                   help="The PREFT checkpoint to convert, as written by --prune_only")
    p.add_argument("--model", required=True, choices=ALL_MODEL_NAMES,
                   help="Model name, used to look up its preprocessing convention")
    p.add_argument("--data_dir", required=True,
                   help="Root ImageNet (contient train/) pour calibration")
    p.add_argument("--output_dir", required=True,
                   help="Dossier de sortie pour le .tflite int8")
    p.add_argument("--num_calib", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--staging_dir", default=None,
                   help="Scratch directory for onnx2tf (default: output_dir/_staging)")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[ERROR] input not found: {in_path}")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(args.staging_dir) if args.staging_dir else (out_dir / "_staging")
    staging.mkdir(parents=True, exist_ok=True)

    stem = in_path.stem
    onnx_path = staging / f"{stem}.onnx"
    float_path = staging / f"{stem}_float32.tflite"
    int8_path = out_dir / f"{stem}_int8.tflite"

    print(f"[02_convert_pruned] {args.model} — {in_path.name}", flush=True)
    print(f"[02_convert_pruned] output → {int8_path}", flush=True)

    # 1. Load the checkpoint. weights_only=False is required: pruning changed
    #    the architecture, so the file stores the whole model, not a state dict.
    t0 = time.time()
    model = torch.load(str(in_path), weights_only=False, map_location="cpu")
    print(f"    [load] PyTorch model loaded in {time.time()-t0:.1f}s", flush=True)

    input_size = get_input_size(args.model)

    # 2. Export ONNX
    t0 = time.time()
    export_to_onnx(model, input_size, str(onnx_path))
    print(f"    [onnx] {time.time()-t0:.1f}s", flush=True)

    # 3. ONNX → TFLite float32
    t0 = time.time()
    convert_onnx_to_tflite(str(onnx_path), str(float_path), args.model, staging)
    print(f"    [tflite_f32] {time.time()-t0:.1f}s", flush=True)

    # 4. TFLite float32 → TFLite int8
    t0 = time.time()
    quantize_to_int8(str(float_path), str(int8_path), args.data_dir,
                     args.model, args.num_calib, args.seed)
    print(f"    [tflite_int8] {time.time()-t0:.1f}s", flush=True)

    # Cleanup staging (garde le .onnx en debug ? non, on cleanup)
    for p_tmp in (onnx_path, float_path):
        if p_tmp.exists():
            p_tmp.unlink()
    # Try to remove staging dir if empty
    try:
        staging.rmdir()
    except OSError:
        pass

    final_size = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"\n[02_convert_pruned] DONE — {int8_path} ({final_size:.2f} MiB)",
          flush=True)


if __name__ == "__main__":
    main()

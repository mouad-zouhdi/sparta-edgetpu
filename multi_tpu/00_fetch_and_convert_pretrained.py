#!/usr/bin/env python3
"""
00_fetch_and_convert_pretrained.py — fetch ImageNet weights and test convertibility.

WHAT THIS PRODUCES
    onnx/<name>.onnx
    tflite_float32/<name>_float32.tflite
    tflite_int8/<name>_int8.tflite
    conversion_report.json    per model: which stages passed, and a consistency
                              check of the outputs

WHAT IT IS FOR
    Two things. First, it downloads the published ImageNet weights for the eight
    architectures of the multi-TPU axis, which is the starting point of the whole
    axis: nothing here is trained from scratch. Second, it is the cheap
    qualification step for a candidate architecture. Before committing an
    architecture to a pruning campaign that costs tens of GPU-hours, this
    establishes that it survives the full export chain at all.

    That matters because several plausible candidates do not. Inception V3 from
    torchvision compiles nowhere near the Edge TPU (see model_zoo.py);
    shufflenet, densenet and efficientnet_lite0 were rejected at this stage on
    conversion or input-size grounds. Finding that out here costs minutes.

PIPELINE
    PyTorch -> ONNX (legacy exporter, opset 18) -> onnx2tf (NHWC)
    -> ai-edge-quantizer (INT8)

CONSISTENCY CHECK
    After conversion, check_inference() runs the same fixed input through PyTorch
    FP32, TFLite FP32 and TFLite INT8, and compares cosine similarity, top-1
    agreement, top-5 overlap, and the presence of NaN or Inf. This catches a
    broken ONNX export or a saturating quantization immediately, rather than
    hours later as an inexplicably poor accuracy.

CALIBRATION NOTE
    Calibration here uses CIFAR-100 images resized to the model's input size.
    That is adequate for a convertibility test, whose question is whether the
    chain produces a valid model, not what its accuracy is. The pruning pipeline
    proper (02_convert_pruned.py) calibrates on ImageNet, which is what the
    accuracy numbers depend on.

SOURCES
    GoogLeNet (Inception V1)     torchvision.models.googlenet
    BN-Inception (Inception V2)  pretrainedmodels.bninception
    Inception V3                 timm (NOT torchvision; see model_zoo.py)
    Inception V4                 timm.inception_v4
    Inception-ResNet V2          timm.inception_resnet_v2
    ResNet-50/101/152            torchvision.models.resnet{50,101,152}

    BN-Inception's checkpoint is hosted at data.lip6.fr, whose TLS certificate
    has expired and which returns 503 intermittently. Retrieve
    bn_inception-52deb4733.pth through the Wayback Machine and place it in
    ~/.cache/torch/hub/checkpoints/.

ADDING AN ARCHITECTURE
    1. Write a _load_<name>() returning an nn.Module in eval mode, with
       aux_logits disabled if the architecture has auxiliary classifiers.
    2. Add a MODELS entry with loader, input_size, mean, std and bgr. Set
       input_range_255 only for BGR [0, 255] models, which so far means
       BN-Inception alone.
    3. Run with --models <name>, then re-run without it to rebuild the report.

USAGE (pytorch-env)
    python 00_fetch_and_convert_pretrained.py
    python 00_fetch_and_convert_pretrained.py --models resnet50 inception_v3
    python 00_fetch_and_convert_pretrained.py --num_calib 50 --force
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE_DIR = Path(__file__).parent
ONNX_DIR = BASE_DIR / "onnx"
FLOAT_DIR = BASE_DIR / "tflite_float32"
INT8_DIR = BASE_DIR / "tflite_int8"
DATA_DIR = BASE_DIR / "calib_data"
REPORT_PATH = BASE_DIR / "conversion_report.json"

for d in [ONNX_DIR, FLOAT_DIR, INT8_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Model definitions: canonical source plus preprocessing
# ─────────────────────────────────────────────
def _load_googlenet():
    """Load torchvision GoogLeNet and strip its auxiliary classifiers.

    aux_logits must be True at construction because the published checkpoint
    contains those weights; they are removed immediately afterwards so the ONNX
    export does not carry them.
    """
    import torchvision.models as tvm
    # torchvision requires aux_logits=True when loading the pretrained weights,
    # since the checkpoint contains the auxiliary classifiers. They are disabled
    # right after construction, as for Inception V3.
    m = tvm.googlenet(weights=tvm.GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=True)
    m.aux_logits = False
    m.aux1 = None
    m.aux2 = None
    return m


def _load_bninception():
    """Load BN-Inception (Inception V2); see the module docstring on its checkpoint."""
    # The original URL (data.lip6.fr) has an expired certificate and returns 503
    # intermittently. Retrieve bn_inception-52deb4733.pth through the Wayback
    # Machine and place it in ~/.cache/torch/hub/checkpoints/ before importing
    # pretrainedmodels, which picks up a cached file automatically.
    import pretrainedmodels
    m = pretrainedmodels.bninception(num_classes=1000, pretrained="imagenet")
    return m


def _load_inception_v3():
    """Load Inception-V3 from timm, whose graph the Edge TPU compiler accepts."""
    # torchvision's Inception V3 carries a `transform_input` block that becomes
    # Gather/Slice/Concat operations on the input channels, producing an
    # intermediate tensor the Edge TPU compiler rejects as too large. timm ships
    # the model without that wrapper, with the same original TF weights;
    # preprocessing is mean = std = 0.5.
    import timm
    return timm.create_model("inception_v3", pretrained=True)


def _load_inception_v4():
    """Load Inception-V4 from timm."""
    import timm
    return timm.create_model("inception_v4", pretrained=True)


def _load_inception_resnet_v2():
    """Load Inception-ResNet-V2 from timm."""
    import timm
    return timm.create_model("inception_resnet_v2", pretrained=True)


def _load_resnet(depth: int):
    """Load a torchvision ResNet of the given depth (50, 101 or 152)."""
    import torchvision.models as tvm
    fn = {50: tvm.resnet50, 101: tvm.resnet101, 152: tvm.resnet152}[depth]
    weights = {50: tvm.ResNet50_Weights, 101: tvm.ResNet101_Weights,
               152: tvm.ResNet152_Weights}[depth].IMAGENET1K_V1
    return fn(weights=weights)


# Stats de preprocessing en *RGB* (range [0,1]).
# BN-Inception is BGR with statistics in [0, 255]; those are expressed here as
# the equivalent RGB [0, 1] form, which is mathematically identical.
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)

MODELS = {
    "inception_v1_googlenet": {
        "loader": _load_googlenet,
        "input_size": 224,
        "mean": IMAGENET_RGB_MEAN,
        "std": IMAGENET_RGB_STD,
        "bgr": False,
    },
    "inception_v2_bninception": {
        "loader": _load_bninception,
        "input_size": 224,
        # bninception : BGR, range [0,255], mean=[104,117,128], std=[1,1,1].
        # preprocess() honours the `bgr` flag and applies the raw statistics.
        "mean": (104.0, 117.0, 128.0),
        "std": (1.0, 1.0, 1.0),
        "bgr": True,
        "input_range_255": True,
    },
    "inception_v3": {
        "loader": _load_inception_v3,
        "input_size": 299,
        # timm.inception_v3 utilise les stats Inception (mean=std=0.5)
        "mean": INCEPTION_MEAN,
        "std": INCEPTION_STD,
        "bgr": False,
    },
    "inception_v4": {
        "loader": _load_inception_v4,
        "input_size": 299,
        "mean": INCEPTION_MEAN,
        "std": INCEPTION_STD,
        "bgr": False,
    },
    "resnet50": {
        "loader": lambda: _load_resnet(50),
        "input_size": 224,
        "mean": IMAGENET_RGB_MEAN,
        "std": IMAGENET_RGB_STD,
        "bgr": False,
    },
    "resnet101": {
        "loader": lambda: _load_resnet(101),
        "input_size": 224,
        "mean": IMAGENET_RGB_MEAN,
        "std": IMAGENET_RGB_STD,
        "bgr": False,
    },
    "resnet152": {
        "loader": lambda: _load_resnet(152),
        "input_size": 224,
        "mean": IMAGENET_RGB_MEAN,
        "std": IMAGENET_RGB_STD,
        "bgr": False,
    },
    "inception_resnet_v2": {
        "loader": _load_inception_resnet_v2,
        "input_size": 299,
        "mean": INCEPTION_MEAN,
        "std": INCEPTION_STD,
        "bgr": False,
    },
}


# ─────────────────────────────────────────────
# Calibration: CIFAR-100 images upscaled to each model's input size. Adequate
# for a convertibility test; the pruning pipeline calibrates on ImageNet.
# ─────────────────────────────────────────────
_CIFAR_CACHE = {"arr": None}


def _cifar100_train_uint8() -> np.ndarray:
    """Retourne le train set CIFAR-100 en (50000, 32, 32, 3) uint8 RGB.
    Downloaded into DATA_DIR on first use (about 170 MB).
    """
    if _CIFAR_CACHE["arr"] is None:
        from torchvision import datasets
        ds = datasets.CIFAR100(root=str(DATA_DIR), train=True, download=True, transform=None)
        _CIFAR_CACHE["arr"] = ds.data  # (50000, 32, 32, 3) uint8 RGB
    return _CIFAR_CACHE["arr"]


def _resize_uint8(img_uint8: np.ndarray, target: int) -> np.ndarray:
    """Bicubic resize of a uint8 NHWC image to the target resolution."""
    if img_uint8.shape[0] == target and img_uint8.shape[1] == target:
        return img_uint8
    pil = Image.fromarray(img_uint8)
    pil = pil.resize((target, target), Image.BICUBIC)
    return np.array(pil, dtype=np.uint8)


def _sample_calib_uint8(n: int, size: int, seed: int = 42) -> np.ndarray:
    """N images de calibration uint8 (n,H,W,3) issues de CIFAR-100 train, resize bicubique."""
    arr = _cifar100_train_uint8()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
    out = np.empty((len(idx), size, size, 3), dtype=np.uint8)
    for k, i in enumerate(idx):
        out[k] = _resize_uint8(arr[i], size)
    return out


def preprocess_image_nhwc_float32(img_uint8: np.ndarray, cfg: dict) -> np.ndarray:
    """Convert uint8 RGB NHWC to the float32 NHWC input this model expects.

    Range and statistics are per model: three incompatible conventions coexist
    in this lineup and the wrong one costs accuracy without raising.
    """
    img = img_uint8.astype(np.float32)
    if cfg.get("input_range_255", False):
        # BN-Inception: no /255 here, its statistics are already in [0, 255]
        pass
    else:
        img = img / 255.0
    if cfg["bgr"]:
        img = img[..., ::-1]  # RGB → BGR
    mean = np.array(cfg["mean"], dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)
    img = (img - mean) / std
    return np.ascontiguousarray(img)


def make_calib_dataset(cfg: dict, n: int, seed: int = 42):
    """Build the calibration sample as float32 (1, H, W, 3) arrays."""
    size = cfg["input_size"]
    raw = _sample_calib_uint8(n, size, seed)
    samples = []
    for i in range(len(raw)):
        x = preprocess_image_nhwc_float32(raw[i], cfg)
        samples.append(np.expand_dims(x, 0))  # (1, H, W, 3)
    return samples


# ─────────────────────────────────────────────
# Stage 1: PyTorch -> ONNX
# ─────────────────────────────────────────────
def export_to_onnx(model: torch.nn.Module, input_size: int, onnx_path: str):
    """Export to ONNX at opset 18 via the legacy TorchScript exporter.

    dynamo=False is deliberate: the newer exporter emits a batch-norm operation the
    Edge TPU compiler rejects once the graph is in NHWC.
    """
    model = model.eval().cpu()
    for mod in model.modules():
        if hasattr(mod, "memory_efficient"):
            mod.memory_efficient = False
    # Inception V3 still honours aux_logits in forward(); the loader already
    # disabled it, this re-asserts it harmlessly.
    if hasattr(model, "aux_logits"):
        model.aux_logits = False

    dummy = torch.randn(1, 3, input_size, input_size)

    print(f"    [onnx] Export (opset 18, legacy TorchScript exporter)…")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
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


# ─────────────────────────────────────────────
# Stage 2: ONNX -> float32 TFLite (NHWC)
# ─────────────────────────────────────────────
def convert_onnx_to_tflite(onnx_path: str, float_path: str, name: str):
    """Convert ONNX (NCHW) to float32 TFLite (NHWC) with onnx2tf."""
    print(f"    [onnx2tf] Conversion → TFLite float32 (NHWC)…")
    tmp_dir = FLOAT_DIR / f"_tmp_{name}_pid{os.getpid()}"
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
        raise FileNotFoundError(f"onnx2tf produced no .tflite for {name}")

    src = None
    for c in candidates:
        if "float32" in c.name:
            src = c
            break
    if src is None:
        src = max(candidates, key=lambda p: p.stat().st_size)

    shutil.copy2(str(src), float_path)
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    size = os.path.getsize(float_path) / (1024 * 1024)
    print(f"    → {float_path} ({size:.1f} MiB)")


# ─────────────────────────────────────────────
# Stage 3: float32 TFLite -> INT8 TFLite
# ─────────────────────────────────────────────
def quantize_to_int8(float_path: str, int8_path: str, cfg: dict, num_calib: int):
    """Quantize float32 TFLite to INT8, with INT8 input and output tensors."""
    from ai_edge_quantizer import quantizer, qtyping
    import ai_edge_litert.interpreter as tfl

    print(f"    [quant] Quantification → int8 (ai-edge-quantizer)…")
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

    calib_arrays = make_calib_dataset(cfg, num_calib, seed=42)
    calib_samples = [{input_name: a} for a in calib_arrays]
    print(f"    [calib] {len(calib_samples)} images CIFAR-100 train "
          f"({cfg['input_size']}×{cfg['input_size']}, NHWC, bicubique)")
    calib_result = q.calibrate({sig_key: calib_samples})
    result = q.quantize(calib_result)

    with open(int8_path, "wb") as f:
        f.write(result.quantized_model)

    size = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"    → {int8_path} ({size:.1f} MiB)")


# ─────────────────────────────────────────────
# Inference consistency check
# ─────────────────────────────────────────────
def _run_tflite(path: str, x_nhwc: np.ndarray) -> np.ndarray:
    """Run one inference on a .tflite model, float32 or INT8 alike."""
    import ai_edge_litert.interpreter as tfl
    interp = tfl.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    # The ai-edge-quantizer config uses 8-bit I/O, so the input is quantized.
    if inp["dtype"] in (np.int8, np.uint8):
        scale, zero = inp["quantization"]
        x = np.round(x_nhwc / scale + zero).astype(inp["dtype"])
        x = np.clip(x, np.iinfo(inp["dtype"]).min, np.iinfo(inp["dtype"]).max).astype(inp["dtype"])
    else:
        x = x_nhwc.astype(np.float32)

    interp.set_tensor(inp["index"], x)
    interp.invoke()
    y = interp.get_tensor(out["index"])
    if out["dtype"] in (np.int8, np.uint8):
        scale, zero = out["quantization"]
        y = (y.astype(np.float32) - zero) * scale
    return y.astype(np.float32)


def check_inference(model: torch.nn.Module, float_path: str, int8_path: str,
                    cfg: dict, seed: int = 7) -> dict:
    """Compare PyTorch FP32, TFLite FP32 and TFLite INT8 on one fixed input.

    Reports cosine similarity, top-1 agreement, top-5 overlap and any NaN or Inf.
    This is the cheap check that catches a broken ONNX export or a saturating
    quantization at conversion time, instead of hours later as an unexplained
    accuracy collapse.
    """
    size = cfg["input_size"]
    raw = _sample_calib_uint8(1, size, seed=seed)
    x_nhwc = preprocess_image_nhwc_float32(raw[0], cfg)
    x_nhwc = np.expand_dims(x_nhwc, 0)  # (1,H,W,3)

    # PyTorch attend NCHW
    x_nchw = torch.from_numpy(np.transpose(x_nhwc, (0, 3, 1, 2)).copy())
    with torch.no_grad():
        y_pt = model.eval().cpu()(x_nchw)
        if isinstance(y_pt, tuple):
            y_pt = y_pt[0]
        y_pt = y_pt.cpu().numpy().reshape(-1)

    y_fp32 = _run_tflite(float_path, x_nhwc).reshape(-1)
    y_int8 = _run_tflite(int8_path, x_nhwc).reshape(-1)

    def _cos(a, b):
        """Cosine similarity, returning NaN rather than dividing by zero."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return float("nan")
        return float(np.dot(a, b) / (na * nb))

    top1_pt = int(np.argmax(y_pt))
    top1_fp32 = int(np.argmax(y_fp32))
    top1_int8 = int(np.argmax(y_int8))
    top5_pt = set(np.argsort(y_pt)[-5:].tolist())
    top5_int8 = set(np.argsort(y_int8)[-5:].tolist())

    has_nan = bool(np.isnan(y_int8).any() or np.isinf(y_int8).any()
                   or np.isnan(y_fp32).any() or np.isinf(y_fp32).any())

    return {
        "top1_pytorch": top1_pt,
        "top1_tflite_fp32": top1_fp32,
        "top1_tflite_int8": top1_int8,
        "top1_match_pt_fp32": top1_pt == top1_fp32,
        "top1_match_pt_int8": top1_pt == top1_int8,
        "cos_pt_vs_tflite_fp32": _cos(y_pt, y_fp32),
        "cos_pt_vs_tflite_int8": _cos(y_pt, y_int8),
        "top5_overlap_pt_int8": len(top5_pt & top5_int8),
        "has_nan_or_inf": has_nan,
        "output_shape_pytorch": list(y_pt.shape),
        "output_shape_tflite_fp32": list(y_fp32.shape),
        "output_shape_tflite_int8": list(y_int8.shape),
    }


# ─────────────────────────────────────────────
# Full pipeline for one model
# ─────────────────────────────────────────────
def convert_and_verify(name: str, cfg: dict, num_calib: int, force: bool = False) -> dict:
    """Run the full chain for one model and record which stages passed.

    Each stage is caught separately so a failure is attributed to the stage that
    caused it, and one architecture failing never aborts the others.
    """
    onnx_path = str(ONNX_DIR / f"{name}.onnx")
    float_path = str(FLOAT_DIR / f"{name}_float32.tflite")
    int8_path = str(INT8_DIR / f"{name}_int8.tflite")

    result = {
        "name": name,
        "input_size": cfg["input_size"],
        "status": "pending",
        "stages": {},
        "files": {},
        "error": None,
        "inference_check": None,
    }
    t0 = time.time()

    try:
        print(f"    [load] {name}…")
        model = cfg["loader"]()
        model.eval()
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"load: {type(e).__name__}: {e}"
        traceback.print_exc()
        return result
    result["stages"]["load"] = "ok"

    try:
        if not os.path.exists(onnx_path) or force:
            export_to_onnx(model, cfg["input_size"], onnx_path)
        else:
            print(f"    [onnx] Existe, skip.")
        result["stages"]["onnx"] = "ok"
        result["files"]["onnx"] = onnx_path
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"onnx: {type(e).__name__}: {e}"
        traceback.print_exc()
        return result

    try:
        if not os.path.exists(float_path) or force:
            convert_onnx_to_tflite(onnx_path, float_path, name)
        else:
            print(f"    [tflite_fp32] Existe, skip.")
        result["stages"]["tflite_fp32"] = "ok"
        result["files"]["tflite_fp32"] = float_path
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"tflite_fp32: {type(e).__name__}: {e}"
        traceback.print_exc()
        return result

    try:
        if not os.path.exists(int8_path) or force:
            quantize_to_int8(float_path, int8_path, cfg, num_calib)
        else:
            print(f"    [tflite_int8] Existe, skip.")
        result["stages"]["tflite_int8"] = "ok"
        result["files"]["tflite_int8"] = int8_path
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"tflite_int8: {type(e).__name__}: {e}"
        traceback.print_exc()
        return result

    try:
        print(f"    [verify] Consistency check: PyTorch / TFLite FP32 / TFLite INT8...")
        result["inference_check"] = check_inference(model, float_path, int8_path, cfg)
        ic = result["inference_check"]
        print(f"      top1 : pt={ic['top1_pytorch']} fp32={ic['top1_tflite_fp32']} int8={ic['top1_tflite_int8']}")
        print(f"      cos(pt,fp32)={ic['cos_pt_vs_tflite_fp32']:.4f}  cos(pt,int8)={ic['cos_pt_vs_tflite_int8']:.4f}")
        print(f"      top5 overlap (pt vs int8) = {ic['top5_overlap_pt_int8']}/5  NaN={ic['has_nan_or_inf']}")
        result["status"] = "ok"
    except Exception as e:
        result["status"] = "verify_failed"
        result["error"] = f"verify: {type(e).__name__}: {e}"
        traceback.print_exc()

    result["elapsed_sec"] = round(time.time() - t0, 1)
    return result


def main():
    """Fetch, convert and verify each requested model, then write the report."""
    parser = argparse.ArgumentParser(description="multi_tpu/models — test conversion TFLite")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models (default: all)")
    parser.add_argument("--num_calib", type=int, default=100,
                        help="Number of calibration images (default 100)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files")
    args = parser.parse_args()

    names = list(MODELS.keys()) if args.models is None else args.models
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"ERROR: unknown models: {unknown}")
        print(f"  Available: {list(MODELS.keys())}")
        return

    print("=" * 70)
    print("multi_tpu/models — TEST DE CONVERSION TFLITE")
    print(f"  Models     : {names}")
    print(f"  Calib      : {args.num_calib} images CIFAR-100 train (bicubique → input size)")
    print(f"  Output     : {BASE_DIR}")
    print("=" * 70)

    report = {"models": {}, "summary": {}}
    for name in names:
        print(f"\n{'─' * 70}")
        print(f"  {name.upper()} ({MODELS[name]['input_size']}×{MODELS[name]['input_size']})")
        print(f"{'─' * 70}")
        res = convert_and_verify(name, MODELS[name], args.num_calib, args.force)
        report["models"][name] = res

    ok = [n for n, r in report["models"].items() if r["status"] == "ok"]
    ko = [n for n, r in report["models"].items() if r["status"] != "ok"]
    report["summary"] = {
        "total": len(report["models"]),
        "success": len(ok),
        "failed": len(ko),
        "success_list": ok,
        "failed_list": ko,
    }

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Succeeded: {len(ok)}/{len(report['models'])}  {ok}")
    print(f"  Failed   : {len(ko)}/{len(report['models'])}  {ko}")
    for n in ko:
        print(f"    - {n}: {report['models'][n]['error']}")

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

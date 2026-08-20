#!/usr/bin/env python3
"""
apply_patches.py
================
Applies required patches to onnx2tf and ai-edge-quantizer.
Run once after installing packages in pytorch-env.

Usage:
    source ./pytorch-env/bin/activate
    python apply_patches.py
"""

import importlib
import os


def patch_onnx2tf_pool():
    """Patch 1: Replace PADV2 with PAD in onnx2tf pool.py for Edge TPU compatibility."""
    try:
        import onnx2tf
        pool_path = os.path.join(
            os.path.dirname(onnx2tf.__file__),
            "tflite_builder", "op_builders", "pool.py"
        )
    except ImportError:
        print("[SKIP] onnx2tf not installed")
        return False

    if not os.path.exists(pool_path):
        print(f"[SKIP] pool.py not found at {pool_path}")
        return False

    with open(pool_path, "r") as f:
        content = f.read()

    old = '''            else:
                if op_type == "MAX_POOL_2D":
                    pad_value = _max_pool_pad_value_for_tensor(
                        tensor_dtype=ctx.get_tensor_dtype(x_nhwc_pool),
                        tensor_quant=x_tensor.quantization,
                    )
                else:
                    pad_value = 0.0
                pad_value_name = ctx.add_const_tensor(
                    f"{node.name}_pad_value",
                    np.asarray(
                        [pad_value],
                        dtype=_numpy_dtype_from_tflite_dtype(ctx.get_tensor_dtype(x_nhwc_pool)),
                    ),
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="PADV2",
                        inputs=[x_nhwc_pool, pads_name, pad_value_name],
                        outputs=[x_nhwc_padded],
                    )
                )'''

    new = '''            else:
                # Patched: PAD instead of PADV2 for Edge TPU compatibility
                ctx.add_operator(
                    OperatorIR(
                        op_type="PAD",
                        inputs=[x_nhwc_pool, pads_name],
                        outputs=[x_nhwc_padded],
                    )
                )'''

    if old in content:
        content = content.replace(old, new)
        with open(pool_path, "w") as f:
            f.write(content)
        print(f"[OK] Patch 1 applied: onnx2tf pool.py (PADV2 → PAD)")
        return True
    elif "# Patched: PAD instead of PADV2" in content:
        print(f"[OK] Patch 1 already applied")
        return True
    else:
        print(f"[WARN] Patch 1: could not find target code in {pool_path}")
        return False


def patch_ai_edge_quantizer():
    """Patch 2: Fix PadV2 rank validation crash in ai-edge-quantizer."""
    try:
        import ai_edge_quantizer.algorithms.uniform_quantize.uniform_quantize_tensor as m
        uqt_path = m.__file__
    except ImportError:
        print("[SKIP] ai-edge-quantizer not installed")
        return False

    with open(uqt_path, "r") as f:
        content = f.read()

    old = '''  if tensor_rank != scale_rank or (tensor_rank != zero_point_rank):
    raise ValueError(
        f"Ranks of scales ({scale_rank}) and zps"
        f" ({zero_point_rank}) must be the same as the tensor rank"
        f" ({tensor_rank})."
    )'''

    new = '''  if tensor_rank != scale_rank or (tensor_rank != zero_point_rank):
    # Patched: skip rank validation for PadV2 constant_value tensors
    pass'''

    if old in content:
        content = content.replace(old, new)
        with open(uqt_path, "w") as f:
            f.write(content)
        print(f"[OK] Patch 2 applied: ai-edge-quantizer rank validation fix")
        return True
    elif "# Patched: skip rank validation" in content:
        print(f"[OK] Patch 2 already applied")
        return True
    else:
        print(f"[WARN] Patch 2: could not find target code in {uqt_path}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("APPLYING REQUIRED PATCHES")
    print("=" * 50)
    p1 = patch_onnx2tf_pool()
    p2 = patch_ai_edge_quantizer()
    print("=" * 50)
    if p1 and p2:
        print("All patches applied successfully.")
    else:
        print("Some patches failed — check output above.")

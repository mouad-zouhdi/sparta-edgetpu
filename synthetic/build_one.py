"""
build_one.py — build, quantize and export ONE synthetic CNN to INT8 TFLite.

WHAT THIS PRODUCES
    outputs/tflite/<tag>.tflite      INT8, ready for edgetpu_compiler
    outputs/metadata/<tag>.json      per-stage status, sizes, timings, opcodes

THE CONVERSION PATH, AND WHY IT IS THIS CONVOLUTED
    Keras -> SavedModel -> tf2onnx -> onnx2tf (flatbuffer_direct backend)
          -> SignatureDef injection -> ai-edge-quantizer -> INT8 TFLite

    The obvious path, Keras straight through TFLiteConverter, produces models
    that work perfectly at N=1 and make `edgetpu_compiler --num_segments N`
    SEGFAULT for every N >= 2. That failure hit all 380 models of the original
    corpus and does not occur on the ImageNet models, which compile to 8 segments
    without complaint.

    Debugging it under gdb, on the stripped compiler binary, located the fault in
    a flatbuffer vtable lookup:

        => mov (%rdi,%rax,4),%eax        # SEGV
           rax read from (%r10) just before: a vtable offset stored in the
           tflite that points outside the valid region

    Six fixes were tried and none worked: downgrading operator versions,
    metadata and signature surgery, injecting QUANTIZE operations by forcing
    float I/O, a full detour through ONNX with onnx2tf 1.x, matching the ResNet-50
    head, and matching its bottleneck block.

    What actually distinguishes a model that compiles is the conversion backend,
    visible in the tflite's description field:

        description                  produced by                  --num_segments 2
        "MLIR Converted."            TFLiteConverter, onnx2tf 1.x  SEGFAULT
        "onnx2tf flatbuffer_direct"  onnx2tf 2.x                   works

    Both flatbuffers are structurally valid. The MLIR path simply lays them out
    in a way that trips the compiler's segmentation heuristic. Short of reverse
    engineering a stripped binary, the exact field responsible is not knowable,
    and the detour works, so this is where the investigation stopped.

    The script asserts on the description at step 3 rather than trusting the
    environment: an accidental onnx2tf downgrade would otherwise produce a corpus
    that only fails hours later, at compile time.

TWO MORE WORKAROUNDS BAKED IN
    SignatureDef injection. onnx2tf emits no SignatureDef, and
    ai-edge-quantizer's calibrate() refuses to run without one ("Invalid
    signature_key"). One is built by hand with flatbuffer_utils before
    quantization.

    BATCH_MATMUL forced to TENSORWISE. ai-edge-quantizer defaults to per-axis
    (channelwise) quantization, and the compiler rejects that on this operator:
    "'dwl.matrix_multiply' op per axis quantization is not supported". Note that
    onnx2tf converts a Keras Dense into BATCH_MATMUL (opcode 126), NOT into
    FULLY_CONNECTED (opcode 9), so configuring FULLY_CONNECTED alone has no
    effect whatsoever, which is an easy hour to lose.

CALIBRATION
    100 samples of Gaussian noise at seed 42. These networks are untrained and
    have no task, so there is no meaningful data distribution to match: the
    quantization scales only have to be plausible and reproducible. The corpus
    exists to characterise memory and transfer behaviour, not accuracy.

ENVIRONMENT
    Requires Python 3.12. On 3.10 or 3.11, pip silently installs onnx2tf 1.x,
    which routes through MLIR and reintroduces the segfault. Verified working with
    onnx2tf 2.4.0 and 2.5.0, whose default backend is flatbuffer_direct. Step 3
    checks the resulting tflite's description rather than the package version, so
    a wrong environment fails here instead of hours later at compile time.

    onnx2tf also calls download_test_image_data(), which crashes under numpy 2
    because it calls np.load without allow_pickle. Either use onnx2tf_wrapper.py,
    which patches np.load, or place a dummy
    calibration_image_sample_data_20x128x128x3_float32.npy in the working
    directory.

USAGE
    python build_one.py --family residual --depth 16 --base_width 32 \\
        --resolution 224 [--out_dir DIR] [--num_calib 100]

    Designed to be spawned as a subprocess by generate_sweep.py: a TensorFlow
    OOM kills the child only, and the parent records build_status="crashed" and
    moves on.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import shutil


METADATA_FIELDS = {
    "family": None, "depth": None, "base_width": None, "resolution": None,
    "num_params": None,
    "onnx_size_mb": None,
    "float32_tflite_size_mb": None,
    "int8_tflite_size_mb": None,
    "build_status": "not_started",
    "onnx_status": "not_started",
    "onnx2tf_status": "not_started",
    "aeq_status": "not_started",
    "compile_status": "pending",
    "build_error": None, "onnx_error": None, "onnx2tf_error": None,
    "aeq_error": None,
    "elapsed_sec": None,
    "tag": None,
    "description_tag": None,
    "opcodes": None,
}


def save_metadata(metadata_path: Path, meta: dict):
    """Write the per-config metadata JSON, after every stage.

    Written incrementally so that a process killed by the OOM killer still leaves a
    record of how far it got and which stage failed.
    """
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    tmp.replace(metadata_path)


def main():
    """Run the full build for one configuration, recording each stage as it passes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=["sequential","residual","dense","branched_2way","branched_4way"])
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--base_width", type=int, required=True)
    ap.add_argument("--resolution", type=int, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--num_calib", type=int, default=100)
    args = ap.parse_args()

    tag = f"{args.family}_d{args.depth}_w{args.base_width}_r{args.resolution}"
    # Root of the generator: holds build_one.py, src/, params_table.json and the
    # dummy calibration .npy that onnx2tf expects in its working directory.
    # Defaults to this script's own directory; override with SYNTH_ROOT when the
    # scripts are deployed somewhere other than the repository.
    SYNTH_ROOT = Path(os.environ.get("SYNTH_ROOT", Path(__file__).resolve().parent))
    OUT = Path(args.out_dir) if args.out_dir else SYNTH_ROOT / "outputs"
    OUT.mkdir(parents=True, exist_ok=True)

    tflite_dir = OUT / "tflite"; tflite_dir.mkdir(exist_ok=True)
    metadata_dir = OUT / "metadata"; metadata_dir.mkdir(exist_ok=True)
    meta_path = metadata_dir / f"{tag}.json"

    meta = dict(METADATA_FIELDS)
    meta.update(family=args.family, depth=args.depth, base_width=args.base_width,
                resolution=args.resolution, tag=tag)
    save_metadata(meta_path, meta)

    t0 = time.time()
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"regen_{tag}_"))
    try:
        # === STEP 1: Keras build + SavedModel ===
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        sys.path.insert(0, str(SYNTH_ROOT))
        try:
            import numpy as np
            import tensorflow as tf
            from src.models import build_model
        except Exception as e:
            meta["build_status"] = "failed"; meta["build_error"] = f"import: {e}"
            save_metadata(meta_path, meta); raise

        try:
            model = build_model(args.family, args.depth, args.base_width, args.resolution)
            meta["num_params"] = int(model.count_params())

            @tf.function(input_signature=[tf.TensorSpec([1, args.resolution, args.resolution, 3], tf.float32)])
            def serve(x):
                """SavedModel serving signature with an explicit batch-1 input shape.

                The shape must be fixed here: a dynamic batch dimension propagates through the
                ONNX export and leaves the Edge TPU compiler unable to size the graph.
                """
                return model(x, training=False)

            sm_dir = tmp_dir / "sm"
            tf.saved_model.save(model, str(sm_dir), signatures={"serving_default": serve})
            meta["build_status"] = "success"
            save_metadata(meta_path, meta)
        except Exception as e:
            meta["build_status"] = "failed"; meta["build_error"] = f"{type(e).__name__}: {e}"
            save_metadata(meta_path, meta); raise

        # === STEP 2: tf2onnx ===
        onnx_path = tmp_dir / f"{tag}.onnx"
        try:
            r = subprocess.run(
                [sys.executable, "-m", "tf2onnx.convert",
                 "--saved-model", str(sm_dir), "--output", str(onnx_path),
                 "--opset", "18", "--inputs-as-nchw", "x"],
                capture_output=True, text=True, timeout=1200,
            )
            if r.returncode != 0 or not onnx_path.exists():
                raise RuntimeError(f"tf2onnx rc={r.returncode}: {r.stderr[-500:]}")
            meta["onnx_size_mb"] = round(onnx_path.stat().st_size / 1e6, 3)
            meta["onnx_status"] = "success"
            save_metadata(meta_path, meta)
        except Exception as e:
            meta["onnx_status"] = "failed"; meta["onnx_error"] = f"{type(e).__name__}: {e}"
            save_metadata(meta_path, meta); raise

        # === STEP 3: onnx2tf flatbuffer_direct ===
        o2t_dir = tmp_dir / "o2t"
        try:
            # onnx2tf calls download_test_image_data() and expects this file in its
            # working directory. Its contents are irrelevant to us (these networks
            # are untrained and calibration proper happens later, in step 4), so it
            # is generated once rather than shipped: 3.8 MB of fixed random noise
            # has no business in a git repository.
            calib_src = SYNTH_ROOT / "calibration_image_sample_data_20x128x128x3_float32.npy"
            if not calib_src.exists():
                import numpy as _np
                _np.save(calib_src, _np.random.default_rng(42)
                         .standard_normal((20, 128, 128, 3)).astype(_np.float32))
            r = subprocess.run(
                [sys.executable, "-m", "onnx2tf",
                 "-i", str(onnx_path), "-o", str(o2t_dir), "-b", "1"],
                capture_output=True, text=True, timeout=1800,
                cwd=str(SYNTH_ROOT),
            )
            fps = list(o2t_dir.glob("*_float32.tflite")) if o2t_dir.exists() else []
            if not fps:
                raise RuntimeError(f"no tflite produced. rc={r.returncode}, stdout tail: {r.stdout[-300:]}")
            float_path = fps[0]
            meta["float32_tflite_size_mb"] = round(float_path.stat().st_size / 1e6, 3)

            # Check description
            from tensorflow.lite.tools import flatbuffer_utils
            mo = flatbuffer_utils.convert_bytearray_to_object(float_path.read_bytes())
            meta["description_tag"] = mo.description.decode() if mo.description else None

            if b"flatbuffer_direct" not in (mo.description or b""):
                raise RuntimeError(f"unexpected description: {mo.description!r} — must be onnx2tf flatbuffer_direct")

            meta["onnx2tf_status"] = "success"
            save_metadata(meta_path, meta)
        except Exception as e:
            meta["onnx2tf_status"] = "failed"; meta["onnx2tf_error"] = f"{type(e).__name__}: {e}"
            save_metadata(meta_path, meta); raise

        # === STEP 4: Add signature + AEQ quantize ===
        try:
            from tensorflow.lite.python.schema_py_generated import SignatureDefT, TensorMapT
            mo = flatbuffer_utils.convert_bytearray_to_object(float_path.read_bytes())
            if not mo.signatureDefs:
                sg = mo.subgraphs[0]
                sd = SignatureDefT(); sd.signatureKey = b"serving_default"; sd.subgraphIndex = 0
                sd.inputs = []; sd.outputs = []
                for idx in sg.inputs:
                    tm = TensorMapT(); tm.name = sg.tensors[idx].name; tm.tensorIndex = idx
                    sd.inputs.append(tm)
                for idx in sg.outputs:
                    tm = TensorMapT(); tm.name = sg.tensors[idx].name; tm.tensorIndex = idx
                    sd.outputs.append(tm)
                mo.signatureDefs = [sd]
                float_path.write_bytes(bytes(flatbuffer_utils.convert_object_to_bytearray(mo)))

            import ai_edge_litert.interpreter as tfl
            from ai_edge_quantizer import quantizer, qtyping

            interp = tfl.Interpreter(model_path=str(float_path))
            interp.allocate_tensors()
            sig = interp.get_signature_list()
            sk = list(sig.keys())[0]; inn = sig[sk]["inputs"][0]
            sh = tuple(int(v) for v in interp.get_input_details()[0]["shape"])
            del interp

            q = quantizer.Quantizer(str(float_path))
            q.add_static_config(regex=".*",
                operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                activation_num_bits=8, weight_num_bits=8)
            q.add_static_config(regex=".*",
                operation_name=qtyping.TFLOperationName.BATCH_MATMUL,
                activation_num_bits=8, weight_num_bits=8,
                weight_granularity=qtyping.QuantGranularity.TENSORWISE)
            q.add_static_config(regex=".*",
                operation_name=qtyping.TFLOperationName.INPUT,
                activation_num_bits=8, weight_num_bits=8)
            q.add_static_config(regex=".*",
                operation_name=qtyping.TFLOperationName.OUTPUT,
                activation_num_bits=8, weight_num_bits=8)

            rng = np.random.default_rng(42)
            calib = [{inn: rng.standard_normal(sh).astype(np.float32)}
                     for _ in range(args.num_calib)]
            cr = q.calibrate({sk: calib})
            res = q.quantize(cr)

            int8_path = tflite_dir / f"{tag}.tflite"
            int8_path.write_bytes(res.quantized_model)
            meta["int8_tflite_size_mb"] = round(int8_path.stat().st_size / 1e6, 3)

            # Save opcodes
            mo2 = flatbuffer_utils.convert_bytearray_to_object(int8_path.read_bytes())
            meta["opcodes"] = [(int(oc.builtinCode), int(oc.version))
                               for oc in mo2.operatorCodes]

            meta["aeq_status"] = "success"
            save_metadata(meta_path, meta)
        except Exception as e:
            meta["aeq_status"] = "failed"; meta["aeq_error"] = f"{type(e).__name__}: {e}"
            save_metadata(meta_path, meta); raise

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        meta["elapsed_sec"] = round(time.time() - t0, 1)
        save_metadata(meta_path, meta)


if __name__ == "__main__":
    main()

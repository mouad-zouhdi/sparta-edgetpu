#!/usr/bin/env python3
"""
onnx2tf_wrapper.py — run the onnx2tf CLI with np.load patched for numpy 2.

WHY THIS EXISTS
    onnx2tf calls download_test_image_data() during conversion, which loads a
    pickled .npy without passing allow_pickle. Under numpy 2 that raises, and the
    conversion dies before producing anything.

    Two fixes are possible: patch np.load, which is what this does, or place a
    dummy calibration_image_sample_data_20x128x128x3_float32.npy in the working
    directory, which is what build_one.py does (it generates one on demand).
    Either works; this wrapper is here for running onnx2tf by hand.

USAGE
    python onnx2tf_wrapper.py -i model.onnx -o out_dir -b 1

    Arguments are passed straight through to onnx2tf.

NOTE
    Importing this module is safe: it patches np.load but does not run the CLI.
    Only executing it as a script does that.
"""
import sys

import numpy as np

_orig_load = np.load


def _patched_load(file, **kw):
    """np.load with allow_pickle defaulted to True.

    Defaulted rather than forced, so an explicit allow_pickle=False from any
    other caller is still honoured.
    """
    kw.setdefault("allow_pickle", True)
    return _orig_load(file, **kw)


np.load = _patched_load


def main() -> int:
    """Invoke the onnx2tf command-line entry point with the patch in place."""
    sys.argv[0] = "onnx2tf"
    import onnx2tf.__main__ as o2t_main

    # onnx2tf renamed its entry point across versions: 2.x exposes `lazy_main`,
    # earlier releases exposed `main`. Accept whichever is present rather than
    # pinning a version here.
    entry = getattr(o2t_main, "lazy_main", None) or getattr(o2t_main, "main", None)
    if entry is None:
        print("ERROR: found no entry point in onnx2tf.__main__ "
              f"(has: {[a for a in dir(o2t_main) if not a.startswith('_')]})",
              file=sys.stderr)
        return 2
    entry()
    return 0


if __name__ == "__main__":
    sys.exit(main())

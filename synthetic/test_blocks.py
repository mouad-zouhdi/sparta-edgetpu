"""
Smoke tests: build a tiny model for each family and check forward pass.

Run with pytest:
    pytest tests/

Or directly:
    python tests/test_blocks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.blocks import FAMILIES  # noqa: E402
from src.models import build_model, validate_forward  # noqa: E402

SMALL_DEPTH = 2
SMALL_WIDTH = 16
SMALL_RES = 96


def _check_family(family: str) -> None:
    """Build a minimal model of one family and confirm the forward pass works."""
    model = build_model(family, SMALL_DEPTH, SMALL_WIDTH, SMALL_RES)
    assert model.input_shape == (None, SMALL_RES, SMALL_RES, 3), (
        f"{family}: unexpected input shape {model.input_shape}"
    )
    out_shape = validate_forward(model)
    assert out_shape == (1, 100), f"{family}: unexpected output shape {out_shape}"
    n = model.count_params()
    assert n > 0, f"{family}: zero params"


def test_sequential():
    """Smoke test: the sequential family builds and runs a forward pass."""
    _check_family("sequential")


def test_residual():
    """Smoke test: the residual family builds and runs a forward pass."""
    _check_family("residual")


def test_dense():
    """Smoke test: the dense family builds and runs a forward pass."""
    _check_family("dense")


def test_branched_2way():
    """Smoke test: the branched_2way family builds and runs a forward pass."""
    _check_family("branched_2way")


def test_branched_4way():
    """Smoke test: the branched_4way family builds and runs a forward pass."""
    _check_family("branched_4way")


def test_all_families_listed():
    """Confirm every family in the plan has a block implementation."""
    expected = {
        "sequential",
        "residual",
        "dense",
        "branched_2way",
        "branched_4way",
    }
    assert set(FAMILIES) == expected


if __name__ == "__main__":
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    for fam in FAMILIES:
        print(f"=== {fam} ===")
        _check_family(fam)
        print(f"  OK")
    test_all_families_listed()
    print("\nAll family smoke tests passed.")

"""
Family-specific building blocks for synthetic CNNs.

Every block takes a tensor `x` and a stage width `width` and returns a tensor.
Conv 3x3 is the dominant op across families. Activations are ReLU6 and norms
are BatchNorm — both Edge TPU-friendly. LayerNorm, GroupNorm, Swish, Mish,
attention are deliberately excluded (poor edgetpu_compiler support).
"""

from typing import Callable

import tensorflow as tf
from tensorflow.keras import layers

KERNEL_INIT = "he_normal"


def _conv_bn_relu(x, channels: int, kernel: int, stride: int = 1):
    """Conv-BatchNorm-ReLU6, the unit every family is built from.

    ReLU6 and BatchNorm are used throughout because the Edge TPU compiler supports
    them well. LayerNorm, GroupNorm, Swish, Mish and attention are deliberately
    absent: they compile poorly or not at all, and would confound a study of memory
    behaviour with a study of operator support.
    """
    x = layers.Conv2D(
        channels,
        kernel,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer=KERNEL_INIT,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    return x


def sequential_block(x, width: int):
    """Plain block: one 3x3 Conv-BN-ReLU6. The topology-free control."""
    return _conv_bn_relu(x, width, 3)


def residual_block(x, width: int):
    """ResNet basic block: two 3x3 convolutions plus the input.

    A 1x1 projection is inserted on the shortcut when the channel count changes.
    """
    shortcut = x
    if int(x.shape[-1]) != width:
        shortcut = layers.Conv2D(
            width, 1, padding="same", use_bias=False, kernel_initializer=KERNEL_INIT
        )(x)
        shortcut = layers.BatchNormalization()(shortcut)

    y = _conv_bn_relu(x, width, 3)
    y = layers.Conv2D(
        width, 3, padding="same", use_bias=False, kernel_initializer=KERNEL_INIT
    )(y)
    y = layers.BatchNormalization()(y)
    y = layers.Add()([y, shortcut])
    y = layers.ReLU(6.0)(y)
    return y


def dense_block(x, width: int):
    """DenseNet-style: concat input with output of a Conv3x3 (growth = width/4)."""
    growth = max(width // 4, 1)
    y = layers.Conv2D(
        growth, 3, padding="same", use_bias=False, kernel_initializer=KERNEL_INIT
    )(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU(6.0)(y)
    return layers.Concatenate(axis=-1)([x, y])


def branched_2way_block(x, width: int):
    """Two parallel branches, 1x1 and 3x3, concatenated."""
    half = width // 2
    other = width - half
    b1 = _conv_bn_relu(x, half, 1)
    b2 = _conv_bn_relu(x, other, 3)
    return layers.Concatenate(axis=-1)([b1, b2])


def branched_4way_block(x, width: int):
    """Inception-A flavour: 1x1 / 1x1->3x3 / 1x1->3x3->3x3 / avgpool->1x1."""
    q = max(width // 4, 1)
    rest = width - 3 * q

    b1 = _conv_bn_relu(x, q, 1)

    b2 = _conv_bn_relu(x, q, 1)
    b2 = _conv_bn_relu(b2, q, 3)

    b3 = _conv_bn_relu(x, q, 1)
    b3 = _conv_bn_relu(b3, q, 3)
    b3 = _conv_bn_relu(b3, q, 3)

    b4 = layers.AveragePooling2D(pool_size=3, strides=1, padding="same")(x)
    b4 = _conv_bn_relu(b4, rest, 1)

    return layers.Concatenate(axis=-1)([b1, b2, b3, b4])


BLOCKS: dict[str, Callable] = {
    "sequential": sequential_block,
    "residual": residual_block,
    "dense": dense_block,
    "branched_2way": branched_2way_block,
    "branched_4way": branched_4way_block,
}

FAMILIES = list(BLOCKS.keys())


def get_block(family: str) -> Callable:
    """Return the block constructor for a family name."""
    if family not in BLOCKS:
        raise ValueError(f"Unknown family '{family}'. Available: {FAMILIES}")
    return BLOCKS[family]

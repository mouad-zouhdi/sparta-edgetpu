"""
Model assembly: stem + 4 stages + head.

Stem        : Conv 3x3 s2 -> BN -> ReLU6 -> MaxPool 3x3 s2     (R -> R/4)
Transition  : Conv 3x3 s2 -> BN -> ReLU6 between stages 1->2->3->4 (each /2)
Stage k     : `depth` blocks at width [C, 2C, 4C, 8C][k]
Head        : GlobalAveragePool -> Dense(100)

For dense family, blocks concat (growth = stage_width/4). A 1x1 projection
is inserted at the end of every dense stage to bring channels back to the
stage width before the next stride-2 transition (keeps channel arithmetic
bounded between stages).
"""

import tensorflow as tf
from tensorflow.keras import Input, Model, layers

from .blocks import get_block

NUM_CLASSES = 100
KERNEL_INIT = "he_normal"


def _conv_bn_relu(x, channels: int, kernel: int, stride: int = 1):
    """Conv-BatchNorm-ReLU6, used by the stem and the stage transitions."""
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


def build_model(family: str, depth: int, base_width: int, resolution: int) -> Model:
    """Assemble one synthetic CNN from a factorial configuration.

    The skeleton is identical across families, so that family is the only thing that
    varies:
        input (R, R, 3)
        -> Conv 3x3 stride 2 + BN + ReLU6 + MaxPool 3x3 stride 2   (R -> R/4)
        -> four stages of widths [C, 2C, 4C, 8C], each `depth` blocks,
           with a stride-2 3x3 convolution between stages
        -> GlobalAveragePooling -> Dense(100)

    Only the block used inside the stages differs between families. He-normal
    initialisation throughout; the networks are never trained.
    """
    block_fn = get_block(family)

    inputs = Input(shape=(resolution, resolution, 3), name="input")

    # Stem: R -> R/2 -> R/4
    x = _conv_bn_relu(inputs, base_width, 3, stride=2)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    widths = [base_width, 2 * base_width, 4 * base_width, 8 * base_width]

    for stage_idx, width in enumerate(widths):
        if stage_idx > 0:
            # Stride-2 transition into the new stage (channel + spatial change).
            x = _conv_bn_relu(x, width, 3, stride=2)
        else:
            # Stage 1 starts at stem width = base_width = stage 1 width.
            # Nothing to do for non-dense families. For dense, channels stay
            # at base_width here (block adds growth on top).
            pass

        for _ in range(depth):
            x = block_fn(x, width)

        if family == "dense":
            # End-of-stage projection back to the nominal stage width.
            # Avoids unbounded channel growth feeding into the next transition.
            x = _conv_bn_relu(x, width, 1)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(NUM_CLASSES, kernel_initializer=KERNEL_INIT)(x)

    return Model(inputs, outputs, name=f"{family}_d{depth}_w{base_width}_r{resolution}")


def validate_forward(model: Model) -> tuple:
    """Run one forward pass on Gaussian noise. Returns output shape."""
    import numpy as np

    shape = model.input_shape[1:]
    x = np.random.default_rng(0).standard_normal((1, *shape)).astype("float32")
    y = model(x, training=False)
    return tuple(y.shape)

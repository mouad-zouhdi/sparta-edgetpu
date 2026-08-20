"""
cifar_vgg.py — CIFAR-native VGG-19 with batch normalisation.

Source: https://github.com/HobbitLong/RepDistiller/blob/master/models/vgg.py
(c) YANG, Wei

WHAT THIS PROVIDES
    vgg19(num_classes), the VGG baseline of the single-TPU axis, in its
    CIFAR-native form (32x32 input).

STRUCTURE
    - five Conv-BN-ReLU stages separated by 2x2 max-pools,
    - spatial resolution 32 -> 16 -> 8 -> 4 -> 2 -> 1 (last stage via an
      adaptive average pool),
    - a single nn.Linear(512, num_classes) classifier, in place of the three
      4096-wide fully connected layers of ImageNet VGG, which would dominate the
      parameter count at this resolution.

WHY BATCH NORM IS NOT OPTIONAL HERE
    The bn_scale pruning criterion ranks channels by the magnitude of their
    BatchNorm gamma, so it can only be applied to a network that has BatchNorm.
    vgg19() therefore always builds the batch-normalised configuration. This is
    also why squeezenet1_1, which has no BatchNorm, has no bn_scale results in
    this repository: those runs are expected failures, not missing data.

CHANGES MADE FOR THIS PROJECT
    - forward() simplified: the `is_feat` / `preact` feature-extraction modes
      used for knowledge distillation upstream were removed.
    - Only vgg19() is exposed.

WHY THIS FILE MUST STAY AT THE TOP LEVEL OF mono_tpu/
    Same pickle-path constraint as cifar_resnet.py; see that file.
"""

import math

import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    """CIFAR-native VGG: five Conv-BN-ReLU stages, one linear classifier."""
    def __init__(self, cfg, batch_norm=True, num_classes=100):
        """Build the convolutional stack from `cfg` plus a single linear classifier.

        `cfg` is a list of per-stage channel lists; the literal 'M' marks a max-pool.
        """
        super().__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        """Run the five stages, adaptively pool to 1x1, flatten, classify."""
        x = F.relu(self.block0(x))
        x = self.pool0(x)
        x = F.relu(self.block1(x))
        x = self.pool1(x)
        x = F.relu(self.block2(x))
        x = self.pool2(x)
        x = F.relu(self.block3(x))
        x = F.relu(self.block4(x))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        """Turn the `cfg` channel list into a Sequential of Conv-BN-ReLU and max-pools.

        BatchNorm is inserted between convolution and ReLU whenever `batch_norm` is
        set, which is always the case here: the bn_scale pruning criterion needs it.
        """
        layers = []
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        # Drop the trailing ReLU: forward() applies it before pooling instead.
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """He-normal init for convolutions, unit gain for BatchNorm, zero biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


_CFG_E = [
    [64, 64], [128, 128],
    [256, 256, 256, 256],
    [512, 512, 512, 512],
    [512, 512, 512, 512],
]


def vgg19(num_classes=100):
    """VGG-19 with BatchNorm for CIFAR: 16 convolutions, about 20.1 M params."""
    return VGG(_CFG_E, batch_norm=True, num_classes=num_classes)

"""
wrn.py — Wide ResNet (Zagoruyko and Komodakis, 2016, arXiv:1605.07146).

WHAT THIS PROVIDES
    wrn_28_10(num_classes), the widest baseline of the single-TPU axis: 28 layers
    deep, widening factor 10, about 36.5 M parameters. It is the most accurate
    baseline of the lineup (81.1 % top-1 on CIFAR-100) and, being the largest, it
    is also the one that streams the most weights from off-chip memory on an
    Edge TPU. That combination makes it the useful extreme of the accuracy /
    memory-regime trade-off this project measures.

STRUCTURE
    Pre-activation residual blocks (BN -> ReLU -> conv, rather than
    conv -> BN -> ReLU), three groups of (depth - 4) / 6 blocks each, with widths
    16, 16*k, 32*k, 64*k for widening factor k. Input is 32x32.

WHY THIS FILE MUST STAY AT THE TOP LEVEL OF mono_tpu/
    Same pickle-path constraint as cifar_resnet.py; see that file.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Pre-activation wide residual block (BN -> ReLU -> conv, twice).

    Pre-activation means normalisation and non-linearity come before each
    convolution rather than after, which keeps the residual path a clean identity
    and is what lets Wide ResNets train at this width.
    """
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        """Build the two 3x3 convolutions, optional dropout, and the shortcut."""
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x):
        """Pre-activate, convolve twice, add the (possibly projected) input."""
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if isinstance(self.shortcut, nn.Conv2d) else x
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    """Wide ResNet for CIFAR: three block groups of widths 16k, 32k, 64k."""
    def __init__(self, depth=28, widen_factor=10, num_classes=100, dropout_rate=0.3):
        """Assemble stem, three groups of (depth - 4) / 6 blocks, and the classifier.

        `depth` must satisfy (depth - 4) % 6 == 0; `widen_factor` is the k that
        multiplies every group width.
        """
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth doit valoir 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        widths = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(widths[0], widths[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(widths[1], widths[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(widths[2], widths[3], n, stride=2, dropout_rate=dropout_rate)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride, dropout_rate):
        """Build one group of `nb_layers` blocks, only the first of which may stride."""
        layers = [BasicBlock(in_planes, out_planes, stride, dropout_rate)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_planes, out_planes, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Stem, three groups, final BN-ReLU, 8x8 average pool, flatten, classify."""
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.flatten(1)
        return self.fc(out)


def wrn_28_10(num_classes=100, dropout_rate=0.3):
    """WRN-28-10 for CIFAR: depth 28, widening factor 10, about 36.5 M params."""
    return WideResNet(depth=28, widen_factor=10,
                      num_classes=num_classes, dropout_rate=dropout_rate)

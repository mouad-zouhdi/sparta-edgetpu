"""
cifar_resnet.py — CIFAR-native ResNet-18 / ResNet-50.

Source: https://github.com/huawei-noah/Efficient-Computing/blob/master/
        Data-Efficient-Model-Compression/DAFL/resnet.py
(c) Huawei Technologies Co., Ltd. <foss@huawei.com>

WHAT THIS PROVIDES
    resnet18(num_classes) and resnet50(num_classes), the two ResNet baselines of
    the single-TPU axis, in their CIFAR-native form (32x32 input).

HOW IT DIFFERS FROM torchvision
    A torchvision ResNet downsamples aggressively in its stem (7x7 stride-2 conv
    followed by a 3x3 stride-2 max-pool), which is right for 224x224 ImageNet and
    destructive at 32x32: the feature map would be 8x8 before the first residual
    block. The CIFAR-native variant used here instead has
        - a 3x3 stride-1 stem convolution,
        - no initial max-pool,
        - three stride-2 stages, so 32 -> 32 -> 16 -> 8 -> 4,
        - a final avg_pool2d(out, 4) reducing 4x4 to 1x1,
        - a plain nn.Linear(512 * expansion, num_classes) classifier.

CHANGES MADE FOR THIS PROJECT
    - forward() simplified: the unused `out_feature` return mode was removed.
    - resnet18() / resnet50() helpers exposed.

WHY THIS FILE MUST STAY AT THE TOP LEVEL OF mono_tpu/
    Checkpoints are saved as whole-model pickles (torch.save(model)), not as
    state dicts, because structured pruning changes the architecture and a state
    dict alone could not rebuild it. Pickle stores the defining module by name,
    so every published .pt refers to "cifar_resnet.BasicBlock". Moving this file
    into a subpackage would rename it to e.g. "archs.cifar_resnet" and make every
    published checkpoint unloadable. Do not move it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two-convolution residual block, used by ResNet-18.

    Keeps the channel count constant through the block (expansion = 1). The
    shortcut is the identity unless the block changes resolution or width, in
    which case a 1x1 convolution projects the input to match.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """Build the two 3x3 convolutions and, if needed, the projection shortcut."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        """conv-BN-ReLU, conv-BN, add the shortcut, then ReLU."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    """Three-convolution residual block, used by ResNet-50.

    Squeezes to `planes` channels with a 1x1, does the spatial work with a 3x3,
    then expands back to 4 * planes with another 1x1 (expansion = 4). This costs
    far fewer parameters than two 3x3 convolutions at full width, which is what
    makes deeper ResNets affordable.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        """Build the 1x1 / 3x3 / 1x1 stack and, if needed, the projection shortcut."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        """Run the three convolutions, add the shortcut, then ReLU."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """CIFAR-native ResNet: 3x3 stride-1 stem, no max-pool, four residual stages."""
    def __init__(self, block, num_blocks, num_classes=100):
        """Assemble stem, four stages of widths 64/128/256/512, and the classifier.

        `num_blocks` gives the block count per stage; `block` is BasicBlock or
        Bottleneck and sets the expansion factor of the classifier input.
        """
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Build one stage: `num_blocks` blocks, only the first of which may stride."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Stem, four stages, 4x4 average pool to 1x1, flatten, classify."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def resnet18(num_classes=100):
    """ResNet-18 for CIFAR: BasicBlock, two blocks per stage, about 11.2 M params."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet50(num_classes=100):
    """ResNet-50 for CIFAR: Bottleneck, [3,4,6,3] blocks, about 23.7 M params."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

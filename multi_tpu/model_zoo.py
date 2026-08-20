#!/usr/bin/env python3
"""
model_zoo.py — the eight ImageNet architectures of the multi-TPU axis.

WHAT THIS PROVIDES
    A single entry point, build_model(name, num_classes, pretrained), plus the
    per-model preprocessing every downstream script needs, for:

        inception_v1_googlenet    6.6 M params   224 px
        inception_v2_bninception 11.3 M          224
        inception_v3             23.8 M          299
        resnet50                 25.6 M          224
        inception_v4             42.7 M          299
        resnet101                44.5 M          224
        inception_resnet_v2      55.8 M          299
        resnet152                60.2 M          224

    The lineup is deliberately two families crossed with a wide size range. The
    two families differ in exactly the structural property that turns out to
    matter, branch count, and the size range spans the point where a model stops
    fitting in one accelerator's SRAM.

HOW THIS DIFFERS FROM THE SINGLE-TPU AXIS
    The mono_tpu architectures are CIFAR-native: their stems were modified for
    32x32 input. These are NOT modified. Stem stride and the initial max-pool are
    left exactly as published, because the object of study here is how these
    architectures behave on an Edge TPU at their native resolution. The only
    change is the final classifier, resized from 1000 outputs to num_classes.

PREPROCESSING IS PER MODEL, AND GETTING IT WRONG IS SILENT
    Three different conventions are in play, and each model must get its own:
        RGB ImageNet     mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)
                         for the ResNets and GoogLeNet
        Inception        mean = std = (0.5, 0.5, 0.5)
                         for the timm models: inception_v3, v4, inception_resnet_v2
        BN-Inception     BGR channel order, range [0, 255],
                         mean (104, 117, 128), std (1, 1, 1)
    Feeding a model the wrong convention does not raise; it just degrades
    accuracy, which is easy to mistake for a pruning effect. get_preprocessing()
    exists so no caller has to remember which is which.

PITFALLS ENCODED IN THIS FILE
    - Classifier adapters check out_features before replacing the head. Without
      that guard, calling build_model(pretrained=True, num_classes=1000) would
      overwrite the pretrained classifier with a randomly initialised one of the
      same shape, and the model would evaluate at chance level while looking
      perfectly well formed. This cost a full baseline run before it was found.
    - GoogLeNet and Inception V3 from torchvision must be constructed with
      aux_logits=True, because their published checkpoints contain the auxiliary
      classifier weights and loading fails otherwise. They are disabled straight
      afterwards (aux_logits=False, aux1=aux2=None), since those extra branches
      would complicate structured pruning for no benefit at inference.
    - Inception V3 is taken from timm, not torchvision. The torchvision version
      applies a `transform_input` step that becomes a Slice/Gather/Concat block
      on the three input channels, producing an intermediate tensor of about
      6 MiB that edgetpu_compiler rejects as a large activation tensor. Neither
      -a nor -d works around it. timm's version has the same TF-Slim weights and
      no wrapper, and compiles cleanly.
    - BN-Inception's canonical checkpoint is hosted at data.lip6.fr, whose TLS
      certificate has expired and which returns 503 intermittently. The file
      bn_inception-52deb4733.pth can be recovered through the Wayback Machine and
      placed in ~/.cache/torch/hub/checkpoints/.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Stats de preprocessing (en sync avec convert_models.py)
# ─────────────────────────────────────────────
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)


# ─────────────────────────────────────────────
# Adaptateurs de classifier (1000 → num_classes)
# ─────────────────────────────────────────────
def _adapt_googlenet(m: nn.Module, num_classes: int) -> nn.Module:
    """Resize torchvision GoogLeNet's head (m.fc) to num_classes, if it differs."""
    # torchvision GoogLeNet: head is m.fc, Linear(1024, 1000)
    if m.fc.out_features != num_classes:
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _adapt_bninception(m: nn.Module, num_classes: int) -> nn.Module:
    """Resize BN-Inception's head (m.last_linear) to num_classes, if it differs."""
    # pretrainedmodels BN-Inception: head is m.last_linear, Linear(1024, 1000)
    if m.last_linear.out_features != num_classes:
        m.last_linear = nn.Linear(m.last_linear.in_features, num_classes)
    return m


def _adapt_inception_v3_timm(m: nn.Module, num_classes: int) -> nn.Module:
    """Resize timm Inception-V3's head (m.fc) to num_classes, if it differs."""
    # timm Inception-V3: head is m.fc, Linear(2048, 1000); no aux branch by default
    if m.fc.out_features != num_classes:
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _adapt_inception_v4_timm(m: nn.Module, num_classes: int) -> nn.Module:
    """Resize timm Inception-V4's head (m.last_linear) to num_classes, if it differs."""
    # timm Inception-V4: head is m.last_linear, Linear(1536, 1000)
    if m.last_linear.out_features != num_classes:
        m.last_linear = nn.Linear(m.last_linear.in_features, num_classes)
    return m


def _adapt_inception_resnet_v2_timm(m: nn.Module, num_classes: int) -> nn.Module:
    """Resize timm Inception-ResNet-V2's head (m.classif) to num_classes, if it differs."""
    # timm Inception-ResNet-V2: head is m.classif, Linear(1536, 1000)
    if m.classif.out_features != num_classes:
        m.classif = nn.Linear(m.classif.in_features, num_classes)
    return m


def _adapt_resnet_torchvision(m: nn.Module, num_classes: int) -> nn.Module:
    """Resize a torchvision ResNet's head (m.fc) to num_classes, if it differs."""
    # torchvision ResNet-50/101/152: head is m.fc, Linear(2048, 1000)
    if m.fc.out_features != num_classes:
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# ─────────────────────────────────────────────
# Loaders: one per architecture, each returning a model with the classifier
# already sized to num_classes
# ─────────────────────────────────────────────
def _load_googlenet(num_classes: int, pretrained: bool) -> nn.Module:
    """Load torchvision GoogLeNet and strip its auxiliary classifiers.

    aux_logits must be True at construction: the published checkpoint contains the
    auxiliary weights and loading fails without them. They are removed immediately
    afterwards, since those branches only complicate structured pruning.
    """
    import torchvision.models as tvm
    if pretrained:
        # Le checkpoint pretrained inclut les aux classifiers.
        m = tvm.googlenet(weights=tvm.GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=True)
    else:
        # Random init, so the aux classifiers can be disabled straight away.
        m = tvm.googlenet(weights=None, aux_logits=False, init_weights=True)
    m.aux_logits = False
    m.aux1 = None
    m.aux2 = None
    return _adapt_googlenet(m, num_classes)


def _load_bninception(num_classes: int, pretrained: bool) -> nn.Module:
    """Load BN-Inception (Inception V2) from pretrainedmodels.

    Its checkpoint lives on a host with an expired certificate that often returns
    503; see the module docstring for how to obtain bn_inception-52deb4733.pth and
    where to place it.
    """
    import pretrainedmodels
    # The checkpoint must already be in the torch hub cache; see the module docstring.
    pretrained_arg = "imagenet" if pretrained else None
    m = pretrainedmodels.bninception(num_classes=1000, pretrained=pretrained_arg)
    return _adapt_bninception(m, num_classes)


def _load_inception_v3(num_classes: int, pretrained: bool) -> nn.Module:
    """Load Inception-V3 from timm, not torchvision.

    torchvision's version carries a transform_input step that expands into a block
    edgetpu_compiler rejects as a large activation tensor. timm's has the same
    TF-Slim weights, no wrapper, and compiles.
    """
    import timm
    # On force timm (pas torchvision) : torchvision V3 a un wrapper
    # transform_input, which creates an intermediate tensor too large for TPU SRAM.
    m = timm.create_model("inception_v3", pretrained=pretrained)
    return _adapt_inception_v3_timm(m, num_classes)


def _load_inception_v4(num_classes: int, pretrained: bool) -> nn.Module:
    """Load Inception-V4 from timm."""
    import timm
    m = timm.create_model("inception_v4", pretrained=pretrained)
    return _adapt_inception_v4_timm(m, num_classes)


def _load_inception_resnet_v2(num_classes: int, pretrained: bool) -> nn.Module:
    """Load Inception-ResNet-V2 from timm."""
    import timm
    m = timm.create_model("inception_resnet_v2", pretrained=pretrained)
    return _adapt_inception_resnet_v2_timm(m, num_classes)


def _load_resnet(depth: int, num_classes: int, pretrained: bool) -> nn.Module:
    """Load a torchvision ResNet of the given depth (50, 101 or 152)."""
    import torchvision.models as tvm
    fn = {50: tvm.resnet50, 101: tvm.resnet101, 152: tvm.resnet152}[depth]
    weights = {50: tvm.ResNet50_Weights, 101: tvm.ResNet101_Weights,
               152: tvm.ResNet152_Weights}[depth].IMAGENET1K_V1 if pretrained else None
    m = fn(weights=weights)
    return _adapt_resnet_torchvision(m, num_classes)


# ─────────────────────────────────────────────
# Model catalogue: loader plus preprocessing, one entry per architecture
# ─────────────────────────────────────────────
MODELS = {
    "inception_v1_googlenet": {
        "loader": lambda nc, p: _load_googlenet(nc, p),
        "input_size": 224,
        "mean": IMAGENET_RGB_MEAN, "std": IMAGENET_RGB_STD,
        "bgr": False, "input_range_255": False,
    },
    "inception_v2_bninception": {
        "loader": lambda nc, p: _load_bninception(nc, p),
        "input_size": 224,
        # bninception : BGR, range [0,255], mean=[104,117,128], std=[1,1,1]
        "mean": (104.0, 117.0, 128.0), "std": (1.0, 1.0, 1.0),
        "bgr": True, "input_range_255": True,
    },
    "inception_v3": {
        "loader": lambda nc, p: _load_inception_v3(nc, p),
        "input_size": 299,
        "mean": INCEPTION_MEAN, "std": INCEPTION_STD,
        "bgr": False, "input_range_255": False,
    },
    "inception_v4": {
        "loader": lambda nc, p: _load_inception_v4(nc, p),
        "input_size": 299,
        "mean": INCEPTION_MEAN, "std": INCEPTION_STD,
        "bgr": False, "input_range_255": False,
    },
    "resnet50": {
        "loader": lambda nc, p: _load_resnet(50, nc, p),
        "input_size": 224,
        "mean": IMAGENET_RGB_MEAN, "std": IMAGENET_RGB_STD,
        "bgr": False, "input_range_255": False,
    },
    "resnet101": {
        "loader": lambda nc, p: _load_resnet(101, nc, p),
        "input_size": 224,
        "mean": IMAGENET_RGB_MEAN, "std": IMAGENET_RGB_STD,
        "bgr": False, "input_range_255": False,
    },
    "resnet152": {
        "loader": lambda nc, p: _load_resnet(152, nc, p),
        "input_size": 224,
        "mean": IMAGENET_RGB_MEAN, "std": IMAGENET_RGB_STD,
        "bgr": False, "input_range_255": False,
    },
    "inception_resnet_v2": {
        "loader": lambda nc, p: _load_inception_resnet_v2(nc, p),
        "input_size": 299,
        "mean": INCEPTION_MEAN, "std": INCEPTION_STD,
        "bgr": False, "input_range_255": False,
    },
}


ALL_MODEL_NAMES = list(MODELS.keys())


def build_model(name: str, num_classes: int = 100,
                pretrained: bool = False) -> nn.Module:
    """Instantiate one architecture by name, with the requested classifier width.

    Args:
        name: a key of MODELS, e.g. "resnet50" or "inception_v3".
        num_classes: width of the final classifier. Pass 1000 to keep the ImageNet
            architecture untouched, which is what the pruning pipeline uses.
        pretrained: load the published ImageNet weights. The classifier is only
            replaced when its width actually differs, so pretrained=True with
            num_classes=1000 preserves the trained head rather than overwriting it
            with a random one of identical shape.

    Returns:
        An nn.Module in eval() mode, with auxiliary logits disabled.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {ALL_MODEL_NAMES}")
    cfg = MODELS[name]
    m = cfg["loader"](num_classes, pretrained)
    m.eval()
    # torchvision Inception V3 still honours aux_logits in forward(); disable it
    # here too, harmlessly, for the timm variants.
    if hasattr(m, "aux_logits"):
        m.aux_logits = False
    return m


def get_input_size(name: str) -> int:
    """Native input resolution of a model, 224 or 299 pixels."""
    return MODELS[name]["input_size"]


def get_preprocessing(name: str) -> dict:
    """Return {input_size, mean, std, bgr, input_range_255} for a model.

    The single source of truth for preprocessing, used by the conversion scripts to
    calibrate quantization and by the benchmarks to build inputs. Three incompatible
    conventions coexist across this lineup and applying the wrong one degrades
    accuracy silently, so no caller should hard-code these values.
    """
    cfg = MODELS[name]
    return {
        "input_size": cfg["input_size"],
        "mean": cfg["mean"], "std": cfg["std"],
        "bgr": cfg["bgr"], "input_range_255": cfg["input_range_255"],
    }


if __name__ == "__main__":
    # Sanity check: build each model with random init, run one forward pass at
    # its native resolution, and confirm the output width.
    print("=" * 70)
    print("model_zoo sanity check (random init, num_classes=100)")
    print("=" * 70)
    for name in ALL_MODEL_NAMES:
        try:
            m = build_model(name, num_classes=100, pretrained=False)
            H = get_input_size(name)
            x = torch.randn(2, 3, H, H)
            with torch.no_grad():
                y = m(x)
            if isinstance(y, tuple):
                y = y[0]
            n_params = sum(p.numel() for p in m.parameters())
            print(f"  {name:30s}  in={H:3d}  out={tuple(y.shape)}  "
                  f"params={n_params/1e6:6.2f}M  OK")
        except Exception as e:
            print(f"  {name:30s}  [FAILED] {type(e).__name__}: {e}")

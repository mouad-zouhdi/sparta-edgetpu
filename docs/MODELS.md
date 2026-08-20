# Published models

Every model measured with this framework is published at
**https://huggingface.co/mouad-zouhdi/sparta-edgetpu-models**

37.38 GiB across 7479 files, which is why it is hosted there rather than in this
repository.

## Downloading

```bash
python setup/fetch_models.py --list                       # what is available
python setup/fetch_models.py --set measurements --out models/
python setup/fetch_models.py --set axis1-edgetpu --out models/
python setup/fetch_models.py --all --out models/          # 37.38 GiB
```

`measurements` is 0.12 GiB of CSVs and is the usual starting point. Downloads
resume: a re-run skips what is already present.

## Publishing an updated collection

```bash
python setup/stage_models.py --config setup/model_sources.json --out /path/to/stage
python setup/upload_models_hf.py --folder /path/to/stage \
    --repo mouad-zouhdi/sparta-edgetpu-models
```

`stage_models.py` builds the tree with hard links, so it costs no extra disk
space. `setup/model_sources.json` holds the source paths, which are specific to
the machine that produced the artefacts; edit that file rather than the script.
Both steps resume cleanly after an interruption.

---

# SPARTA — models and measurements for structured pruning on Edge TPU

Every model measured with the SPARTA framework, together with the benchmark CSVs
and the compiler reports that describe them.

**Code:** https://github.com/mouad-zouhdi/sparta-edgetpu
The repository README defines every column of every CSV published here.

Internship work at LAAS-CNRS. Author: Mouad Zouhdi.

---

## What this collection is

The framework measures how structured pruning behaves once a model actually runs
on a Google Coral Edge TPU: latency, accuracy, and the compiler's split between
parameters cached in on-chip memory and parameters streamed from the host.

These artefacts are published so the measurements can be checked, re-run on other
hardware, or extended without repeating the pruning, which costs thousands of
GPU-hours.

---

## Contents

| Path | Files | Size | What it is |
|---|---:|---:|---|
| `axis1_cifar100/baselines/` | 7 | 0.38 GiB | CIFAR-100 baselines, PyTorch |
| `axis1_cifar100/pruned_pytorch/` | 405 | 11.37 GiB | pruned and recovered, PyTorch |
| `axis1_cifar100/tflite_int8/` | 411 | 2.97 GiB | quantized, before compilation |
| `axis1_cifar100/edgetpu/` | 410 | 3.20 GiB | compiled binaries, the ones benchmarked |
| `axis1_cifar100/logs/` | 410 | 12 MiB | per-run logs: accuracies, achieved rates, layer structures |
| `axis2_imagenet/pruned_pytorch/` | 38 | 1.15 GiB | final models, plus the checkpoints that won their loop |
| `axis2_imagenet/edgetpu/` | 1548 | 8.00 GiB | compiled segments, 1 to 8 per model |
| `axis2_imagenet/logs/` | 392 | 4 MiB | training logs, pipeline summaries, compiler reports |
| `synthetic/tflite_int8/` | 307 | 8.01 GiB | synthetic corpus, quantized |
| `synthetic/edgetpu/` | 656 | 2.16 GiB | compiled, N = 1 to 8 |
| `synthetic/metadata/` | 416 | 2 MiB | structural metadata, successes and failures alike |
| `synthetic/compile_reports/` | 2456 | 12 MiB | per (model, N) compiler reports |
| `measurements/` | 21 | 0.12 GiB | every benchmark CSV, with a README describing each |

Total: 7479 files, 37.38 GiB.

### axis1_cifar100

Seven architectures (`resnet18`, `resnet50`, `vgg19`, `wrn_28_10`,
`mobilenetv2`, `googlenet`, `squeezenet1_1`) crossed with seven pruning criteria
(`magnitude_l1`, `magnitude_l2`, `bn_scale`, `fpgm`, `taylor`, `obdc`, `random`)
and nine target rates. Each combination is an independent run from the baseline.

Of 441 combinations, 404 produced a model. The rest are criterion and
architecture pairs that do not apply: `bn_scale` needs BatchNorm, and `obdc` does
not support depthwise convolutions or Fire modules.

Naming: `<architecture>_pruned<P>pct_<criterion>`, where `<P>` is the
**requested** target. The **achieved** reduction differs and is recorded in the
logs as `param_reduction_pct`; use that one.

### axis2_imagenet

Eight ImageNet architectures pruned from their published weights to a size
target, then compiled across 1 to 8 segments so that several accelerators share
the model. The `*_pipeline_summary.json` files record each iteration of the loop
that determined how much pruning was needed.

### synthetic

400 configurations: 5 topology families x 4 depths x 5 widths x 4 input
resolutions, of which 291 built. The networks are untrained; they exist to sample
memory and transfer behaviour more densely than a handful of real models can.

These `.tflite` files were produced through `onnx2tf`'s `flatbuffer_direct`
backend rather than `TFLiteConverter`, because models from the other path cannot
be compiled with `--num_segments` above 1. The repository documents this.

---

## What is not here

**The intermediate artefacts of the guided loop.** To make an ImageNet model fit
N accelerators, the pipeline prunes, quantizes, compiles, checks the result and
prunes again. Every iteration but the last is discarded, and those intermediate
checkpoints, quantized models and compiled segments are not published: nothing
was measured on them.

Their numbers are kept, in `axis2_imagenet/logs/*_pipeline_summary.json`, which
records the target and the resulting memory split for every iteration.

**Failed synthetic builds are documented rather than hidden.** Of 400
configurations, 291 built and 109 did not, through memory exhaustion during
conversion or an excessive parameter count. Their metadata files are included,
with their failure status.

---

## Using the models

The `.tflite` files are INT8, with INT8 input and output tensors. Those under
`edgetpu/` are already compiled for the accelerator; the others still need
`edgetpu_compiler`.

**Read them with `ai_edge_litert`, not with `tflite_runtime` 2.5**, the version
shipped with the Coral stack. That version misreads the quantization produced by
`ai-edge-quantizer`: the output collapses onto the zero point and accuracy reads
at chance level, with no error raised.

```python
from ai_edge_litert.interpreter import Interpreter, load_delegate
interp = Interpreter(
    model_path="axis1_cifar100/edgetpu/resnet18_pruned50pct_taylor_int8_edgetpu.tflite",
    experimental_delegates=[load_delegate("libedgetpu.so.1")],
)
interp.allocate_tensors()
```

The PyTorch checkpoints are **whole-model pickles**, not state dicts, because
structured pruning changes the architecture and a state dict alone could not
rebuild it. Load them with `weights_only=False`, and with `mono_tpu/` on
`sys.path` so that `cifar_resnet`, `cifar_vgg` and `wrn` resolve:

```python
import sys; sys.path.insert(0, "path/to/sparta-edgetpu/mono_tpu")
import torch
model = torch.load("axis1_cifar100/pruned_pytorch/resnet18_pruned50pct_taylor.pt",
                   weights_only=False)
```

---

## Downloading

A subset, which is usually what you want:

```python
from huggingface_hub import snapshot_download
snapshot_download("mouad-zouhdi/sparta-edgetpu-models", local_dir="models",
                  allow_patterns=["axis1_cifar100/edgetpu/*", "measurements/*"])
```

Or through the repository's helper, which names the subsets:

```bash
python setup/fetch_models.py --list
python setup/fetch_models.py --set measurements axis1-edgetpu --out models/
```

The whole collection, 37.38 GiB:

```python
snapshot_download("mouad-zouhdi/sparta-edgetpu-models", local_dir="models")
```

---

## Licence

MIT for the artefacts produced here. They derive from CIFAR-100 and ImageNet-1k
and from architectures whose original licences apply.

# Published models

Every model measured in this study is published at
**https://huggingface.co/mouad-zouhdi/sparta-edgetpu-models**

37.38 GiB across 7479 files, which is why it is hosted there rather than in this
repository.

## Download

```bash
python setup/fetch_models.py --list                       # what is available
python setup/fetch_models.py --set measurements --out models/
python setup/fetch_models.py --set axis1-edgetpu --out models/
python setup/fetch_models.py --all --out models/          # 37.38 GiB
```

Start with `measurements`: 0.13 GiB of CSVs, which is what every result in the
README is computed from.

## Publishing an updated collection

```bash
python setup/stage_models.py --config setup/model_sources.json --out /path/to/stage
python setup/upload_models_hf.py --folder /path/to/stage \
    --repo mouad-zouhdi/sparta-edgetpu-models
```

`stage_models.py` builds the tree with hard links, so it costs no disk space and
takes about a second. `model_sources.json` holds the source paths, which are
specific to the machine this research ran on; edit that file rather than the
script. Both operations resume cleanly after an interruption.

---

# SPARTA — measured models for structured pruning on Edge TPU

Every model measured in the SPARTA study, together with the measurement CSVs and
the compiler reports that describe them.

**Code:** https://github.com/mouad-zouhdi/sparta-edgetpu
The repository README documents every column of every CSV here.

Internship work at LAAS-CNRS. Author: Mouad Zouhdi.

---

## What this collection is for

The study measures how a structured-pruning mask turns into latency on Coral
Edge TPU accelerators. The central observation is that an Edge TPU holds about
8 MB of parameters in on-chip SRAM: parameters that fit are loaded once,
parameters that do not are re-streamed on **every inference**. That boundary,
not the operation count, dominates latency, which is why two pruning criteria at
equal accuracy can deliver different speedups.

These artefacts are published so the measurements can be checked, re-run on other
hardware, or extended without repeating several thousand GPU-hours of pruning.

---

## Contents

| Path | Files | Size | What it is |
|---|---:|---:|---|
| `axis1_cifar100/baselines/` | 7 | 0.38 GiB | CIFAR-100 baselines, PyTorch |
| `axis1_cifar100/pruned_pytorch/` | 405 | 11.37 GiB | pruned and recovered, PyTorch |
| `axis1_cifar100/tflite_int8/` | 411 | 2.97 GiB | quantized, before compilation |
| `axis1_cifar100/edgetpu/` | 410 | 3.20 GiB | **compiled binaries, the ones benchmarked** |
| `axis1_cifar100/logs/` | 410 | 12 MiB | per-run logs, accuracies, layer structures |
| `axis2_imagenet/pruned_pytorch/` | 38 | 1.15 GiB | final models + the PREFT checkpoints that won their loop |
| `axis2_imagenet/edgetpu/` | 1548 | 8.00 GiB | **compiled segments, 1 to 8 per model** |
| `axis2_imagenet/logs/` | 392 | 4 MiB | training logs, pipeline summaries, compiler reports |
| `synthetic/tflite_int8/` | 307 | 8.01 GiB | synthetic corpus, quantized |
| `synthetic/edgetpu/` | 656 | 2.16 GiB | **compiled, N = 1 to 8** |
| `synthetic/metadata/` | 416 | 2 MiB | structural metadata, successes and failures alike |
| `synthetic/compile_reports/` | 2456 | 12 MiB | per (model, N) compiler reports |
| `measurements/` | 21 | 0.12 GiB | every benchmark CSV, plus a README explaining each |

Total: 7479 files, 37.38 GiB.

## What is deliberately NOT here

**The guided loop's own probes.** To make an ImageNet model fit N accelerators,
the pipeline prunes, quantizes, compiles, reads how many bytes still stream, and
prunes again. Each iteration produces a checkpoint, a quantized model and a set
of compiled segments, and every one of them except the last is discarded. Those
200 intermediate artefacts (3.8 GiB) are not published: nothing was ever measured
on them.

Their **numbers** are kept, in `axis2_imagenet/logs/*_pipeline_summary.json`,
which records the target and the resulting on-chip and off-chip bytes for every
iteration. That is what the fixed-overhead regression quoted below was fitted on,
so it stays reproducible without the binaries.

The 12 PREFT checkpoints that ARE here are the ones that won their loop: each is
the exact input to the fine-tuning that produced a published model.

**Failed synthetic builds are documented, not hidden.** Of 400 configurations,
291 built and 109 did not, through `tf2onnx` running out of memory or exceeding
the parameter ceiling. Their metadata JSONs are included with their failure
status: they map the empirical boundary of what this toolchain handles, which is
a result rather than an omission.


---

## Axis 1 — comparing pruning criteria (CIFAR-100)

Seven architectures x seven criteria x nine pruning targets, each an independent
run from the baseline. 441 combinations attempted, **404 successful**; the 37
failures are known criterion/architecture incompatibilities (OBDC on depthwise
and Fire modules, bn_scale on an architecture without BatchNorm), not lost data.

| Architecture | Params | Baseline top-1 |
|---|---:|---:|
| resnet18 | 11.2 M | 76.89 % |
| resnet50 | 23.7 M | 77.81 % |
| vgg19 | 20.1 M | 73.49 % |
| wrn_28_10 | 36.5 M | 81.10 % |
| mobilenetv2 | 2.4 M | 64.86 % |
| googlenet | 5.7 M | 77.49 % |
| squeezenet1_1 | 0.8 M | 55.50 % |

Criteria: `magnitude_l1`, `magnitude_l2`, `bn_scale`, `fpgm`, `taylor`, `obdc`,
`random`. Protocol follows PruningBench (Li et al., arXiv:2406.12315): one
uniform recipe for every architecture, deliberately untuned per model, so results
remain comparable with that leaderboard.

Naming: `<arch>_pruned<P>pct_<criterion>`, where `<P>` is the **requested**
target. The **achieved** reduction differs and is recorded in the logs as
`param_reduction_pct`; use that one for any plot or correlation.

## Axis 2 — fitting real models to N accelerators (ImageNet)

ResNet-50/101/152, GoogLeNet, BN-Inception, Inception V3/V4 and
Inception-ResNet-V2, pruned from published ImageNet weights to a size target
rather than to a grid, then compiled across N segments so that N accelerators
pool their SRAM.

Pruning to a parameter count does not make a model fit: the compiler reports a
fixed per-architecture overhead that does not shrink with pruning (2.3 MiB for
ResNet-101, 5.2 MiB for Inception-ResNet-V2). The pipeline therefore measures the
compiled result and re-prunes until nothing streams, landing 6 to 35 points
deeper than the arithmetic predicts.

## Synthetic corpus

400 configurations: 5 topology families x 4 depths x 5 widths x 4 resolutions,
291 of which built successfully. Untrained: these networks exist to sample memory
and transfer behaviour far more densely than 15 real models can.

Note that these `.tflite` files were produced through `onnx2tf`'s
`flatbuffer_direct` backend, not through `TFLiteConverter`. That is not a
stylistic choice: models converted through the MLIR path make
`edgetpu_compiler --num_segments N` segfault for every N >= 2. The repository
documents the diagnosis.

---

## Reading the models

The `.tflite` files are INT8 with INT8 input and output tensors. Those under
`edgetpu/` are already compiled for the accelerator; the others still need
`edgetpu_compiler`.

**Do not read them with `tflite_runtime` 2.5**, the version shipped with the
Coral stack. It misreads the quantization produced by `ai-edge-quantizer`:
outputs collapse onto the zero point, every prediction lands on the same class,
and accuracy reads as chance **without any error being raised**. Use
`ai_edge_litert` instead.

```python
from ai_edge_litert.interpreter import Interpreter, load_delegate
interp = Interpreter(
    model_path="axis1_cifar100/edgetpu/resnet18_pruned50pct_taylor_int8_edgetpu.tflite",
    experimental_delegates=[load_delegate("libedgetpu.so.1")],
)
interp.allocate_tensors()
```

The PyTorch checkpoints are **whole-model pickles**, not state dicts, because
structured pruning changes the architecture and a state dict could not rebuild
it. Loading them therefore needs `weights_only=False` and needs the defining
modules importable under their original names (`cifar_resnet`, `cifar_vgg`,
`wrn`), which is why those files sit at the top level of `mono_tpu/` in the code
repository:

```python
import sys; sys.path.insert(0, "path/to/sparta-edgetpu/mono_tpu")
import torch
model = torch.load("axis1_cifar100/pruned_pytorch/resnet18_pruned50pct_taylor.pt",
                   weights_only=False)
```

---

## Downloading

Whole collection (41 GiB):

```python
from huggingface_hub import snapshot_download
snapshot_download("mouad-zouhdi/sparta-edgetpu-models", local_dir="models")
```

One subset, which is usually what you want:

```python
snapshot_download("mouad-zouhdi/sparta-edgetpu-models", local_dir="models",
                  allow_patterns=["axis1_cifar100/edgetpu/*", "measurements/*"])
```

---

## Licence

MIT for the artefacts produced here. They derive from CIFAR-100 and ImageNet-1k
and from architectures whose original licences apply.

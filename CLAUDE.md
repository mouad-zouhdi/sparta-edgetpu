# SPARTA: working notes

What this project measured, what it found, and what it is easy to get wrong.
Written for whoever picks the work up next, agent or human. Everything here is
either in this repository or on the Hub collection it names; nothing depends on
the machine the research ran on.

- **Code:** https://github.com/mouad-zouhdi/sparta-edgetpu
- **Models and measurements:** https://huggingface.co/mouad-zouhdi/sparta-edgetpu-models
- Internship at LAAS-CNRS. Author: Mouad Zouhdi.

Read `docs/PITFALLS.md` before running anything, `report/rapport.pdf` for the
written-up version, and `paper/DSD.pdf` for the four-page one.

---

## The question

Structured pruning removes whole channels, so it shrinks a network on paper by a
predictable factor. On a Google Coral Edge TPU it does not deliver a predictable
speedup, and the reason is memory.

The accelerator has about 8 MiB of on-chip SRAM. Whatever fits there is loaded
once and kept; whatever does not is streamed from the host on **every single
inference**. Pruning that moves a model across that threshold buys an enormous
speedup. Pruning that does not buys almost nothing. So the question is not how
much a criterion prunes but where its mask lands relative to that threshold, and
the ranking of criteria can invert between the two regimes.

Two axes answer it.

- **Axis 1, one accelerator, CIFAR-100.** 7 architectures x 7 importance
  criteria x 9 target rates, each an independent run from a common baseline.
  441 combinations, 404 models, all benchmarked.
- **Axis 2, several accelerators, ImageNet-1k.** 8 architectures pruned from
  pretrained weights to a size target, compiled across 1 to 8 segments so that
  several accelerators share one model, then measured over every way of
  splitting eight accelerators between instances.

A third, supporting corpus of **291 synthetic networks** samples the memory and
transfer behaviour more densely than a handful of real models can.

---

## Repository map

```
setup/          environments, patches, staging and fetching of the artefacts
requirements/   pinned dependencies of the three environments
mono_tpu/       axis 1: baselines, pruning, quantization, compilation, benchmarks
multi_tpu/      axis 2: ImageNet pruning to a size target, segmentation, benchmarks
  bench/                phases A to D, cold start, chained latency
  bench/partition_grid/ the campaign every multi-accelerator result rests on
synthetic/      the synthetic corpus and its build chain
report/         the internship report, its figures and their generators
paper/          the DSD paper, the conference talk, the defence slides
docs/           PITFALLS.md, MODELS.md, TESTING.md, the smoke test
```

Three environments, because they cannot coexist: `pytorch-env` (training,
pruning, conversion), `coral-env` (`tflite_runtime`, `pycoral`, **no torch**),
`synth-env` (Python 3.12, `onnx2tf` >= 2.5). `setup/setup_envs.sh` builds all
three; `setup/apply_patches.py` must run on every `pytorch-env`, or GoogLeNet and
SqueezeNet are rejected by the compiler.

---

## Axis 1: one accelerator, CIFAR-100

| # | Script | What it does |
|---|---|---|
| 0 | `mono_tpu/00_prepare_baselines.py` | 7 baselines, 200 epochs, the PruningBench recipe |
| 1 | `mono_tpu/01_prune.py` | progressive pruning, then 100 epochs of recovery. Logs float top-1 and top-5 with a confidence interval, and the post-pruning layer structure |
| 2 | `mono_tpu/02_convert_tflite_int8.py` | PyTorch to ONNX to TFLite float32 NHWC to INT8, calibrated on 100 training images |
| 3 | `mono_tpu/03_compile_edgetpu.py` | INT8 to Edge TPU, keeping the compiler's metrics |
| 4 | `mono_tpu/04_benchmark.py` | steady-state CPU and TPU latency, top-1 and top-5 over the 10k test set |
| 5 | `mono_tpu/05_benchmark_coldstart.py` | the warm-up curve: K passes of N inferences on the **same** interpreter |
|  | `mono_tpu/aggregate_pruning_logs.py` | per-run logs to `fp32_accuracy.json` and `layer_sparsity.json` |

### The lineup

| Model | Params | Baseline top-1 |
|---|---:|---:|
| ResNet-18 | 11.2M | 76.89 % |
| ResNet-50 | 23.7M | 77.81 % |
| VGG-19 | 20.1M | 73.49 % |
| WRN-28-10 | 36.5M | 81.10 % |
| MobileNetV2 | 2.4M | 64.86 % |
| GoogLeNet | 5.7M | 77.49 % |
| SqueezeNet 1.1 | 0.8M | 55.50 % |

MNASNet-1.0 was dropped as redundant with MobileNetV2. ShuffleNet, DenseNet,
Inception-V3 and EfficientNet-Lite0 were rejected on conversion or input size.

### The recipe, uniform on purpose

Uniformity is what makes the numbers comparable with the PruningBench
leaderboard (Li et al. 2024, arXiv:2406.12315). It is not tuned per model.

- **Baseline**: SGD lr 0.1, momentum 0.9, weight decay 5e-4, MultiStep
  [120, 150, 180] gamma 0.1, **200 epochs**, batch 128, no pretraining.
- **Recovery**: SGD lr 0.01, same momentum and decay, MultiStep [60, 80],
  **100 epochs**.
- **Sparsity learning** (for `bn_scale`): SGD lr 0.01, decay 0, MultiStep
  [60, 80], 100 epochs, reg 1e-5. Cached per architecture, skipped for
  SqueezeNet which has no BatchNorm.
- **Pruning**: `iterative_steps=400`, global, `max_pruning_ratio=0.9`. The
  data-driven criteria (Taylor, OBD-C, Fisher, HRank) run 10 forward and
  backward batches before each `pruner.step()`.
- 7 criteria at seed 42: `magnitude_l1`, `magnitude_l2`, `bn_scale`, `fpgm`,
  `taylor`, `obdc`, `random`. 9 rates: 10 to 90 % of parameters removed.

**Post-training quantization, not QAT**, for three reasons: leaderboard
comparability, isolation of the pruning effect, and because it is what makes the
quantization-friendliness of a pruned model measurable at all.

### What did not run, and why

37 of the 441 combinations produced nothing, and they are structural
incompatibilities propagated across all nine rates, not failures: OBD-C against
VGG-19, MobileNetV2 and SqueezeNet, and `bn_scale` against SqueezeNet. One
compilation failed, `mobilenetv2_pruned80pct_random`, which is why it is absent
from the benchmark CSV.

### Always use the achieved rate

`prune_pct` in a filename is the **target**. The achieved reduction is
`param_reduction_pct` in the logs, and the two differ. Every analysis uses the
achieved one.

---

## Axis 2: several accelerators, ImageNet-1k

Training eight ImageNet architectures from scratch on CIFAR-100 would have cost
about 4000 GPU-hours. The pivot was to start from published ImageNet weights,
prune towards a **size target**, and recover with a short cosine schedule.

**The recipe**: SGD lr 0.01, momentum 0.9, weight decay 1e-4, linear warmup then
per-step `CosineAnnealingLR`, batch 128, AMP fp16, `max_pruning_ratio=0.95`,
best-state selection with no early stopping.

**The epoch budget is decided after pruning, from the rate actually achieved**
(`FT_BUDGET_BANDS` in `multi_tpu/pipeline_full.py`): 15 epochs below 10 %, 20 up
to 30, 40 up to 45, 60 up to 70, 75 up to 85, 90 beyond. An earlier wave budgeted
on the *predicted* rate, which ran 6 to 35 points low in an architecture-dependent
way; `--epochs_from_actual` exists because of that.

### The guided loop

`pipeline_full.py` does not trust a size prediction. It prunes, quantizes,
compiles, reads the **external volume the compiler reports**, and prunes a little
more while that volume is above zero. Only then does it launch recovery.

⚠️ `--run_tag` is mandatory. Without it the ResNet-101 runs for N = 6, 7 and 8
collide, their initial `target_mb` being identical.

⚠️ Before launching a training for a given N, read `actual_pct` in the previous
N's `_pipeline_summary.json`. If the rate is the same, recompiling is enough.

---

## The central result: a two-regime latency model

With `E` the external volume and `I` the internal volume in MiB, `C` the compute
time, `k` the transfer cost and `c` a fixed cost:

```
steady-state latency   = C + k·E
first-inference latency = C + k·E + k·I + c
```

The internal part is loaded once and kept. The external part is retransmitted on
every inference. That is the whole model, and it holds without fitting: the four
USB models that stream all pay the **same** first-inference surcharge, 25.1 to
25.4 ms, agreeing to 1.6 %, although their sizes span 10.8 to 35.1 MiB. What they
share is their **internal** volume, 7.58 to 7.66 MiB.

### The values

| Quantity | Value | How |
|---|---|---|
| `k` (USB, Pi 4) | **3.29 ms/MiB** = 304 MiB/s | steady-state latency regressed on E, 410 models, R² = 0.9995 |
| `k` (PCIe, x86) | **2.76 ms/MiB** = 363 MiB/s | same, same binaries, R² = 0.9996 |
| `c` | **≈ 0.5 ms** | first-to-second inference gap, 7 architectures |
| `C` | 0.57 to 2.23 ms | measured on the 241 models that fit in SRAM |

Per architecture the slope is 3.384 (ResNet-18), 3.247 (ResNet-50), 3.271
(VGG-19), 3.249 (WRN), a 4.2 % spread, each at R² > 0.9994. **The intercept is
C**, and it falls inside the independently measured range. The cold-start route
gives 3.25 on 7 baselines, 1.3 % away: two estimates with no data in common.

⚠️ **Do not use the older 3.3 / 2.67 / 0.43.** Those came from single-variable
cold-start fits, biased by the overlap term.

### Crossing the threshold

Crossing the SRAM threshold divides latency by **2.7 in median, 1.2 to 6.3**,
measured over the 20 pairs of configurations that straddle it at fixed criterion
and architecture.

⚠️ **The threshold is at 70 % reduction, not 60**, and WRN-28-10 is an explicit
exception: it still streams at 80 and 90 %, all seven of its variants at each
rate. ResNet-50 only fits from 70 % on.

### The host matters as much as the link

Same 241 binaries, same Coral USB stick: **3.292 ms/MiB on a Pi 4 against 2.317
on an x86 host**, a 30 % difference, larger than the 16 % between buses.

⚠️ **At constant host, USB beats the PCIe card**: 2.32 ms/MiB for x86 + Coral
USB against 2.80 for x86 + PCIe card. Never read the 3.29 to 2.80 gap as the cost
of changing bus; those two rows differ by host as well.

⚠️ **One convention throughout.** Going from USB to PCIe takes `k` from 3.29 to
2.76: that is **16 % less cost**, or 19 % more bandwidth, depending on which
quantity you name. The whole write-up reasons in **cost**, 16 % for the link and
30 % for the host, and gives the bandwidth reading only in passing.

### External validation

Gao, Choi and Wang (RTSS 2025 WiP) measure 340 MiB/s directly on a Coral USB
with a **Pi 5**, against our 304 on a **Pi 4**; a per-segment overhead of 1.0 ms
(range 0.3 to 2.0) against our ≈ 0.5; and they bound the compute-transfer overlap
between 0 and 100 % of compute. They add an output-transfer term at only 35 to
87 MiB/s, which is negligible for 100 or 1000 classes and would be fatal for a
dense-output task.

### Not established, and it does not matter

The overlap coefficient `beta` on USB is **not identifiable**: the two corpora
disagree (0.42 ± 0.05 on 116 synthetic models, 0.155 ± 0.072 on 241 axis-1
models, F = 14.1), while on PCIe they agree (0.580 ± 0.064 and 0.631 ± 0.040,
F = 1.6). `beta` is only identifiable when the range of C is wide **and**
uncorrelated with I. Forcing either value moves `k` by 1 %, so nothing rests on
it.

Compute is **1.54x slower behind USB** than behind PCIe, at identical silicon and
identical binaries, proportionally rather than by a fixed offset (slope 1.628,
zero intercept). The mechanism is unproven and this is outside the write-up's
scope.

---

## What the memory budget actually is

### The SRAM budget for weights shrinks as input resolution grows

On-chip memory holds the weights **and** the activation buffers, so the budget
left for weights is not the 7.6 MiB constant that CIFAR-100 suggests. Same
weights, only the resolution varying, at one accelerator:

| Resolution | on-chip weights (MiB) | off-chip (MiB) | accelerators needed |
|---:|---:|---:|---:|
| 96 | 7.71 | 16.72 | 4 |
| 224 | 7.00 | 17.43 | 4 |
| 512 | 4.62 | 19.81 | 5 |
| 768 | 3.91 | 20.81 | 7 |

Confirmed on the eight ImageNet baselines: 7.14 MiB for the ResNets at 224x224,
6.59 for GoogLeNet, 5.53 for Inception-V3 and Inception-ResNet-V2 at 299x299.
**This is why GoogLeNet fits on CIFAR-100 and overflows by 0.19 MiB on
ImageNet-1k**, at nearly identical file size.

⚠️ **A model's need cannot be deduced from its file size. It has to be measured.**

### A fixed per-architecture overhead

Regressing the volume the compiler reports on the parameter target in MiB, over
every iteration of the guided loop:

| Model | n | Slope | **Fixed overhead (MiB)** | R² |
|---|---:|---:|---:|---:|
| ResNet-101 | 18 | 0.962 ± 0.013 | **2.29 ± 0.36** | 0.997 |
| Inception-V4 | 49 | 1.027 ± 0.011 | **2.86 ± 0.32** | 0.994 |
| Inception-ResNet-V2 | 60 | 0.970 ± 0.009 | **5.24 ± 0.28** | 0.995 |

A slope near 1 says the INT8 "one byte per weight" assumption is sound. It is the
constant term that a naive prediction ignores, and it does **not** shrink with
pruning: fused BatchNorms, per-channel int32 biases, memory alignment. That is
why Inception-ResNet-V2 has to be pruned to 95 % to fit on one accelerator.

### Three sizes, never to be confused

Conflating these has already produced one incoherent table. For an unpruned
ResNet-152:

| Quantity | Source | Value |
|---|---|---|
| INT8 TFLite file size | `os.path.getsize` of the input | 59.39 MiB |
| **Compiled weight volume** = `on_chip_used_mb + off_chip_used_mb` | the compiler's report | **58.19 MiB** |
| Edge TPU binary size | `output_mb`, `size_mb` in the metadata CSV | 61.54 MiB |

**Only the second one splits into internal and external**, so only it decides the
need. `on + off` is the binary minus 0.2 to 0.7 MiB of graph, code and metadata,
very regularly.

⭐ **The gap between file and compiled volume separates pruned models from
unpruned ones.** Over the 42 checkpoints compiled to a single segment: median
−0.04 MiB on the 8 unpruned, range [−1.20, +0.43], both signs; median **+1.03 MiB**
on the 34 pruned, range [+0.20, +3.83], **always positive**. Pruning leaves layer
widths that are no longer multiples of the systolic array's granularity, and the
compiler pads them back up. It is the fixed overhead above, seen from the file.

### `off_chip` is not monotonic in N

A ResNet-101 checkpoint prepared for six accelerators fits at six, **overflows at
seven** (2.02 MiB), and fits again at eight. The compiler's segmentation
heuristic does not optimise for fitting in internal memory. This invalidates the
intuition that more segments cannot hurt.

Segment imbalance also grows with N, from 1.00 to 1.33 in max-over-mean internal
volume, and up to a factor 5.4 between first and last stage at N = 5. Since a
pipeline's throughput is the inverse of its slowest stage, that is one of the
causes of the throughput ceiling, and the entry point for a smarter partitioner.

---

## The multi-accelerator results

Card: ASUS CRL-G18U-P3DF, eight Coral Edge TPUs. A **partition of eight** splits
the accelerators between independent instances; an instance of N stages runs the
binary compiled into N segments. There are exactly p(8) = **22** such partitions,
from one eight-stage pipeline to eight independent copies. Notation: `4x2` is two
instances of four stages, `1x8` is eight instances of one.

The final campaign is 43 checkpoints x 22 partitions = **946 configurations**,
8.1 M timed inferences, 32 hours, zero failures.

1. **Deployment rule.** Partition onto the smallest number of accelerators that
   removes streaming (the "need"), then replicate. **30 of 31 in latency, 26 of
   31 in throughput.** The 5 exceptions fall in two families: a small residual of
   streaming that costs more to remove than to endure (GoogLeNet, 0.19 MiB, where
   `1x8` wins on both counts and which is also the only latency exception; and
   Inception-ResNet-V2 at 87 %); and a heterogeneous shape that beats the full
   pipeline on throughput because a pipeline of eight is rate-limited by its
   heaviest segment (unpruned ResNet-101 with `6+2`, Inception-V4 at 26 % and
   Inception-ResNet-V2 at 31 % with `7+1`).
2. **Divisibility.** When the need divides eight, **24 of 26** have a single
   optimum. When it does not, **9 of 17** present a trade-off.
3. ⭐ **The trade-off is very asymmetric.** Over the 11 cases, choosing latency
   costs **3.9 %** of throughput (median); choosing throughput costs **117 %** of
   latency; and the second is larger in **11 of 11**. So optimise for latency by
   default. That is a rule needing no further measurement.
4. **What to do with the remainder** (12 configurations): grouping beats
   scattering on both counts (+3.3 % throughput, −41 % worst latency). Grouping
   against absorbing depends on the objective: +22 % throughput but +32 % latency.
5. **Instances do not interfere; throughput composes.** The throughput of an
   N-stage instance does not depend on the partition around it (CV 0.2 to 0.6 %,
   at the noise floor), so total throughput is the sum over instances.
   ⚠️ Shown **between saturating partitions** only.
6. **Repeatability is driven by pipeline depth, not by the model.** The CV of
   instantaneous throughput rises from 0.16 % at one stage to 1.38 % at eight,
   because a pipeline's rate is the maximum of N noisy stage durations rather than
   their mean. Segment imbalance rises alongside it (max/mean 1.00 to 1.96,
   max/min 1.0 to 6.1).

⚠️ **Accelerator assignment has no effect.** 990 measurements at fixed assignment
and 990 at random, interleaved: F = 0.905, p = 0.94. An earlier note claiming an
effect was refuted. The method trap behind that error: never compare a range to a
standard deviation.

### Two conclusions that look contradictory and are not

"Optimise latency by default" is about choosing a partition for a **given model**.
"Throughput is bought more cheaply" is about what a **deeper cut** returns at
equal accuracy given up. Both hold; they answer different questions.

---

## Reading the published data

Everything below is on the Hub. Use a Python environment with `pandas`.

| File | Grain | Use it for |
|---|---|---|
| `measurements/benchmark_results.csv` | model | axis 1: latency, accuracy, memory split, 411 rows |
| `measurements/cold_start_results.json` | baseline | the warm-up curve, K x N matrix |
| `measurements/coldstart_*.csv` | model | first inference, 5 campaigns over 2 hosts x 2 accelerator types |
| `axis2_imagenet/measurements/accuracy_vs_tpu.csv` | checkpoint | **the synthesis**: need, accuracy, best partition per objective, trade-off. 43 rows |
| `axis2_imagenet/measurements/partitions_fixedn_par_config.csv` | (checkpoint, partition) | most multi-accelerator figures. 946 rows |
| `axis2_imagenet/measurements/partitions_fixedn_par_instance.csv` | + instance | per-stage analysis, critical stage, imbalance |
| `axis2_imagenet/measurements/segment_metadata_fixedn.csv` | (checkpoint, N, segment) | sizes, internal and external volume |
| `axis2_imagenet/measurements/partitions_sweepN*.csv` | ten passes | **error bars**, and non-saturating shapes |
| `axis1_cifar100/logs/*.json` | run | achieved rate, float accuracy, layer structure |

Join key everywhere: `(model, pct)` for axis 2, plus `corpus` where the same name
denotes two different trainings. `pct` is the **achieved** rate. `shape` is the
partition, parts in decreasing order joined by `+`. `fit_tpu` is the accelerator
count targeted, which is **not** `n_stages`.

### Four traps in the data

1. **`df.shape` is a pandas attribute, not the column.** Write `df['shape']`.
   `df.shape == '5+3'` returns an empty result, silently.
2. **`thr_fps_total` is repeated on every instance row** of one configuration: it
   is already the cumulative throughput. Take `.first()` per group.
   `thr_fps_instance` is the one that sums.
3. **A partition's latency is that of its slowest instance**, never the mean over
   instances. Use `lat_worst_ms_mean`, and do not rebuild it by taking a maximum
   over the per-instance file: the two summarise different underlying statistics
   and differ by about 0.3 ms, small enough to pass unnoticed and larger than the
   confidence interval.
4. **The fixed-count campaign has one pass, so its `*_std` is not an error bar.**
   It is the spread across the thousand inferences of one measurement. Dividing it
   by the square root of a thousand would claim a precision eleven times better
   than reality. Error bars come from the ten-pass campaign.

⚠️ **The latency columns of `pipeline_canonical.csv` and `hybrid_bench.csv` are
inflated by pycoral's runner** (50 ms to 17 s for models under 5 ms). Only their
**throughput** is usable. Pipeline latency is measured by chaining the segments
by hand, which is what `bench/partition_grid/` does.

---

## Traps, all of them met in practice

### Conversion and compilation

1. **`tflite_runtime` 2.5 silently saturates models quantized by
   `ai-edge-quantizer`**: every logit equal, top-1 at chance, cosine similarity
   NaN, and no error raised. Install `ai_edge_litert` and use its `Interpreter`.
2. **torchvision's Inception-V3 will not compile**: `transform_input` builds a
   6 MiB tensor. Use `timm.create_model("inception_v3")`.
3. **GoogLeNet and Inception-V3** must be loaded with `aux_logits=True` and have
   `aux1` / `aux2` neutralised before the ONNX export.
4. **`edgetpu_compiler --timeout_sec`** takes an underscore. Without it the flag
   fails silently.
5. **`setup/apply_patches.py` is required on every `pytorch-env`**: it fixes
   `onnx2tf`'s PADV2-to-PAD in `pool.py` and a rank validation in
   `ai_edge_quantizer`. Without it GoogLeNet and SqueezeNet are rejected by the
   compiler.
6. **`PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True`** is mandatory on ImageNet.
7. **A race on the conversion temp directory**: the directory name carries the
   PID for that reason.
8. `tf2onnx` is killed by the OOM reaper beyond about 145M parameters.

### ⭐ The compiler segfault, do not re-investigate

`edgetpu_compiler --num_segments N` with N >= 2 **segfaults** on any `.tflite`
whose description reads `MLIR Converted.` (TensorFlow's native converter, or
`onnx2tf` 1.x) and works on `onnx2tf flatbuffer_direct` (`onnx2tf` >= 2.5,
Python 3.12+). Six ways of fixing it failed; going through `onnx2tf` 2.5 is the
fix. Short of reverse-engineering a stripped binary, this goes no further.

The corrected chain: Keras to SavedModel to `tf2onnx` (`--inputs-as-nchw`, opset
18) to `onnx2tf` 2.5 on the `flatbuffer_direct` backend, then inject a
SignatureDef with `flatbuffer_utils`, then quantize with `BATCH_MATMUL` forced to
**TENSORWISE**, then compile.

Its own traps: `onnx2tf` >= 2.5 needs Python 3.12; its
`download_test_image_data()` breaks on numpy 2 (patch `np.load`); it emits
`.tflite` files **without a SignatureDef**, which the quantizer refuses; and the
quantizer puts per-axis quantization on `BATCH_MATMUL` (opcode 126, which is what
`onnx2tf` turns `Dense` into, **not** FULLY_CONNECTED), which the compiler
rejects.

### The Coral USB stick

1. **`load_delegate` fails about one time in five** when chaining interpreters.
   Retry up to eight times.
2. **The device enumerates as USB 2.0 before its firmware loads** (`1a6e:089a`,
   480 Mb/s), then reappears as `18d1:9302` at 5000 Mb/s. Do not blame the port
   before running an inference.
3. **Creating an interpreter costs 2.7 s**, 97 % of a campaign's wall time. Size
   a campaign in passes, not in models.

### The eight-accelerator card

1. **Never run two campaigns at once on the card.** The processes fight over
   accelerators and take the driver down with them.
2. The `apex` driver falls over beyond roughly 1.5 GB of simultaneous mappings,
   which is why model loading is isolated in subprocesses.
3. `/dev/apex_*` revert to `root:root` on reboot; re-trigger the udev rules.
4. **`RuntimeError: Pipeline was turned off before`** is harmless noise from
   `PipelinedModelRunner.__del__`. Never filter a monitor on `Traceback`; filter
   on the process exit status.

### Shell and process traps

1. **`pkill -f <pattern>` kills itself** when the pattern appears in the calling
   shell's own command line. Met three times. Write `[u]pload...`, or check the
   log instead.
2. **Never wait on a background job with `while pgrep -f name`**: the loop matches
   itself and spins forever.
3. **Always count files after an `rsync`.** One push delivered 38 of 684 segments
   and exited successfully.
4. A detached run survives a dropped connection with
   `setsid ... > log 2>&1 < /dev/null &`.

### Known bugs, patched

1. **MNASNet `_metadata`**: the `OrderedDict._metadata` must be preserved when
   cloning the state dict.
2. **OBD-C `_prepare_model`** needs a patch for `torch_pruning` 1.6.0, and still
   fails on depthwise convolutions, Fire modules, and VGG-19 (a 4608 / 4609 size
   mismatch).
3. **MobileNetV2 converges chaotically on CIFAR-100.** That is an artefact of
   running an ImageNet-native architecture at 32x32, not a bug.
4. **A shortened baseline makes achieved pruning rates look wrong**, and that is
   a second, independent cause from the one above.

---

## The report

`report/rapport.pdf`, in French, 70 pages. `report/README.md` says how to rebuild
it and which script writes which figure. Its scope is deliberately that of
`paper/DSD.pdf` minus the scheduling section: no more, no less, than the
single-accelerator results plus the multi-accelerator ones described here.

⚠️ **The paper's Table I differs from the report's**, deliberately: its memory
columns are computed, the report's are read from the compiler. The compiler's are
the ones to trust.

Style rules, fixed and requested: French throughout, including axis labels; **no
em dashes**; at most four panels per row; one legend per set of similar data; a
panel title is the model name alone, with the detail in the figure caption; an
asymptote of reference on any figure with accuracy on an axis; **no logarithmic
scale anywhere**; numbers that count or measure something written as digits.

### pgfplots traps met while building it

1. **Without `scale only axis`, `width` and `height` include titles and tick
   labels**, so a panel declared 5.0 x 4.2 cm draws only 2.4 cm of axis.
2. **A blank line inside an `axis`'s options breaks the parser**, with an error
   naming something else. Filter empty entries out of a generated option list.
3. **The decimal comma is pgfplots' list separator.** `xticklabels={0,8}` makes
   two ticks. Brace each label: `xticklabels={{0,8},{0,6}}`.
4. **`\pgfkeysvalueof{/pgfplots/xmin}` does not work under autoscale.** For a
   full-width rule, use
   `\draw ({rel axis cs:0,0}|-{axis cs:0,Y}) -- ({rel axis cs:1,0}|-{axis cs:0,Y});`
5. **In maths mode the decimal comma must be braced**, `$74{,}37$`, or LaTeX sets
   "74, 37" with a space. And a **negative sign must be in maths mode**, `$-$2,71`,
   or it comes out as the visibly shorter hyphen.
6. **The page count is not driven by prose.** Cutting 900 words gained no page,
   because with figures placed `[H]` every cut is absorbed by the whitespace a
   figure was already leaving. Removing or merging a float is what works.

---

## What is deliberately not published

- **The guided loop's own probes**: 108 pre-fine-tuning checkpoints from rejected
  iterations, and everything derived from them. Nothing was ever measured on them.
  Their numbers survive in the `*_pipeline_summary.json` files. The 31 winning
  checkpoints **are** published.
- **The float32 TFLite intermediates**, 13 GB and regenerable. `04_benchmark`
  skips the float path cleanly when they are absent.
- **The superseded synthetic chain**, the one whose models cannot be segmented.
  The corpus that replaced it is published in full, failures included.
- An early **EfficientNet-Lite0 on Imagenette** campaign is published under
  `archive_imagenette_efficientnet_lite0/` for completeness only. It is **not**
  comparable with axis 1: different dataset, protocol, resolution and criterion
  names, and no result rests on it.

## Where the numbers came from, for anything not published

`provenance_logs/` on the Hub holds the scheduler logs of the runs that produced
the models, and `axis1_cifar100/compile_logs/` the compiler's output for each
binary. Nothing is derived from them; they record what actually ran, and with
which options, which no aggregated file keeps.

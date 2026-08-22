# SPARTA — Structured pruning on Edge TPU accelerators

Measurement framework for **structured pruning** of convolutional networks
deployed on **Google Coral Edge TPU** accelerators, in two hardware regimes: a
single accelerator (Raspberry Pi 4 with a Coral USB stick) and an
eight-accelerator PCIe card (ASUS CRL-G18U-P3DF).

> Internship work at LAAS-CNRS. Author: Mouad Zouhdi.

---

## What this is for

The framework exists to measure how a structured-pruning mask behaves once the
model actually runs on an Edge TPU, rather than how much accuracy it costs. It
provides the tooling to:

- train baselines, prune them under a chosen criterion, and recover accuracy;
- quantize to INT8 and compile for the accelerator;
- measure latency, accuracy, throughput and the compiler's memory split, and
  write all of it to CSV.

An Edge TPU holds a limited amount of parameters in on-chip SRAM; anything above
that is streamed from host memory. The compiler reports that split, and the
benchmarks record it alongside the timings, so the two can be related.

The repository covers three bodies of work:

| Directory | Corpus | What varies |
|---|---|---|
| `mono_tpu/` | CIFAR-100, 7 architectures | pruning criterion and target rate |
| `multi_tpu/` | ImageNet, 8 architectures | size target and number of accelerators |
| `synthetic/` | 400 generated CNNs | topology, depth, width, input resolution |

---

## Repository layout

```
setup/            environment creation, model download and upload
requirements/     one dependency set per environment, pinned
mono_tpu/         single-accelerator pipeline (CIFAR-100)
multi_tpu/        multi-accelerator pipeline (ImageNet)
  bench/          pipeline, parallel and cold-start benchmarks
synthetic/        factorial CNN generator
docs/             pitfalls, model catalogue, testing notes
run_pipeline_mono_tpu.sh    runs the single-accelerator pipeline
run_pipeline_multi_tpu.sh   runs the multi-accelerator pipeline
```

Every script begins with a header describing what it produces and how it works,
and every function has a docstring. Start with the header of the script you
intend to run, or with `--help`.

---

## Installation

### Prerequisites

**Two Python versions must be on `PATH`**, and this is the only prerequisite the
setup script cannot resolve for you: `pycoral` ships wheels for Python 3.6 to 3.9
only, while `onnx2tf` 2.x requires 3.12. The script checks for each one and
refuses to build an environment on the wrong interpreter, because a `pytorch-env`
on 3.11 silently installs `onnx2tf` 1.x, whose output makes `edgetpu_compiler`
segfault under segmentation, hours later.

```bash
# Debian / Ubuntu, if 3.9 is missing
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9-venv python3.12-venv
```

Either version can also come from conda (`conda create -n py39 python=3.9`); the
script only needs to find `python3.9` and `python3.12` on `PATH`.

### Hardware

Neither accelerator is needed to install the environments or to run `--smoke`.
Which stage needs what is in the pipeline tables below; in short, a GPU for
training and pruning, `edgetpu_compiler` to compile, a Coral device only to
measure. The GPU is a matter of time and not of correctness: every stage runs on
CPU and produces the same artefacts, but a CIFAR-100 baseline is 200 epochs and
each recovery 100 more, and the ImageNet axis costs 0.27 to 0.50 GPU-hour per
epoch. Conversion, compilation and generation are CPU-only by construction.

### If the machine has one or more GPUs

No modification to any script. Each one resolves its own device, `cuda` when
available and `cpu` otherwise, `--device` overrides, and fp16 is enabled on the
ImageNet fine-tuning whenever the device is CUDA (`--no_amp` disables it).

A second GPU does not make one run faster: there is no `DataParallel` and no
`DistributedDataParallel`, so one process uses one device. The workload is a
sweep of independent runs, scheduled one per GPU through a SLURM array. To use
several, run several jobs at once:

```bash
CUDA_VISIBLE_DEVICES=0 MODELS=resnet18 ./run_pipeline_mono_tpu.sh &
CUDA_VISIBLE_DEVICES=1 MODELS=vgg19    ./run_pipeline_mono_tpu.sh &
```

On the ImageNet axis a batch of 128 at 224 px needs around 10 GB of VRAM;
`--batch_size 64` on a smaller card.

### Creating the environments

```bash
./setup/setup_envs.sh                      # creates envs/{pytorch,coral,synth}-env
./setup/setup_envs.sh pytorch coral        # a subset
./setup/setup_envs.sh --prefix /opt/sparta # somewhere other than the repository
./setup/setup_envs.sh --edgetpu-compiler   # instructions for the compiler
```

The environments are created inside the repository whatever the current
directory, since that is where the pipeline runners and `docs/run_smoke.sh` look
for them. Re-running is safe: an existing environment is reused and its packages
re-resolved. The script exits non-zero if any environment fails to import what it
needs, so a broken install is caught here rather than in a job.

Then check the result, which parses, imports and `--help`s every script in the
environment it is meant to run in:

```bash
bash docs/run_smoke.sh
```

Three environments are required because they are mutually incompatible, and
merging them breaks results rather than failing loudly. Beyond the Python
version split above, installing TensorFlow next to `onnx2tf` moves the pins the
PyTorch conversion path depends on, which is why the synthetic generator gets
its own. `coral-env` deliberately contains no torch, since it is deployed to a
Raspberry Pi and to the 8x Edge TPU host.

| Environment | Python | Used by |
|---|---|---|
| `pytorch-env` | 3.12 | training, pruning, PyTorch to TFLite conversion |
| `coral-env` | 3.9 | Edge TPU compilation, inference, benchmarking |
| `synth-env` | 3.12 | the synthetic generator |

`edgetpu_compiler` is a closed-source Google binary, x86-64 Linux only, and is
not a Python package. All results here were produced with version 16.0.

A Coral device additionally needs the runtime library `libedgetpu1-std`, and on
a freshly rebooted host the `/dev/apex_*` nodes can come back owned by root:
`sudo udevadm trigger --subsystem-match=apex`.

### Two things the install prints that are expected

`pip` reports a dependency conflict for `onnx2tf` and `tf-keras`, and the
`--- installing without dependency resolution ---` step is what puts them there.
Both declare pins that contradict versions this project is known to run on:
`onnx2tf` pins `numpy==1.26.4` and `onnx==1.20.1` against the numpy 2 / onnx 1.21
the conversion was validated on, and `tf-keras` pulls a second distribution that
would overwrite the `tensorflow` module already installed. Honouring either makes
the requirements file unsatisfiable or silently swaps a package for a different
build of the same module. Each requirements file lists these at the bottom, with
the reason. The conflict lines are expected; the import checks are what decides
whether the environment is sound.

---

## Running a pipeline

Two scripts run a complete pipeline end to end, one per axis. Every parameter is
a variable in a CONFIGURATION block at the top of the file, each with a comment
saying what it does and what to change it to.

```bash
./run_pipeline_mono_tpu.sh --help      # single accelerator, CIFAR-100
./run_pipeline_multi_tpu.sh --help     # multiple accelerators, ImageNet
```

```bash
./run_pipeline_mono_tpu.sh             # every stage, with the settings in the file
./run_pipeline_mono_tpu.sh --smoke     # minutes, to check the setup
./run_pipeline_mono_tpu.sh --from 02   # resume at a stage
./run_pipeline_mono_tpu.sh --only 04   # a single stage
./run_pipeline_mono_tpu.sh --dry-run   # print the commands, run nothing
```

Any variable can be overridden from the environment without editing the file:

```bash
MODELS="resnet18 vgg19" CRITERIA="taylor random" TARGETS="30 60" \
    ./run_pipeline_mono_tpu.sh

MODELS="resnet50 inception_v4" TARGETS_MB="8 16" \
    ./run_pipeline_multi_tpu.sh
```

`--smoke` shrinks every cost knob at once: one model, one criterion, minimal
epochs. It exercises the same code paths as a real run and the numbers it
produces mean nothing, which is the point. Run it first on a new machine.

Stages needing hardware are skipped with a message when the hardware is absent,
so the scripts are usable on a workstation without a Coral device. Each stage can
also be run on its own; `--dry-run` prints the exact command for every one.

## The single-accelerator pipeline (`mono_tpu/`)

Seven CIFAR-100 architectures (`resnet18`, `resnet50`, `vgg19`, `wrn_28_10`,
`mobilenetv2`, `googlenet`, `squeezenet1_1`), eleven pruning criteria
(`magnitude_l1`, `magnitude_l2`, `bn_scale`, `fpgm`, `taylor`, `obdc`, `random`,
`lamp`, `fisher`, `group_lasso`, `hrank`) and any set of target rates. Each
combination is an independent run starting from the baseline.

The protocol follows **PruningBench** (Li et al. 2024, arXiv:2406.12315): one
uniform recipe for every architecture, so results stay comparable with that
leaderboard.

| Stage | Script | Environment | Hardware |
|---|---|---|---|
| 00 | `00_prepare_baselines.py` | pytorch-env | GPU strongly advised |
| 01 | `01_prune.py` | pytorch-env | GPU strongly advised |
| — | `aggregate_pruning_logs.py` | pytorch-env | none |
| 02 | `02_convert_tflite_int8.py` | pytorch-env | none, CPU by construction |
| 03 | `03_compile_edgetpu.py` | coral-env | `edgetpu_compiler` |
| 04 | `04_benchmark.py` | coral-env | Coral device for `--device tpu` |
| 05 | `05_benchmark_coldstart.py` | coral-env | Coral device for `--device tpu` |

Some criterion and architecture pairs do not apply: `bn_scale` needs BatchNorm,
and `obdc` does not support depthwise convolutions or Fire modules. Those runs
are logged and skipped rather than aborting the sweep.

## The multi-accelerator pipeline (`multi_tpu/`)

Eight ImageNet architectures (GoogLeNet, BN-Inception, Inception V3/V4,
Inception-ResNet-V2, ResNet-50/101/152), pruned from published weights to a
**size target** rather than to a fixed rate, then compiled across N segments with
`edgetpu_compiler --num_segments N` so that N accelerators share the load.

| Size target | Segments | Configuration |
|---|---|---|
| 8 MB | 1 | 8 independent models in parallel |
| 16 MB | 2 | 4 pipelines of 2 accelerators |
| 32 MB | 4 | 2 pipelines of 4 |
| 64 MB | 8 | 1 pipeline across all 8 |

`pipeline_full.py` drives the whole thing: it prunes, quantizes, compiles, reads
how many bytes the compiler still streams off-chip, and prunes further if any
remain, before launching the recovery fine-tuning. Every iteration is recorded in
a pipeline summary JSON.

| Stage | Script | Environment | Hardware |
|---|---|---|---|
| 00 | `00_fetch_and_convert_pretrained.py` | pytorch-env | none, CPU by construction |
| — | `pipeline_full.py` (drives 01 to 03) | pytorch-env + compiler | GPU, and `edgetpu_compiler` |
| 01 | `01_prune_imagenet.py` | pytorch-env | GPU required in practice |
| 02 | `02_convert_pruned.py` | pytorch-env | none, CPU by construction |
| 03 | `03_compile_edgetpu_segments.py` | coral-env | `edgetpu_compiler` |
| — | `verify_tpu.py` | coral-env | Coral device |
| bench | `bench/bench_pipeline.py`, `bench_parallel.py`, `bench_coldstart_*.py`, `bench_latency_chained.py` | coral-env | Coral device |

## The synthetic generator (`synthetic/`)

400 configurations: 5 topology families (`sequential`, `residual`, `dense`,
`branched_2way`, `branched_4way`) x 4 depths x 5 widths x 4 input resolutions.
The networks are never trained; they exist to sample memory and transfer
behaviour more densely than a handful of real models can.

```bash
python synthetic/generate_sweep.py --workers 2 --num_calib 100 --skip-existing
python synthetic/compile_sweep.py --tflite-dir outputs/tflite --out-root outputs_pipeline
```

Conversion goes through `onnx2tf`'s `flatbuffer_direct` backend rather than
`TFLiteConverter`; `synthetic/build_one.py` explains why and checks that the
produced file took that path.

---

## Models and measurements

Every model measured with this framework, and every benchmark CSV, is published
separately because the collection is about 37 GB. See `docs/MODELS.md`.

```bash
python setup/fetch_models.py --list                        # what is available
python setup/fetch_models.py --set measurements --out models/
python setup/fetch_models.py --set axis1-edgetpu --out models/
```

---

# Benchmark metrics

Every benchmark writes a CSV. This section defines each column.

Two conventions hold throughout:

- **A missing measurement is written as an empty field, never as zero**, so an
  absent value is never mistaken for a measured one.
- **Signed deltas are negative for a loss.** `quant_drop_top1 = -1.2` means 1.2
  points of top-1 were lost.

---

## `benchmark_results.csv` — steady-state, single accelerator

Written by `mono_tpu/04_benchmark.py`, one row per model. The file is cumulative:
running with `--device cpu` and later `--device tpu` fills in the same file, and
the derived columns are recomputed each pass from whatever is present.

### Identity

| Column | Definition |
|---|---|
| `model` | full model name, e.g. `resnet18_pruned50pct_taylor` |
| `tag` | `finetuned` for a baseline, `pruned` otherwise |
| `importance` | pruning criterion; empty for a baseline |
| `prune_pct` | the **requested** target, in % of parameters |

`param_reduction_pct` below is the **achieved** reduction. The two differ,
because global pruning removes whole dependency groups of varying size. Use the
achieved value on plots and in correlations.

### Size and cost

| Column | Definition |
|---|---|
| `size_int8_mib` | size of the INT8 `.tflite` file |
| `params_int8` | parameter count in the quantized model |
| `macs_int8` | estimated multiply-accumulate operations per inference |

### Memory split, from the compiler report

| Column | Definition |
|---|---|
| `tpu_on_chip_mib` | parameters cached in on-chip SRAM |
| `tpu_off_chip_mib` | parameters streamed from host memory on each inference |
| `tpu_streaming_ratio` | off-chip divided by total |
| `tpu_sram_util_pct` | SRAM occupancy against the device budget |
| `tpu_subgraphs` | number of Edge TPU subgraphs |
| `tpu_ops_count`, `cpu_ops_count` | operations mapped to each device |
| `tpu_ops_coverage_pct` | share of operations running on the accelerator |

A large `cpu_ops_count` means part of the graph fell back to the host, and the
measured latency then includes host-side work. A few operations at the tail
(softmax, reshape) are normal.

### Accuracy

| Column | Definition |
|---|---|
| `top1_cpu_f32_pct`, `top5_cpu_f32_pct` | FP32 reference accuracy |
| `top1_int8_pct`, `top5_int8_pct` | accuracy after quantization, measured on the accelerator |
| `*_ci95_lo`, `*_ci95_hi` | bootstrap 95 % confidence bounds |
| `n_eval_acc` | number of images evaluated |

INT8 accuracy is taken from the Edge TPU rather than from the INT8 CPU path: the
compiler's kernels are not bit-identical to the CPU ones, and the Coral stack's
`tflite_runtime` 2.5 misreads models produced by `ai-edge-quantizer` without
raising an error.

### Latency and throughput

| Column | Definition |
|---|---|
| `lat_{cpu,tpu}_{f32,int8}_ms_{mean,std,median,p95,p99}` | steady-state latency |
| `lat_*_cv` | coefficient of variation, std over mean |
| `throughput_{cpu_f32,cpu_int8,tpu_int8}_fps` | inferences per second |

Steady state means: warmup inferences discarded, then N inferences timed on one
interpreter that stays alive. Model loading and the first-inference weight
transfer are therefore excluded, and only `invoke()` is inside the timed region.
`cold_start_results.csv` covers the excluded part.

### Accuracy deltas

| Column | Definition |
|---|---|
| `quant_drop_top{1,5}` | INT8 minus FP32, same weights |
| `prune_drop_top{1,5}_f32` | pruned minus baseline, both FP32 |
| `prune_drop_top{1,5}_i8` | pruned minus baseline, both INT8 |
| `combined_drop_top{1,5}` | pruned INT8 minus baseline FP32 |

### Speedups and compression

| Column | Definition |
|---|---|
| `tpu_speedup_int8` | CPU INT8 latency divided by TPU INT8 latency |
| `quant_speedup_cpu` | CPU FP32 divided by CPU INT8 |
| `prune_speedup_{cpu,tpu}` | baseline latency divided by pruned latency |
| `theoretical_speedup_macs` | baseline MACs divided by pruned MACs |
| `tpu_realization_efficiency` | `prune_speedup_tpu` divided by `theoretical_speedup_macs` |
| `param_reduction_pct`, `size_reduction_pct`, `macs_reduction_pct` | reductions against the baseline |
| `compression_ratio` | baseline size divided by pruned size |

---

## `cold_start_results.csv` — first-inference cost

Written by `mono_tpu/05_benchmark_coldstart.py`, one row per model, with the
block below repeated for `cpu_int8`, `cpu_f32` and `tpu_int8`:

| Column | Definition |
|---|---|
| `<mode>_n_passes` | passes recorded for this mode |
| `<mode>_cold_{mean,std,median,p95,p99,min,max,cv,n}` | position 1: the first inference after loading |
| `<mode>_steady_{...}` | positions 6 to 10 pooled: after warm-up |

One pass is N inferences on a **fresh** interpreter, then the next model. Model
order is reshuffled between passes so that any effect drifting over the run is
not absorbed into the per-model result. The JSON alongside also keeps the full
K x N matrix and per-position statistics.

---

## `coldstart_axis1.csv`, `coldstart_synth.csv` — raw first-inference series

Written by `multi_tpu/bench/bench_coldstart_{pcie,usb}.py`, in long format:

| Column | Definition |
|---|---|
| `pass` | pass index |
| `tag` | model |
| `position` | 1 is the first inference, 2 to N follow it |
| `lat_ms` | measured latency |
| `timestamp` | when the measurement was taken |

Raw rather than summarised, so any aggregation can be done downstream.

---

## `N1_baseline.csv`, `pipeline_canonical.csv`, `pipeline_spread.csv`

Written by `multi_tpu/bench/bench_pipeline.py`, phases A, B and C, sharing one
schema:

| Column | Definition |
|---|---|
| `tag` | model |
| `N` | number of pipeline stages, hence accelerators |
| `perm` | TPU order, comma-separated |
| `kind` | `baseline` at N=1, `canonical` for order 0..N-1, `random` otherwise |
| `perm_idx` | permutation index within phase C |
| `throughput_fps` | inferences per second |
| `lat_ms_{mean,std,median,p95,p99,min,max}` | latency |
| `cold_first_lat_ms` | first inference of the series |
| `warmup`, `reps` | measurement parameters |

**Use the throughput columns, not the latency ones.** These phases run through
pycoral's `PipelinedModelRunner`, which is built for throughput and whose
per-item timing does not represent end-to-end latency. Pipeline latency is
measured separately by `bench_latency_chained.py`, which chains the segments
manually.

---

## `parallel_bench.csv` — N copies on N accelerators

Written by `multi_tpu/bench/bench_parallel.py`, in long format, one row per
`(mode, tag, n_tpu, tpu_idx, rep)`:

| Column | Definition |
|---|---|
| `mode` | `steady` or `cold` |
| `tag`, `n_tpu`, `tpu_idx`, `rep` | model, accelerator count, which one, repetition |
| `latency_ms` | that inference |
| `family`, `depth`, `base_width`, `resolution`, `num_params` | structural metadata |
| `tflite_size_mb`, `edgetpu_size_mb` | sizes |
| `on_chip_mb`, `off_chip_mb` | memory split |
| `num_ops_tpu`, `num_ops_cpu_fallback`, `num_subgraphs` | compiler output |

This measures the **parallel** regime: N independent copies of one model, each on
its own accelerator and its own inference stream. It complements
`bench_pipeline.py`, which splits one model across N accelerators.

In `steady` mode a barrier before each timed repetition makes every accelerator
start together. In `cold` mode a fresh interpreter per repetition forces SRAM
re-initialisation.

---

## Compiler reports (JSON)

`compiler_metrics.json` (single accelerator) and `<basename>_compile_report.json`
(multi) hold the parsed `edgetpu_compiler` output: `on_chip_used_mb`,
`off_chip_used_mb`, `ops_edgetpu`, `ops_cpu`, `num_subgraphs`, `compile_ms`, per
segment, plus totals.

---

## Reproducibility

Seed 42 throughout, propagated to `random`, `numpy` and `torch` (CPU and CUDA),
and re-set at the start of each run so a run behaves the same alone or inside a
sweep. Calibration draws use seed 42; benchmark inputs and verification use seed
123, kept distinct so quantization is never evaluated on the images that
calibrated it.

**What the seed does and does not guarantee.** Pruning is bit-reproducible: two
runs of the same triple at the same seed select the same channels and reach the
same parameter reduction to the last digit. Fine-tuning is not: two such runs
were measured at 24.55 % and 24.20 % top-1. The difference comes from GPU
kernel non-determinism, which the seed does not control. Set
`torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG` if you
need the training reproducible too, at some cost in speed.

So a reported accuracy carries a run-to-run spread of a few tenths of a point,
independently of the evaluation noise the bootstrap intervals describe. Treat
differences smaller than that as unresolved.

Three sources of variability can be separated: bootstrap confidence intervals on
accuracy, the calibration draw (`--calibration_seed`), and the training seed.

---

## Testing

`docs/run_smoke.sh` checks that every script parses, imports in its environment
and answers `--help`. `docs/TESTING.md` describes what is covered.

```bash
bash docs/run_smoke.sh
```

## Known pitfalls

`docs/PITFALLS.md` collects the failure modes worth knowing before running
anything, several of which produce plausible-looking output rather than an error.

## References

- Li et al., *PruningBench: A Comprehensive Benchmark of Structural Pruning*,
  arXiv:2406.12315 (2024).
- Fang et al., *DepGraph: Towards Any Structural Pruning*, CVPR 2023, the
  Torch-Pruning library used throughout.
- Li et al., *Pruning Filters for Efficient ConvNets*, ICLR 2017.
- Renda et al., *Comparing Rewinding and Fine-tuning in Neural Network Pruning*,
  ICLR 2020.

## Licence

MIT for the code. The published models derive from CIFAR-100 and ImageNet-1k and
from architectures whose original licences apply.

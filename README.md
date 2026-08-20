# SPARTA — Structured pruning on Edge TPU accelerators

Measurement framework for **structured pruning** of convolutional networks
deployed on **Google Coral Edge TPU** accelerators, across two hardware regimes:
a single accelerator (Raspberry Pi 4 + Coral USB) and an eight-accelerator PCIe
card (ASUS CRL-G18U-P3DF).

The question it exists to answer is not "how much accuracy does pruning cost",
which is well covered elsewhere, but **how a pruning mask turns into latency on
real hardware**. Those are not the same thing, and the gap between them is where
the results live.

> Internship work at LAAS-CNRS. Author: Mouad Zouhdi (`mouad.zouhdi@laas.fr`).

---

## The central finding, in one paragraph

An Edge TPU holds about **8 MB of parameters in on-chip SRAM**. Parameters that
fit are loaded once and reused; parameters that do not fit are **re-streamed
across the host bus on every single inference**. That boundary, not the operation
count, dominates latency.

The consequence is that pruning does not deliver its arithmetic saving. Two
criteria that remove the same fraction of parameters, at the same accuracy, can
produce different speedups, because they distribute sparsity differently across
layers and therefore land on different sides of that boundary. The ranking
between criteria can invert between the SRAM-resident regime and the streaming
regime. The algorithm never sees the hardware; what changes is the mapping from
mask to latency.

Quantitatively, the measured latency model is:

```
steady-state latency  = C + k·E
first-inference cost  = C + k·E + k·I + c
```

with `C` the compute time, `E` the parameter volume streamed from off-chip on
every inference, `I` the volume cached in SRAM once, `k` the transfer cost per
MiB and `c` a fixed overhead. Measured values:

| Quantity | Value | Method |
|---|---|---|
| `k` (USB, Pi 4) | **3.3 ms/MiB** = 307 MiB/s | first- minus second-inference gap, 7 architectures |
| `c` (USB) | **0.43 ms** | same fit, max residual 0.47 ms |
| `k` (PCIe) | **2.67 ms/MiB** = 375 MiB/s | 116 synthetic models + an overlap term, R² = 0.92 |
| overlap | **58 %** of compute time masks loading | two-variable regression |

The decomposition is not merely a fitted form. The four USB models that stream
all pay nearly the same first-inference surcharge (25.1 to 25.5 ms, agreeing to
1.6 %) although their total sizes range from 10.8 to 35.1 MiB. What they share is
not size but **internal volume**: 7.58 to 7.66 MiB.

An independent measurement (Gao, Choi and Wang, RTSS 2025 WiP) reports
340 MiB/s host-to-device on a Coral USB with a Pi 5, against our 307 MiB/s on a
Pi 4: 11 % apart, in the expected direction.

---

## Repository layout

```
setup/            environment creation and upstream patches
requirements/     one pinned dependency set per environment
mono_tpu/         axis 1 — CIFAR-100, 7 architectures x 7 criteria x 9 targets
multi_tpu/        axis 2 — ImageNet, size-targeted pruning, N-segment pipelines
  bench/          pipeline, parallel and cold-start benchmarks
synthetic/        400-configuration factorial CNN generator
docs/             pitfalls and hardware notes
```

Every script carries a header explaining what it produces and why it works the
way it does, and every function has a docstring. Start with the header of the
script you intend to run.

---

## Two experimental axes

### Axis 1 — comparing pruning criteria (`mono_tpu/`)

Seven CIFAR-100 architectures, seven importance criteria, nine pruning targets,
each an independent run from the baseline: **441 combinations, 404 successful**.
The 37 failures are known criterion/architecture incompatibilities, listed in
`mono_tpu/01_prune.py`, not lost data.

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
`random`, plus `lamp`, `fisher`, `group_lasso` and `hrank`. The protocol follows
**PruningBench** (Li et al. 2024, arXiv:2406.12315) so results stay comparable
with that leaderboard: one uniform recipe for every architecture, deliberately
untuned per model.

`random` is not filler. It is the control that separates what a criterion
contributes from what merely removing capacity and retraining contributes.

### Axis 2 — fitting real models to N accelerators (`multi_tpu/`)

Eight ImageNet architectures (GoogLeNet, BN-Inception, Inception V3/V4,
Inception-ResNet-V2, ResNet-50/101/152), pruned from published weights to a
**size target** rather than to a grid, then compiled across N segments with
`edgetpu_compiler --num_segments N`:

| INT8 target | Segments | Multi-TPU configuration |
|---|---|---|
| 8 MB | 1 | 8 independent models in parallel |
| 16 MB | 2 | 4 pipelines of 2 accelerators |
| 32 MB | 4 | 2 pipelines of 4 |
| 64 MB | 8 | 1 pipeline across all 8 |

**Pruning to a parameter count does not make a model fit.** The compiler reports
consistently more bytes than the weight count implies, and the gap grows as the
model shrinks. Regressed over the guided loop's own iterations:

| Model | Slope (MiB per MiB of params) | Fixed overhead |
|---|---:|---:|
| ResNet-101 | 0.962 ± 0.013 | **2.29 ± 0.36 MiB** |
| Inception-V4 | 1.027 ± 0.011 | **2.86 ± 0.32 MiB** |
| Inception-ResNet-V2 | 0.970 ± 0.009 | **5.24 ± 0.28 MiB** |

The slope near 1.0 confirms the INT8 one-byte-per-weight assumption. It is the
**constant** the naive prediction ignores, and it does not shrink with pruning:
fused batch-norm constants, int32 biases and per-tensor alignment padding all
scale with the number of tensors and channels, not with the number of weights.
Architectures with many branches and 1x1 convolutions suffer most. For
Inception-ResNet-V2 at one segment, of roughly 6.1 MiB fitting in SRAM, 3.9 MiB
are overhead and only 2.2 MiB are weights, which is why it converges at 95 %
pruned rather than the 86 % arithmetic suggests.

`pipeline_full.py` therefore **measures instead of predicting**: prune, quantize,
compile, read the compiler's off-chip figure, and prune again by that much if
anything still streams. Convergence takes 2-3 iterations and lands 6 to 35 points
deeper than the naive prediction.

### Synthetic corpus (`synthetic/`)

400 configurations: 5 topology families x 4 depths x 5 widths x 4 resolutions,
sampling memory and transfer behaviour far more densely than 15 real models can.
Nothing is trained; these networks exist to be measured, not to classify.

---

## Quick start

```bash
./setup/setup_envs.sh                  # creates envs/{pytorch,coral,synth}-env
./setup/setup_envs.sh --edgetpu-compiler
```

Three environments, because they are mutually incompatible: `pycoral` ships
wheels for Python 3.6-3.9 only while `onnx2tf` 2.x needs 3.12, and installing
TensorFlow next to `onnx2tf` breaks the pins the PyTorch conversion path depends
on. `coral-env` deliberately has no torch: it is deployed to a Raspberry Pi.

### Axis 1, end to end

```bash
P=envs/pytorch-env/bin/python ; C=envs/coral-env/bin/python
$P mono_tpu/00_prepare_baselines.py --output_dir models --data_dir data
$P mono_tpu/01_prune.py --data_dir data --num_classes 100 \
      --checkpoints 10 20 30 40 50 60 70 80 90
$P mono_tpu/02_convert_tflite_int8.py --data_dir data --num_calib 100
$C mono_tpu/03_compile_edgetpu.py
$C mono_tpu/04_benchmark.py --device both --warmup 30 --runs 200 --num_images 0
$C mono_tpu/05_benchmark_coldstart.py --device both --passes 30
$P mono_tpu/aggregate_pruning_logs.py
```

### Axis 2, one model at one target

```bash
$P multi_tpu/00_fetch_and_convert_pretrained.py --models resnet50
$P multi_tpu/pipeline_full.py --model resnet50 --target_mb 16 \
      --importance taylor --data_dir /datasets/Imagenet_1k \
      --epochs_from_actual --run_tag N2
```

### Synthetic corpus

```bash
S=envs/synth-env/bin/python
$S synthetic/generate_sweep.py --workers 2 --num_calib 100 --skip-existing
$C synthetic/compile_sweep.py --tflite-dir outputs/tflite --out-root outputs_pipeline
$C multi_tpu/bench/bench_pipeline.py --phase all
$C multi_tpu/bench/bench_parallel.py --mode both --orchestrate --resume
```

Run the parallel benchmark **after** the pipeline phases, never alongside: two
processes competing for the accelerators crash each other.

---

## Pretrained models and measured artefacts

Every model measured here is published separately on the Hugging Face Hub,
because the collection is about 37 GB and exceeds what a git repository should
carry. See `docs/MODELS.md` for the layout and a download script.

---

# Benchmark metrics

Every benchmark writes a CSV. This section documents each column.

Two conventions hold throughout:

- **A missing measurement is written as an empty field, never as zero.** A zero
  is a measured value; an empty field means the measurement was not made.
- **Signed deltas are negative for a loss.** `quant_drop_top1 = -1.2` means
  quantization cost 1.2 points of top-1.

---

## `benchmark_results.csv` — steady-state, single TPU

Produced by `mono_tpu/04_benchmark.py`. One row per model. Cumulative: running
with `--device cpu` and later `--device tpu` fills in the same file, and the
derived metrics are recomputed each pass from whatever is present.

### Identity

| Column | Meaning |
|---|---|
| `model` | full model name, e.g. `resnet18_pruned50pct_taylor` |
| `tag` | `finetuned` (the baseline) or `pruned` |
| `importance` | pruning criterion; empty for a baseline |
| `prune_pct` | **requested** target, in % of parameters |

> **Use `param_reduction_pct`, not `prune_pct`, on every axis and in every
> correlation.** Global pruning removes whole dependency groups of varying size,
> so the achieved rate is never exactly the target. Plotting against the target
> silently misplaces every point.

### Size and cost

| Column | Meaning |
|---|---|
| `size_int8_mib` | INT8 `.tflite` file size |
| `params_int8` | parameter count in the quantized model |
| `macs_int8` | estimated multiply-accumulate operations per inference |

### Memory regime (from the compiler report)

| Column | Meaning |
|---|---|
| `tpu_on_chip_mib` | parameters cached in SRAM |
| `tpu_off_chip_mib` | parameters streamed **on every inference** |
| `tpu_streaming_ratio` | off-chip / total |
| `tpu_sram_util_pct` | SRAM occupancy, against the ~8 MB budget |
| `tpu_subgraphs` | Edge TPU subgraph count |
| `tpu_ops_count`, `cpu_ops_count` | operations mapped to each device |
| `tpu_ops_coverage_pct` | share of operations running on the accelerator |

`tpu_off_chip_mib == 0` is the decisive property. It separates the two regimes
and explains most of the variance in `tpu_realization_efficiency` below. A few
CPU operations at the tail (softmax, reshape) are normal; a large `cpu_ops_count`
means part of the graph fell back to the host and the latency is not measuring
what you think.

### Accuracy

| Column | Meaning |
|---|---|
| `top1_cpu_f32_pct`, `top5_cpu_f32_pct` | FP32 reference |
| `top1_int8_pct`, `top5_int8_pct` | after quantization, measured **on the TPU** |
| `*_ci95_lo`, `*_ci95_hi` | bootstrap 95 % confidence bounds |
| `n_eval_acc` | images evaluated (10 000 for the full test set) |

INT8 accuracy comes from the Edge TPU, not from the INT8 CPU path, for two
reasons: the compiler's kernels are not bit-identical to the CPU ones, so the
target hardware's number is the honest one; and the Coral stack's
`tflite_runtime` 2.5 misreads models produced by `ai-edge-quantizer`, collapsing
their output onto the zero point and reporting chance-level accuracy **without
raising**.

The confidence intervals are what make a difference interpretable. Without them,
a 0.3-point gap between two criteria cannot be distinguished from evaluation
noise.

### Latency and throughput

| Column | Meaning |
|---|---|
| `lat_{cpu,tpu}_{f32,int8}_ms_{mean,std,median,p95,p99}` | steady-state latency |
| `lat_*_cv` | coefficient of variation, std / mean |
| `throughput_{cpu_f32,cpu_int8,tpu_int8}_fps` | inferences per second |

**Steady state** means: warmup inferences discarded, then N inferences timed on
one interpreter that stays alive. Model loading and the first-inference weight
transfer are excluded, and only `invoke()` is inside the timed region; input
preparation is host-side work and stays outside it.

Mean and median are both kept because they disagree when the distribution is
skewed by scheduling noise; p95 and p99 say whether that skew is a tail or a
shift. Cold-start costs are measured separately, see below.

### Derived: accuracy deltas

| Column | Definition | Isolates |
|---|---|---|
| `quant_drop_top{1,5}` | INT8 − FP32, same weights | the cost of quantization alone |
| `prune_drop_top{1,5}_f32` | pruned − baseline, both FP32 | the cost of pruning alone |
| `prune_drop_top{1,5}_i8` | pruned − baseline, both INT8 | pruning, after quantization |
| `combined_drop_top{1,5}` | pruned INT8 − baseline FP32 | what a user actually gives up |

`quant_drop_top1` is only interpretable because quantization is post-training.
With quantization-aware training the weights would adapt, and the number would
measure the training rather than the model. As it stands it measures how
quantization-friendly each criterion's output is, which is a property of the
mask.

### Derived: speedups

| Column | Definition |
|---|---|
| `tpu_speedup_int8` | CPU INT8 latency / TPU INT8 latency |
| `quant_speedup_cpu` | CPU FP32 / CPU INT8 |
| `prune_speedup_{cpu,tpu}` | baseline latency / pruned latency |
| `theoretical_speedup_macs` | baseline MACs / pruned MACs |
| **`tpu_realization_efficiency`** | `prune_speedup_tpu / theoretical_speedup_macs` |

`tpu_realization_efficiency` is the central quantity of the whole study: **the
fraction of the arithmetic saving that the hardware actually delivers.** A value
near 1 means the model was compute-bound and pruning paid off as predicted. Well
below 1 means it is limited by weight transfer, and that is exactly where two
criteria at equal accuracy stop being interchangeable.

### Derived: compression

`param_reduction_pct`, `size_reduction_pct`, `macs_reduction_pct`,
`compression_ratio` — all against the baseline of the same architecture.

---

## `cold_start_results.csv` — first-inference cost, single TPU

Produced by `mono_tpu/05_benchmark_coldstart.py`. One row per model, with the
block below repeated for each of `cpu_int8`, `cpu_f32` and `tpu_int8`:

| Column | Meaning |
|---|---|
| `<mode>_n_passes` | passes recorded for this mode |
| `<mode>_cold_{mean,std,median,p95,p99,min,max,cv,n}` | position 1: the genuinely cold inference |
| `<mode>_steady_{...}` | positions 6-10 pooled: after warm-up |

**`cold_mean − steady_mean` is the quantity of interest.** It is the weight
transfer, and combined with the model's internal and external volumes it yields
the transfer coefficient `k` in the model at the top of this README.

Protocol: one pass is N inferences on a **fresh** interpreter, then the next
model. Model order is reshuffled between passes, which is not cosmetic: without
it, any effect drifting over the run (thermal throttling above all) is absorbed
into the per-model result, making the models measured last look uniformly slower
with no trace of the bias in the output.

---

## `coldstart_axis1.csv`, `coldstart_synth.csv` — raw first-inference series

Produced by `multi_tpu/bench/bench_coldstart_{pcie,usb}.py`. Long format, one row
per measurement:

| Column | Meaning |
|---|---|
| `pass` | pass index |
| `tag` | model |
| `position` | 1 = cold, 2..N = the warm-up climb |
| `lat_ms` | latency |
| `timestamp` | when it was taken |

Raw rather than summarised, because the fit that extracts `k`, `c` and the
overlap term needs the individual points.

**Fitting note.** A one-variable fit `d = c + k·I` returns a **negative**
intercept (−1.29 ± 0.36), which is physically impossible. The missing term is
overlap: the residual correlates at −0.66 with the operation count. The
two-variable form `d = c + k·I − β·C` gives `k = 2.668 ± 0.072`,
`β = 0.580 ± 0.064` and an intercept indistinguishable from zero, R² = 0.924.

Confirmed on a second corpus: the 241 single-TPU-axis models that fit in SRAM
give `β = 0.631 ± 0.040`, within 0.7 σ, and `k = 2.800 ± 0.006`, R² = 0.9995.
Overlap is a property of the platform, not of the corpus.

---

## `N1_baseline.csv`, `pipeline_canonical.csv`, `pipeline_spread.csv` — pipelining

Produced by `multi_tpu/bench/bench_pipeline.py`, phases A, B and C. One schema:

| Column | Meaning |
|---|---|
| `tag` | model |
| `N` | pipeline stages (accelerators) |
| `perm` | TPU order, comma-separated |
| `kind` | `baseline` (N=1), `canonical` (order 0..N−1), `random` |
| `perm_idx` | permutation index within phase C |
| `throughput_fps` | inferences per second |
| `lat_ms_{mean,std,median,p95,p99,min,max}` | latency |
| `cold_first_lat_ms` | first inference of the series |
| `warmup`, `reps` | measurement parameters |

**The latency columns of these files are not trustworthy; the throughput columns
are.** pycoral's `PipelinedModelRunner` inflates per-item latency badly (50 ms to
17 s for models whose real inference is under 5 ms) because it is built for
throughput and queues work accordingly. Correct pipeline latency comes from
`bench_latency_chained.py`, which bypasses the runner and chains the segments by
hand.

**On `perm`: the TPU ordering has no effect.** This was initially reported as an
asymmetry and is **refuted**. Interleaving 990 fixed-order measurements with 990
random-order ones in one session gave F = 0.905, p = 0.94 on the variances and
p = 0.45 on the means:

| | mean | σ | range |
|---|---:|---:|---:|
| fixed order | 112.25 fps | 1.59 % | 11.26 % |
| random orders | 112.19 fps | 1.51 % | 12.40 % |

Repeating one assignment 990 times disperses as much as using 990 different ones.
There is no optimal assignment to search for. What the number does measure is
**repeatability: σ ≈ 1.6 % of throughput per measurement**, so two configurations
measured once each are only distinguishable beyond about 2 %. Repeatability
varies between sessions (0.64 % one day, 1.55 % another), so only compare
measurements taken in the same session.

The original error is worth recording: it compared a **range** (12.45 % over 990
draws) against a **standard deviation** (0.63 %), and estimated a within-group
variance on 9 and 15 degrees of freedom. Never do either.

---

## `parallel_bench.csv` — N copies on N accelerators

Produced by `multi_tpu/bench/bench_parallel.py`. Long format, one row per
`(mode, tag, n_tpu, tpu_idx, rep)`:

| Column | Meaning |
|---|---|
| `mode` | `steady` or `cold` |
| `tag`, `n_tpu`, `tpu_idx`, `rep` | model, accelerator count, which one, repetition |
| `latency_ms` | that inference |
| `family`, `depth`, `base_width`, `resolution`, `num_params` | structure |
| `tflite_size_mb`, `edgetpu_size_mb` | sizes |
| `on_chip_mb`, `off_chip_mb` | memory regime |
| `num_ops_tpu`, `num_ops_cpu_fallback`, `num_subgraphs` | compiler output |

This is the **parallel** regime: N independent copies of one model, each on its
own accelerator and its own inference stream. Contrast with pipelining, which
splits one model across N accelerators. Together they answer when to pipeline and
when to parallelise: pipelining buys throughput at the cost of latency and is the
only option when a model does not fit one accelerator; parallelism buys
throughput more cheaply but requires each copy to fit alone.

The derived quantity is **slowdown against the single-accelerator baseline**: 1.0
is perfect scaling, above 1.0 is contention.

At 8 accelerators, median cold slowdown is **1.60x** and median steady-state
slowdown is **1.00x**. Contention is real while weights move and essentially
absent once they are resident.

A counter-intuitive result from the same data: cold slowdown is **inversely**
correlated with model size. Models under 2 MB show 15-25x; models over 100 MB
show 1.1-1.3x. The fixed per-inference overhead dominates on small models and is
contended for, while on large models the per-device transfer dominates and scales
cleanly.

`mode` matters. In `steady`, a barrier before **each** timed repetition forces all
accelerators to start together, making it a worst-case contention measurement;
without the barrier the copies drift apart and stop competing. In `cold`, a fresh
interpreter per repetition forces SRAM re-initialisation so each repetition
genuinely pays the transfer.

---

## Compiler reports (JSON)

`compiler_metrics.json` (axis 1) and `<basename>_compile_report.json` (axis 2)
hold the per-segment parse of `edgetpu_compiler` output: `on_chip_used_mb`,
`off_chip_used_mb`, `ops_edgetpu`, `ops_cpu`, `num_subgraphs`, `compile_ms`, plus
totals.

**`off_chip` is not monotonic in N.** A ResNet-101 checkpoint that fits at 6
segments overflows by 2.02 MiB at 7 and fits again at 8. The segmentation
heuristic balances on some other criterion and does not optimise for fitting, so
"more segments can never be worse" is false, and a failure at N does not rule out
N+1.

---

## Reproducibility

Seed 42 throughout, propagated to `random`, `numpy` and `torch` (CPU and CUDA),
and re-seeded at the start of every run so a run gives the same result alone or
inside a sweep. Calibration draws use seed 42; benchmark inputs use seed 123, and
verification uses 123 as well, kept distinct so that quantization is never
evaluated on the images that calibrated it.

Three layers of variability are separable: bootstrap confidence intervals on
accuracy (no retraining needed), the calibration draw (re-quantize at another
`--calib_seed`), and training seeds (43, 44 for multi-seed runs).

---

## Testing

`docs/TESTING.md` records what has been verified and what has not, and
`docs/run_smoke.sh` re-runs the static and import checks over every script:

```bash
bash docs/run_smoke.sh          # 32 checks, seconds
```

## Known pitfalls

Collected in `docs/PITFALLS.md`. The four that cost the most time:

1. **`tflite_runtime` 2.5 silently saturates `ai-edge-quantizer` models.** Output
   collapses onto the zero point, every prediction lands on the same class, top-1
   reads as chance. No exception is raised. Use `ai_edge_litert`.
2. **`edgetpu_compiler --num_segments N` segfaults on TFLite produced through the
   MLIR path** (`TFLiteConverter`, or `onnx2tf` 1.x). Only `onnx2tf` 2.x, whose
   `flatbuffer_direct` backend leaves a different description in the file, works.
   Diagnosed under gdb as an out-of-range flatbuffer vtable lookup; see
   `synthetic/build_one.py`.
3. **The apex driver aborts above ~1.5 GB of simultaneous mappings**, at the C
   level, which SIGKILLs the Python process and cannot be caught. Every
   measurement runs in a subprocess for this reason alone.
4. **`/dev/apex_*` reverts to root ownership on reboot.** Symptom:
   `list_edge_tpus()` sees all eight devices but `make_interpreter()` fails to
   load the delegate. Fix: `sudo udevadm trigger --subsystem-match=apex`.

---

## References

- Li et al., *PruningBench: A Comprehensive Benchmark of Structural Pruning*,
  arXiv:2406.12315 (2024) — the protocol axis 1 follows.
- Fang et al., *DepGraph: Towards Any Structural Pruning*, CVPR 2023 — the
  Torch-Pruning library used throughout.
- Gao, Choi and Wang, *Work-in-Progress: Modeling Inference Latency on Edge TPUs*,
  RTSS 2025 — independent measurement of Coral USB transfer bandwidth.
- Li et al., *Pruning Filters for Efficient ConvNets*, ICLR 2017 — the
  fine-tuning recipe of axis 2.
- Renda et al., *Comparing Rewinding and Fine-tuning in Neural Network Pruning*,
  ICLR 2020.

## Licence

MIT for the code. The models published on the Hugging Face Hub derive from
CIFAR-100 and ImageNet-1k and from architectures whose original licences apply.

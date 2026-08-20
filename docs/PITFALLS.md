# Pitfalls

Failure modes encountered while building this framework. The ones marked
**silent** are the expensive ones: they produce plausible-looking results rather
than errors, and can go unnoticed for days.

---

## Toolchain

### 1. `tflite_runtime` 2.5 saturates `ai-edge-quantizer` models — **silent**

The Coral stack ships `tflite_runtime` 2.5, frozen since 2021. It misreads the
quantization that `ai-edge-quantizer` produces: every output collapses onto the
zero point, so every prediction lands on the same class and top-1 comes out at
chance level. **No exception is raised.**

Symptom: top-1 around 1 % on 100 classes, or a constant predicted class index,
or a cosine similarity of NaN.

Fix: install `ai_edge_litert` (>= 2.1.0) and use
`ai_edge_litert.interpreter.Interpreter`. It also supports
`load_delegate("libedgetpu.so.1")`, so nothing else changes.

This is why `04_benchmark.py` takes INT8 accuracy from the Edge TPU path only,
never from the INT8 CPU path.

### 2. `edgetpu_compiler --num_segments N` segfaults on MLIR-produced TFLite

Every model converted through `TFLiteConverter`, or through `onnx2tf` 1.x,
compiles fine at N=1 and **segfaults for every N >= 2**. Models converted with
`onnx2tf` 2.x do not.

Diagnosed under gdb on the stripped compiler binary:

```
=> 0x7ffff7e9220e: mov (%rdi,%rax,4),%eax   # SEGV
   rax read from (%r10) just before: a flatbuffer vtable offset pointing
   outside the valid region
```

Six fixes were tried and none worked: downgrading operator versions,
metadata/signature surgery, injecting QUANTIZE operations via float I/O, a full
detour through ONNX with `onnx2tf` 1.x, matching the ResNet-50 head, matching its
bottleneck block.

What distinguishes a working model is the conversion backend, visible in the
tflite description field:

| Description | Produced by | `--num_segments 2` |
|---|---|---|
| `MLIR Converted.` | `TFLiteConverter`, `onnx2tf` 1.x | **segfault** |
| `onnx2tf flatbuffer_direct` | `onnx2tf` 2.x | works |

Both flatbuffers are structurally valid. Short of reverse engineering a stripped
binary, the exact field responsible is not knowable, and the detour works, so the
investigation stopped there.

`synthetic/build_one.py` asserts on the description rather than on the package
version, so a wrong environment fails at conversion instead of hours later.

### 3. `onnx2tf` emits no SignatureDef

`ai-edge-quantizer`'s `calibrate()` then refuses to run with "Invalid
signature_key". Inject one by hand with `flatbuffer_utils` before quantizing;
see step 4 of `synthetic/build_one.py`.

### 4. `ai-edge-quantizer` puts per-axis quantization on `BATCH_MATMUL`

The compiler rejects it: `'dwl.matrix_multiply' op per axis quantization is not
supported`. Force `TENSORWISE` granularity for that operator.

Note that `onnx2tf` converts a Keras `Dense` into **`BATCH_MATMUL`** (opcode 126),
**not** into `FULLY_CONNECTED` (opcode 9). Configuring `FULLY_CONNECTED` alone has
no effect at all, which is an easy hour to lose.

### 5. `onnx2tf`'s `download_test_image_data()` crashes under numpy 2

It calls `np.load` without `allow_pickle`. Either patch `np.load`
(`synthetic/onnx2tf_wrapper.py`) or place a dummy
`calibration_image_sample_data_20x128x128x3_float32.npy` in the working
directory. `build_one.py` generates it on demand.

### 6. The PyTorch 2.9 ONNX exporter emits an operator the compiler rejects

The newer exporter produces `_native_batch_norm_legit_no_training`, which
`edgetpu_compiler` rejects with "non-broadcastable operands" once the graph has
been moved to NHWC. Use the legacy TorchScript exporter (`dynamo=False`), which
folds batch norm into a standard ONNX `BatchNormalization`.

### 7. The compiler's timeout flag is `--timeout_sec`, with an underscore

With a hyphen the binary rejects the option, in a way easily missed in a batch
log.

### 8. `onnx2tf` and `apply_patches.py`

Two upstream source files need patching before they emit Edge TPU-compatible
models: `onnx2tf`'s `pool.py` (PADV2 to PAD) and `ai_edge_quantizer`'s rank
validation. Without them, googlenet and squeezenet are rejected outright. Run
`setup/apply_patches.py` with the target environment's interpreter.

---

## Hardware and drivers

### 9. The apex driver aborts above ~1.5 GB of simultaneous mappings

Beyond roughly that, the kernel driver fails with
`Could not map pages : 137 (Cannot allocate memory)` and then
`F driver/mmio_driver.cc:119] Non-OK-status: Inconsistent parameter mapping`.

That is a **C-level abort**: it SIGKILLs the whole Python process and **cannot be
caught by try/except**. The only defences are subprocess isolation and a
predictive guard (`--max-total-map-mb 1500`).

Empirical: 220 MB across 4 accelerators is fine, 220 MB across 8 is not.

### 10. `/dev/apex_*` reverts to root ownership on reboot

Symptom: `list_edge_tpus()` reports all eight devices but `make_interpreter()`
fails with "Failed to load delegate from libedgetpu.so.1".

Fix: `sudo udevadm trigger --subsystem-match=apex`, after every reboot unless a
permanent udev rule is added.

### 11. Coral USB re-enumerates after its first inference

Before the firmware loads it appears as `1a6e:089a Global Unichip` on the
480 Mb/s bus; afterwards as `18d1:9302 Google Inc.` at 5000 Mb/s. Do not conclude
the port is USB 2.0 without running an inference first.

### 12. Coral USB `load_delegate` fails intermittently

Roughly one attempt in five when interpreters are created back to back: the
device needs a moment between them. Retry with increasing waits, before the timed
region. Without this, a cold-start campaign is unusable.

### 13. Interpreter creation costs 2.7 s on Coral USB

Against about 0.06 s over PCIe, which makes it **97 % of a cold-start campaign's
runtime**. Budget in passes, not in models: 30 passes over 410 models is 9.4 h,
and 10 passes are usually enough, since the dispersion that matters is between
models rather than between repetitions.

That 2.7 s is 2.68 s on x86 and 2.73 s on a Pi 4, a 2 % difference, which places
the cost on the accelerator rather than on the host.

### 14. Two benchmark processes crash each other

They contend for the accelerators. Check with `pgrep -af bench_` before
launching, and run the parallel benchmark only after the pipeline phases.

---

## Measurement methodology

### 15. Never compare a range against a standard deviation

This produced a wrong published-adjacent claim. A 12.45 % **range** over 990
draws was compared against a 0.63 % **standard deviation** and read as an effect.
There was none: F = 0.905, p = 0.94.

Related: never estimate a within-group variance on fewer than about 20 degrees of
freedom. Estimates on 9 and 15 suggested an effect at p = 0.002 that does not
exist.

### 16. pycoral's `PipelinedModelRunner` inflates latency — **silent**

It reports 50 ms to 17 s for models whose real inference is under 5 ms. It is
built for throughput and queues work accordingly, so its per-item timing is not a
latency. Its **throughput** figures are fine.

Correct pipeline latency needs manual segment chaining; see
`multi_tpu/bench/bench_latency_chained.py`.

### 17. Fixing a training budget from a predicted pruning rate — **silent**

The guided loop converges 6 to 35 points deeper than the "bytes ~ parameters"
prediction. Budgets fixed in advance from the predicted rate under-trained
several runs, and the shortfall **correlated with architecture**, which biased a
comparison between model families in a way that was invisible in the outputs.

Always derive the budget from the achieved rate, after convergence. See
`FT_BUDGET_BANDS` and step [3b] in `multi_tpu/pipeline_full.py`.

### 18. Shuffle model order between passes

Any effect drifting over a run, thermal throttling above all, is otherwise
absorbed into the per-model result: models measured last look uniformly slower
and nothing in the output reveals it.

### 19. `off_chip` is not monotonic in N

A ResNet-101 checkpoint fits at 6 segments, overflows by 2.02 MiB at 7, and fits
again at 8. The segmentation heuristic does not optimise for fitting, so a
failure at N does not rule out N+1.

### 20. Recompute data-driven importance at every pruning step

`torch_pruning` rebases channel indices after each removal. Scores cached once
are then read through indices that no longer denote the same channels, and the
pruning becomes effectively random. Observed on HRank: 1.00 % validation accuracy
with a one-shot cache (chance level on 100 classes) against 49.84 % with
recomputation.

### 21. Use the achieved pruning rate, not the target

Global pruning removes whole dependency groups of varying size, so the achieved
rate is never exactly the requested one. Plot against `param_reduction_pct`;
plotting against `prune_pct` silently misplaces every point.

---

## Infrastructure

### 22. Orphaned temporary directories after an OOM kill

`build_one.py` cleans its scratch directory in a `finally` block, which does not
run under SIGKILL. Each orphan is around 6 GB; 27 of them filled a disk to 97 %
within hours. `generate_sweep.py` cleans `/tmp/regen_*` at start-up and refuses
to start a new configuration below `--min_disk_gb`.

### 23. `tf2onnx` is disproportionately memory-hungry on wide, shallow models

It OOMs on models above roughly 100 M parameters. Empirical thresholds: 80 M with
2 workers on a 15 GB host, 100 M with 2 workers on 31 GB, 145 M with 1 worker.
A model with few but very wide blocks OOMs where a deeper one of the same size
does not.

### 24. SLURM defaults are too small

`--mem=24G` at minimum: the 2 GB default makes `onnx2tf` die with a bus error on
resnet50 and larger. Several jobs sharing one `--job-name` queue behind each
other rather than running in parallel.

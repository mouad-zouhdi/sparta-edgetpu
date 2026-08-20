# Testing

What has been verified about this code, and how to re-run it.

The point of this file is to be honest about coverage: which claims are backed by
an execution, which by a static check, and which are not covered at all.

---

## 1. Static and import checks (34 files, seconds)

Every script must parse, import in its target environment, and answer `--help`.

```bash
bash docs/run_smoke.sh
```

This catches the failure mode that matters most when porting code between
machines: a module that resolves at install time but cannot load, or an import
left pointing at a file that has since been renamed. It found two real problems
here, a stale module name and an entry point that upstream had renamed.

Two files are checked differently, for reasons that are properties of the code
rather than gaps:

- `synthetic/src/*.py` use relative imports and are checked as a package, not
  loaded as standalone files.
- `multi_tpu/bench/bench_parallel.py` needs `pycoral` and `pandas` in one
  interpreter. It is parsed rather than imported unless such an environment
  exists; `requirements/coral-env.txt` declares both.

## 2. Unit-level checks on the parts that carry logic

Run against real inputs, not mocks:

| What | Checked against |
|---|---|
| Compiler stdout parser | real `edgetpu_compiler` output, including the unit switch from `0.00B` to MiB |
| Segment report parser | a real multi-segment compile, verifying per-segment and total figures |
| Cold-start statistics | a synthetic matrix with a known cold/steady split |
| Fine-tuning budget bands | every band boundary, plus values between them |
| Segment-count inference | each size target, and one that is not in the table |
| Architecture builders | all four CIFAR architectures, checking output shape and parameter count against the published figures (11.2 M, 23.7 M, 20.1 M, 36.5 M) |
| Learning-rate schedule | full-length and shortened runs, confirming milestones survive the filter |
| Model zoo | all eight ImageNet architectures built with pretrained weights, forward pass, and a check that the classifier was not silently replaced |

## 3. End-to-end: the synthetic generator

The full conversion chain, run on real configurations:

```
Keras -> SavedModel -> tf2onnx -> onnx2tf flatbuffer_direct
      -> SignatureDef injection -> ai-edge-quantizer -> INT8 TFLite
      -> edgetpu_compiler --num_segments {1,2,4}
```

Verified on one configuration per family, plus variations in depth, width and
resolution, plus one configuration deliberately too large, which must fail
gracefully and leave a metadata record rather than taking the sweep down.

The decisive assertion is the multi-segment compile. `--num_segments 2` and `4`
succeed here, where a model converted through the MLIR path segfaults the
compiler. That is the workaround this repository exists to record, and it is
checked rather than asserted: the produced file's description field is verified
to read `onnx2tf flatbuffer_direct`, and the compiles are actually run.

## 4. End-to-end: axis 1, on a GPU cluster

`00 -> 01 -> aggregate -> 02`, at reduced epoch counts:

- two baselines trained, one CIFAR-native and one torchvision architecture
  adapted to 32x32, since those follow different code paths;
- **all ten importance criteria** exercised on one architecture;
- the two **known-incompatible** pairs run deliberately, to confirm they are
  handled rather than crashing: `bn_scale` on an architecture without BatchNorm
  must skip, and `obdc` on Fire modules must fail gracefully. A run that crashed
  here would take a whole sweep with it;
- three pruning targets, which exercises the gap between the requested and the
  achieved rate;
- log aggregation, then conversion of everything produced.

## 5. End-to-end: axis 2, on a GPU cluster

- every model-zoo loader with real pretrained weights;
- `--prune_only`, producing a PREFT checkpoint and its metadata sidecar;
- `--ft_only --resume_from`, the resume path the guided loop depends on;
- conversion of the result, calibrated on ImageNet.

---

## What is NOT covered

**Anything requiring the accelerators.** The benchmark scripts
(`04_benchmark.py`, `05_benchmark_coldstart.py`, everything under
`multi_tpu/bench/`) are checked statically and by unit tests of their statistics,
but a full run needs a Coral device or the 8x PCIe card. Their measurement logic
has been exercised in production over the campaigns whose results are published,
which is stronger evidence than a test would give, but it is not reproducible on
a machine without the hardware.

**Full-length training.** Everything is run at reduced epoch counts. The recipes
themselves are the published ones and were exercised at full length during the
campaigns; what the tests verify is that the code paths execute, not that the
accuracies reproduce.

**The cluster job scripts.** The SLURM wrappers used for the original campaigns
are not part of this repository: they encode one site's partitions, paths and
quotas, and would mislead more than help.

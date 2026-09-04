# The partition grid

The campaign the multi-accelerator results rest on. Eight accelerators are split
between independent instances, an instance of N stages runs the binary compiled
into N segments, and there are exactly p(8) = 22 such partitions, from one
eight-stage pipeline to eight independent copies. Crossed with 43 checkpoints
that is 946 configurations.

Its measurements are published on the Hub, under
`axis2_imagenet/measurements/`, whose README documents the protocol and every
column.

**These scripts carry their comments in French**, unlike the rest of this
repository. They were written during the campaign and are published verbatim,
as the record of what ran, rather than rewritten afterwards. The tables below
say in English what each one does.

## Measurement

| Script | What it does |
|---|---|
| `bench_partitions_fixedn.py` | the final campaign: every partition of eight, a **fixed inference count** per configuration (1000 latency, 1000 throughput, 200 first inference). Stages are chained by hand with threads rather than through pycoral's `PipelinedModelRunner`, which hides per-stage times and inflates latency |
| `bench_partitions.py` | the earlier campaign, same grid but **fixed-duration** windows (4.5 s and 6.5 s) and ten passes. It is what gives the error bars |
| `bench_scaling.py` | pipeline against parallelism over non-saturating shapes, where accelerators are left idle |
| `bench_accuracy_v2.py` | INT8 top-1 and top-5 on ImageNet validation, one accelerator, one point per checkpoint |

Two traps the fixed-count rule creates, both handled in
`bench_partitions_fixedn.py`. Every instance keeps running until the slowest has
finished, so that no measurement ends on a half-idle card, in a contention
regime that is not the one being measured; and only the first thousand
inferences of each are kept. The gain over fixed windows is in the tails, not in
the means: repeating passes remains the only honest way to get an error bar.

## Aggregation

| Script | What it does |
|---|---|
| `aggregate_partitions_fixedn.py` | raw rows to the two aggregates, one per (checkpoint, partition) and one per instance |
| `aggregate_partitions.py` | the same for the ten-pass campaign, confidence intervals included |
| `extract_segment_metadata_all.py` | sizes and memory split per (checkpoint, N, segment), as the compiler reports them. `on_chip_mb` is loaded once and kept, `off_chip_mb` is retransmitted on every inference: the two terms of the latency model |
| `merge_accuracy_5000.py` | carries a re-measured accuracy into the synthesis file, touching only the three accuracy columns |

## Catalogues

| Script | What it does |
|---|---|
| `multi_tpu_corpus.py` | the 43 checkpoints behind one key and one path resolution. The three campaigns each had their own binary root, segment-directory naming and identifier |
| `wave_configs_v2.py`, `sweepN_configs.py` | the checkpoint lists of the wave and of the N = 1..8 sweep |

⚠️ **43 entries, 42 distinct names.** `resnet101_pruned88pct_taylor` exists in
both the wave and the sweep and denotes **two different trainings**: 75 epochs
budgeted on the predicted rate in one, 90 on the achieved rate in the other. That
is why every key carries its corpus and the two roots stay separate. Do not merge
them.

⚠️ **A single pass has no error bar.** In the fixed-count files, `*_std` and the
percentiles describe the spread across the thousand inferences of one
measurement, not the uncertainty on their mean. The two differ by a factor of
eleven: the standard error of a mean over a thousand inferences is 0.018 %, while
the spread of that mean between sessions is 0.197 %, measured over the ten-pass
campaign. Dividing a `*_std` by the square root of a thousand would claim a
precision eleven times better than reality.

## Paths

These scripts carry absolute paths from the machine the campaign ran on, and the
model roots they name are published on the Hub under `axis2_imagenet/edgetpu/`.
Running them again needs eight Coral accelerators; reading their output does not.

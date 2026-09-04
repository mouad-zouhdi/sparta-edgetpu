# Paper and talks

## `DSD.pdf`

*Pruning Deep Neural Networks for Edge TPU Deployment*, submitted to DSD 2026.
It covers the same measurements as the report, in four pages and in English, and
its section V on scheduling is the one part with no counterpart in the report.

⚠️ **Its Table I differs from the report's tables**, and deliberately. The
memory columns there are computed from parameter counts; the report's are read
from the compiler. Where the two disagree, the compiler is right: the gap
between them is the fixed per-architecture overhead the paper's own analysis is
about.

`figures/build_resnet50_page.py` writes the self-contained TikZ fragment for the
paper's five-panel ResNet-50 figure, to be `\input{}` inside an IEEEtran
`figure*`. It has nothing to do with the report's figures, which have their own
generators under `report/generators/`.

## `slides_dsd2026/`

The conference talk, beamer sources and compiled PDF. `bench/` holds the CSVs
its plots read, so it rebuilds on its own.

## `slides_defence/`

The internship defence, in French. Sources and images; no PDF was kept.

## Two pgfplots traps, learned building these

1. `at={([yshift=-Xcm]p2.outer south)}` is **silently ignored**. Write
   `at=(p2.outer south), anchor=outer north, yshift=-Xcm`.
2. **A blank line inside an `axis`'s options breaks the parser**, with an error
   that names something else entirely. A generator that emits an empty string
   conditionally will hit this.

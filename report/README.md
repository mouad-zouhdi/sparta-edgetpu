# Internship report

The report this work was written up in, in French, with everything needed to
rebuild it: `rapport.pdf` is the compiled document, `rapport.tex` its single
source, and the rest is what LaTeX needs to turn one into the other.

```bash
./build.sh          # full build, latexmk + biber, 3 to 5 minutes
./build.sh fast     # one pass, cross-references possibly stale
./build.sh watch    # rebuild on every save
./build.sh clean    # remove intermediates, and the PDF with them
```

`INSTALL.md` lists the TeX Live packages. The build was checked from this
directory on TeX Live 2023: 70 pages, no error, no unresolved cross-reference.

## What is where

| Path | What it is |
|---|---|
| `rapport.tex` | the whole document, one file |
| `figs/*.tex` | the figures, as self-contained pgfplots fragments |
| `figs/data/*.csv` | the numbers those figures plot |
| `figs/tables/*.tex`, `plots/` | the appendix tables and the curve data they read |
| `generators/*.py` | the scripts that wrote `figs/` from the measurements |
| `*.png`, `*.jpg` | photographs and diagrams, included as-is |

`figs/` is committed already generated, so the report builds without running
anything in `generators/`. Run those only to change a figure.

## Rebuilding a figure

| Script | Writes | Figures |
|---|---|---|
| `make_panels.py` | `panel_{speedup,accuracy,post_prune,pareto}.tex` | 5.1, 5.2, 5.3, 5.5 |
| `make_ft_curves.py` | `ft_a1_<model>_corps.tex`, `ft_a2_sweep.tex` | 5.4, 5.9 |
| `make_pipepar.py` | `panel_pipe_par.tex` | 5.6 |
| `make_pareto_acc.py` | `panel_pareto_acc.tex` | 5.7 |
| `make_accloss.py` | `panel_precision.tex` | 5.8 |
| `make_annexe_tables.py` | `figs/tables/*.tex` | appendix A |
| `make_annexe_multitpu.py` | `annexe_multitpu.tex` | appendix B |

`figs/single_latency_ntpu.tex` (figure 5.10) and `figs/annexe_tables.tex` have no
generator; they are written by hand.

**The generators carry absolute paths from the machine the research ran on.**
They are published as the record of what produced each figure, not as a portable
tool. Every input they name is published, and this is where it now lives:

| Path in the scripts | Where it is now |
|---|---|
| `mesures_brut/benchmark_results.csv` | `measurements/benchmark_results.csv` on the Hub |
| `second/pruning_logs/` | `axis1_cifar100/logs/` on the Hub |
| `multi_tpu/models/wave_bench/results_raw/` | `axis2_imagenet/measurements/` on the Hub |
| `work/final_tables_and_plots/` | `plots/` and `figs/tables/` here |
| `final_plots/pipeline_vs_parallel/data.csv` | `plots/pipeline_vs_parallel/data.csv` here |

The Hub collection is
[mouad-zouhdi/sparta-edgetpu-models](https://huggingface.co/mouad-zouhdi/sparta-edgetpu-models).

## Two traps this document sets

1. **The images are the real files here, and were symbolic links in the working
   copy.** Writing a `.png` over one of them from a plotting script destroyed the
   original twice during the work. Delete the file before writing to its name.
2. **Do not reintroduce `\counterwithout{figure}{section}`.** It strips the
   chapter prefix from `\thefigure` without stopping the counter from resetting,
   which produces several figures numbered 1 and ambiguous cross-references.

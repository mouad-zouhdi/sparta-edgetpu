#!/usr/bin/env python3
"""make_curves.py — pgfplots LaTeX curves for the SPARTA benchmark.

Reads /home/a131/Desktop/results/benchmark_results.json (full 10..90 % sweep)
and emits LaTeX-only output, grouped **by curve type** (not by model).

Layout under work/final_tables_and_plots/:

    curves/
        accuracy_vs_compression/
            resnet18.tex … squeezenet1_1.tex    one figure per model
        accuracy_vs_compression.tex             groupplots panel of all models
        speedup_vs_compression/…
        speedup_vs_compression.tex
        macs_reduction_vs_compression/…
        macs_reduction_vs_compression.tex
        pareto_accuracy_vs_latency/…
        pareto_accuracy_vs_latency.tex
        curves.tex                              index \\input'ing the 4 panels

    pareto_fronts/
        <model>_front.txt                       Pareto configs per model
        global_pareto.tex                       all-models Pareto pgfplots
        global_pareto.txt                       configs on the global front

Required Overleaf preamble:
    \\usepackage{pgfplots} \\pgfplotsset{compat=1.18}
    \\usepgfplotslibrary{groupplots}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
RESULTS = Path("/home/a131/Desktop/results/benchmark_results.json")
OUT_ROOT = Path("/home/a131/Desktop/Project/work/final_tables_and_plots")
CURVES_DIR = OUT_ROOT / "curves"
PARETO_DIR = OUT_ROOT / "pareto_fronts"

# ── Run plan ────────────────────────────────────────────────────────────────
MODELS = ["resnet18", "resnet50", "vgg19", "wrn_28_10",
          "mobilenetv2", "googlenet", "squeezenet1_1"]
DISPLAY = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50",
    "vgg19": "VGG-19", "wrn_28_10": "WRN-28-10",
    "mobilenetv2": "MobileNetV2", "googlenet": "GoogLeNet",
    "squeezenet1_1": "SqueezeNet 1.1",
}
IMPORTANCES = ["magnitude_l1", "magnitude_l2", "bn_scale", "fpgm",
               "taylor", "obdc", "random"]
RATIOS = [10, 20, 30, 40, 50, 60, 70, 80, 90]

EDGETPU_SRAM_MIB = 8.0

# ── Style ───────────────────────────────────────────────────────────────────
# pgfplots-friendly colour names (xcolor HTML codes).
IMP_COLOR_HEX = {
    "magnitude_l1": "1F77B4", "magnitude_l2": "AEC7E8",
    "bn_scale":     "FF7F0E", "fpgm":         "2CA02C",
    "taylor":       "D62728", "obdc":         "9467BD",
    "random":       "7F7F7F",
}
IMP_MARKER = {
    "magnitude_l1": "o",         # circle
    "magnitude_l2": "square",
    "bn_scale":     "diamond",
    "fpgm":         "triangle",
    "taylor":       "triangle*",
    "obdc":         "pentagon",
    "random":       "x",
}
IMP_LABEL = {
    "magnitude_l1": "Magnitude L1", "magnitude_l2": "Magnitude L2",
    "bn_scale":     "BN-Scale",     "fpgm":         "FPGM",
    "taylor":       "Taylor",       "obdc":         "OBD-C",
    "random":       "Random",
}

# Short column-name prefix used in the per-panel CSVs. Must be valid
# pgfplots column identifiers (letters, digits, underscore).
IMP_COL = {
    "magnitude_l1": "magl1", "magnitude_l2": "magl2",
    "bn_scale":     "bnscale", "fpgm": "fpgm",
    "taylor":       "taylor",  "obdc": "obdc",
    "random":       "random",
}

MODEL_COLOR_HEX = {
    "resnet18":      "1F77B4",
    "resnet50":      "FF7F0E",
    "vgg19":         "2CA02C",
    "wrn_28_10":     "D62728",
    "mobilenetv2":   "9467BD",
    "googlenet":     "8C564B",
    "squeezenet1_1": "17BECF",
}
MODEL_MARKER = {
    "resnet18":      "o",
    "resnet50":      "square",
    "vgg19":         "diamond",
    "wrn_28_10":     "triangle",
    "mobilenetv2":   "triangle*",
    "googlenet":     "pentagon",
    "squeezenet1_1": "x",
}


# ── Data ────────────────────────────────────────────────────────────────────
def load_data() -> dict:
    return json.loads(RESULTS.read_text())["models"]


def per_model_rows(bench: dict, model: str) -> dict:
    rows = defaultdict(list)
    for key, val in bench.items():
        if not key.startswith(model + "_pruned"):
            continue
        imp = val.get("importance")
        if imp is None:
            continue
        rows[imp].append(val)
    for imp in rows:
        rows[imp].sort(key=lambda r: r.get("prune_pct") or 0)
    return dict(rows)


def baseline_row(bench: dict, model: str) -> dict | None:
    return bench.get(f"{model}_finetuned")


def is_pareto(pts: np.ndarray) -> np.ndarray:
    """Pareto mask, two-objective minimisation."""
    n = len(pts)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                mask[i] = False
                break
    return mask


def compute_8mb_crossing(by_imp: dict, base_size: float) -> float | None:
    if base_size is None or base_size <= EDGETPU_SRAM_MIB:
        return None
    crossings = []
    for imp, rows in by_imp.items():
        rows_sorted = sorted(rows, key=lambda r: r.get("param_reduction_pct") or 0)
        prev = (0.0, base_size)
        for r in rows_sorted:
            pct = r.get("param_reduction_pct")
            sz = r.get("size_int8_mib")
            if pct is None or sz is None:
                continue
            if sz <= EDGETPU_SRAM_MIB:
                if prev[1] == sz:
                    crossings.append(pct)
                else:
                    crossings.append(
                        prev[0] + (EDGETPU_SRAM_MIB - prev[1])
                        * (pct - prev[0]) / (sz - prev[1])
                    )
                break
            prev = (pct, sz)
    return float(np.median(crossings)) if crossings else None


# ── LaTeX scaffolding ───────────────────────────────────────────────────────
def standalone_wrap(body: str) -> str:
    """Wrap a pgfplots body in a tikzpicture; the file is meant to be
    \\input'd inside a figure environment."""
    return (
        "% Auto-generated by make_curves.py — do not edit by hand.\n"
        "% Requires: \\usepackage{pgfplots}\\pgfplotsset{compat=1.18}\n"
        "\\begin{tikzpicture}\n"
        f"{body}\n"
        "\\end{tikzpicture}\n"
    )


def groupplot_wrap(body: str, ncols: int, nmodels: int) -> str:
    return (
        "% Auto-generated by make_curves.py — do not edit by hand.\n"
        "% Requires: \\usepackage{pgfplots}\\pgfplotsset{compat=1.18}\n"
        "%           \\usepgfplotslibrary{groupplots}\n"
        "\\begin{tikzpicture}\n"
        f"{body}\n"
        "\\end{tikzpicture}\n"
    )


def define_colors_block() -> str:
    """xcolor definitions emitted at the top of every figure."""
    out = []
    for imp, hexv in IMP_COLOR_HEX.items():
        out.append(rf"\definecolor{{imp{imp.replace('_','')}}}{{HTML}}{{{hexv}}}")
    for m, hexv in MODEL_COLOR_HEX.items():
        out.append(rf"\definecolor{{mdl{m.replace('_','')}}}{{HTML}}{{{hexv}}}")
    return "\n".join(out)


def imp_color_name(imp: str) -> str:
    return "imp" + imp.replace("_", "")


def model_color_name(model: str) -> str:
    return "mdl" + model.replace("_", "")


def coords(xs, ys) -> str:
    parts = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        parts.append(f"({x:.4f},{y:.4f})")
    return " ".join(parts)


# ── CSV writing for line-style panels ──────────────────────────────────────
def write_line_csv(csv_path: Path, by_imp: dict, x_field: str, y_field: str,
                   base_xy: tuple | None) -> None:
    """Write one CSV with paired (x, y) columns per importance, one row per
    pruning target (plus an optional baseline row at target = 0)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    imp_data = {}
    for imp in IMPORTANCES:
        rows = by_imp.get(imp, [])
        imp_data[imp] = {r.get("prune_pct"): (r.get(x_field), r.get(y_field))
                         for r in rows}

    header = []
    for imp in IMPORTANCES:
        col = IMP_COL[imp]
        header.append(f"{col}_x")
        header.append(f"{col}_y")

    rows_out = []
    if base_xy is not None:
        bx, by = base_xy
        row = []
        for _ in IMPORTANCES:
            row.append(f"{bx:.4f}")
            row.append(f"{by:.4f}")
        rows_out.append(row)
    for target in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        row = []
        for imp in IMPORTANCES:
            xy = imp_data[imp].get(target)
            if xy is None or xy[0] is None or xy[1] is None:
                row.append("")
                row.append("")
            else:
                row.append(f"{xy[0]:.4f}")
                row.append(f"{xy[1]:.4f}")
        rows_out.append(row)

    with csv_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for r in rows_out:
            f.write(",".join(r) + "\n")


def write_accuracy_csv(csv_path: Path, base: dict, by_imp: dict) -> None:
    write_line_csv(csv_path, by_imp,
                   x_field="param_reduction_pct", y_field="top1_int8_pct",
                   base_xy=(0.0, base.get("top1_int8_pct")))


def write_speedup_csv(csv_path: Path, base: dict, by_imp: dict) -> None:
    write_line_csv(csv_path, by_imp,
                   x_field="param_reduction_pct", y_field="prune_speedup_tpu",
                   base_xy=(0.0, 1.0))


def write_macs_csv(csv_path: Path, base: dict, by_imp: dict) -> None:
    write_line_csv(csv_path, by_imp,
                   x_field="param_reduction_pct", y_field="macs_reduction_pct",
                   base_xy=(0.0, 0.0))


def _addplot_table(imp: str, csv_rel: str) -> str:
    col = IMP_COL[imp]
    return (
        rf"\addplot[color={imp_color_name(imp)}, mark={IMP_MARKER[imp]}, "
        rf"mark size=1.5pt, line width=0.8pt, unbounded coords=jump] "
        rf"table[x={col}_x, y={col}_y, col sep=comma] {{{csv_rel}}};"
    )


# ── Curve type 1: accuracy vs compression ───────────────────────────────────
def _axis_open(opts: list[str]) -> str:
    """Emit \\begin{axis}[…] with options joined on a single logical line,
    skipping empty fragments. Empty lines inside option brackets become
    \\par at parse time and break pgfplots."""
    flat = " ".join(o for o in opts if o)
    return r"\begin{axis}[" + flat + r"]"


LEGEND_OUTSIDE = (
    r"legend style={font=\scriptsize, at={(1.02,1.0)}, anchor=north west,"
    r" cells={anchor=west}, draw=gray!50},"
    r" legend columns=1,"
)


def axis_accuracy(model: str, base: dict, by_imp: dict,
                  crossing: float | None, with_legend: bool = True,
                  title_suffix: str = "", csv_rel: str = "") -> str:
    base_acc = base.get("top1_int8_pct")
    title = DISPLAY.get(model, model) + title_suffix
    lines = []
    lines.append(_axis_open([
        f"width=8cm, height=6cm, title={{{title}}},",
        r"xlabel={Parameter reduction (\%)},",
        r"ylabel={INT8 Top-1 acc.\ (\%)},",
        LEGEND_OUTSIDE if with_legend else "",
        r"grid=both, grid style={dashed,gray!30},",
        r"tick label style={font=\scriptsize},",
        r"label style={font=\scriptsize}, title style={font=\small},",
        r"xmin=-2, xmax=95, ymin=0, ymax=90,",
    ]))
    if base_acc is not None:
        lines.append(rf"\addplot[mark=none, domain=-2:95, samples=2, dashed, black, thin] {{{base_acc:.3f}}};")
        if with_legend:
            lines.append(rf"\addlegendentry{{Baseline ({base_acc:.2f}\%)}}")

    for imp in IMPORTANCES:
        if not by_imp.get(imp):
            continue
        lines.append(_addplot_table(imp, csv_rel))
        if with_legend:
            lines.append(rf"\addlegendentry{{{IMP_LABEL[imp]}}}")

    if crossing is not None:
        lines.append(
            rf"\addplot[mark=none, color=red!70!black, dash dot, thick] "
            rf"coordinates {{({crossing:.2f},0) ({crossing:.2f},90)}};"
        )
        if with_legend:
            lines.append(
                rf"\addlegendentry{{Fits 8\,MB SRAM ($\approx${crossing:.1f}\%)}}"
            )

    lines.append(r"\end{axis}")
    return "\n".join(lines)


# ── Curve type 2: speedup vs compression ────────────────────────────────────
def axis_speedup(model: str, base: dict, by_imp: dict,
                 crossing: float | None, with_legend: bool = True,
                 title_suffix: str = "", csv_rel: str = "") -> str:
    title = DISPLAY.get(model, model) + title_suffix
    lines = []
    lines.append(_axis_open([
        f"width=8cm, height=6cm, title={{{title}}},",
        r"xlabel={Parameter reduction (\%)},",
        r"ylabel={Edge TPU speedup ($\times$)},",
        LEGEND_OUTSIDE if with_legend else "",
        r"grid=both, grid style={dashed,gray!30},",
        r"tick label style={font=\scriptsize},",
        r"label style={font=\scriptsize}, title style={font=\small},",
        r"xmin=-2, xmax=95, ymin=0, ymax=70,",
    ]))
    lines.append(r"\addplot[mark=none, domain=-2:95, samples=2, dashed, black, thin] {1};")
    if with_legend:
        lines.append(r"\addlegendentry{No speedup (1$\times$)}")
    for imp in IMPORTANCES:
        if not by_imp.get(imp):
            continue
        lines.append(_addplot_table(imp, csv_rel))
        if with_legend:
            lines.append(rf"\addlegendentry{{{IMP_LABEL[imp]}}}")
    if crossing is not None:
        lines.append(
            rf"\addplot[mark=none, color=red!70!black, dash dot, thick] "
            rf"coordinates {{({crossing:.2f},0) ({crossing:.2f},70)}};"
        )
        if with_legend:
            lines.append(
                rf"\addlegendentry{{Fits 8\,MB SRAM ($\approx${crossing:.1f}\%)}}"
            )
    lines.append(r"\end{axis}")
    return "\n".join(lines)


# ── Curve type 3: MACs reduction vs compression ─────────────────────────────
def axis_macs(model: str, base: dict, by_imp: dict,
              crossing: float | None, with_legend: bool = True,
              title_suffix: str = "", csv_rel: str = "") -> str:
    title = DISPLAY.get(model, model) + title_suffix
    lines = []
    lines.append(_axis_open([
        f"width=8cm, height=6cm, title={{{title}}},",
        r"xlabel={Parameter reduction (\%)},",
        r"ylabel={MACs reduction (\%)},",
        LEGEND_OUTSIDE if with_legend else "",
        r"grid=both, grid style={dashed,gray!30},",
        r"tick label style={font=\scriptsize},",
        r"label style={font=\scriptsize}, title style={font=\small},",
        r"xmin=-2, xmax=95, ymin=0, ymax=100,",
    ]))
    lines.append(r"\addplot[mark=none, domain=0:95, samples=2, dotted, black] {x};")
    if with_legend:
        lines.append(r"\addlegendentry{$y = x$}")
    for imp in IMPORTANCES:
        if not by_imp.get(imp):
            continue
        lines.append(_addplot_table(imp, csv_rel))
        if with_legend:
            lines.append(rf"\addlegendentry{{{IMP_LABEL[imp]}}}")
    if crossing is not None:
        lines.append(
            rf"\addplot[mark=none, color=red!70!black, dash dot, thick] "
            rf"coordinates {{({crossing:.2f},0) ({crossing:.2f},100)}};"
        )
        if with_legend:
            lines.append(
                rf"\addlegendentry{{Fits 8\,MB SRAM ($\approx${crossing:.1f}\%)}}"
            )
    lines.append(r"\end{axis}")
    return "\n".join(lines)


# ── Curve type 4: Pareto accuracy vs latency ────────────────────────────────
def pareto_points(by_imp: dict, model: str = ""):
    pts = []
    for imp in IMPORTANCES:
        for r in by_imp.get(imp, []):
            lat = r.get("lat_tpu_int8_ms_mean")
            acc = r.get("top1_int8_pct")
            if lat is None or acc is None:
                continue
            pct = r.get("prune_pct")
            key = (f"{model}_pruned{pct}pct_{imp}"
                   if model and pct is not None else r.get("tag") or "?")
            pts.append({
                "lat": lat, "acc": acc, "imp": imp,
                "prune_pct": pct,
                "param_red": r.get("param_reduction_pct"),
                "size_mib": r.get("size_int8_mib"),
                "key": key,
            })
    return pts


def pareto_mask(pts):
    if not pts:
        return []
    arr = np.array([[p["lat"], -p["acc"]] for p in pts])
    return is_pareto(arr)


def _pareto_opacity(prune_pct):
    """Linearly decrease opacity from 1.0 at 10 % prune to 0.2 at 90 %.
    Baseline / None prune_pct → fully opaque."""
    if prune_pct is None:
        return 1.0
    return max(0.2, 1.0 - 0.8 * (prune_pct - 10) / 80.0)


def axis_pareto(model: str, base: dict, by_imp: dict,
                with_legend: bool = True,
                title_suffix: str = "") -> tuple[str, list]:
    pts = pareto_points(by_imp, model=model)
    mask = pareto_mask(pts)
    pareto = sorted([p for p, m in zip(pts, mask) if m], key=lambda p: p["lat"])

    title = DISPLAY.get(model, model) + title_suffix
    lines = []
    lines.append(_axis_open([
        f"width=8cm, height=6cm, title={{{title}}},",
        r"xlabel={Edge TPU latency (ms)},",
        r"ylabel={INT8 Top-1 acc.\ (\%)},",
        LEGEND_OUTSIDE if with_legend else "",
        r"grid=both, grid style={dashed,gray!30},",
        r"tick label style={font=\scriptsize},",
        r"label style={font=\scriptsize}, title style={font=\small},",
    ]))

    # One \addplot per point (color = importance, opacity = f(prune_pct)).
    # Single marker shape (circle) for visual consistency.
    forget = "forget plot, "
    for p in pts:
        op = _pareto_opacity(p["prune_pct"])
        lines.append(
            rf"\addplot[only marks, {forget}"
            rf"color={imp_color_name(p['imp'])}, "
            rf"mark=*, mark size=2.0pt, opacity={op:.2f}, "
            rf"mark options={{fill={imp_color_name(p['imp'])}, "
            rf"fill opacity={op:.2f}, draw opacity={op:.2f}}}] "
            rf"coordinates {{({p['lat']:.4f},{p['acc']:.4f})}};"
        )

    # Pareto front line on top
    if pareto:
        xs = [p["lat"] for p in pareto]
        ys = [p["acc"] for p in pareto]
        lines.append(
            r"\addplot[mark=none, color=black, dotted, line width=0.9pt] "
            rf"coordinates {{{coords(xs, ys)}}};"
        )
        if with_legend:
            lines.append(r"\addlegendentry{Pareto front}")

    # Baseline star
    if base:
        b_lat = base.get("lat_tpu_int8_ms_mean")
        b_acc = base.get("top1_int8_pct")
        if b_lat is not None and b_acc is not None:
            lines.append(
                r"\addplot[only marks, mark=star, mark size=4pt, "
                r"color=black, mark options={fill=yellow!90!orange}] "
                rf"coordinates {{({b_lat:.4f},{b_acc:.4f})}};"
            )
            if with_legend:
                lines.append(r"\addlegendentry{Baseline (unpruned)}")

    # Importance colour legend (one entry per criterion, opaque sample)
    if with_legend:
        for imp in IMPORTANCES:
            if not by_imp.get(imp):
                continue
            lines.append(
                rf"\addplot[only marks, color={imp_color_name(imp)}, "
                rf"mark=*, mark size=2.5pt] coordinates {{(nan,nan)}};"
            )
            lines.append(rf"\addlegendentry{{{IMP_LABEL[imp]}}}")
        # Compression-opacity hint
        lines.append(r"\addlegendimage{empty legend}")
        lines.append(r"\addlegendentry{\emph{Opacity $\propto$ 1\,$-$\,prune\,\%}}")

    lines.append(r"\end{axis}")
    return "\n".join(lines), pareto


# ── Groupplot wrapping ──────────────────────────────────────────────────────
def render_groupplot_panel(axes_bodies: list[str], ncols: int = 2,
                           horizontal_sep: str = "1.8cm",
                           vertical_sep: str = "1.4cm",
                           subplot_width: str = "8cm",
                           subplot_height: str = "5.5cm",
                           legend_entries: list[tuple[str, str]] | None = None,
                           global_title: str = "") -> str:
    """Group 7 panels in a 2x4 grid; if `legend_entries` is provided, the 8th
    cell hosts a hidden-axis shared legend. Optional `global_title` is
    rendered as a tikz node above the whole groupplot."""
    nplots = len(axes_bodies)
    with_legend_cell = bool(legend_entries)
    total = nplots + (1 if with_legend_cell else 0)
    nrows = (total + ncols - 1) // ncols
    head = (
        r"\pgfplotsset{every axis/.append style={"
        rf"width={subplot_width}, height={subplot_height}, "
        r"tick label style={font=\scriptsize}, "
        r"label style={font=\scriptsize}, "
        r"title style={font=\small}}}"
        "\n"
        r"\begin{groupplot}[group style={"
        rf"group size={ncols} by {nrows}, "
        rf"horizontal sep={horizontal_sep}, vertical sep={vertical_sep}"
        r"}]" "\n"
    )
    body = []
    for ax in axes_bodies:
        opt_start = ax.find("[")
        opt_end = ax.find("]\n", opt_start)
        if opt_start < 0 or opt_end < 0:
            body.append(ax)
            continue
        opts = ax[opt_start + 1: opt_end]
        inner = ax[opt_end + 2:].rstrip()
        if inner.endswith(r"\end{axis}"):
            inner = inner[: -len(r"\end{axis}")].rstrip()
        body.append(rf"\nextgroupplot[{opts}]")
        body.append(inner)

    if with_legend_cell:
        body.append(
            r"\nextgroupplot["
            r"hide axis, xmin=0, xmax=1, ymin=0, ymax=1, "
            r"legend pos=north west, "
            r"legend style={font=\small, draw=gray!50, fill=white,"
            r" cells={anchor=west}, inner sep=4pt}, "
            r"legend cell align=left"
            r"]"
        )
        for image, label in legend_entries:
            body.append(image)
            body.append(rf"\addlegendentry{{{label}}}")

    out = head + "\n".join(body) + "\n" + r"\end{groupplot}"
    if global_title:
        out += (
            "\n"
            r"\node[font=\large, anchor=south, align=center] "
            r"at (current bounding box.north) "
            rf"[yshift=3mm] {{{global_title}}};"
        )
    return out


# ── Per-model Pareto config dump ────────────────────────────────────────────
def write_pareto_txt(model: str, base: dict, by_imp: dict) -> list[dict]:
    pts = pareto_points(by_imp, model=model)
    mask = pareto_mask(pts)
    pareto = sorted([p for p, m in zip(pts, mask) if m], key=lambda p: p["lat"])

    header = (f"# {'importance':<14}  {'target':>6}  "
              f"{'size_mib':>9}  {'lat_tpu_ms':>10}  {'top1_int8_pct':>13}")
    sep    = "# " + "-" * (len(header) - 2)
    lines = [
        f"# Pareto front (accuracy vs Edge TPU latency) — {DISPLAY.get(model, model)}",
        f"# Source: {RESULTS}",
        header,
        sep,
    ]
    if base:
        b_lat = base.get("lat_tpu_int8_ms_mean")
        b_acc = base.get("top1_int8_pct")
        b_sz  = base.get("size_int8_mib")
        lines.append(
            f"  {'baseline':<14}  {'--':>6}  "
            f"{(b_sz if b_sz is not None else 0):>9.2f}  "
            f"{b_lat:>10.3f}  {b_acc:>13.2f}"
        )
    for p in pareto:
        target = f"{p['prune_pct']}%" if p["prune_pct"] is not None else "--"
        sz = p.get("size_mib")
        sz_str = f"{sz:>9.2f}" if sz is not None else f"{'--':>9}"
        lines.append(
            f"  {p['imp']:<14}  {target:>6}  {sz_str}  "
            f"{p['lat']:>10.3f}  {p['acc']:>13.2f}"
        )
    PARETO_DIR.mkdir(parents=True, exist_ok=True)
    (PARETO_DIR / f"{model}_front.txt").write_text("\n".join(lines) + "\n")
    return pareto


# ── Global Pareto ──────────────────────────────────────────────────────────
def collect_global_points(bench: dict):
    pts = []
    for model in MODELS:
        base = baseline_row(bench, model)
        if base:
            pts.append({
                "model": model, "imp": "baseline",
                "prune_pct": None, "param_red": 0.0,
                "lat": base.get("lat_tpu_int8_ms_mean"),
                "acc": base.get("top1_int8_pct"),
                "size_mib": base.get("size_int8_mib"),
                "key": f"{model}_finetuned",
            })
        for key, val in bench.items():
            if not key.startswith(model + "_pruned"):
                continue
            lat = val.get("lat_tpu_int8_ms_mean")
            acc = val.get("top1_int8_pct")
            if lat is None or acc is None:
                continue
            pts.append({
                "model": model,
                "imp": val.get("importance") or "?",
                "prune_pct": val.get("prune_pct"),
                "param_red": val.get("param_reduction_pct"),
                "size_mib": val.get("size_int8_mib"),
                "lat": lat, "acc": acc,
                "key": key,
            })
    return [p for p in pts if p["lat"] is not None and p["acc"] is not None]


def render_global_pareto(bench: dict):
    pts = collect_global_points(bench)
    arr = np.array([[p["lat"], -p["acc"]] for p in pts])
    mask = is_pareto(arr)
    pareto = sorted([p for p, m in zip(pts, mask) if m], key=lambda p: p["lat"])

    # Write .txt
    header = (f"# {'model':<14}  {'importance':<14}  {'target':>6}  "
              f"{'size_mib':>9}  {'lat_tpu_ms':>10}  {'top1_int8_pct':>13}")
    sep = "# " + "-" * (len(header) - 2)
    lines = [
        "# Global Pareto front (accuracy vs Edge TPU latency) — all models, all configs",
        f"# Source: {RESULTS}",
        f"# {len(pts)} candidate configurations, {len(pareto)} on the global front.",
        header,
        sep,
    ]
    for p in pareto:
        target = "--" if p["prune_pct"] is None else f"{p['prune_pct']}%"
        sz = p.get("size_mib")
        sz_str = f"{sz:>9.2f}" if sz is not None else f"{'--':>9}"
        lines.append(
            f"  {DISPLAY.get(p['model'], p['model']):<14}  "
            f"{p['imp']:<14}  "
            f"{target:>6}  "
            f"{sz_str}  "
            f"{p['lat']:>10.3f}  "
            f"{p['acc']:>13.2f}"
        )
    PARETO_DIR.mkdir(parents=True, exist_ok=True)
    (PARETO_DIR / "global_pareto.txt").write_text("\n".join(lines) + "\n")

    # pgfplots — colour by model, single mark shape, opacity ∝ compression.
    body = []
    body.append(define_colors_block())
    body.append(
        r"\begin{axis}[" "\n"
        r"  width=14cm, height=10cm,"
        r"  title={Global Pareto front across all models and configurations"
        r" (INT8 on Edge TPU)}," "\n"
        r"  xlabel={Edge TPU latency (ms)},"
        r" ylabel={INT8 Top-1 accuracy (\%)}," "\n"
        r"  grid=both, grid style={dashed,gray!30},"
        r" tick label style={font=\footnotesize},"
        r" label style={font=\footnotesize},"
        r" title style={font=\small},"
        r" xmode=log, log basis x=10,"
        r" legend style={font=\scriptsize, at={(1.02,1.0)}, anchor=north west,"
        r" cells={anchor=west}, draw=gray!50},"
        r" legend columns=1,"
        r"]"
    )

    # One \addplot per point (color = model, opacity = compression).
    forget = "forget plot, "
    for p in pts:
        op = _pareto_opacity(p["prune_pct"])
        body.append(
            rf"\addplot[only marks, {forget}"
            rf"color={model_color_name(p['model'])}, "
            rf"mark=*, mark size=1.6pt, opacity={op:.2f}, "
            rf"mark options={{fill={model_color_name(p['model'])}, "
            rf"fill opacity={op:.2f}, draw opacity={op:.2f}}}] "
            rf"coordinates {{({p['lat']:.4f},{p['acc']:.4f})}};"
        )

    # Pareto front line on top
    xs = [p["lat"] for p in pareto]
    ys = [p["acc"] for p in pareto]
    body.append(
        r"\addplot[mark=none, color=black, dotted, line width=1.0pt] "
        rf"coordinates {{{coords(xs, ys)}}};"
    )
    body.append(r"\addlegendentry{Global Pareto front}")

    # Model legend (opaque colour sample, no marker variation)
    for model in MODELS:
        if not any(p["model"] == model for p in pts):
            continue
        body.append(
            rf"\addplot[only marks, color={model_color_name(model)}, "
            rf"mark=*, mark size=3pt] coordinates {{(nan,nan)}};"
        )
        body.append(rf"\addlegendentry{{{DISPLAY.get(model, model)}}}")
    body.append(r"\addlegendimage{empty legend}")
    body.append(r"\addlegendentry{\emph{Opacity $\propto$ 1\,$-$\,prune\,\%}}")

    body.append(r"\end{axis}")
    tex = standalone_wrap("\n".join(body))
    (PARETO_DIR / "global_pareto.tex").write_text(tex)


# ── Curve emission ──────────────────────────────────────────────────────────
GROUP_TITLES = {
    "accuracy_vs_compression":
        r"INT8 Top-1 accuracy after recovery fine-tuning,"
        r" per importance criterion (CIFAR-100, Edge TPU)",
    "speedup_vs_compression":
        r"Edge TPU latency speedup over the unpruned baseline,"
        r" per importance criterion (INT8, 200-run mean)",
    "macs_reduction_vs_compression":
        r"MACs reduction vs.\ parameter reduction, per importance criterion",
    "pareto_accuracy_vs_latency":
        r"Per-model Pareto front: INT8 Top-1 accuracy vs Edge TPU latency",
}


def importance_legend_entries(with_baseline: bool = True,
                              with_sram: bool = False,
                              extra: list[tuple[str, str]] | None = None
                              ) -> list[tuple[str, str]]:
    entries = []
    if with_baseline:
        entries.append((
            r"\addlegendimage{mark=none, dashed, black, thin}",
            "Baseline (unpruned)",
        ))
    for imp in IMPORTANCES:
        entries.append((_legend_image(imp), IMP_LABEL[imp]))
    if with_sram:
        entries.append((
            r"\addlegendimage{mark=none, color=red!70!black,"
            r" dash dot, thick}",
            r"Fits 8\,MB Edge TPU SRAM",
        ))
    if extra:
        entries.extend(extra)
    return entries


def _legend_image(imp: str) -> str:
    return (
        rf"\addlegendimage{{color={imp_color_name(imp)}, "
        rf"mark={IMP_MARKER[imp]}, line width=0.8pt, mark size=2.5pt}}"
    )


def emit_curve_type(curve_name: str, axis_fn, bench: dict, with_8mb: bool,
                    legend_entries: list[tuple[str, str]] | None = None,
                    csv_writer=None):
    """Emit per-model individual .tex + a 2x4 groupplot .tex for `curve_name`.

    If `csv_writer` is provided, the panel data is dumped to
    curves/data/<curve_name>/<model>.csv and the axis function receives a
    `csv_rel` (path relative to the Overleaf job root, where the wrapper
    lives). Pareto-style axes pass None — they inline coordinates."""
    sub = CURVES_DIR / curve_name
    sub.mkdir(parents=True, exist_ok=True)
    data_sub = CURVES_DIR / "data" / curve_name
    if csv_writer is not None:
        data_sub.mkdir(parents=True, exist_ok=True)

    axes_for_group = []
    for model in MODELS:
        base = baseline_row(bench, model)
        if base is None:
            continue
        by_imp = per_model_rows(bench, model)
        if not by_imp:
            continue
        crossing = (compute_8mb_crossing(by_imp, base.get("size_int8_mib"))
                    if with_8mb else None)

        # Write CSV (line panels only).
        csv_rel = ""
        if csv_writer is not None:
            csv_abs = data_sub / f"{model}.csv"
            csv_writer(csv_abs, base, by_imp)
            csv_rel = f"curves/data/{curve_name}/{model}.csv"

        # Individual file (full legend, csv path embedded).
        if csv_writer is not None:
            ax_full = axis_fn(model, base, by_imp, crossing,
                              with_legend=True, csv_rel=csv_rel)
        else:
            ax_full = axis_fn(model, base, by_imp, crossing, with_legend=True)
        if isinstance(ax_full, tuple):
            ax_full = ax_full[0]
        body = define_colors_block() + "\n" + ax_full
        (sub / f"{model}.tex").write_text(standalone_wrap(body))

        # Groupplot subplot (no per-panel legend; shared in 8th cell).
        if csv_writer is not None:
            ax_compact = axis_fn(model, base, by_imp, crossing,
                                 with_legend=False, csv_rel=csv_rel)
        else:
            ax_compact = axis_fn(model, base, by_imp, crossing,
                                 with_legend=False)
        if isinstance(ax_compact, tuple):
            ax_compact = ax_compact[0]
        axes_for_group.append(ax_compact)

    entries = legend_entries or importance_legend_entries(with_sram=with_8mb)
    group_body = define_colors_block() + "\n" + render_groupplot_panel(
        axes_for_group, ncols=2,
        legend_entries=entries,
        global_title=GROUP_TITLES.get(curve_name, ""),
    )
    (CURVES_DIR / f"{curve_name}.tex").write_text(groupplot_wrap(
        group_body, ncols=2, nmodels=len(axes_for_group)
    ))


def axis_pareto_wrap(model, base, by_imp, crossing, with_legend=True):
    # Pareto ignores `crossing`.
    return axis_pareto(model, base, by_imp, with_legend=with_legend)[0]


def pareto_legend_entries() -> list[tuple[str, str]]:
    """Legend specific to the Pareto plots: solid colour dot per importance,
    plus baseline star and Pareto-front line. Marker shape is uniform; only
    colour and opacity carry information."""
    entries = [
        (r"\addlegendimage{mark=none, dotted, color=black, line width=0.9pt}",
         "Pareto front"),
        (r"\addlegendimage{only marks, mark=star, mark size=4pt,"
         r" color=black, mark options={fill=yellow!90!orange}}",
         "Baseline (unpruned)"),
    ]
    for imp in IMPORTANCES:
        entries.append((
            rf"\addlegendimage{{only marks, color={imp_color_name(imp)}, "
            rf"mark=*, mark size=3pt}}",
            IMP_LABEL[imp],
        ))
    entries.append((
        r"\addlegendimage{empty legend}",
        r"\emph{Opacity $\propto$ 1\,$-$\,prune\,\%}",
    ))
    return entries


# ── Driver ──────────────────────────────────────────────────────────────────
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    PARETO_DIR.mkdir(parents=True, exist_ok=True)

    bench = load_data()

    # Per-curve-type LaTeX (CSV-backed for line plots; Pareto stays inline
    # because each point has individual opacity).
    emit_curve_type("accuracy_vs_compression", axis_accuracy, bench,
                    with_8mb=True, csv_writer=write_accuracy_csv)
    emit_curve_type("speedup_vs_compression", axis_speedup, bench,
                    with_8mb=True, csv_writer=write_speedup_csv)
    emit_curve_type("macs_reduction_vs_compression", axis_macs, bench,
                    with_8mb=True, csv_writer=write_macs_csv)
    emit_curve_type("pareto_accuracy_vs_latency", axis_pareto_wrap, bench,
                    with_8mb=False, legend_entries=pareto_legend_entries())

    # Index file — input from final_tables_and_plots/.
    # Note: post_prune_acc.tex and dispersion_vs_target.tex are produced by
    # training_curves.py; this index references them so the user does not need
    # to remember which script writes what.
    index = [
        r"% Auto-generated curve index — \input{curves/curves.tex} from the"
        r" parent document to pull in every aggregated panel.",
        r"% Panels marked (groupplots) require \usepgfplotslibrary{groupplots}.",
        r"\input{curves/accuracy_vs_compression.tex}",
        r"\input{curves/speedup_vs_compression.tex}",
        r"\input{curves/macs_reduction_vs_compression.tex}",
        r"\input{curves/pareto_accuracy_vs_latency.tex}",
        r"\input{curves/post_prune_acc.tex}",
    ]
    (CURVES_DIR / "curves.tex").write_text("\n".join(index) + "\n")

    # Per-model Pareto config .txt
    for model in MODELS:
        base = baseline_row(bench, model)
        if base is None:
            continue
        by_imp = per_model_rows(bench, model)
        if not by_imp:
            continue
        write_pareto_txt(model, base, by_imp)

    # Global Pareto (.tex + .txt)
    render_global_pareto(bench)

    print(f"[ok] curves grouped by type under {CURVES_DIR}")
    print(f"[ok] Pareto fronts under {PARETO_DIR}")


if __name__ == "__main__":
    main()

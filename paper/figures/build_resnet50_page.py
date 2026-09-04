#!/usr/bin/env python3
"""Build a self-contained tikzpicture fragment for the ResNet-50 panel page.

Layout
------
Row 1 (3 panels):
  (1) Top-1 accuracy after recovery fine-tuning      [vs param reduction %]
  (2) Latency speedup over unpruned baseline         [vs param reduction %]
  (3) Pareto front: INT8 Top-1 vs Edge TPU latency   [opacity = compression]

Row 2, aligned right (2 panels in columns 2 & 3):
  (4) Top-1 val accuracy immediately after pruning   [vs param reduction %]
  (5) Recovery fine-tuning trajectory at 90% pruning [val acc vs epoch]

Shared legend sits in the empty column 1 of row 2 (under panel 1).

Panels 1, 2, 4 carry a parallel top axis showing the corresponding INT8
model size (MiB), computed from the ResNet-50 baseline (23.30 MiB).
"""
import csv
import glob
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent
TEX_OUT = OUT_DIR / "resnet50_page.tex"
BENCH_CSV = Path("/home/a131/Desktop/mesures_brut/benchmark_results.csv")
CURVES_DATA = Path("/home/a131/Desktop/Project/work/final_tables_and_plots/curves/data")
LOGS_70_90 = Path("/home/a131/Desktop/second/pruning_logs")

BASELINE_SIZE_MIB = 23.30          # ResNet-50 INT8 finetuned size
BASELINE_LAT_MS = 50.9249           # Edge TPU baseline latency
BASELINE_TOP1 = 77.40               # baseline INT8 accuracy (CSV)
REF_ACC_FP32 = 77.81                # FP32 val accuracy (CLAUDE.md / pruning logs)

IMP_ORDER = ["magnitude_l1","magnitude_l2","bn_scale","fpgm","taylor","obdc","random"]
IMP_LABEL = {
    "magnitude_l1":"Magnitude L1", "magnitude_l2":"Magnitude L2",
    "bn_scale":"BN-Scale", "fpgm":"FPGM", "taylor":"Taylor",
    "obdc":"OBD-C", "random":"Random",
}
IMP_COLOR = {
    "magnitude_l1":"impmagnitudel1", "magnitude_l2":"impmagnitudel2",
    "bn_scale":"impbnscale", "fpgm":"impfpgm",
    "taylor":"imptaylor", "obdc":"impobdc", "random":"imprandom",
}
IMP_MARK = {
    "magnitude_l1":"o", "magnitude_l2":"square",
    "bn_scale":"diamond", "fpgm":"triangle",
    "taylor":"triangle*", "obdc":"pentagon", "random":"x",
}
IMP_COL = {"magnitude_l1":"magl1","magnitude_l2":"magl2","bn_scale":"bnscale",
           "fpgm":"fpgm","taylor":"taylor","obdc":"obdc","random":"random"}

COLOR_DEFS = """\
\\definecolor{impmagnitudel1}{HTML}{1F77B4}
\\definecolor{impmagnitudel2}{HTML}{AEC7E8}
\\definecolor{impbnscale}{HTML}{FF7F0E}
\\definecolor{impfpgm}{HTML}{2CA02C}
\\definecolor{imptaylor}{HTML}{D62728}
\\definecolor{impobdc}{HTML}{9467BD}
\\definecolor{imprandom}{HTML}{7F7F7F}"""


# ── data extraction ─────────────────────────────────────────────────────────
def load_paired_csv(path: Path) -> dict[str, list[tuple[float, float]]]:
    out = {imp: [] for imp in IMP_ORDER}
    with open(path) as f:
        for row in csv.DictReader(f):
            for imp in IMP_ORDER:
                col = IMP_COL[imp]
                x = row.get(f"{col}_x", "")
                y = row.get(f"{col}_y", "")
                if x and y:
                    out[imp].append((float(x), float(y)))
    return out


def load_pareto_per_imp_pct():
    data = {imp: {} for imp in IMP_ORDER}
    with open(BENCH_CSV) as f:
        for row in csv.DictReader(f):
            m = row["model"]
            if m.startswith("resnet50_pruned"):
                imp = row.get("importance")
                if imp not in IMP_ORDER:
                    continue
                try:
                    pct = int(row["prune_pct"])
                    lat = float(row["lat_tpu_int8_ms_mean"])
                    acc = float(row["top1_int8_pct"])
                    data[imp][pct] = (lat, acc)
                except Exception:
                    continue
    return data


def compute_pareto_front(pareto_data):
    pts = [(BASELINE_LAT_MS, BASELINE_TOP1)]
    for imp in IMP_ORDER:
        pts.extend(pareto_data[imp].values())
    front = []
    for p in pts:
        dominated = False
        for q in pts:
            if q is p:
                continue
            if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort()
    return front


def load_trajectory_90():
    out = {}
    for f in sorted(LOGS_70_90.glob("resnet50_*_90pct.json")):
        d = json.loads(Path(f).read_text())
        imp = d.get("importance")
        if imp not in IMP_ORDER:
            continue
        traj = [(h["epoch"], h["val_acc"]) for h in d.get("ft_history", [])
                if "val_acc" in h]
        out[imp] = traj
    return out


# ── tex emitters ────────────────────────────────────────────────────────────
def coord_str(pts):
    return " ".join(f"({x:.4f},{y:.4f})" for x, y in pts)


PARAM_TICK = "xtick={0,20,40,60,80}, xticklabels={0,20,40,60,80},"
SIZE_TICK = "xtick={{0,20,40,60,80}}, xticklabels={{{lbl}}}".format(
    lbl=",".join(f"{BASELINE_SIZE_MIB*(1-p/100):.1f}" for p in (0, 20, 40, 60, 80))
)

PANEL_W = "5.0cm"   # natural panel width — \resizebox will scale to \linewidth
PANEL_H = "3.8cm"
COL_TIGHTEN = "0pt"     # 0 = adjacent outer boxes touch (no overlap, minimal gap)
ROW_GAP = "0.8cm"   # tight gap; captions live inside this gap
TITLE_YSHIFT = "14pt"   # uniform top padding so all panels share a top baseline
CAPTION_WIDTH = "5.8cm"   # ≈ panel outer width so the caption spans the figure


def panel_with_twin(name, prev_anchor, title, ylabel, ymin, ymax,
                    plots_tex, extra_baseline=None, ymode="",
                    has_top_twin=True, position="east"):
    """Main axis + twin top axis (Model size MiB).

    prev_anchor: pgf coordinate spec (e.g. None for first, or "(p1.outer east)").
    """
    if prev_anchor is None:
        at_clause = ""
    elif position == "south":
        at_clause = (f"at={prev_anchor}, anchor=outer north, "
                     f"yshift=-{ROW_GAP},")
    else:
        at_clause = (f"at={prev_anchor}, anchor=outer west, "
                     f"xshift=-{COL_TIGHTEN},")

    baseline_line = ""
    if extra_baseline is not None:
        baseline_line = (
            f"\\addplot[mark=none, domain=-2:95, samples=2, dashed, black, thin] "
            f"{{{extra_baseline:.3f}}};"
        )

    # Top padding (where the panel title used to live) is kept so that
    # all five panels share a single horizontal baseline at the main-axis top.
    option_lines = [
        f"  name={name},",
        f"  {at_clause}" if at_clause else None,
        f"  width={PANEL_W}, height={PANEL_H}, scale only axis,",
        f"  title={{\\vphantom{{Xy}}}}, title style={{font=\\small, yshift={TITLE_YSHIFT}}},",
        f"  xlabel={{Parameter reduction (\\%)}},",
        f"  ylabel={{{ylabel}}},",
        f"  {ymode}" if ymode else None,
        f"  xmin=-2, xmax=95, ymin={ymin}, ymax={ymax},",
        f"  {PARAM_TICK}",
        f"  grid=both, grid style={{dashed,gray!30}},",
        f"  tick label style={{font=\\scriptsize}},",
        f"  label style={{font=\\scriptsize}},",
    ]
    option_block = "\n".join(line for line in option_lines if line is not None)
    main = (
        f"\\begin{{axis}}[\n{option_block}\n]\n"
        f"{baseline_line}\n"
        f"{plots_tex}\n"
        f"\\end{{axis}}\n"
    )

    twin = (
        f"\\begin{{axis}}[\n"
        f"  at=({name}.south west), anchor=south west,\n"
        f"  width={PANEL_W}, height={PANEL_H}, scale only axis,\n"
        f"  hide y axis, axis x line*=top,\n"
        f"  xmin=-2, xmax=95, ymin={ymin}, ymax={ymax},\n"
        f"  {SIZE_TICK},\n"
        f"  xlabel={{INT8 model size (MiB)}}, xlabel near ticks,\n"
        f"  tick label style={{font=\\scriptsize}},\n"
        f"  xlabel style={{font=\\scriptsize}},\n"
        f"]\n"
        f"\\end{{axis}}\n"
    )
    return main + twin


def panel_plain(name, prev_anchor, title, xlabel, ylabel,
                xmin, xmax, ymin, ymax, plots_tex,
                extra_baseline=None, ymode="", xtick=None, position="east",
                has_top_twin=False):
    """A standard axis with no twin (used for Pareto and Trajectory)."""
    if prev_anchor is None:
        at_clause = ""
    elif position == "south":
        at_clause = (f"at={prev_anchor}, anchor=outer north, "
                     f"yshift=-{ROW_GAP},")
    else:
        at_clause = (f"at={prev_anchor}, anchor=outer west, "
                     f"xshift=-{COL_TIGHTEN},")

    baseline_line = ""
    if extra_baseline is not None:
        baseline_line = (
            f"\\addplot[mark=none, domain={xmin}:{xmax}, samples=2, dashed, black, thin] "
            f"{{{extra_baseline:.3f}}};"
        )

    option_lines = [
        f"  name={name},",
        f"  {at_clause}" if at_clause else None,
        f"  width={PANEL_W}, height={PANEL_H}, scale only axis,",
        f"  title={{\\vphantom{{Xy}}}}, title style={{font=\\small, yshift={TITLE_YSHIFT}}},",
        f"  xlabel={{{xlabel}}}, ylabel={{{ylabel}}},",
        f"  {ymode}" if ymode else None,
        f"  xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax},",
        f"  {xtick}" if xtick else None,
        f"  grid=both, grid style={{dashed,gray!30}},",
        f"  tick label style={{font=\\scriptsize}},",
        f"  label style={{font=\\scriptsize}},",
    ]
    option_block = "\n".join(line for line in option_lines if line is not None)
    return (
        f"\\begin{{axis}}[\n{option_block}\n]\n"
        f"{baseline_line}\n"
        f"{plots_tex}\n"
        f"\\end{{axis}}\n"
    )


def plots_for_imp_curves(data):
    """data: {imp: list[(x,y)]} - emit one \\addplot per importance."""
    out = []
    for imp in IMP_ORDER:
        pts = data.get(imp, [])
        if not pts:
            continue
        out.append(
            f"\\addplot[color={IMP_COLOR[imp]}, mark={IMP_MARK[imp]}, "
            f"mark size=1.5pt, line width=0.8pt] coordinates {{{coord_str(pts)}}};"
        )
    return "\n".join(out)


def plots_for_pareto(pareto, opacity_map):
    """One marker per (imp, pct) — same shape/color as the curve panels,
    opacity proportional to (1 - compression). Baseline drawn once as star."""
    out = []
    for imp in IMP_ORDER:
        for pct in sorted(pareto[imp]):
            lat, acc = pareto[imp][pct]
            op = opacity_map[pct]
            out.append(
                f"\\addplot[only marks, forget plot, color={IMP_COLOR[imp]}, "
                f"mark={IMP_MARK[imp]}, mark size=2.5pt, "
                f"mark options={{fill={IMP_COLOR[imp]}, fill opacity={op:.2f}, "
                f"draw opacity={op:.2f}}}] coordinates "
                f"{{({lat:.4f},{acc:.4f})}};"
            )
    return "\n".join(out)


def plots_for_pareto_front(front):
    return (
        f"\\addplot[mark=none, color=black, dotted, line width=0.9pt] "
        f"coordinates {{{coord_str(front)}}};"
    )


def plots_for_trajectory(traj):
    out = []
    for imp in IMP_ORDER:
        pts = traj.get(imp, [])
        if not pts:
            continue
        out.append(
            f"\\addplot[color={IMP_COLOR[imp]}, mark=none, line width=0.9pt] "
            f"coordinates {{{coord_str(pts)}}};"
        )
    return "\n".join(out)


def build_legend(at_clause):
    legend_entries = []
    legend_entries.append(
        (r"\addlegendimage{mark=none, dashed, black, thin}", "Baseline (unpruned)")
    )
    for imp in IMP_ORDER:
        legend_entries.append((
            f"\\addlegendimage{{color={IMP_COLOR[imp]}, mark={IMP_MARK[imp]}, "
            f"line width=0.8pt, mark size=2.5pt}}",
            IMP_LABEL[imp],
        ))
    body = []
    for image, label in legend_entries:
        body.append(image)
        body.append(f"\\addlegendentry{{{label}}}")
    return (
        f"\\begin{{axis}}[\n"
        f"  name=legendbox,\n"
        f"  {at_clause}\n"
        f"  width={PANEL_W}, height={PANEL_H}, scale only axis,\n"
        f"  hide axis, xmin=0, xmax=1, ymin=0, ymax=1,\n"
        f"  legend pos=north west,\n"
        f"  legend style={{font=\\small, draw=gray!50, fill=white,\n"
        f"    cells={{anchor=west}}, inner sep=4pt}},\n"
        f"  legend cell align=left,\n"
        f"]\n"
        + "\n".join(body) + "\n"
        f"\\end{{axis}}\n"
    )


def main():
    acc_data    = load_paired_csv(CURVES_DATA / "accuracy_vs_compression" / "resnet50.csv")
    spd_data    = load_paired_csv(CURVES_DATA / "speedup_vs_compression" / "resnet50.csv")
    ppa_data    = load_paired_csv(CURVES_DATA / "post_prune_acc" / "resnet50.csv")
    pareto      = load_pareto_per_imp_pct()
    front       = compute_pareto_front(pareto)
    traj_90     = load_trajectory_90()

    opacity_map = {10:0.95, 20:0.85, 30:0.75, 40:0.65,
                   50:0.55, 60:0.45, 70:0.35, 80:0.25, 90:0.18}

    # ── panels ───────────────────────────────────────────────────────────
    p1 = panel_with_twin(
        "p1", None,
        "Top-1 accuracy after recovery fine-tuning",
        "INT8 Top-1 (\\%)", 65, 80,
        plots_for_imp_curves(acc_data),
        extra_baseline=BASELINE_TOP1,
    )
    p2 = panel_with_twin(
        "p2", "(p1.outer east)",
        "Edge TPU latency speedup over baseline",
        "Speedup ($\\times$)", 0, 60,
        plots_for_imp_curves(spd_data),
        extra_baseline=1.0,
    )

    # Pareto plot — linear axis across the full latency range.
    pareto_plots = (
        plots_for_pareto(pareto, opacity_map) + "\n" +
        plots_for_pareto_front(front) + "\n" +
        f"\\addplot[only marks, mark=star, mark size=4pt, color=black, "
        f"mark options={{fill=yellow!90!orange}}] coordinates "
        f"{{({BASELINE_LAT_MS:.4f},{BASELINE_TOP1:.4f})}};"
    )
    p3 = panel_plain(
        "p3", "(p2.outer east)",
        "Pareto front: INT8 Top-1 vs Edge TPU latency",
        "Edge TPU latency (ms)", "INT8 Top-1 (\\%)",
        0, 55, 65, 80,
        pareto_plots,
    )

    # Row 2: aligned right (cols 2 and 3)
    p4 = panel_with_twin(
        "p4", "(p2.outer south)",
        "Top-1 val acc immediately after pruning",
        "Top-1 val (\\%)", 0, 85,
        plots_for_imp_curves(ppa_data),
        extra_baseline=REF_ACC_FP32,
        position="south",
    )

    p5 = panel_plain(
        "p5", "(p3.outer south)",
        "Recovery fine-tuning trajectory at 90\\% pruning",
        "Epoch", "Top-1 val (\\%)",
        0, 100, 0, 80,
        plots_for_trajectory(traj_90),
        xtick="xtick={0,20,40,60,80,100},",
        position="south",
    )

    # Add LR-decay vertical guides at ep=60, 80 to panel 5 (inside its axis env)
    decay_lines = (
        "\\addplot[mark=none, color=gray!50, dashed, thin] "
        "coordinates {(60,0) (60,80)};\n"
        "\\addplot[mark=none, color=gray!50, dashed, thin] "
        "coordinates {(80,0) (80,80)};\n"
    )
    p5 = p5.replace("\\end{axis}\n", decay_lines + "\\end{axis}\n", 1)

    # Shared legend: top-aligned with the row-2 main axes, centered under p1.
    legend = build_legend(
        "at=(p4.north -| p1.center), anchor=north,"
    )

    # Title above row 1.
    title_node = (
        r"\node[font=\large\bfseries, anchor=south, align=center] "
        r"at (current bounding box.north) [yshift=2mm] "
        r"{ResNet-50: structured pruning effects on accuracy and Edge TPU latency};"
    )

    # Per-panel "Fig. X. <title>" captions placed below each axis. Each
    # caption calls \refstepcounter{figure} so the document's figure counter
    # advances by one per panel, preserving global Fig. X numbering.
    panel_titles = [
        ("p1", "Top-1 accuracy after recovery fine-tuning."),
        ("p2", "Edge TPU latency speedup over the unpruned baseline."),
        ("p3", "Pareto front: INT8 Top-1 accuracy vs Edge TPU latency."),
        ("p4", "Top-1 validation accuracy immediately after pruning "
               "(before recovery fine-tuning)."),
        ("p5", "Recovery fine-tuning trajectory at 90\\% pruning."),
    ]
    caption_nodes = []
    for pname, ptitle in panel_titles:
        # inner ysep=0 + small yshift pulls the caption close to the xlabel
        # without overlapping it.
        caption_nodes.append(
            f"\\node[anchor=north, font=\\footnotesize, align=center, "
            f"text width={CAPTION_WIDTH}, inner xsep=2pt, inner ysep=0pt] "
            f"at ({pname}.outer south) "
            f"{{\\refstepcounter{{figure}}Fig.~\\thefigure.~{ptitle}}};"
        )
    captions = "\n".join(caption_nodes)

    # Inset note inside the Pareto panel explaining the opacity encoding.
    pareto_note = (
        "\\node[anchor=south east, font=\\scriptsize, "
        "fill=white, draw=gray!50, rounded corners=1pt, inner sep=2pt] "
        "at (p3.south east) [xshift=-2pt, yshift=2pt] "
        "{marker opacity $\\propto$ compression ratio};"
    )

    body = "\n".join([p1, p2, p3, p4, p5, legend, title_node,
                      captions, pareto_note])
    out = (
        "% Auto-generated by build_resnet50_page.py — do not edit by hand.\n"
        "% Requires: \\usepackage{pgfplots}\\pgfplotsset{compat=1.18}\n"
        "% Wrap the tikzpicture in \\resizebox so the figure auto-fits the\n"
        "% host container's \\linewidth (e.g. inside \\begin{figure*}).\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{tikzpicture}\n"
        + COLOR_DEFS + "\n"
        + body + "\n"
        "\\end{tikzpicture}%\n"
        "}\n"
    )
    TEX_OUT.write_text(out)
    print(f"Wrote {TEX_OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-model benchmark tables from benchmark_results.json.

Reads /home/a131/Desktop/results/benchmark_results.json (full sweep, 10..90 %).

Outputs:
    work/final_tables_and_plots/tables/{model}.tex   one LaTeX table per model
    work/final_tables_and_plots/tables.tex           index \\input'ing them all
    work/final_tables_and_plots/csv/{model}.csv      same data, CSV intermediate
    stdout                                           ASCII tables (debug view)
"""
import csv
import json
from itertools import groupby
from pathlib import Path

JSON_PATH = Path("/home/a131/Desktop/results/benchmark_results.json")
OUT_ROOT = Path("/home/a131/Desktop/Project/work/final_tables_and_plots")
TEX_DIR = OUT_ROOT / "tables"
CSV_DIR = OUT_ROOT / "csv"
INDEX_TEX = OUT_ROOT / "tables.tex"

MODEL_ORDER = ["resnet18", "resnet50", "vgg19", "wrn_28_10",
               "mobilenetv2", "googlenet", "squeezenet1_1"]
DISPLAY = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vgg19": "VGG-19",
    "wrn_28_10": "WRN-28-10",
    "mobilenetv2": "MobileNetV2",
    "googlenet": "GoogLeNet",
    "squeezenet1_1": "SqueezeNet 1.1",
}
IMP_ORDER = ["magnitude_l1", "magnitude_l2", "bn_scale", "fpgm",
             "taylor", "obdc", "random"]
PCT_ORDER = [10, 20, 30, 40, 50, 60, 70, 80, 90]

HEADERS = ["Target", "Importance", "Acc", "Δ Acc",
           "Lat (ms)", "TPU spd", "CPU spd",
           "Red Params (%)", "Red MACs (%)",
           "Size (mb)", "On Chip (mb)", "Off Chip (mb)"]

CSV_HEADERS = ["target_pct", "importance", "acc", "d_acc",
               "lat_ms", "tpu_spd", "cpu_spd",
               "red_params_pct", "red_macs_pct",
               "size_mb", "on_chip_mb", "off_chip_mb"]


# ── ASCII rendering (debug view) ────────────────────────────────────────────
def fmt(value, spec, na="-"):
    if value is None:
        return na
    return format(value, spec)


def cell_widths(headers, rows):
    return [max(len(h), max((len(r[i]) for r in rows), default=0))
            for i, h in enumerate(headers)]


def hline(widths, left, mid, right, fill="─"):
    return left + mid.join(fill * (w + 2) for w in widths) + right


def render_row_ascii(cells, widths, sep="│"):
    return sep + sep.join(f" {c:<{w}} " for c, w in zip(cells, widths)) + sep


def title_bar(title, total_width):
    inner = total_width - 2
    return (
        "┌" + "─" * inner + "┐\n"
        "│" + title.center(inner) + "│"
    )


def format_row_ascii(r):
    tgt, imp, acc, dacc, lat, sp_tpu, sp_cpu, red_p, red_m, sz, on, off = r
    if tgt is None:
        tgt_s = "—"
    elif tgt == "":
        tgt_s = ""
    else:
        tgt_s = f"{tgt}%"
    return [
        tgt_s,
        imp,
        fmt(acc, ".2f"),
        fmt(dacc, "+.2f"),
        fmt(lat, ".2f"),
        fmt(sp_tpu, ".2f") + ("×" if sp_tpu is not None else ""),
        fmt(sp_cpu, ".2f") + ("×" if sp_cpu is not None else ""),
        fmt(red_p, ".1f"),
        fmt(red_m, ".1f"),
        fmt(sz, ".2f"),
        fmt(on, ".2f"),
        fmt(off, ".2f"),
    ]


def render_table_ascii(model, baseline_row, pruned_rows):
    groups = []
    for tgt, grp in groupby(pruned_rows, key=lambda r: r[0]):
        glist = [list(r) for r in grp]
        center = (len(glist) - 1) // 2
        for i, row in enumerate(glist):
            if i != center:
                row[0] = ""
        groups.append(glist)

    fmt_baseline = format_row_ascii(baseline_row)
    fmt_groups = [[format_row_ascii(r) for r in g] for g in groups]
    all_fmt = [fmt_baseline] + [r for g in fmt_groups for r in g]
    widths = cell_widths(HEADERS, all_fmt)
    total = sum(widths) + 3 * len(widths) + 1

    out = [title_bar(model, total),
           hline(widths, "├", "┬", "┤"),
           render_row_ascii(HEADERS, widths),
           hline(widths, "├", "┼", "┤"),
           render_row_ascii(fmt_baseline, widths)]
    for fmt_group in fmt_groups:
        out.append(hline(widths, "├", "┼", "┤"))
        for cells in fmt_group:
            out.append(render_row_ascii(cells, widths))
    out.append(hline(widths, "└", "┴", "┘"))
    return "\n".join(out)


# ── CSV writing ─────────────────────────────────────────────────────────────
def write_csv(model, baseline_row, pruned_rows):
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    with (CSV_DIR / f"{model}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADERS)
        for row in [baseline_row] + pruned_rows:
            w.writerow(["" if v is None else v for v in row])


# ── LaTeX rendering ─────────────────────────────────────────────────────────
def latex_num(v, spec, na="--"):
    if v is None or v == "":
        return na
    try:
        if spec.endswith("d"):
            return format(int(float(v)), spec)
        return format(float(v), spec)
    except (ValueError, TypeError):
        return na


def latex_spd(v):
    s = latex_num(v, ".2f")
    return s + r"$\times$" if s != "--" else s


def latex_method(name):
    if name == "baseline":
        return "baseline"
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def latex_row(target_cell, row):
    tgt, imp, acc, dacc, lat, sp_tpu, sp_cpu, red_p, red_m, sz, on, off = row
    cells = [
        target_cell,
        latex_method(imp),
        latex_num(acc, ".2f"),
        latex_num(dacc, "+.2f"),
        latex_num(lat, ".2f"),
        latex_spd(sp_tpu),
        latex_spd(sp_cpu),
        latex_num(red_p, ".1f"),
        latex_num(red_m, ".1f"),
        latex_num(sz, ".2f"),
        latex_num(on, ".2f"),
        latex_num(off, ".2f"),
    ]
    return " & ".join(cells) + r" \\"


def render_table_latex(model, baseline_row, pruned_rows):
    out = [
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption{{{DISPLAY.get(model, model)} pruning benchmark (INT8).}}",
        rf"  \label{{tab:bench-{model}}}",
        r"  \centerline{\resizebox{\benchscale\linewidth}{!}{%",
        r"  \begin{tabular}{c l rr rrr rr rrr}",
        r"    \toprule",
        r"    & & \multicolumn{2}{c}{Accuracy} & \multicolumn{3}{c}{Time} "
        r"& \multicolumn{2}{c}{Reduction (\%)} & \multicolumn{3}{c}{Memory (mb)} \\",
        r"    \cmidrule(lr){3-4} \cmidrule(lr){5-7} "
        r"\cmidrule(lr){8-9} \cmidrule(lr){10-12}",
        r"    Target & Importance & Acc & $\Delta$ Acc "
        r"& Lat (ms) & TPU spd & CPU spd "
        r"& Params & MACs "
        r"& Size & On Chip & Off Chip \\",
        r"    \midrule",
        "    " + latex_row("--", baseline_row),
        r"    \midrule",
    ]

    groups = [(tgt, list(g)) for tgt, g in groupby(pruned_rows, key=lambda r: r[0])]
    for gi, (tgt, glist) in enumerate(groups):
        n = len(glist)
        for i, r in enumerate(glist):
            left = rf"\multirow{{{n}}}{{*}}{{{tgt}\%}}" if i == 0 else ""
            out.append("    " + latex_row(left, r))
        if gi < len(groups) - 1:
            out.append(r"    \midrule")

    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}}}")
    out.append(r"\end{table}")
    return "\n".join(out)


INDEX_PREAMBLE = r"""% Auto-generated by make_tables.py — do not edit by hand.
% Required packages: booktabs, graphicx, multirow, float.
%
% Per-model tables are split into tables/<model>.tex and \input'd below.
% Scale knob: redefine \benchscale to enlarge/shrink every table.
\providecommand{\benchscale}{1.0}
"""


def render_baseline_speedup_latex(models: dict) -> str:
    out = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Baseline TPU vs CPU latency (INT8).}",
        r"  \label{tab:baseline-speedup}",
        r"  \centerline{%",
        r"  \begin{tabular}{l r rrr}",
        r"    \toprule",
        r"    Model & Size (MiB) & CPU INT8 (ms) & TPU INT8 (ms)"
        r" & Speedup TPU/CPU \\",
        r"    \midrule",
    ]
    for model in MODEL_ORDER:
        base = models.get(f"{model}_finetuned")
        if not base:
            continue
        sz  = base.get("size_int8_mib")
        cpu = base.get("lat_cpu_int8_ms_mean")
        tpu = base.get("lat_tpu_int8_ms_mean")
        sp  = base.get("tpu_speedup_int8")
        cells = [
            DISPLAY.get(model, model),
            latex_num(sz, ".2f"),
            latex_num(cpu, ".2f"),
            latex_num(tpu, ".2f"),
            latex_spd(sp),
        ]
        out.append("    " + " & ".join(cells) + r" \\")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}}")
    out.append(r"\end{table}")
    return "\n".join(out)


def render_quant_speedup_latex(models: dict) -> str:
    out = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Baseline CPU quantization (FP32 vs INT8).}",
        r"  \label{tab:quant-speedup}",
        r"  \centerline{%",
        r"  \begin{tabular}{l rrrr}",
        r"    \toprule",
        r"    Model & CPU FP32 (ms) & CPU INT8 (ms) & Speedup INT8/FP32"
        r" & $\Delta$ Acc \\",
        r"    \midrule",
    ]
    for model in MODEL_ORDER:
        base = models.get(f"{model}_finetuned")
        if not base:
            continue
        f32  = base.get("lat_cpu_f32_ms_mean")
        i8   = base.get("lat_cpu_int8_ms_mean")
        sp   = base.get("quant_speedup_cpu")
        dacc = base.get("quant_drop_top1")
        cells = [
            DISPLAY.get(model, model),
            latex_num(f32, ".2f"),
            latex_num(i8, ".2f"),
            latex_spd(sp),
            latex_num(dacc, "+.2f"),
        ]
        out.append("    " + " & ".join(cells) + r" \\")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}}")
    out.append(r"\end{table}")
    return "\n".join(out)


# ── Driver ──────────────────────────────────────────────────────────────────
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    d = json.loads(JSON_PATH.read_text())
    models = d["models"]

    index_lines = [INDEX_PREAMBLE]
    n_models = 0

    for model in MODEL_ORDER:
        base = models.get(f"{model}_finetuned")
        if not base:
            print(f"[skip] no baseline for {model}")
            continue
        base_acc = base.get("top1_int8_pct")
        base_size = base.get("size_int8_mib")

        baseline_row = [
            None,
            "baseline",
            base_acc,
            0.0,
            base.get("lat_tpu_int8_ms_mean"),
            1.0,
            1.0,
            0.0,
            0.0,
            base_size,
            base.get("tpu_on_chip_mib"),
            base.get("tpu_off_chip_mib"),
        ]

        pruned_rows = []
        seen = {}  # (importance, param_reduction_pct) -> dedupe
        for pct in PCT_ORDER:
            for imp in IMP_ORDER:
                m = models.get(f"{model}_pruned{pct}pct_{imp}")
                if not m:
                    continue
                red = m.get("param_reduction_pct")
                key = (imp, red)
                if key in seen:
                    continue
                seen[key] = True
                pacc = m.get("top1_int8_pct")
                dacc = (pacc - base_acc) if (pacc is not None and base_acc is not None) else None
                pruned_rows.append([
                    pct,
                    imp,
                    pacc,
                    dacc,
                    m.get("lat_tpu_int8_ms_mean"),
                    m.get("prune_speedup_tpu"),
                    m.get("prune_speedup_cpu"),
                    red,
                    m.get("macs_reduction_pct"),
                    m.get("size_int8_mib"),
                    m.get("tpu_on_chip_mib"),
                    m.get("tpu_off_chip_mib"),
                ])
        if not pruned_rows:
            print(f"[skip] no pruned rows for {model}")
            continue

        print(render_table_ascii(model, baseline_row, pruned_rows))
        print()

        write_csv(model, baseline_row, pruned_rows)

        tex = render_table_latex(model, baseline_row, pruned_rows)
        (TEX_DIR / f"{model}.tex").write_text(tex + "\n")
        index_lines.append(rf"\input{{tables/{model}.tex}}")
        n_models += 1

    # Speedup summary tables (one row per model, no multirow).
    bs = render_baseline_speedup_latex(models)
    (TEX_DIR / "baseline_speedup.tex").write_text(bs + "\n")
    index_lines.append(r"\input{tables/baseline_speedup.tex}")

    qs = render_quant_speedup_latex(models)
    (TEX_DIR / "quant_speedup.tex").write_text(qs + "\n")
    index_lines.append(r"\input{tables/quant_speedup.tex}")

    INDEX_TEX.write_text("\n\n".join(index_lines) + "\n")
    print(f"wrote {INDEX_TEX} ({n_models} per-model tables + 2 speedup tables)")


if __name__ == "__main__":
    main()

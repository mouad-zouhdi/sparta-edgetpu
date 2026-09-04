#!/usr/bin/env python3
"""Baseline CPU INT8 vs TPU INT8 speedup table."""
import csv
import json
from pathlib import Path

JSON_PATH = Path("/home/a131/Desktop/Project/benchmark_results.json")
OUT_PATH = Path("/home/a131/Desktop/Project/work/baseline_speedup.txt")
CSV_PATH = Path("/home/a131/Desktop/Project/work/csv/baseline_speedup.csv")
MODEL_ORDER = ["resnet18", "resnet50", "vgg19", "wrn_28_10",
               "mobilenetv2", "googlenet", "squeezenet1_1"]

HEADERS = ["Model", "CPU INT8 (ms)", "TPU INT8 (ms)", "Speedup TPU/CPU"]
CSV_HEADERS = ["model", "cpu_int8_ms", "tpu_int8_ms", "speedup_tpu_cpu"]


def fmt(value, spec, na="-"):
    if value is None:
        return na
    return format(value, spec)


def cell_widths(headers, rows):
    return [max(len(h), max((len(r[i]) for r in rows), default=0))
            for i, h in enumerate(headers)]


def hline(widths, left, mid, right, fill="─"):
    return left + mid.join(fill * (w + 2) for w in widths) + right


def render_row(cells, widths, sep="│"):
    return sep + sep.join(f" {c:<{w}} " for c, w in zip(cells, widths)) + sep


def title_bar(title, total_width):
    inner = total_width - 2
    return (
        "┌" + "─" * inner + "┐\n"
        "│" + title.center(inner) + "│"
    )


def main():
    d = json.loads(JSON_PATH.read_text())
    models = d["models"]
    rows = []
    raw = []
    for model in MODEL_ORDER:
        base = models.get(f"{model}_finetuned")
        if not base:
            continue
        cpu = base.get("lat_cpu_int8_ms_mean")
        tpu = base.get("lat_tpu_int8_ms_mean")
        sp = base.get("tpu_speedup_int8")
        rows.append([
            model,
            fmt(cpu, ".2f"),
            fmt(tpu, ".2f"),
            fmt(sp, ".2f") + "×",
        ])
        raw.append([model, cpu, tpu, sp])

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADERS)
        for r in raw:
            w.writerow(["" if v is None else v for v in r])

    widths = cell_widths(HEADERS, rows)
    total = sum(widths) + 3 * len(widths) + 1

    out = []
    out.append(title_bar("Baseline INT8 — CPU vs TPU", total))
    out.append(hline(widths, "├", "┬", "┤"))
    out.append(render_row(HEADERS, widths))
    out.append(hline(widths, "├", "┼", "┤"))
    for r in rows:
        out.append(render_row(r, widths))
    out.append(hline(widths, "└", "┴", "┘"))
    text = "\n".join(out)
    print(text)
    OUT_PATH.write_text(text + "\n")


if __name__ == "__main__":
    main()

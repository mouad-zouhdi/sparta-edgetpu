#!/usr/bin/env python3
"""
03_compile_edgetpu.py — compile INT8 TFLite models for the Edge TPU.

WHAT THIS PRODUCES
    edgetpu_compiled/<name>_edgetpu.tflite   the binary the accelerator runs
    compiler_metrics.json                    parsed compiler output, per model
    compilation_report.txt                   the raw logs, kept for auditing

WHY THE COMPILER METRICS MATTER AS MUCH AS THE BINARY
    The Edge TPU has roughly 8 MB of on-chip SRAM. A model whose parameters fit
    entirely in it is read once and then reused; a model that does not fit has
    the excess streamed over the host bus on EVERY inference. That single
    distinction dominates latency far more than the operation count does, and it
    is the reason a pruning gain measured in MACs so often fails to materialise
    on the device.

    The compiler reports the split explicitly, and this script parses it into:
        on_chip_mib            parameters cached in SRAM
        on_chip_remaining_mib  SRAM left over, i.e. headroom before streaming
        off_chip_mib           parameters streamed on every inference
        num_subgraphs          Edge TPU subgraph count
        total_ops / ops mapped how much of the graph runs on the accelerator

    off_chip_mib == 0 is the property everything downstream keys on: it is the
    boundary between the two memory regimes, and the multi-TPU pipeline exists
    precisely to reach it by pooling the SRAM of several accelerators.

HOW IT WORKS
    Shells out to `edgetpu_compiler`, then parses its stdout with regular
    expressions, since the tool offers no machine-readable output. Sizes are
    normalised to MiB because the compiler switches units (B, KiB, MiB) depending
    on magnitude, which silently breaks any naive numeric comparison.

REQUIREMENTS
    edgetpu_compiler v16.0, x86-64 Linux only. Note that its timeout flag is
    spelled --timeout_sec with an underscore; with a hyphen the binary rejects
    the option in a way that is easy to miss in a batch log.

USAGE (coral-env, or any shell with edgetpu_compiler on PATH)
    python 03_compile_edgetpu.py
    python 03_compile_edgetpu.py --models mobilenetv2
    python 03_compile_edgetpu.py --tflite-dir tflite_int8 --output-dir edgetpu_compiled
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEFAULT_TFLITE_DIR = BASE_DIR / "tflite_int8"
DEFAULT_OUTPUT_DIR = BASE_DIR / "edgetpu_compiled"
DEFAULT_REPORT_FILE = BASE_DIR / "compilation_report.txt"
DEFAULT_METRICS_FILE = BASE_DIR / "compiler_metrics.json"

# Reassigned by main() once the CLI arguments are parsed
TFLITE_DIR = DEFAULT_TFLITE_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
REPORT_FILE = DEFAULT_REPORT_FILE
METRICS_FILE = DEFAULT_METRICS_FILE


def parse_size_to_mib(value_str, unit_str):
    """Normalise a compiler size to MiB.

    The compiler prints B, KiB, MiB or GiB depending on magnitude, so comparing the
    raw numbers across models would silently mix units.
    """
    val = float(value_str)
    unit = unit_str.strip()
    if unit == "B":
        return val / (1024 * 1024)
    elif unit in ("KiB", "KB"):
        return val / 1024
    elif unit in ("MiB", "MB"):
        return val
    elif unit in ("GiB", "GB"):
        return val * 1024
    return val


def parse_stdout(stdout: str) -> dict:
    """Extract the memory split and operation counts from the compiler's stdout.

    There is no machine-readable output mode, so this parses the human-readable
    report. Every field defaults to 0 and is only overwritten on a regex match, so a
    change in the compiler's wording degrades to a zero rather than to a crash;
    check compilation_report.txt if the metrics look implausibly empty.
    """
    metrics = {
        "on_chip_mib": 0.0, "on_chip_remaining_mib": 0.0,
        "off_chip_mib": 0.0, "num_subgraphs": 0, "total_ops": 0,
    }

    m = re.search(
        r"On-chip memory used for caching model parameters:\s*([\d.]+)\s*(\S+)",
        stdout)
    if m:
        metrics["on_chip_mib"] = parse_size_to_mib(m.group(1), m.group(2))

    m = re.search(
        r"On-chip memory remaining for caching model parameters:\s*([\d.]+)\s*(\S+)",
        stdout)
    if m:
        metrics["on_chip_remaining_mib"] = parse_size_to_mib(m.group(1), m.group(2))

    m = re.search(
        r"Off-chip memory used for streaming uncached model parameters:\s*([\d.]+)\s*(\S+)",
        stdout)
    if m:
        metrics["off_chip_mib"] = parse_size_to_mib(m.group(1), m.group(2))

    m = re.search(r"Number of Edge TPU subgraphs:\s*(\d+)", stdout)
    if m:
        metrics["num_subgraphs"] = int(m.group(1))

    m = re.search(r"Total number of operations:\s*(\d+)", stdout)
    if m:
        metrics["total_ops"] = int(m.group(1))

    return metrics


def parse_log(log_path: Path) -> dict:
    """Parse a saved compiler log file, for re-reading results without recompiling."""
    ops_tpu, ops_cpu = 0, 0
    details = []
    if not log_path.exists():
        return {"ops_tpu": 0, "ops_cpu": 0, "details": []}
    with open(log_path) as f:
        for line in f:
            m = re.match(r"(\S+)\s+(\d+)\s+(.*)", line.strip())
            if m and m.group(1) != "Operator":
                op, count, status = m.group(1), int(m.group(2)), m.group(3).strip()
                if "Mapped to Edge TPU" in status:
                    ops_tpu += count
                    details.append((op, count, "TPU"))
                else:
                    ops_cpu += count
                    details.append((op, count, "CPU"))
    return {"ops_tpu": ops_tpu, "ops_cpu": ops_cpu, "details": details}


def compile_for_edgetpu(tflite_path: Path, skip_existing: bool = True) -> dict:
    """Compile one INT8 TFLite model and return its parsed metrics.

    Returns None on failure rather than raising: several models in the sweep are
    expected to be rejected (large activation tensors, unsupported operations), and
    a batch run must record those and continue.
    """
    expected = OUTPUT_DIR / (tflite_path.stem + "_edgetpu.tflite")
    if skip_existing and expected.exists():
        return {"model": tflite_path.stem, "skipped": True}
    result = {
        "model": tflite_path.stem, "success": False, "skipped": False,
        "ops_tpu": 0, "ops_cpu": 0, "details": [],
        "on_chip_mib": 0, "off_chip_mib": 0,
        "on_chip_remaining_mib": 0, "num_subgraphs": 0, "total_ops": 0,
        "log": "",
    }
    cmd = ["edgetpu_compiler", "--out_dir", str(OUTPUT_DIR), str(tflite_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        stdout = proc.stdout + proc.stderr
        result["log"] = stdout
        result.update(parse_stdout(stdout))

        expected = OUTPUT_DIR / (tflite_path.stem + "_edgetpu.tflite")
        log_file = OUTPUT_DIR / (tflite_path.stem + "_edgetpu.log")
        if expected.exists():
            result["success"] = True
            ops = parse_log(log_file)
            result["ops_tpu"] = ops["ops_tpu"]
            result["ops_cpu"] = ops["ops_cpu"]
            result["details"] = ops["details"]
    except FileNotFoundError:
        result["log"] = "edgetpu_compiler introuvable"
    except subprocess.TimeoutExpired:
        result["log"] = "Timeout (>300s)"
    return result


def main():
    """Discover the INT8 models, compile each, and write the metrics and report."""
    parser = argparse.ArgumentParser(description="SPARTA — Compilation Edge TPU")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filtre par nom de base (correspondance partielle)")
    parser.add_argument("--tflite-dir", type=Path, default=DEFAULT_TFLITE_DIR,
                        help=f"Dossier source des .tflite int8 (def: {DEFAULT_TFLITE_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Dossier de sortie des .tflite Edge TPU (def: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--report-file", type=Path, default=None,
                        help="Fichier rapport texte (def: <output-dir>/../compilation_report.txt)")
    parser.add_argument("--metrics-file", type=Path, default=None,
                        help="JSON metrics file (default: <output-dir>/../compiler_metrics.json)")
    parser.add_argument("--force", action="store_true",
                        help="Recompile even when the _edgetpu.tflite output already exists.")
    args = parser.parse_args()

    global TFLITE_DIR, OUTPUT_DIR, REPORT_FILE, METRICS_FILE
    TFLITE_DIR = args.tflite_dir
    OUTPUT_DIR = args.output_dir
    REPORT_FILE = args.report_file if args.report_file is not None \
        else OUTPUT_DIR.parent / "compilation_report.txt"
    METRICS_FILE = args.metrics_file if args.metrics_file is not None \
        else OUTPUT_DIR.parent / "compiler_metrics.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPARTA — COMPILATION EDGE TPU")
    print("=" * 70)
    print(f"Source : {TFLITE_DIR}")
    print(f"Sortie : {OUTPUT_DIR}")
    print(f"Rapport: {REPORT_FILE}")
    print(f"Metrics: {METRICS_FILE}\n")

    if not TFLITE_DIR.exists():
        print(f"[ERREUR] {TFLITE_DIR} introuvable."); return

    files = sorted(TFLITE_DIR.glob("*.tflite"))
    if args.models:
        files = [f for f in files
                 if any(m in f.stem for m in args.models)]

    if not files:
        print("[ERROR] No .tflite file found."); return

    print(f"{len(files)} models to compile.\n")
    reports = []
    all_metrics = {}

    n_skipped = 0
    for f in files:
        print(f"--- {f.name} ---")
        r = compile_for_edgetpu(f, skip_existing=not args.force)
        if r.get("skipped"):
            print(f"  [skip] {f.stem}_edgetpu.tflite already exists (use --force to recompile)")
            n_skipped += 1
            continue
        reports.append(r)
        if r["success"]:
            total = r["ops_tpu"] + r["ops_cpu"]
            pct = 100 * r["ops_tpu"] / total if total else 0
            print(f"  Ops: {r['ops_tpu']} TPU / {r['ops_cpu']} CPU ({pct:.0f}% TPU)")
            print(f"  SRAM: {r['on_chip_mib']:.2f} MiB used, "
                  f"{r['on_chip_remaining_mib']*1024:.1f} KiB free")
            print(f"  Off-chip streaming: {r['off_chip_mib']:.2f} MiB")
            print(f"  Subgraphs: {r['num_subgraphs']}")

            model_key = r["model"].replace("_int8", "")
            all_metrics[model_key] = {
                "on_chip_mib": r["on_chip_mib"],
                "on_chip_remaining_mib": r["on_chip_remaining_mib"],
                "off_chip_mib": r["off_chip_mib"],
                "num_subgraphs": r["num_subgraphs"],
                "ops_tpu": r["ops_tpu"],
                "ops_cpu": r["ops_cpu"],
                "total_ops": r["total_ops"],
            }
        else:
            print(f"  [FAILED] {r['log'][:200]}")

    # Summary table
    header = (f"{'Model':<55s} {'TPU':>5s} {'CPU':>5s} "
              f"{'%TPU':>5s} {'SRAM':>8s} {'Stream':>8s} {'SubG':>5s}")
    sep = "-" * len(header)
    print(f"\n{'='*70}\nRAPPORT\n{'='*70}")
    print(header); print(sep)

    with open(REPORT_FILE, "w") as log:
        log.write("SPARTA — Edge TPU Compilation Report\n" + header + "\n" + sep + "\n")
        for r in reports:
            total = r["ops_tpu"] + r["ops_cpu"]
            pct = 100 * r["ops_tpu"] / total if total else 0
            line = (f"{r['model']:<55s} "
                    f"{r['ops_tpu']:>5d} {r['ops_cpu']:>5d} "
                    f"{pct:>4.0f}% "
                    f"{r['on_chip_mib']:>7.2f} {r['off_chip_mib']:>7.2f} "
                    f"{r['num_subgraphs']:>5d}")
            print(line); log.write(line + "\n")

    existing = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            existing = json.load(f)
    existing.update(all_metrics)

    with open(METRICS_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n→ {REPORT_FILE}\n→ {METRICS_FILE}")
    if n_skipped:
        print(f"({n_skipped} file(s) skipped, already compiled. Use --force to recompile.)")


if __name__ == "__main__":
    main()

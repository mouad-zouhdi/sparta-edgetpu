#!/usr/bin/env python3
"""
03_compile_edgetpu_segments.py — compile one INT8 model across N Edge TPUs.

WHAT THIS PRODUCES
    <output_dir>/<basename>_segments/  the N compiled segment binaries
    <report_out>                       a structured compile_report.json

    The report is the point of this script. edgetpu_compiler prints a
    human-readable summary and offers no machine-readable output, so this parses
    its stdout into per-segment figures that pipeline_full.py can act on.

WHY SEGMENTATION EXISTS
    A single Edge TPU holds roughly 8 MB of parameters in SRAM. Anything beyond
    that is streamed from host memory on EVERY inference, which costs far more
    than the arithmetic does. `--num_segments N` splits the model across N
    accelerators, pooling their SRAM: a model too large for one TPU can be held
    entirely on-chip across four.

    The decisive field is therefore totals.off_chip_used_mb. Zero means nothing
    is streamed and the model is in the fast regime; anything above zero means
    the excess is re-transferred on every inference. pipeline_full.py treats
    "off_chip == 0" as its fit criterion, exactly, not approximately.

A PROPERTY WORTH KNOWING: off_chip IS NOT MONOTONIC IN N
    More segments does not guarantee a better fit. One ResNet-101 checkpoint fits
    at 6 segments, overflows by 2.02 MiB at 7, and fits again at 8. The
    compiler's segmentation heuristic balances the split on some other criterion
    and does not optimise for fitting, so a failure at N does not rule out N+1,
    and a success at N does not imply one at N+1.

REPORT FORMAT
    {
      "input": ..., "output_dir": ...,
      "num_segments_requested": 4, "num_segments_produced": 4,
      "compile_success": true, "compiler_exit_code": 0,
      "segments": [
         {"idx": 0, "input_mb": 3.28, "output_mb": 3.69,
          "on_chip_used_mb": 3.54, "on_chip_remaining_mb": 3.05,
          "off_chip_used_mb": 0.0, "num_subgraphs": 1, "total_ops": 93,
          "ops_edgetpu": 89, "ops_cpu": 4, "compile_ms": 1063},
         ...
      ],
      "totals": {...}
    }

    ops_cpu is reported but does not by itself mark a failure. Four CPU
    operations at the tail (softmax and reshape) are normal; a much larger count
    signals something worth investigating.

REQUIREMENTS
    edgetpu_compiler on PATH. Note its timeout flag is --timeout_sec, with an
    underscore; with a hyphen the binary rejects the option in a way that is easy
    to miss in a batch log. On a cluster without root, unpack the .deb into
    $HOME/local and put $HOME/local/usr/bin on PATH from the job script.

USAGE
    python 03_compile_edgetpu_segments.py \\
        --input tflite_int8/foo_int8.tflite \\
        --num_segments 4 \\
        --output_dir edgetpu_compiled/ \\
        --report_out foo_compile_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ─────────────────────────────────────────────
# Parsing the compiler's stdout
# ─────────────────────────────────────────────
_UNIT_TO_MB = {"B": 1 / (1024 * 1024), "KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}


def _to_mb(val_str: str) -> float:
    """Parse '3.28MiB', '199.19KiB', '0.00B', '1.02GiB' → MB float."""
    m = re.match(r"([\d.]+)\s*(B|KiB|MiB|GiB)", val_str.strip())
    if not m:
        return 0.0
    return float(m.group(1)) * _UNIT_TO_MB[m.group(2)]


def parse_compile_stdout(stdout: str, num_segments_requested: int) -> dict:
    """Parse the compiler's stdout into per-segment metrics and their totals.

    edgetpu_compiler has no machine-readable output, so this reads the human-readable
    report with regular expressions. It also checks that the number of segments
    produced matches the number requested: the compiler can silently emit fewer, and
    a pipeline built on the wrong segment count would run but measure the wrong
    thing.

    The field that decides everything downstream is off_chip_used_mb, summed over
    segments: zero means the model is held entirely in the pooled SRAM, anything
    else means the excess is re-streamed on every inference.
    """
    # Split par bloc "Model compiled successfully" : 1 bloc = 1 segment
    # (ou 1 seul bloc pour N=1)
    blocks = re.split(r"Model compiled successfully in (\d+) ms\.", stdout)
    segments = []
    seg_idx = 0
    for i in range(1, len(blocks), 2):
        try:
            compile_ms = int(blocks[i])
        except (ValueError, IndexError):
            compile_ms = -1
        chunk = blocks[i + 1] if (i + 1) < len(blocks) else ""

        def _find(regex, cast=str, default=None):
            """Pull one regex group out of the current segment chunk, with a default.

            Every field defaults rather than raising, so a change in the compiler wording
            degrades to a missing value instead of a crash mid-campaign.
            """
            m = re.search(regex, chunk)
            if not m:
                return default
            return cast(m.group(1).strip())

        seg = {
            "idx": seg_idx,
            "compile_ms": compile_ms,
            "input_model": _find(r"Input model:\s*(\S+)", str, ""),
            "input_mb": _find(r"Input size:\s*([\d.]+\s*\w+)", _to_mb, 0.0),
            "output_model": _find(r"Output model:\s*(\S+)", str, ""),
            "output_mb": _find(r"Output size:\s*([\d.]+\s*\w+)", _to_mb, 0.0),
            "on_chip_used_mb":
                _find(r"On-chip memory used for caching model parameters:\s*([\d.]+\s*\w+)",
                      _to_mb, 0.0),
            "on_chip_remaining_mb":
                _find(r"On-chip memory remaining for caching model parameters:\s*([\d.]+\s*\w+)",
                      _to_mb, 0.0),
            "off_chip_used_mb":
                _find(r"Off-chip memory used for streaming uncached model parameters:\s*([\d.]+\s*\w+)",
                      _to_mb, 0.0),
            "num_subgraphs":
                _find(r"Number of Edge TPU subgraphs:\s*(\d+)", int, 0),
            "total_ops":
                _find(r"Total number of operations:\s*(\d+)", int, 0),
            "ops_edgetpu":
                _find(r"Number of operations that will run on Edge TPU:\s*(\d+)",
                      int, 0),
            "ops_cpu":
                _find(r"Number of operations that will run on CPU:\s*(\d+)",
                      int, 0),
        }
        # When no "ops_edgetpu" line appears, the whole graph mapped to the TPU:
        # ops_edgetpu == total_ops and ops_cpu == 0.
        if seg["ops_edgetpu"] == 0 and seg["ops_cpu"] == 0 and seg["total_ops"] > 0:
            seg["ops_edgetpu"] = seg["total_ops"]
        segments.append(seg)
        seg_idx += 1

    # Totals
    totals = {
        "on_chip_used_mb": sum(s["on_chip_used_mb"] for s in segments),
        "off_chip_used_mb": sum(s["off_chip_used_mb"] for s in segments),
        "ops_edgetpu": sum(s["ops_edgetpu"] for s in segments),
        "ops_cpu": sum(s["ops_cpu"] for s in segments),
    }
    return {
        "num_segments_requested": num_segments_requested,
        "num_segments_produced": len(segments),
        "segments": segments,
        "totals": totals,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    """Compile one INT8 model into N segments and write the structured JSON report."""
    p = argparse.ArgumentParser(
        description="Compile an INT8 model into N Edge TPU segments and report the result.")
    p.add_argument("--input", required=True,
                   help="The INT8 .tflite to compile")
    p.add_argument("--num_segments", type=int, required=True,
                   help="Nombre de segments Edge TPU (1, 2, 4, 8...)")
    p.add_argument("--output_dir", required=True,
                   help="Directory where edgetpu_compiler writes the *_edgetpu.tflite segments")
    p.add_argument("--report_out", required=True,
                   help="Path of the JSON report to write")
    p.add_argument("--timeout_sec", type=int, default=600,
                   help="Compilation timeout in seconds (default 600). Note the compiler spells\n                        this flag --timeout_sec, with an underscore.")
    p.add_argument("--compiler_bin", default="edgetpu_compiler",
                   help="Explicit path to the compiler binary (default: found on PATH)")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[ERROR] input tflite not found: {in_path}")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [args.compiler_bin,
           "-o", str(out_dir),
           "-n", str(args.num_segments),
           "-t", str(args.timeout_sec),
           str(in_path)]
    print(f"[compile] {' '.join(cmd)}", flush=True)

    t0 = time.time()
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=args.timeout_sec + 30,
        )
        stdout = res.stdout
        stderr = res.stderr
        exit_code = res.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if e.stdout else ""
        stderr = e.stderr.decode() if e.stderr else ""
        exit_code = -1
        print(f"[compile] TIMEOUT after {args.timeout_sec}s", flush=True)

    duration_s = time.time() - t0
    print(stdout)
    if stderr:
        print("STDERR:", stderr, file=sys.stderr)

    # Parsing
    parsed = parse_compile_stdout(stdout, args.num_segments)
    compile_success = (
        exit_code == 0
        and parsed["num_segments_produced"] == args.num_segments
        and "Compilation succeeded" in stdout
    )

    report = {
        "input": str(in_path),
        "output_dir": str(out_dir),
        "compile_success": compile_success,
        "compiler_exit_code": exit_code,
        "compiler_duration_s": duration_s,
        "compiler_stderr": stderr[-2000:] if stderr else "",
        **parsed,
    }

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Summary console
    print(f"\n[compile_report] → {report_path}", flush=True)
    print(f"  success={compile_success}, segments_produced="
          f"{parsed['num_segments_produced']}/{args.num_segments}", flush=True)
    for s in parsed["segments"]:
        print(f"    seg {s['idx']}: on_chip={s['on_chip_used_mb']:.2f} MB, "
              f"off_chip={s['off_chip_used_mb']:.3f} MB, "
              f"ops_edgetpu={s['ops_edgetpu']}, ops_cpu={s['ops_cpu']}", flush=True)
    print(f"  TOTAL: on_chip={parsed['totals']['on_chip_used_mb']:.2f} MB, "
          f"off_chip={parsed['totals']['off_chip_used_mb']:.3f} MB", flush=True)

    # Exit non-zero when the model does not fit or the compile failed, so the
    # caller can branch on it; the report holds the reason either way.
    sys.exit(0 if compile_success else 1)


if __name__ == "__main__":
    main()

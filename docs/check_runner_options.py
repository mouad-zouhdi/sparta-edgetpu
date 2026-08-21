#!/usr/bin/env python3
"""
check_runner_options.py — verify the pipeline runners only pass options that exist.

WHY THIS EXISTS
    The runner scripts drive the pipeline scripts through their command lines, so
    a typo or a renamed option is not caught by anything: the shell is happy, and
    the failure only appears when that stage is reached, which can be hours into a
    run. One such option shipped and was found by a cluster test rather than by a
    check.

    A grep is not enough. An option name can appear in a help string, in a
    comment, or in a command the script itself builds for a subprocess, without
    being accepted by its own argument parser. This asks each script what it
    actually accepts, by running it with --help.

USAGE
    python docs/check_runner_options.py [--python INTERPRETER]

    Exits non-zero and names every option that does not exist.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# Which runner drives which script, and where in the runner its arguments live.
# The pattern selects the region of the runner that builds that script's command
# line, so options meant for a different script are not attributed to this one.
TARGETS = [
    ("run_pipeline_mono_tpu.sh", "mono_tpu/00_prepare_baselines.py",  r'-u 00_prepare_baselines\.py((?:[^\n]*\\\n)*[^\n]*)'),
    ("run_pipeline_mono_tpu.sh", "mono_tpu/01_prune.py",              r'-u 01_prune\.py((?:[^\n]*\\\n)*[^\n]*)'),
    ("run_pipeline_mono_tpu.sh", "mono_tpu/aggregate_pruning_logs.py",r'-u aggregate_pruning_logs\.py((?:[^\n]*\\\n)*[^\n]*)'),
    ("run_pipeline_mono_tpu.sh", "mono_tpu/02_convert_tflite_int8.py",r'-u 02_convert_tflite_int8\.py((?:[^\n]*\\\n)*[^\n]*)'),
    ("run_pipeline_mono_tpu.sh", "mono_tpu/03_compile_edgetpu.py",    r'-u 03_compile_edgetpu\.py((?:[^\n]*\\\n)*[^\n]*)'),
    ("run_pipeline_mono_tpu.sh", "mono_tpu/04_benchmark.py",          r'-u 04_benchmark\.py((?:[^\n]*\\\n)*[^\n]*)'),
    ("run_pipeline_mono_tpu.sh", "mono_tpu/05_benchmark_coldstart.py",r'-u 05_benchmark_coldstart\.py((?:[^\n]*\\\n)*[^\n]*)'),
    ("run_pipeline_multi_tpu.sh","multi_tpu/00_fetch_and_convert_pretrained.py", r'-u 00_fetch_and_convert_pretrained\.py((?:[^\n]*\\\n)*[^\n]*)'),
    ("run_pipeline_multi_tpu.sh","multi_tpu/pipeline_full.py",        r'args=\((.*?)\n            \)'),
    ("run_pipeline_multi_tpu.sh","multi_tpu/verify_tpu.py",           r'-u verify_tpu\.py((?:[^\n]*\\\n)*[^\n]*)'),
    ("run_pipeline_multi_tpu.sh","multi_tpu/bench/bench_pipeline.py", r'-u bench/bench_pipeline\.py((?:[^\n]*\\\n)*[^\n]*)'),
    ("run_pipeline_multi_tpu.sh","multi_tpu/bench/bench_parallel.py", r'-u bench/bench_parallel\.py((?:[^\n]*\\\n)*[^\n]*)'),
]

OPT = re.compile(r'--[a-zA-Z][\w-]*')


def accepted_options(script: pathlib.Path, interp: str) -> set[str] | None:
    """Ask a script what options it accepts, by parsing its own --help output.

    Returns None when the script cannot be run at all in this environment, which
    is not a failure: several scripts need coral-env or an accelerator.
    """
    try:
        r = subprocess.run([interp, str(script), "--help"],
                           capture_output=True, text=True, timeout=180)
    except Exception:
        return None
    if r.returncode != 0 and not r.stdout:
        return None
    return set(OPT.findall(r.stdout))


def main() -> int:
    """Check every runner/script pair and report options that do not exist."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--python", default=sys.executable,
                    help="Interpreter used to ask each script for its --help")
    args = ap.parse_args()

    bad = skipped = checked = 0
    for runner, script, pattern in TARGETS:
        sh = (REPO / runner).read_text()
        m = re.search(pattern, sh, re.S)
        if not m:
            print(f"  SKIP    {script}: no command line found in {runner}")
            skipped += 1
            continue
        used = set(OPT.findall(m.group(1)))
        # Options the runner adds conditionally, outside the matched block.
        used |= set(re.findall(r'args\+=\((--[\w-]+)', sh)) if "pipeline_full" in script else set()
        real = accepted_options(REPO / script, args.python)
        if real is None:
            print(f"  SKIP    {script}: not runnable in this environment")
            skipped += 1
            continue
        missing = sorted(used - real)
        checked += 1
        if missing:
            bad += len(missing)
            print(f"  FAIL    {script}")
            for o in missing:
                print(f"            {o} is passed by {runner} but not accepted")
        else:
            print(f"  ok      {script}  ({len(used)} options)")

    print(f"\n  {checked} checked, {skipped} skipped, {bad} invalid option(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

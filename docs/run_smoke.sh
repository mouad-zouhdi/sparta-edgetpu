#!/usr/bin/env bash
# ===========================================================================
# run_smoke.sh — parse, import and --help every script in the repository.
#
# Each script is checked in the environment it is meant to run in, because the
# whole point of having three environments is that a script importable in one is
# not importable in another.
#
# Set PYTORCH_PY / CORAL_PY to point at your own interpreters; the defaults
# assume ./envs, which is what setup/setup_envs.sh creates.
#
# Exit status is the number of failures, so this is usable in CI.
# ===========================================================================
set -uo pipefail

R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="${PYTORCH_PY:-$R/envs/pytorch-env/bin/python}"
C="${CORAL_PY:-$R/envs/coral-env/bin/python}"

for py in "$P" "$C"; do
  [ -x "$py" ] || { echo "ERROR: no interpreter at $py"; echo "Run setup/setup_envs.sh, or set PYTORCH_PY / CORAL_PY."; exit 2; }
done

pass=0; fail=0; declare -a FAILED

# check <label> <interpreter> <file> <yes|no: does it take --help?>
check() {
  local label="$1" py="$2" f="$3" want_help="$4"
  local dir; dir="$(dirname "$f")"
  if ! python3 -c "import ast;ast.parse(open('$f').read())" 2>/dev/null; then
    echo "  PARSE  FAIL  $label"; FAILED+=("$label:parse"); ((fail++)); return
  fi
  if ! (cd "$dir" && "$py" -c "
import sys, importlib.util
sys.path.insert(0, '.'); sys.path.insert(0, '$R/multi_tpu/bench')
s = importlib.util.spec_from_file_location('smoke_mod', '$(basename "$f")')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)" 2>/dev/null); then
    echo "  IMPORT FAIL  $label"; FAILED+=("$label:import"); ((fail++)); return
  fi
  if [ "$want_help" = yes ] && ! (cd "$dir" && timeout 180 "$py" "$(basename "$f")" --help >/dev/null 2>&1); then
    echo "  HELP   FAIL  $label"; FAILED+=("$label:help"); ((fail++)); return
  fi
  echo "  ok           $label"; ((pass++))
}

echo "=== mono_tpu (pytorch-env) ==="
for f in 00_prepare_baselines.py 01_prune.py 02_convert_tflite_int8.py aggregate_pruning_logs.py; do
  check "mono_tpu/$f" "$P" "$R/mono_tpu/$f" yes; done
for f in cifar_resnet.py cifar_vgg.py wrn.py; do
  check "mono_tpu/$f" "$P" "$R/mono_tpu/$f" no; done

echo "=== mono_tpu (coral-env) ==="
for f in 03_compile_edgetpu.py 04_benchmark.py 05_benchmark_coldstart.py; do
  check "mono_tpu/$f" "$C" "$R/mono_tpu/$f" yes; done

echo "=== multi_tpu (pytorch-env) ==="
check "multi_tpu/model_zoo.py" "$P" "$R/multi_tpu/model_zoo.py" no
for f in 00_fetch_and_convert_pretrained.py 01_prune_imagenet.py 02_convert_pruned.py pipeline_full.py; do
  check "multi_tpu/$f" "$P" "$R/multi_tpu/$f" yes; done

echo "=== multi_tpu (coral-env) ==="
for f in 03_compile_edgetpu_segments.py verify_tpu.py; do
  check "multi_tpu/$f" "$C" "$R/multi_tpu/$f" yes; done
check "multi_tpu/bench/bench_utils.py" "$C" "$R/multi_tpu/bench/bench_utils.py" no
for f in bench_pipeline.py bench_coldstart_pcie.py bench_coldstart_usb.py bench_latency_chained.py; do
  check "multi_tpu/bench/$f" "$C" "$R/multi_tpu/bench/$f" yes; done
# Needs pycoral and pandas in one interpreter; parsed unless such an env exists.
if "$C" -c "import pycoral, pandas" 2>/dev/null; then
  check "multi_tpu/bench/bench_parallel.py" "$C" "$R/multi_tpu/bench/bench_parallel.py" yes
elif python3 -c "import ast;ast.parse(open('$R/multi_tpu/bench/bench_parallel.py').read())" 2>/dev/null; then
  echo "  ok (parse)   multi_tpu/bench/bench_parallel.py [needs pycoral+pandas]"; ((pass++))
else
  echo "  PARSE  FAIL  multi_tpu/bench/bench_parallel.py"; FAILED+=("bench_parallel:parse"); ((fail++))
fi

echo "=== synthetic ==="
check "synthetic/onnx2tf_wrapper.py" "$P" "$R/synthetic/onnx2tf_wrapper.py" yes
# src/ is a package: relative imports, so import it as one.
if (cd "$R/synthetic" && "$P" -c "
from src.plan import all_configs
from src.blocks import get_block
from src.models import build_model, validate_forward
assert len(all_configs()) == 400" 2>/dev/null); then
  echo "  ok           synthetic/src (package: plan, blocks, models)"; ((pass++))
else
  echo "  IMPORT FAIL  synthetic/src (package)"; FAILED+=("synthetic/src:import"); ((fail++))
fi
for f in build_one.py generate_sweep.py; do
  check "synthetic/$f" "$P" "$R/synthetic/$f" yes; done
check "synthetic/compile_sweep.py" "$C" "$R/synthetic/compile_sweep.py" yes

echo "=== setup ==="
check "setup/apply_patches.py" "$P" "$R/setup/apply_patches.py" no
for f in stage_models.py fetch_models.py upload_models_hf.py; do
  check "setup/$f" "$P" "$R/setup/$f" yes; done

echo
echo "==================================================="
echo "  PASS: $pass    FAIL: $fail"
[ $fail -gt 0 ] && printf '  %s\n' "${FAILED[@]}"
exit $fail

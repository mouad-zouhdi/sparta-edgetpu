#!/usr/bin/env bash
# ===========================================================================
# run_pipeline_mono_tpu.sh — the complete single-accelerator pipeline.
#
# Trains CIFAR-100 baselines, prunes them under one or more criteria, quantizes
# to INT8, compiles for the Edge TPU, and benchmarks the result.
#
#   00  train baselines                 pytorch-env   GPU strongly advised
#   01  prune + recovery fine-tuning    pytorch-env   GPU strongly advised
#   --  aggregate the per-run logs      pytorch-env
#   02  convert to INT8 TFLite          pytorch-env
#   03  compile for the Edge TPU        coral-env     needs edgetpu_compiler
#   04  benchmark, steady state         coral-env     needs a Coral device for --device tpu
#   05  benchmark, first inference      coral-env     needs a Coral device for --device tpu
#
# USAGE
#   ./run_pipeline_mono_tpu.sh                 # every stage, with the settings below
#   ./run_pipeline_mono_tpu.sh --smoke         # a few minutes, to check the setup
#   ./run_pipeline_mono_tpu.sh --from 02       # resume at a stage
#   ./run_pipeline_mono_tpu.sh --only 04       # a single stage
#   ./run_pipeline_mono_tpu.sh --dry-run       # print the commands, run nothing
#
# CUSTOMISING
#   Edit the CONFIGURATION block below, or override any variable from the
#   environment without touching the file:
#
#     MODELS="resnet18 vgg19" CRITERIA="taylor random" ./run_pipeline_mono_tpu.sh
#
#   Every variable is documented where it is defined, with the values used for
#   the published runs and what to change them to.
# ===========================================================================
set -uo pipefail

# ===========================================================================
# CONFIGURATION — everything you are likely to change lives here.
# ===========================================================================

# --- Where things live -----------------------------------------------------

# Working directory: holds models/, pytorch_pruned/, pruning_logs/, onnx_models/,
# tflite_float32/, tflite_int8/ and edgetpu_compiled/.
# Point this somewhere with room to spare: a full sweep produces tens of GB, and
# leaving it inside the repository means committing it by accident.
WORK_DIR="${WORK_DIR:-$(pwd)/work_mono}"

# CIFAR-100 root: must contain the cifar-100-python/ subfolder. Stage 00
# downloads it here if it is missing.
DATA_DIR="${DATA_DIR:-$WORK_DIR/data}"

# Where the benchmark CSVs and JSONs are written.
RESULTS_DIR="${RESULTS_DIR:-$WORK_DIR/results}"

# Interpreters. The defaults assume setup/setup_envs.sh created ./envs.
# Set these if your environments are elsewhere.
PYTORCH_PY="${PYTORCH_PY:-$(pwd)/envs/pytorch-env/bin/python}"
CORAL_PY="${CORAL_PY:-$(pwd)/envs/coral-env/bin/python}"

# --- What to run -----------------------------------------------------------

# Architectures. The full lineup is:
#   resnet18 resnet50 vgg19 wrn_28_10 mobilenetv2 googlenet squeezenet1_1
# Fewer models means a proportionally shorter run; the pipeline does not care
# how many there are.
MODELS="${MODELS:-resnet18 squeezenet1_1}"

# Pruning criteria. Available:
#   magnitude_l1 magnitude_l2 bn_scale fpgm taylor obdc random
#   lamp fisher group_lasso hrank
# Notes worth knowing before choosing:
#   - bn_scale needs BatchNorm, so it does not apply to squeezenet1_1, and it
#     first trains a sparsity-learning model, which costs an extra SPARSITY_EPOCHS.
#   - obdc does not support depthwise convolutions or Fire modules; those pairs
#     are logged as failures and skipped.
#   - taylor, obdc, fisher and hrank are data-driven and run gradient batches
#     before every pruning step, so they take noticeably longer. hrank is the
#     slowest by a wide margin, since it computes a rank per feature map.
#   - random is the control: it says how much of a result comes from the
#     criterion rather than from removing capacity and retraining.
CRITERIA="${CRITERIA:-magnitude_l2 taylor random}"

# Target pruning rates, in % of parameters removed. The published grid is
#   10 20 30 40 50 60 70 80 90
# Each target is an independent run from the baseline, so the list length
# multiplies the runtime directly.
TARGETS="${TARGETS:-30 60}"

# --- Training and pruning recipe -------------------------------------------
# The defaults reproduce the published runs, which follow PruningBench. Change
# them only deliberately: they are what makes results comparable with that
# leaderboard, and with each other.

# Baseline training epochs. Published: 200. Lower this for a quick trial; the
# learning-rate milestones are adjusted automatically for short runs.
BASELINE_EPOCHS="${BASELINE_EPOCHS:-200}"

# Recovery fine-tuning epochs after pruning. Published: 100.
FT_EPOCHS="${FT_EPOCHS:-100}"

# Sparsity-learning epochs, used by bn_scale only. Published: 100. The result is
# cached per model and reused across targets, so this is paid once per model.
SPARSITY_EPOCHS="${SPARSITY_EPOCHS:-100}"

# Gradient batches accumulated before each pruning step by the data-driven
# criteria. Published: 10. Lowering it speeds those criteria up at the cost of a
# noisier importance estimate.
DD_BATCHES="${DD_BATCHES:-10}"

BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"

# --- Quantization ----------------------------------------------------------

# Calibration images, drawn from the CIFAR-100 TRAINING split. Published: 100.
NUM_CALIB="${NUM_CALIB:-100}"

# Calibration seed. Change it and re-run to measure how sensitive the INT8
# result is to which images were drawn; 42 keeps the published filenames.
CALIB_SEED="${CALIB_SEED:-42}"

# --- Benchmarking ----------------------------------------------------------

# Which devices to measure: cpu, tpu, or both. "tpu" needs a Coral device
# attached; "cpu" runs anywhere. The output file is cumulative, so you can
# measure cpu now and tpu later on another machine.
BENCH_DEVICE="${BENCH_DEVICE:-both}"

# Discarded warmup inferences, then timed inferences. Published: 30 and 200.
# Raise the warmup for sub-millisecond models.
BENCH_WARMUP="${BENCH_WARMUP:-30}"
BENCH_RUNS="${BENCH_RUNS:-200}"

# Evaluation images. 0 means the full 10 000-image test set, which is what the
# published accuracies use. A smaller number is much faster and much noisier.
BENCH_IMAGES="${BENCH_IMAGES:-0}"

# Cold-start campaign: K passes of N inferences. Published: 30 and 10.
# The output is cumulative, so passes can be added later by re-running.
COLD_PASSES="${COLD_PASSES:-30}"
COLD_INFERENCES="${COLD_INFERENCES:-10}"

# ===========================================================================
# END OF CONFIGURATION — the rest is plumbing.
# ===========================================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FROM=""; ONLY=""; DRY=0; SMOKE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)     FROM="$2"; shift 2 ;;
        --only)     ONLY="$2"; shift 2 ;;
        --dry-run)  DRY=1; shift ;;
        --smoke)    SMOKE=1; shift ;;
        -h|--help)  sed -n '2,36p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
done

# --smoke shrinks every cost knob at once. It exercises the same code paths as a
# real run and produces meaningless numbers, which is the point: it answers
# "is the setup correct", not "how good is this model".
if [[ $SMOKE -eq 1 ]]; then
    MODELS="${MODELS%% *}"          # the first model only
    CRITERIA="${CRITERIA%% *}"      # the first criterion only
    TARGETS="30"
    BASELINE_EPOCHS=1; FT_EPOCHS=1; SPARSITY_EPOCHS=1; DD_BATCHES=2
    NUM_CALIB=16; BENCH_WARMUP=2; BENCH_RUNS=5; BENCH_IMAGES=100
    COLD_PASSES=2; COLD_INFERENCES=4
    echo "### SMOKE MODE: one model, one criterion, minimal epochs ###"
    echo "### Judge it on whether the stages run, never on the numbers.  ###"
    echo "### A baseline this short has learned nothing, so the pruning  ###"
    echo "### rates and accuracies it yields are meaningless.            ###"
fi

# Should this stage run, given --from and --only?
should_run() {
    local stage="$1"
    [[ -n "$ONLY" ]] && { [[ "$stage" == "$ONLY" ]] && return 0 || return 1; }
    [[ -n "$FROM" ]] && { [[ "$stage" > "$FROM" || "$stage" == "$FROM" ]] && return 0 || return 1; }
    return 0
}

run() {
    local label="$1"; shift
    echo
    echo "==============================================================="
    echo "  $label"
    echo "==============================================================="
    printf '  %q' "$@"; echo
    [[ $DRY -eq 1 ]] && return 0
    "$@" || { echo "STAGE FAILED: $label" >&2; exit 1; }
}

for py in "$PYTORCH_PY" "$CORAL_PY"; do
    [[ $DRY -eq 1 ]] && break
    [[ -x "$py" ]] || { echo "ERROR: no interpreter at $py"; echo "Run setup/setup_envs.sh, or set PYTORCH_PY / CORAL_PY."; exit 2; }
done

mkdir -p "$WORK_DIR" "$DATA_DIR" "$RESULTS_DIR"
cd "$REPO/mono_tpu"

cat <<EOF

  Work dir   : $WORK_DIR
  Data dir   : $DATA_DIR
  Results    : $RESULTS_DIR
  Models     : $MODELS
  Criteria   : $CRITERIA
  Targets    : $TARGETS %
  Epochs     : baseline $BASELINE_EPOCHS, fine-tune $FT_EPOCHS
  Benchmark  : device=$BENCH_DEVICE, warmup=$BENCH_WARMUP, runs=$BENCH_RUNS, images=$BENCH_IMAGES
EOF

# --- 00: baselines ---------------------------------------------------------
if should_run 00; then
    run "00  train the baselines" \
        "$PYTORCH_PY" -u 00_prepare_baselines.py \
            --output_dir "$WORK_DIR/models" --data_dir "$DATA_DIR" \
            --models $MODELS --epochs "$BASELINE_EPOCHS" \
            --batch_size "$BATCH_SIZE" --seed "$SEED"
fi

# --- 01: pruning -----------------------------------------------------------
# One invocation per criterion, so that a criterion which does not apply to an
# architecture cannot take the rest of the sweep with it.
if should_run 01; then
    for crit in $CRITERIA; do
        run "01  prune with $crit at $TARGETS %" \
            "$PYTORCH_PY" -u 01_prune.py \
                --work_dir "$WORK_DIR" --data_dir "$DATA_DIR" --num_classes 100 \
                --checkpoints $TARGETS --models $MODELS --importance "$crit" \
                --batch_size "$BATCH_SIZE" --final_epochs "$FT_EPOCHS" \
                --sparsity_epochs "$SPARSITY_EPOCHS" --dd_batches "$DD_BATCHES" \
                --seed "$SEED"
    done

    run "--  aggregate the per-run logs" \
        "$PYTORCH_PY" -u aggregate_pruning_logs.py \
            --logs_dir "$WORK_DIR/pruning_logs" --models_dir "$WORK_DIR/models" \
            --fp32_output "$WORK_DIR/fp32_accuracy.json" \
            --layer_output "$WORK_DIR/layer_sparsity.json"
fi

# --- 02: quantization ------------------------------------------------------
if should_run 02; then
    run "02  convert to INT8 TFLite" \
        "$PYTORCH_PY" -u 02_convert_tflite_int8.py \
            --work_dir "$WORK_DIR" --data_dir "$DATA_DIR" \
            --num_calib "$NUM_CALIB" --calibration_seed "$CALIB_SEED"
fi

# --- 03: compilation -------------------------------------------------------
if should_run 03; then
    if ! command -v edgetpu_compiler >/dev/null 2>&1; then
        echo "ERROR: edgetpu_compiler is not on PATH. See setup/setup_envs.sh --edgetpu-compiler." >&2
        exit 2
    fi
    run "03  compile for the Edge TPU" \
        "$CORAL_PY" -u 03_compile_edgetpu.py \
            --tflite-dir "$WORK_DIR/tflite_int8" \
            --output-dir "$WORK_DIR/edgetpu_compiled" \
            --metrics-file "$WORK_DIR/compiler_metrics.json"
fi

# --- 04 and 05: benchmarks -------------------------------------------------
# Both read models from --platform_dir and write to --results_dir, which lets
# the model directory stay read-only on a deployment target such as a Pi.
if should_run 04; then
    run "04  benchmark, steady state" \
        "$CORAL_PY" -u 04_benchmark.py \
            --platform_dir "$WORK_DIR" --results_dir "$RESULTS_DIR" \
            --data_dir "$DATA_DIR" --device "$BENCH_DEVICE" \
            --warmup "$BENCH_WARMUP" --runs "$BENCH_RUNS" --num_images "$BENCH_IMAGES"
fi

if should_run 05; then
    run "05  benchmark, first inference" \
        "$CORAL_PY" -u 05_benchmark_coldstart.py \
            --platform_dir "$WORK_DIR" --results_dir "$RESULTS_DIR" \
            --device "$BENCH_DEVICE" --passes "$COLD_PASSES" \
            --inferences_per_pass "$COLD_INFERENCES"
fi

cat <<EOF

===============================================================
  Done.

  Models      $WORK_DIR/{models,pytorch_pruned,tflite_int8,edgetpu_compiled}
  Logs        $WORK_DIR/pruning_logs
  Results     $RESULTS_DIR/benchmark_results.{json,csv}
              $RESULTS_DIR/cold_start_results.{json,csv}

  Every CSV column is defined in the repository README, under
  "Benchmark metrics".
===============================================================
EOF

#!/usr/bin/env bash
# ===========================================================================
# run_pipeline_multi_tpu.sh — the complete multi-accelerator pipeline.
#
# Prunes ImageNet models from their published weights down to a size target, so
# that the result fits across N Edge TPUs, then compiles and benchmarks them.
#
#   00  fetch pretrained weights, check convertibility   pytorch-env
#   --  pipeline_full.py, which drives 01 to 03 in a loop:
#         01  prune to the size target                   pytorch-env  GPU required
#         02  quantize to INT8                           pytorch-env
#         03  compile into N segments                    needs edgetpu_compiler
#       then fine-tunes the checkpoint that fits.
#   --  verify_tpu.py, CPU against accelerator           coral-env    needs a Coral device
#   bench  pipeline, parallel, latency, cold start       coral-env    needs the 8x card
#
# WHY A LOOP RATHER THAN A SINGLE PASS
#   Pruning to a parameter count does not guarantee the model fits: the compiler
#   reports more bytes than the weight count implies. pipeline_full.py therefore
#   compiles the candidate, reads how many bytes still stream from off-chip
#   memory, and prunes further if any remain, before committing to the expensive
#   fine-tuning run. Each iteration is recorded in a pipeline summary JSON.
#
# USAGE
#   ./run_pipeline_multi_tpu.sh                # every stage, with the settings below
#   ./run_pipeline_multi_tpu.sh --smoke        # a short run, to check the setup
#   ./run_pipeline_multi_tpu.sh --only prune   # a single stage
#   ./run_pipeline_multi_tpu.sh --dry-run      # print the commands, run nothing
#
#   Stages: fetch, prune, verify, bench
#
# CUSTOMISING
#   Edit the CONFIGURATION block below, or override any variable from the
#   environment without touching the file:
#
#     MODELS="resnet50 inception_v4" TARGETS_MB="8 16" ./run_pipeline_multi_tpu.sh
#
# COST WARNING
#   The pruning stage fine-tunes on ImageNet. A single model at one size target
#   is tens of GPU-hours; the whole grid is measured in days. Run --smoke first,
#   and consider whether you need every model at every target.
# ===========================================================================
set -uo pipefail

# ===========================================================================
# CONFIGURATION — everything you are likely to change lives here.
# ===========================================================================

# --- Where things live -----------------------------------------------------

# Working directory: receives pytorch_pruned_imagenet/, pruning_logs_imagenet/,
# tflite_int8_pruned/ and edgetpu_compiled_pruned/. Budget tens of GB.
WORK_DIR="${WORK_DIR:-$(pwd)/work_multi}"

# ImageNet root. Must contain train/ and val/ in ImageFolder layout, that is one
# subdirectory per class. There is no download step: the dataset requires
# registration and must be provided.
DATA_DIR="${DATA_DIR:-/datasets/Imagenet_1k}"

# Interpreters. The defaults assume setup/setup_envs.sh created ./envs.
PYTORCH_PY="${PYTORCH_PY:-$(pwd)/envs/pytorch-env/bin/python}"
CORAL_PY="${CORAL_PY:-$(pwd)/envs/coral-env/bin/python}"

# --- What to run -----------------------------------------------------------

# Architectures. The full lineup is:
#   inception_v1_googlenet inception_v2_bninception inception_v3 inception_v4
#   inception_resnet_v2 resnet50 resnet101 resnet152
# They range from 6.6 M to 60.2 M parameters, so the cost per model varies by
# roughly an order of magnitude. Two families are represented on purpose:
# ResNets and Inceptions differ in branch count, which is one of the things the
# framework is set up to compare.
MODELS="${MODELS:-resnet50}"

# Size targets in MB of INT8 model. Each target corresponds to a multi-TPU
# configuration, because one accelerator holds roughly 8 MB in SRAM:
#     8 MB  -> 1 segment,  8 independent models running in parallel
#    16 MB  -> 2 segments, 4 pipelines of 2 accelerators
#    32 MB  -> 4 segments, 2 pipelines of 4
#    64 MB  -> 8 segments, 1 pipeline across all 8
# The segment count is derived from the target; set NUM_SEGMENTS below to force
# a different one.
TARGETS_MB="${TARGETS_MB:-8 16}"

# Segment count. Empty means derive it from the size target, which is what you
# usually want. Set it to compile a given model at a segment count that does not
# match its target.
NUM_SEGMENTS="${NUM_SEGMENTS:-}"

# Pruning criterion: taylor or magnitude_l2.
#   taylor        data-driven, accumulates gradients before each pruning step
#   magnitude_l2  data-free, faster, no calibration data needed
CRITERION="${CRITERION:-taylor}"

# Gradient batches accumulated before each pruning step, for taylor only.
TAYLOR_BATCHES="${TAYLOR_BATCHES:-10}"

# --- Recovery fine-tuning --------------------------------------------------

# Derive the epoch budget from the pruning rate the loop actually reached,
# rather than from the requested target. Recommended: the loop converges deeper
# than the size target predicts, so a budget fixed in advance under-trains the
# deepest runs. The bands live in FT_BUDGET_BANDS in multi_tpu/pipeline_full.py.
EPOCHS_FROM_ACTUAL="${EPOCHS_FROM_ACTUAL:-1}"

# Used only when EPOCHS_FROM_ACTUAL=0. Fixed epoch and warmup counts.
FT_EPOCHS="${FT_EPOCHS:-60}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"

# Batch size. 128 fits a 48 GB GPU at 224 px; drop to 64 at 299 px, or on a
# smaller card, if you hit an out-of-memory error.
BATCH_SIZE="${BATCH_SIZE:-128}"

# Mixed precision. Roughly twice as fast on recent GPUs, with no measurable
# accuracy cost. Set to 0 as a control, or if numerical instability is suspected.
USE_AMP="${USE_AMP:-1}"

# --- The guided loop -------------------------------------------------------

# Maximum iterations before giving up on making a model fit. The loop usually
# converges in two or three.
MAX_ITERS="${MAX_ITERS:-8}"

# Margin subtracted alongside the reported off-chip bytes at each iteration, in
# MB. Raise it if the loop keeps needing extra iterations on large models.
MARGIN_MB="${MARGIN_MB:-0.5}"

# Calibration images for quantization, drawn from the ImageNet training split.
# Used by the fetch stage. pipeline_full.py does not expose it: it passes its own
# default down to 02_convert_pruned.py.
NUM_CALIB="${NUM_CALIB:-100}"

# --- Benchmarking ----------------------------------------------------------
# These stages need the 8x Edge TPU card and are skipped when no device is found.

# Which pipeline phases to run: A is the single-accelerator baseline, B is the
# canonical order at N=2..8, C samples several TPU orderings, all runs them all.
BENCH_PHASES="${BENCH_PHASES:-all}"

# Parallel benchmark mode: steady, cold, or both.
#   steady  one interpreter, synchronised repetitions
#   cold    a fresh interpreter per repetition, so each pays the weight transfer
PARALLEL_MODE="${PARALLEL_MODE:-both}"
PARALLEL_REPS="${PARALLEL_REPS:-100}"

# Refuse configurations whose total memory mapping would exceed this, in MB.
# The accelerator driver aborts the process above roughly 1500 MB, and that
# abort cannot be caught, so this guard is what keeps a sweep alive.
MAX_TOTAL_MAP_MB="${MAX_TOTAL_MAP_MB:-1500}"

# ===========================================================================
# END OF CONFIGURATION — the rest is plumbing.
# ===========================================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLY=""; DRY=0; SMOKE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)     ONLY="$2"; shift 2 ;;
        --dry-run)  DRY=1; shift ;;
        --smoke)    SMOKE=1; shift ;;
        -h|--help)  sed -n '2,42p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
done

# --smoke shrinks every cost knob at once: same code paths, meaningless numbers.
if [[ $SMOKE -eq 1 ]]; then
    MODELS="${MODELS%% *}"
    TARGETS_MB="${TARGETS_MB%% *}"
    EPOCHS_FROM_ACTUAL=0; FT_EPOCHS=1; WARMUP_EPOCHS=0
    TAYLOR_BATCHES=2; NUM_CALIB=16; BATCH_SIZE=32
    MAX_ITERS=2; PARALLEL_REPS=5; BENCH_PHASES="A"
    echo "### SMOKE MODE: one model, one target, 1 fine-tuning epoch ###"
fi

should_run() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

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

if [[ $DRY -eq 0 ]]; then
    [[ -x "$PYTORCH_PY" ]] || { echo "ERROR: no interpreter at $PYTORCH_PY. Run setup/setup_envs.sh."; exit 2; }
    [[ -d "$DATA_DIR/train" && -d "$DATA_DIR/val" ]] || {
        echo "ERROR: expected ImageNet at $DATA_DIR/{train,val}/ in ImageFolder layout."
        echo "Set DATA_DIR to point at it."; exit 2; }
fi

mkdir -p "$WORK_DIR"
cd "$REPO/multi_tpu"

cat <<EOF

  Work dir   : $WORK_DIR
  Data dir   : $DATA_DIR
  Models     : $MODELS
  Targets    : $TARGETS_MB MB
  Criterion  : $CRITERION
  Fine-tune  : $( [[ "$EPOCHS_FROM_ACTUAL" == "1" ]] && echo "budget derived from the achieved rate" || echo "$FT_EPOCHS epochs, $WARMUP_EPOCHS warmup" )
EOF

# --- fetch -----------------------------------------------------------------
# Downloads the published weights and checks each architecture survives the
# export chain. Cheap, and it fails here rather than hours into a pruning run.
if should_run fetch; then
    run "fetch  pretrained weights and convertibility check" \
        "$PYTORCH_PY" -u 00_fetch_and_convert_pretrained.py \
            --models $MODELS --num_calib "$NUM_CALIB"
fi

# --- prune -----------------------------------------------------------------
# One pipeline_full.py invocation per (model, target). --run_tag keeps the
# filenames distinct when two segment counts share an initial target.
if should_run prune; then
    if ! command -v edgetpu_compiler >/dev/null 2>&1; then
        echo "ERROR: edgetpu_compiler is not on PATH; the guided loop cannot check the fit." >&2
        echo "See setup/setup_envs.sh --edgetpu-compiler." >&2
        exit 2
    fi
    for model in $MODELS; do
        for target in $TARGETS_MB; do
            args=(
                --model "$model" --target_mb "$target" --importance "$CRITERION"
                --taylor_batches "$TAYLOR_BATCHES" --data_dir "$DATA_DIR"
                --pruned_dir "$WORK_DIR/pytorch_pruned_imagenet"
                --int8_dir   "$WORK_DIR/tflite_int8_pruned"
                --edgetpu_dir "$WORK_DIR/edgetpu_compiled_pruned"
                --log_dir    "$WORK_DIR/pruning_logs_imagenet"
                --batch_size "$BATCH_SIZE"
                --max_iters "$MAX_ITERS" --refine_margin_mb "$MARGIN_MB"
                --run_tag "T${target}MB" --python "$PYTORCH_PY"
            )
            [[ -n "$NUM_SEGMENTS" ]] && args+=(--num_segments "$NUM_SEGMENTS")
            if [[ "$EPOCHS_FROM_ACTUAL" == "1" ]]; then
                args+=(--epochs_from_actual)
            else
                args+=(--ft_epochs "$FT_EPOCHS" --warmup_epochs "$WARMUP_EPOCHS")
            fi
            [[ "$USE_AMP" == "0" ]] && args+=(--no_amp)

            run "prune  $model to $target MB, then fine-tune" \
                "$PYTORCH_PY" -u pipeline_full.py "${args[@]}"
        done
    done
fi

# --- verify ----------------------------------------------------------------
# Compares CPU and accelerator outputs on identical inputs. Needs a Coral device.
if should_run verify; then
    if [[ $DRY -eq 0 ]] && ! "$CORAL_PY" -c "
from pycoral.utils.edgetpu import list_edge_tpus
import sys; sys.exit(0 if list_edge_tpus() else 1)" 2>/dev/null; then
        echo "SKIP verify: no Edge TPU device found."
    else
        run "verify  CPU against accelerator" \
            "$CORAL_PY" -u verify_tpu.py --models $MODELS
    fi
fi

# --- bench -----------------------------------------------------------------
# Needs the 8x card. Run the pipeline phases BEFORE the parallel benchmark:
# two processes competing for the accelerators interfere with each other.
if should_run bench; then
    if [[ $DRY -eq 0 ]] && ! "$CORAL_PY" -c "
from pycoral.utils.edgetpu import list_edge_tpus
import sys; sys.exit(0 if len(list_edge_tpus()) >= 2 else 1)" 2>/dev/null; then
        echo "SKIP bench: fewer than two Edge TPU devices found."
    else
        run "bench  pipeline phases $BENCH_PHASES" \
            "$CORAL_PY" -u bench/bench_pipeline.py --phase "$BENCH_PHASES"

        run "bench  parallel, mode $PARALLEL_MODE" \
            "$CORAL_PY" -u bench/bench_parallel.py \
                --mode "$PARALLEL_MODE" --reps "$PARALLEL_REPS" \
                --max-total-map-mb "$MAX_TOTAL_MAP_MB" --orchestrate --resume
    fi
fi

cat <<EOF

===============================================================
  Done.

  Checkpoints    $WORK_DIR/pytorch_pruned_imagenet
  INT8 models    $WORK_DIR/tflite_int8_pruned
  Compiled       $WORK_DIR/edgetpu_compiled_pruned
  Logs           $WORK_DIR/pruning_logs_imagenet
                   <model>_<tag>_pipeline_summary.json  loop iterations
                   <model>_<criterion>_<tag>.json       fine-tuning history

  Every CSV column is defined in the repository README, under
  "Benchmark metrics".
===============================================================
EOF

#!/usr/bin/env bash
# ===========================================================================
# setup_envs.sh — create the virtual environments this project needs.
#
# WHAT THIS PRODUCES
#   Up to three venvs under $PREFIX (default: ./envs), plus the Edge TPU
#   compiler if you ask for it:
#
#     envs/pytorch-env   Python 3.12  training, structured pruning,
#                                     PyTorch -> ONNX -> TFLite int8
#     envs/coral-env     Python 3.9   Edge TPU inference and benchmarking
#     envs/synth-env     Python 3.12  synthetic CNN generator (TensorFlow)
#
# WHY THREE ENVIRONMENTS AND NOT ONE
#   They are mutually incompatible, and merging them silently breaks results
#   rather than failing loudly:
#     - pycoral 2.0.0 ships wheels for Python 3.6-3.9 only, while the onnx2tf 2.x
#       line needs Python >= 3.12. No single interpreter satisfies both.
#     - Installing a full TensorFlow next to onnx2tf moves the numpy/protobuf
#       pins and breaks the PyTorch -> TFLite path, which is why the synthetic
#       generator gets its own env.
#     - coral-env deliberately contains no torch: it is deployed to a Raspberry
#       Pi and to the 8x Edge TPU host, where torch is neither available nor
#       wanted.
#
# LOGIC THIS SCRIPT FOLLOWS
#   1. Resolve one interpreter per env, and refuse to continue if the version
#      is wrong (a 3.11 pytorch-env installs onnx2tf 1.x, which routes through
#      the MLIR backend and yields models that make edgetpu_compiler segfault
#      under segmentation; that failure would only surface hours later).
#   2. Create the venv, upgrade pip, install the matching requirements file.
#   3. Install the handful of packages that must bypass the resolver, with
#      --no-deps. These are listed at the bottom of each requirements file with
#      the reason: their declared pins contradict versions this project is known
#      to run on, and honouring them makes the file unsatisfiable or silently
#      replaces a package with a different build of the same module.
#   4. For pytorch-env only, run setup/apply_patches.py, which rewrites two
#      upstream source files in place (see that script for the rationale).
#   5. Verify each env by importing its critical modules.
#
# USAGE
#   ./setup/setup_envs.sh                      # all three envs
#   ./setup/setup_envs.sh pytorch coral        # a subset
#   ./setup/setup_envs.sh --prefix /opt/sparta # somewhere else
#   ./setup/setup_envs.sh --edgetpu-compiler   # also install the compiler
#
# NOTE ON THE EDGE TPU COMPILER
#   edgetpu_compiler is a closed-source Google binary, x86-64 Linux only. It is
#   not a Python package and cannot be pip-installed, so --edgetpu-compiler
#   reports whether one is already on PATH and otherwise prints the two ways of
#   getting it: a system-wide `apt install edgetpu-compiler`, or unpacking the
#   .deb into $PREFIX without root, which is what the cluster jobs do.
# ===========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Anchored to the repository, not to the current directory: run_smoke.sh and the
# two pipeline runners look for the environments there, so a setup launched from
# elsewhere would otherwise put them where nothing finds them.
PREFIX="$REPO_ROOT/envs"
WANT_COMPILER=0
TARGETS=()

# --- Argument parsing -------------------------------------------------------
# Positional arguments name the envs to build; with none given, build all.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)            PREFIX="$2"; shift 2 ;;
        --edgetpu-compiler)  WANT_COMPILER=1; shift ;;
        -h|--help)           awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; exit 0 ;;
        pytorch|coral|synth) TARGETS+=("$1"); shift ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
done
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(pytorch coral synth)

REQ_DIR="$REPO_ROOT/requirements"
mkdir -p "$PREFIX"

# --- Helpers ---------------------------------------------------------------

# Find an interpreter matching an exact "major.minor", trying the versioned
# name first so that a system python3 of the wrong version never wins silently.
find_python() {
    local want="$1" cand
    for cand in "python${want}" "python3"; do
        if command -v "$cand" >/dev/null 2>&1; then
            if "$cand" -c "import sys; sys.exit(0 if '.'.join(map(str,sys.version_info[:2]))=='${want}' else 1)"; then
                command -v "$cand"; return 0
            fi
        fi
    done
    return 1
}

# Create a venv from a requirements file. Idempotent: an existing venv is
# reused and its packages re-resolved, so re-running after adding a dependency
# is cheap and safe.
make_env() {
    local name="$1" pyver="$2" reqfile="$3" interp
    local venv="$PREFIX/$name"

    echo
    echo "=============================================================="
    echo " $name  (Python $pyver)  ->  $venv"
    echo "=============================================================="

    if ! interp="$(find_python "$pyver")"; then
        echo "ERROR: no Python $pyver on PATH." >&2
        case "$pyver" in
            3.9)  echo "  coral-env needs 3.9 because pycoral 2.0.0 has no newer wheel." >&2
                  echo "  Debian/Ubuntu: use deadsnakes, or conda create -n py39 python=3.9" >&2 ;;
            3.12) echo "  onnx2tf 2.x needs 3.12; on 3.10/3.11 pip silently installs" >&2
                  echo "  onnx2tf 1.x, whose MLIR output segfaults edgetpu_compiler -n N." >&2 ;;
        esac
        return 1
    fi
    echo "interpreter: $interp"

    [[ -d "$venv" ]] || "$interp" -m venv "$venv"
    "$venv/bin/python" -m pip install --upgrade pip setuptools wheel -q
    "$venv/bin/python" -m pip install -r "$reqfile"
    echo "$name: dependencies installed"
}

# Install packages the resolver must not see. pip has no per-requirement
# --no-deps, so these cannot live in the requirements file itself; the file
# lists them in a comment block instead, with the reason for each.
install_no_deps() {
    local name="$1"; shift
    local venv="$PREFIX/$name"
    echo "--- installing without dependency resolution: $* ---"
    "$venv/bin/python" -m pip install --no-deps "$@"
}

# Import-check an env. Cheap, but catches the failure mode that matters here:
# a package that resolves at install time yet cannot load (ABI mismatch between
# numpy and pycoral, or a CUDA build that does not match the driver).
VERIFY_FAILURES=0
verify_env() {
    local name="$1"; shift
    local venv="$PREFIX/$name"
    echo "--- verifying $name ---"
    for mod in "$@"; do
        if "$venv/bin/python" -c "import $mod" 2>/dev/null; then
            echo "  ok    $mod"
        else
            echo "  FAIL  $mod"
            VERIFY_FAILURES=$((VERIFY_FAILURES + 1))
        fi
    done
}

# Unpack the Edge TPU compiler .deb into $PREFIX without root. Used by the
# cluster jobs, which cannot apt-install anything.
install_edgetpu_compiler() {
    local target="$PREFIX/edgetpu"
    echo
    echo "=============================================================="
    echo " edgetpu_compiler -> $target"
    echo "=============================================================="
    if command -v edgetpu_compiler >/dev/null 2>&1; then
        echo "already on PATH: $(command -v edgetpu_compiler)"
        edgetpu_compiler --version | head -1
        return 0
    fi
    mkdir -p "$target"
    cat <<'MSG'
The compiler is distributed through Google's apt repository. Either:

  sudo sh -c 'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
      > /etc/apt/sources.list.d/coral-edgetpu.list'
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt update && sudo apt install edgetpu-compiler

or, without root (this is what the cluster jobs do), download the .deb and:

  dpkg-deb -x edgetpu-compiler_*_amd64.deb "$PREFIX/edgetpu"
  export PATH="$PREFIX/edgetpu/usr/bin:$PATH"

Version 16.0 is the one every result in this repository was produced with.
MSG
}

# --- Build the requested environments --------------------------------------
for t in "${TARGETS[@]}"; do
    case "$t" in
        pytorch)
            make_env pytorch-env 3.12 "$REQ_DIR/pytorch-env.txt"
            install_no_deps pytorch-env onnx2tf==2.4.0 tf_keras==2.21.0
            # onnx2tf and ai-edge-quantizer both need a source patch before they
            # emit Edge TPU-compatible models. Without it, googlenet and
            # squeezenet are rejected outright by the compiler.
            echo "--- applying upstream patches (onnx2tf, ai-edge-quantizer) ---"
            "$PREFIX/pytorch-env/bin/python" "$REPO_ROOT/setup/apply_patches.py"
            verify_env pytorch-env torch torchvision torch_pruning onnx onnx2tf \
                       ai_edge_quantizer ai_edge_litert timm
            ;;
        coral)
            make_env coral-env 3.9 "$REQ_DIR/coral-env.txt"
            verify_env coral-env pycoral ai_edge_litert numpy PIL
            ;;
        synth)
            make_env synth-env 3.12 "$REQ_DIR/synth-env.txt"
            install_no_deps synth-env tf-keras==2.21.0
            verify_env synth-env tensorflow tf2onnx onnx2tf ai_edge_quantizer
            ;;
    esac
done

[[ $WANT_COMPILER -eq 1 ]] && install_edgetpu_compiler

cat <<EOF

==============================================================
Done. Activate with:

  source $PREFIX/pytorch-env/bin/activate   # pruning / conversion
  source $PREFIX/coral-env/bin/activate     # Edge TPU benchmarking
  source $PREFIX/synth-env/bin/activate     # synthetic generator

A Coral device also needs the runtime library (libedgetpu1-std) and, on a
freshly rebooted host, the /dev/apex_* nodes may come back owned by root:
  sudo udevadm trigger --subsystem-match=apex
==============================================================
EOF

# Exit non-zero when a module failed to import, so that a broken env is caught
# here rather than hours later in a job. The messages above say which one.
if [[ $VERIFY_FAILURES -gt 0 ]]; then
    echo "$VERIFY_FAILURES module(s) failed to import; see the FAIL lines above." >&2
    exit 1
fi

#!/usr/bin/env bash
# One-time setup of the chosen laptop (WSL2 Ubuntu + GPU TensorFlow): docs/laptop_runbook.md section 0 as a script.
#
#   bash scripts/setup_wsl.sh            # build ~/.venv (Python 3.13) and install the pinned packages
#   bash scripts/setup_wsl.sh --check    # only verify an existing setup (GPU, packages, fast tests)
#   bash scripts/setup_wsl.sh --rebuild  # delete and rebuild ~/.venv
#
# Run from inside the clone (keep it in ~/, not /mnt/c). Before this: update the NVIDIA Windows driver and run
# `wsl --install -d Ubuntu` in an administrator PowerShell. Never install a Linux NVIDIA driver inside WSL.
# Environment: VENV (default ~/.venv).
set -euo pipefail

MODE=install
case "${1:-}" in
  "") ;;
  --check) MODE=check ;;
  --rebuild) MODE=rebuild ;;
  -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  *) echo "unknown option: $1 (use --check, --rebuild or nothing)" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/.venv}"
PY="$VENV/bin/python"
cd "$REPO"

say()  { printf '\n== %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

say "machine"
grep -qi microsoft /proc/version || warn "this does not look like WSL; the runbook assumes WSL2 Ubuntu"
case "$REPO" in /mnt/*) warn "the repo is under /mnt (Windows disk): slow and line-ending trouble; clone into ~/ instead" ;; esac
if command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
  :
else
  warn "nvidia-smi failed: update the Windows NVIDIA driver and run 'wsl --update'; the pipeline still runs on CPU"
fi

if [ "$MODE" != check ]; then
  say "system packages"
  need=()
  for p in python3-venv tmux git make; do dpkg -s "$p" >/dev/null 2>&1 || need+=("$p"); done
  if [ ${#need[@]} -gt 0 ]; then
    sudo apt-get update && sudo apt-get install -y "${need[@]}"
  else
    echo "already installed: python3-venv tmux git make"
  fi

  if [ "$MODE" = rebuild ] && [ -d "$VENV" ]; then
    say "removing $VENV"
    rm -rf "$VENV"
  fi

  if [ -x "$PY" ] && "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 13))'; then
    say "Python 3.13 venv already at $VENV"
  else
    # TensorFlow 2.21 has wheels for 3.10-3.13 only and Ubuntu 26.04 ships 3.14: uv fetches 3.13 under
    # ~/.local/share/uv (no sudo, the system Python is untouched)
    say "creating the Python 3.13 venv at $VENV (via uv)"
    BOOT="$HOME/.local/share/uv-bootstrap"
    if [ ! -x "$BOOT/bin/uv" ]; then
      tmp="$(mktemp -d)"
      python3 -m venv "$tmp/v"
      "$tmp/v/bin/python" -m pip install -q --target "$BOOT" uv
      rm -rf "$tmp"
    fi
    PYTHONPATH="$BOOT" "$BOOT/bin/uv" python install 3.13
    PYTHONPATH="$BOOT" "$BOOT/bin/uv" venv --clear --seed --python 3.13 "$VENV"
  fi

  say "packages"
  # WSL's /tmp is a RAM disk smaller than the unpacked CUDA wheels: keep pip's temp files on disk
  export TMPDIR="$HOME/.cache/pip-tmp"
  mkdir -p "$TMPDIR"
  "$PY" -m pip install -q --upgrade pip
  if [ -f requirements.txt ]; then
    # exact versions of the thesis run; the editable line pins a GitHub commit, so install this clone instead
    req="$TMPDIR/requirements.pinned.txt"
    grep -v '^-e ' requirements.txt > "$req"
    "$PY" -m pip install -r "$req"
    "$PY" -m pip install --no-deps -e .
  else
    "$PY" -m pip install -e . -r requirements-dev.txt "tensorflow[and-cuda]" joblib
  fi
  rm -rf "$TMPDIR"
  unset TMPDIR

  # TensorFlow 2.21's library search path omits nvidia/cusolver/lib: link the file into a folder it does search
  SP="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
  if [ -e "$SP/nvidia/cusolver/lib/libcusolver.so.11" ] && [ -d "$SP/nvidia/cusparse/lib" ]; then
    ln -sfn ../../cusolver/lib/libcusolver.so.11 "$SP/nvidia/cusparse/lib/libcusolver.so.11"
    echo "linked libcusolver.so.11"
  else
    warn "CUDA wheels not found in $SP/nvidia; TensorFlow will run on CPU"
  fi
fi

say "check"
[ -x "$PY" ] || { echo "no venv at $VENV: run 'bash scripts/setup_wsl.sh' first" >&2; exit 1; }
"$PY" - <<'EOF'
import sys
import pasig  # noqa: F401  (installed from this clone)
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"Python {sys.version.split()[0]} | TensorFlow {tf.__version__} | GPUs: {[g.name for g in gpus]}")
if not gpus:
    print("WARNING: TensorFlow sees no GPU (see the runbook's symptom table); the pipeline will use the CPU")
EOF
"$PY" -m pytest -m fast -q
say "done: run 'source $VENV/bin/activate' in every new terminal"

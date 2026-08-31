#!/usr/bin/env bash
# Provision a Ubuntu GPU VM (T4) for training:
#   NVIDIA driver + CUDA 12, Python venv with the CUDA PyTorch build.
#
# Usage: bash scripts/setup_gpu_vm.sh   (run as a user with sudo)
set -euo pipefail

echo "==> Installing system packages"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential python3-venv python3-pip curl git unzip

echo "==> NVIDIA driver + CUDA toolkit (12.x via pip wheels is enough for torch)"
# The T4 is fully supported by the open nvidia-driver-535+; install driver:
sudo apt-get install -y --no-install-recommends nvidia-driver-535 || \
  echo "driver install deferred; continue with venv"

echo "==> Python venv (uv)"
VENV="${VENV:-$HOME/pad-venv}"
UV_BIN="$(command -v uv || echo "$HOME/.local/bin/uv")"
"$UV_BIN" venv "$VENV" --python 3.13
# Python available inside the venv via `uv run --python "$VENV"` or activation

echo "==> Project install (uv sync — includes CUDA-less core; see note below)"
# On the T4 you typically want the CUDA torch build. Two options:
#   A) keep PyPI torch (CPU/CUDA-less):   uv sync --extra dev
#   B) CUDA wheels via the PyTorch index:
#        uv pip install --python "$VENV/bin/python" torch torchvision \
#            --index-url https://download.pytorch.org/whl/cu121
#        then: uv pip install --python "$VENV/bin/python" --editable . --extra dev
"$UV_BIN" sync --python "$VENV" --extra dev || true
"$UV_BIN" pip install --python "$VENV/bin/python" -e . --extra dev

echo "==> Sanity check"
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "==> Done. Activate with: source $VENV/bin/activate"
echo "    Then: export PYTHONPATH=src"
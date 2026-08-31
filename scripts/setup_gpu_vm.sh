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

echo "==> Python venv"
VENV="${VENV:-$HOME/pad-venv}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip setuptools wheel

echo "==> PyTorch CUDA wheels"
"$VENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "==> Project deps"
"$VENV/bin/pip" install -r requirements.txt

echo "==> Sanity check"
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "==> Done. Activate with: source $VENV/bin/activate"
echo "    Then: export PYTHONPATH=src"
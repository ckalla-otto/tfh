#!/usr/bin/env bash
# Ship the subset manifests (and optionally the depth cache) to the training VM.
#
# Usage:
#   VM=user@host  bash scripts/rsync_data.sh [--with-depth-cache]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${VM:?set VM=user@host}"
REMOTE_DIR="${REMOTE_DIR:-$HOME/tfh}"

echo ">>> syncing repo (code only)"
rsync -avz --exclude data --exclude results --exclude .git \
  "$ROOT"/ "${VM}:${REMOTE_DIR}/"

echo ">>> syncing subset manifests"
rsync -avz "$ROOT/data/subsets/" "${VM}:${REMOTE_DIR}/data/subsets/"

if [ "${1:-}" = "--with-depth-cache" ]; then
  echo ">>> syncing depth cache"
  rsync -avz "$ROOT/data/caches/" "${VM}:${REMOTE_DIR}/data/caches/"
fi

echo ">>> Done. On the VM: cd tfh && source ~/pad-venv/bin/activate && export PYTHONPATH=src"
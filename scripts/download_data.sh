#!/usr/bin/env bash
# Download the Celeba-Spoof mirror from Kaggle using env-var credentials only.
#
# Required env vars (never write a kaggle.json):
#   KAGGLE_USERNAME, KAGGLE_KEY, PAD_DATASET_SLUG (e.g. owner/celeba-spoof-mirror)
#
# Usage:
#   export KAGGLE_USERNAME=... KAGGLE_KEY=... PAD_DATASET_SLUG=...
#   bash scripts/download_data.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data/raw}"

: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
: "${KAGGLE_KEY:?set KAGGLE_KEY or KAGGLE_API_KEY}"
: "${PAD_DATASET_SLUG:?set PAD_DATASET_SLUG}"

# Accept legacy KAGGLE_API_KEY as an alias for KAGGLE_KEY
export KAGGLE_KEY="${KAGGLE_KEY:-${KAGGLE_API_KEY:-}}"
: "${KAGGLE_KEY:?set KAGGLE_KEY or KAGGLE_API_KEY}"

# Optional: source a local .env (never committed) if present
if [ -f "$ROOT/.env" ]; then
  set -a; source "$ROOT/.env"; set +a
  export KAGGLE_KEY="${KAGGLE_KEY:-${KAGGLE_API_KEY:-}}"
fi

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if command -v kaggle >/dev/null 2>&1; then
  echo ">>> kaggle CLI found; downloading $PAD_DATASET_SLUG"
  kaggle datasets download -d "$PAD_DATASET_SLUG" -p "$DATA_DIR"
else
  echo ">>> kaggle CLI not found; trying kagglehub (python)"
  python - <<PY
import os, kagglehub
os.environ.setdefault("KAGGLE_USERNAME", os.environ["KAGGLE_USERNAME"])
os.environ.setdefault("KAGGLE_KEY", os.environ["KAGGLE_KEY"])
p = kagglehub.dataset_download(os.environ["PAD_DATASET_SLUG"])
print("downloaded to:", p)
PY
fi

# unzip any archives not yet extracted
find "$DATA_DIR" -maxdepth 1 -name "*.zip" | while read -r z; do
  target="${z%.zip}"
  if [ ! -d "$target" ]; then
    echo ">>> unzipping $z"
    unzip -q "$z" -d "$DATA_DIR" && rm -f "$z"
  fi
done

echo ">>> Done. Source mirror under $DATA_DIR"
echo ">>> Next: build the crawl manifest (crawl.csv) or point data.crawl_meta at the mirror"
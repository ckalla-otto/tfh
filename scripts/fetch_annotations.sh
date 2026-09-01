#!/usr/bin/env bash
# Fetch the official CelebA-Spoof annotation table (label.csv) from Kaggle and
# verify its SHA-256 checksum.
#
# Needed before `pad prepare` on a fresh machine. Downloads only the ~60 MB
# labels file (not the images). Requires Kaggle credentials and the `kaggle` CLI.
#
# Usage:
#   bash scripts/fetch_annotations.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL_CSV="$ROOT/data/labels/label.csv"
CHECKSUM_FILE="$ROOT/data/labels/label.csv.sha256"

KAGGLE_SLUG="${KAGGLE_SLUG:-tungnguyentien/celeba-spoof-crop-1-9}"
KAGGLE_FILE="${KAGGLE_FILE:-CelebA_Spoof_crop_1_9/data_1.0_128/label.csv}"

mkdir -p "$ROOT/data/labels"

echo "==> Downloading $KAGGLE_FILE from $KAGGLE_SLUG"
uv run kaggle datasets download -d "$KAGGLE_SLUG" -f "$KAGGLE_FILE" -p "$ROOT/data/labels"

# The kaggle CLI sometimes writes a .zip; extract if present.
if [[ -f "$ROOT/data/labels/label.csv.zip" ]]; then
  echo "==> Extracting label.csv.zip"
  (cd "$ROOT/data/labels" && unzip -o -q label.csv.zip && rm -f label.csv.zip)
fi

echo "==> Verifying checksum (from repo root)"
(cd "$ROOT" && shasum -a 256 -c "$CHECKSUM_FILE")

echo "==> OK: $LABEL_CSV"
# tfh — Single-image Face Presentation Attack Detection (PAD)

PyTorch implementation of a **two-stream PAD model on a shared DINOv2 backbone**,
trained on a limited, stratified subset of **CelebA-Spoof** (from Kaggle):

- **binary live/spoof** head → BCE
- **depth-regression** head → Smooth-L1 / MSE on face-masked pseudo-depth:
  live + physical-3D spoof classes regress a Depth-Anything depth map, the six
  flat 2D attack classes regress a flat zero plane
- **high-frequency texture branch** (luminance − Gaussian blur) → catches borders,
  edges, moiré, print/screen artifacts of the presentation medium
- **spoof-type auxiliary head** (10-way CE) → semantic multi-task supervision

Per ISO/IEC 30107-3-style metrics (APCER / BPCER / ACER), per-spoof-type
breakdown, confusion matrix, and per-class **hard-sample reports**.

> See [`docs/architecture.md`](docs/architecture.md) for the diagram and loss spec.

## Quick start

```bash
# uv manages the environment (see pyproject.toml)
uv venv --python 3.13
uv sync --extra dev

# 1. download the mirror (env-var creds only)
export KAGGLE_USERNAME=... KAGGLE_KEY=... PAD_DATASET_SLUG=...
bash scripts/download_data.sh

# 2. build the crawl manifest CSV (or reuse the Kaggle mirror's own csv)
#    -> set data.crawl_meta, then:

# 3. (optional) generate pseudo-depth cache for the estimated classes
uv run python -m pad depth_targets --config configs/exp_smoke.yaml --splits "train val test"

# 4. train (builds splits automatically on first run; add --rebuild-splits to redo)
uv run python -m pad train --config configs/exp_smoke.yaml --run-name smoke

# 5. evaluate a checkpoint
uv run python -m pad evaluate --config configs/exp_smoke.yaml \
    --ckpt results/smoke/best.pt --split test
```

All CLIs use **Google Fire** (no argparse): flags are just keyword arguments
(`--config`, `--run-name`, `--rebuild-splits`, `--no-tta`, ...). The
module-level forms (`uv run python -m pad.train`, `python -m pad.evaluate`,
`python -m pad.depth_targets`) work identically.

When running from inside the activated venv, drop the `uv run` prefix
(`python -m pad train ...`).

## Repo layout

```
tfh/
├── configs/              # base.yaml (+ include-merging experiment configs)
├── docs/architecture.md
├── scripts/              # download_data.sh, setup_gpu_vm.sh, rsync_data.sh
├── src/pad/
│   ├── split.py          # equal-per-spoof-type, identity-exclusive subsetting
│   ├── data.py           # dataset, extended crop, face masks, HF maps, aug, samplers
│   ├── depth_targets.py  # offline Depth-Anything pseudo-depth cache
│   ├── model.py          # DINOv2 + HF branch + heads
│   ├── losses.py         # BCE + Smooth-L1(depth) + BCE(hf) + CE(spoof-type)
│   ├── inference.py      # shared scoring (used by train + evaluate)
│   ├── train.py          # T4 training loop (AMP, guard, early stop)
│   └── evaluate.py       # metrics, per-type, confusion, hard samples
├── tests/test_split.py   # stratification invariants
└── data/ results/        # gitignored
```

## Key design decisions (locked during planning)

| Decision | Value | Why |
|---|---|---|
| Subset | ~18–24k images (≈2k/class, Live + 9 attacks) | stable per-type metrics; identity-exclusive |
| Crop | extended bbox ×1.3; **depth supervised on the face rect only** | borders/edges → HF branch; geometry stays anatomical |
| Depth targets | flat=0 for 2D attacks; pseudo-depth for live + 3D classes | 3D masks have real relief; flat target would fight pixels |
| Depth loss | Smooth-L1 (β=0.1) + per-image confidence weight | noisy pseudo-labels; cap outliers |
| Backbone | DINOv2-Base (timm) | strongest cheap transfer |
| Eval | APCER/BPCER/ACER @ BPCER≈1%, per-type, hard samples, TTA flip | ISO-style + where it fails |

## Experiment matrix

| # | Config | Question |
|---|--------|----------|
| 0 | BCE-only | backbone baseline |
| 1 | BCE + depth | core method |
| 2 | λ sweep {0.1, 1, 10} | loss balance |
| 3 | HF branch (γ=0) | texture value alone |
| 4 | full (BCE+depth+HF+spoof-type) | proposed method |
| 4b | spoof-type off | semantic aux value |
| 4c | MSE vs Smooth-L1, conf-weight on/off | depth loss design |
| 5 | tight vs extended crop | context hypothesis |
| 6 | depth strategy {per-type, all-flat, all-estimated} | 3D-mask handling |
| 7 | subset 5k/15k/40k | data efficiency |
| 8 | leave-one-attack-out | generalization |

## Dev notes

- Dev happens on macOS (CPU/MPS). Training runs on the **T4 VM** (CUDA); see
  `scripts/setup_gpu_vm.sh` + `scripts/rsync_data.sh`.
- The HF map is computed **on-the-fly after augmentation** (not cached) so the
  RGB and HF streams stay perfectly consistent under grayscale/geometric aug.
  The depth cache is the only offline artifact.
- Dependencies are managed by **uv**: `uv sync` installs the locked resolve
  (`uv.lock` is gitignored to avoid platform/torch-build differences; regenerate
  per host with `uv lock --upgrade` if needed). Tests: `uv run pytest`.

## Known limitations (documented in the write-up)

- 3D-mask detection does NOT rest on depth (depth treats 3D classes as live);
  discriminators are the HF branch + spoof-type semantics.
- Single-image, RGB-only → no rPPG / motion cues.
- Pseudo-depth labels are noisy → mitigated by Smooth-L1 + confidence weighting.
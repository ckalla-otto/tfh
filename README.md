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

# 1. prepare train/val/test from the on-disk mirror + the official labels
#    (data-root = folder containing Data/{train,test}/<subject>/{live,spoof}/<img>)
uv run python -m pad prepare \
    --data-root /Users/christiankalla/Downloads/CelebA_Spoof \
    --labels data/labels/label.csv \
    --config configs/base.yaml
cat data/subsets/balance_report.md      # verify RESULT: PASS

# 2. (optional) pseudo-depth cache for the estimated classes (live + 3D masks)
uv run python -m pad depth_targets --config configs/base.yaml --splits "train val test"

# 3. train + evaluate
uv run python -m pad train --config configs/base.yaml --run-name smoke
uv run python -m pad evaluate --config configs/base.yaml --ckpt results/smoke/best.pt --split test

# 4. predict a single image with a probability
uv run python -m pad predict --image_path path/to/img.jpg --ckpt results/smoke/best.pt
uv run python -m pad predict --image_path img.jpg --ckpt best.pt --bbox "10 20 260 300"
```

All CLIs use **Google Fire** (no argparse): flags are just keyword arguments
(`--config`, `--run-name`, `--no-tta`, ...). The module-level forms
(`uv run python -m pad.train`, `python -m pad.evaluate`, ...) work identically.

When running from inside the activated venv, drop the `uv run` prefix
(`python -m pad train ...`).

## Preparing the dataset (`pad prepare`)

Assumes the mirror is already downloaded to disk:

```
<data-root>/Data/{train,test}/<subject>/{live,spoof}/<img>.png|jpg
<data-root>/Data/{train,test}/<subject>/{live,spoof}/<img>_BB.txt   # bbox x y w h [conf]
```

and the official annotation table at `data/labels/label.csv` (indexed by
`split/subject/class/<img>`; column 40 = spoof type 0–9, 41 = illumination,
42 = environment). The official `label.csv` can be obtained from the Kaggle
dataset `tungnguyentien/celeba-spoof-crop-1-9`.

What `pad prepare` does:

1. **Walks all images** under `Data/` — the official `train/` and `test/`
   folders are **pooled together** (the folder names are arbitrary download
   parts; this lets the missing attack classes in `train/` be covered by
   `test/`).
2. **Reads each bbox** from the sibling `<img>_BB.txt` (`x y w h`).
3. **Joins `label.csv`** → true `spoof_type` (0–9), `environment`,
   `illumination`; images without a valid label (or bbox) are dropped unless
   `--include-unknown`.
4. **Builds a stratified, identity-exclusive 70/15/15 subset** with equal
   per-spoof-type counts; writes `data/subsets/{train,val,test}.csv` +
   `balance_report.md` (fails loudly on any invariant violation).

Re-run `pad prepare` with a different `--subsets-dir` / config to change the
budget or seed (e.g. `configs/exp_smoke.yaml` for a quick pilot).

## Repo layout

```
tfh/
├── configs/              # base.yaml (+ include-merging experiment configs)
├── docs/architecture.md
├── src/pad/
│   ├── prepare.py        # build train/val/test from on-disk data + label.csv
│   ├── split.py          # equal-per-spoof-type, identity-exclusive subsetting
│   ├── data.py           # dataset, extended crop, face masks, HF maps, aug, samplers
│   ├── depth_targets.py  # offline Depth-Anything pseudo-depth cache
│   ├── model.py          # DINOv2 + HF branch + heads
│   ├── losses.py         # BCE + Smooth-L1(depth) + BCE(hf) + CE(spoof-type)
│   ├── inference.py      # shared scoring (used by train + evaluate)
│   ├── train.py          # training loop (AMP, guard, early stop)
│   ├── evaluate.py       # metrics, per-type, confusion, hard samples
│   └── predict.py        # single-image live/spoof probability (+ depth diagnostic)
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

## Prediction (single image)

```bash
uv run python -m pad predict --image_path path/to/img.jpg \
    --ckpt results/run/best.pt --config configs/base.yaml

# optional face bbox "x1 y1 x2 y2" (the model was trained on extended face crops);
# without it, InsightFace (SCRFD) auto-detects the face:
uv run python -m pad predict --image_path img.jpg --ckpt best.pt --bbox "10 20 260 300"
uv run python -m pad predict --image_path img.jpg --ckpt best.pt       # auto face detection

# disable flip TTA, pick device, disable auto-detection:
uv run python -m pad predict ... --no-tta --device cpu --auto-face false
```

Output:
```
P(live)=0.1396 P(spoof)=0.8604  type=region_mask (idx 6)  decision=spoof
```

Conventions:
- `live probability` = sigmoid of the binary head (with optional horizontal-flip
  TTA, matching eval), `decision` = thresholded at 0.5.
- `spoof_type` = argmax of the 10-way head (0=live, 1..9=attacks).
- Face box used for the extended crop: explicit `--bbox` if given, else
  **InsightFace (SCRFD, `buffalo_l`)** auto-detection; if nothing is found the
  full image is center-cropped. (The model was trained on extended face crops,
  so a bbox/auto-face improves accuracy.)

## Dev notes

- Dev happens on macOS (CPU/MPS). Training runs wherever a GPU is available
  (`train` auto-uses CUDA, else MPS/CPU).
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
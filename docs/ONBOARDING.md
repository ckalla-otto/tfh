# Onboarding — catching up on this repo

This document is the fastest path for a new agent (or a fresh machine) to go from
zero to a trained PAD model. Everything else (`README.md`, `docs/architecture.md`,
`docs/data_pipeline.md`) goes deeper; this one is the sequence.

## 1. What this repo is

Single-image **face presentation attack detection (PAD)** on a stratified subset
of **CelebA-Spoof**. A shared **DINOv2-Base** backbone is supervised by four
signals:

| Head | Output | Loss |
|---|---|---|
| binary | P(live) | BCE |
| depth | 112×112 face-masked depth map | Smooth-L1 (confidence-weighted) |
| high-frequency branch | spoof vs live from texture | BCE (aux) |
| spoof-type | 10-way attack class | CE |

Evaluation is ISO/IEC 30107-3-style (APCER / BPCER / ACER), per-spoof-type,
plus confusion matrix and per-class hard-sample reports.

## 2. Environment (any machine)

```bash
uv venv --python 3.13
uv sync                 # installs deps from pyproject.toml
uv run pytest           # 7 tests, all pass
uv run ruff format      # run before finishing any change
```

- Python ≥ 3.10 (we use 3.13). GPU not required — `train` auto-selects
  `cuda` → `mps` → `cpu`.
- `uv.lock` is intentionally NOT committed (platform/torch differences); each
  host runs `uv sync` to re-resolve.

## 3. The canonical pipeline (run in this order)

```bash
# (0) fetch + verify the official annotations (one time per machine)
bash scripts/fetch_annotations.sh

# (a) create the stratified subset CSVs from the on-disk mirror + label.csv
uv run python -m pad prepare \
    --data-root /path/to/CelebA_Spoof \
    --labels data/labels/label.csv \
    --config configs/base.yaml
cat data/subsets/balance_report.md     # must end "RESULT: PASS"

# (b) organize images into a self-contained, class-folder dataset under data/
uv run python -m pad export --subsets-dir data/subsets --out-dir data/dataset

# (c) optional: pseudo-depth cache for the ESTIMATED classes (live + 3D masks)
uv run python -m pad depth_targets --config configs/base.yaml --splits "train val test"

# (d) train + evaluate
uv run python -m pad train --config configs/base.yaml --run-name exp1
uv run python -m pad evaluate --config configs/base.yaml --ckpt results/exp1/best.pt --split test

# (e) predict on an arbitrary image
uv run python -m pad predict --config configs/base.yaml --ckpt results/exp1/best.pt \
    --image_path some_face.jpg      # optional --bbox "x1 y1 x2 y2"
```

Every command is a **Google Fire** CLI (`python -m pad <cmd> --flag value`); flag
names are the function parameters.

## 4. What the data pipeline produces

Inputs (must exist before `prepare`):
- The celebrity mirror on disk: `Data/{train,test}/<subject>/{live,spoof}/<img>.png|jpg`
  each with a sibling `<img>_BB.txt` (`x y w h [conf]` face bbox).
- `data/labels/label.csv` — the official annotation table (see below).

Outputs after `prepare` + `export`:
- `data/subsets/{train,val,test}.csv` — **the metadata truth**: one row per image
  with `image_path` now pointing into `data/dataset/...`.
- `data/dataset/{train,val,test}/{spoof_name}/<img>` — **the images themselves**
  (real copies), organized by spoof name.

```
data/
├── labels/label.csv            # official annotations (spoof_type, env, illum, ...)
├── subsets/{train,val,test}.csv  # metadata; image_path -> data/dataset
└── dataset/
    ├── train/{live,photo,poster,a4,face_mask,upper_body_mask,region_mask,pc_pad,phone,3d_mask}/
    ├── val/   …same…
    └── test/  …same…
```

The `data/` directory is gitignored. `data/dataset` is fully self-contained
(no symlinks): the machine can drop the original mirror after `export`.

> Getting `data/labels/label.csv`: **one command** fetches it from Kaggle and
> verifies its committed SHA-256 checksum:
> ```bash
> bash scripts/fetch_annotations.sh
> ```
> (Downloads only `CelebA_Spoof_crop_1_9/data_1.0_128/label.csv` from
> `tungnguyentien/celeba-spoof-crop-1-9`, ~60 MB — no images.) The official
> annotation vector is indexed by the relative path `split/subject/class/<img>`;
> column 40 = spoof type 0-9, 41 = illumination, 42 = environment. `pad prepare`
> verifies this file against `data/labels/label.csv.sha256` and fails loudly on
> mismatch.

## 5. Config (`configs/base.yaml`)

- `data.root` — path to the raw mirror (only used by `prepare`).
- `data.subsets_dir` / `data.dataset_dir` — CSV + image-tree output locations.
- `data.subset.budget_total` — **20,000** total → train 14k / val 3k / test 3k
  (1,400 / 300 / 300 per spoof type; identity-disjoint 70/15/15).
- `model.backbone` — `vit_base_patch14_dinov2`.
- `loss.*` / `depth.*` / `train.*` — loss weights, depth cache, hyperparams.

## 6. CLI cheatsheet

| Command | What it does |
|---|---|
|`pad prepare` | Walk mirror + join label.csv → stratified identity-excl subset CSVs |
|`pad export` | Copy subset images into `data/dataset/{split}/{class}/`, re-point CSVs |
|`pad depth_targets` | Pregenerate Depth-Anything pseudo-depth cache (ESTIMATED classes) |
|`pad train` | Train DINOv2 two-stream PAD model (AMP, depth guard, early stop) |
|`pad evaluate` | ISO metrics + per-type + confusion + hard-samples |
|`pad predict` | Live/spoof probability + attack type for one image |

## 7. Where to look for more

- `docs/architecture.md` — model diagram, loss, training, eval.
- `docs/data_pipeline.md` — CSV schema, transform details, depth cache format.
- `README.md` — short overview + quick start.
- `src/pad/*.py` — one module per concern (data/split/prepare/model/losses/…).
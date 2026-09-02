# tfh — Single-image Face Presentation Attack Detection (PAD)

PyTorch implementation of a **PAD model on a shared DINOv2 backbone**,
trained on a limited, stratified subset of **CelebA-Spoof** (from Kaggle):

- **binary live/spoof** head → BCE
- **depth-regression** head → Smooth-L1 / MSE on face-masked pseudo-depth
  (live + physical-3D spoof classes regress a Depth-Anything depth map, the
  six flat 2D attack classes regress a flat zero plane) — **optional**,
  off by default (`model.use_depth_head: false`), an experiment to enable
- **high-frequency texture branch** (luminance − Gaussian blur) → catches borders,
  edges, moiré, print/screen artifacts of the presentation medium
- **spoof-type auxiliary head** (10-way CE) → semantic multi-task supervision

Per ISO/IEC 30107-3-style metrics (APCER / BPCER / ACER), per-spoof-type
breakdown, confusion matrix, and per-class **hard-sample reports**.

> See [`docs/architecture.md`](docs/architecture.md) for the diagram and loss spec.

## Quick start

> New to this repo? Read **`docs/ONBOARDING.md`** first — it's the fastest
> path for another agent / fresh machine. `docs/data_pipeline.md` has the
> full data-layer reference.

```bash
# uv manages the environment (see pyproject.toml)
uv venv --python 3.13
uv sync --extra dev

# 0. fetch + verify the official annotations (one time per machine)
bash scripts/fetch_annotations.sh

# 1. build the stratified train/val/test CSVs from the on-disk mirror + labels
uv run python -m pad prepare \
    --data-root /path/to/CelebA_Spoof \      # folder containing Data/{train,test}/...
    --labels data/labels/label.csv \
    --config configs/base.yaml
cat data/subsets/balance_report.md          # verify RESULT: PASS

# 2. organize the images into a self-contained, class-folder dataset under data/
uv run python -m pad export --subsets-dir data/subsets --out-dir data/dataset

# 3. (optional) pseudo-depth cache for the estimated classes (live + 3D masks)
uv run python -m pad depth_targets --config configs/base.yaml --splits "train val test"

# 4. train + evaluate
uv run python -m pad train --config configs/base.yaml --run-name smoke
uv run python -m pad evaluate --config configs/base.yaml --ckpt results/smoke/best.pt --split test

# 5. predict a single image with a probability
uv run python -m pad predict --image_path path/to/img.jpg --ckpt results/smoke/best.pt
uv run python -m pad predict --image_path img.jpg --ckpt best.pt --bbox "10 20 260 300"
```

All CLIs use **Google Fire** (no argparse): flags are just keyword arguments
(`--config`, `--run-name`, `--no-tta`, ...). The module-level forms
(`uv run python -m pad.train`, `python -m pad.evaluate`, ...) work identically.

When running from inside the activated venv, drop the `uv run` prefix
(`python -m pad train ...`).

## The dataset layout

After `pad prepare` + `pad export` the dataset is **self-contained under `data/`**:

```
data/
├── labels/label.csv            # official annotations (needed by prepare)
├── subsets/{train,val,test}.csv  # METADATA truth (bbox, spoof_type, env, illum), image_path -> data/dataset
└── dataset/
    ├── train/{live,photo,poster,a4,face_mask,upper_body_mask,region_mask,pc_pad,phone,3d_mask}/
    ├── val/   …same…
    └── test/  …same…
```

- `data/dataset` contains the actual image files (copies) organized by spoof
  name; no references to the original mirror.
- The CSVs carry all labels/bboxes and point `image_path` at the images; every
  downstream script (`train`, `evaluate`, `depth_targets`, `predict`) reads
  them.
- Default subset: **20,000** images — train 14k / val 3k / test 3k (1,400 /
  300 / 300 per spoof type), identity-exclusive and equal per class.

### Ready-made archives

Both build directories are published and can be restored by downloading and
unpacking:

- `data/` (self-contained dataset: labels, subset CSVs, and the image tree):
  ```bash
  curl -L -o data_remote.zip https://storage.googleapis.com/tfh_data_ck/data_remote.zip
  unzip -q data_remote.zip -d ./data_remote  # note: archive may nest under a top folder;
  # then point configs at it or copy into the expected data/ layout.
  ```
- `results/` (trained model artifacts + per-run eval reports):
  ```bash
  curl -L -o results.tar.gz https://storage.googleapis.com/tfh_data_ck/results.tar.gz
  tar xzf results.tar.gz -C .  # restores a `results/` directory
  ```

> `data/` and `results/` are gitignored build artifacts, so this is the
> canonical way to retrieve them without re-downloading the source mirror or re-training.

## Preparing the dataset

Two steps, documented in detail in **[`docs/data_pipeline.md`](docs/data_pipeline.md)**:

1. **`pad prepare`** — walk the on-disk mirror (`Data/{train,test}/<subject>/{live,spoof}/<img>.png|jpg` + sibling `<img>_BB.txt` bboxes), join the official `label.csv` (fetched via `bash scripts/fetch_annotations.sh`, indexed by the image's relative path, col 40 = spoof type 0–9, SHA-256-verified against the committed `data/labels/label.csv.sha256`), and build a stratified, identity-exclusive 70/15/15 subset with equal per-spoof-type counts → `data/subsets/{train,val,test}.csv` + `balance_report.md` (fails loudly on violation).
2. **`pad export`** — copy the subset images into the self-contained class-folder tree `data/dataset/{train,val,test}/{spoof_name}/` (real files) and re-point the CSVs' `image_path` there.

Run with a different `--subsets-dir` / `--config` to change budget or seed
(e.g. `configs/exp_smoke.yaml` for a quick pilot).

## Repo layout

```
tfh/
├── configs/                  # base.yaml (+ include-merging experiment configs)
├── docs/
│   ├── ONBOARDING.md         # fastest path for new agent / fresh machine
│   ├── architecture.md       # model diagram, loss, training, eval
│   └── data_pipeline.md      # data-layer reference (schema, prepare/export)
├── src/pad/
│   ├── prepare.py            # build train/val/test CSVs from on-disk data + label.csv (+ `export`)
│   ├── split.py              # equal-per-spoof-type, identity-exclusive subsetting
│   ├── data.py               # dataset, extended crop, face masks, HF maps, aug, samplers
│   ├── depth_targets.py      # offline Depth-Anything pseudo-depth cache
│   ├── model.py              # DINOv2 + HF branch + heads
│   ├── losses.py             # BCE + Smooth-L1(depth) + BCE(hf) + CE(spoof-type)
│   ├── inference.py          # shared scoring (used by train + evaluate)
│   ├── train.py              # training loop (AMP, guard, early stop)
│   ├── evaluate.py           # metrics, per-type, confusion, hard samples
│   └── predict.py            # single-image live/spoof probability (+ depth diagnostic)
├── tests/test_split.py       # stratification invariants
└── data/ results/            # gitignored (data/dataset is built by export)
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

Moved to report.md section 7 (Experiment matrix & status), which marks which experiment was actually run.

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
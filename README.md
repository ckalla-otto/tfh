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

# 1. download ONLY the metadata tables from the mirror (no images yet)
mkdir -p data/raw/celeba-spoof data/subset
for f in train.csv test.csv val.csv metadata.csv; do
  uv run kaggle datasets download -d "$PAD_DATASET_SLUG" -f "$f" -p data/raw/celeba-spoof
done

# 2. crawl the metadata -> normalized manifest (no image files needed)
uv run python -m pad make_crawl --root data/raw/celeba-spoof --out data/crawl.csv --from-metadata true

# 3. stratify + sample the subset (equal per spoof-type, identity-exclusive)
uv run python -m pad make_splits --config configs/base.yaml

# 4. download ONLY the sampled images
uv run python -m pad download_subset --subset-dir data/subsets --out-dir data/subset --slug "$PAD_DATASET_SLUG"

# 5. (optional) pseudo-depth cache
uv run python -m pad depth_targets --config configs/base.yaml --splits "train val test"

# 6. train + evaluate
uv run python -m pad train --config configs/base.yaml --run-name smoke
uv run python -m pad evaluate --config configs/base.yaml --ckpt results/smoke/best.pt --split test

# 7. predict a single image with a probability
uv run python -m pad predict --image_path path/to/img.jpg --ckpt results/smoke/best.pt
uv run python -m pad predict --image_path img.jpg --ckpt best.pt --bbox "10 20 260 300"
```

All CLIs use **Google Fire** (no argparse): flags are just keyword arguments
(`--config`, `--run-name`, `--rebuild-splits`, `--no-tta`, ...). The
module-level forms (`uv run python -m pad.train`, `python -m pad.evaluate`, ...)
work identically.

When running from inside the activated venv, drop the `uv run` prefix
(`python -m pad train ...`).

## Crawling the dataset (and downloading only the subset, via the Kaggle API)

This flow uses **only the Kaggle API** — no full-archive download. Verified
end-to-end against the official-layout mirror
`attentionlayer241/celeba-spoof-for-face-antispoofing`.

```bash
# 1. credentials (gitignored). The Python CLIs auto-load this via python-dotenv,
#    so no `source` and no `$PAD_DATASET_SLUG` shell expansion is needed:
cp .env.example .env   # = KAGGLE_USERNAME, KAGGLE_KEY, PAD_DATASET_SLUG

# 2. dump the mirror's file listing (paths only — cheap, paginated)
#    Writes each page to data/mirror_files.txt immediately and keeps a resume
#    token (data/mirror_files.txt.listing_token); re-running continues where it
#    left off instead of restarting. Just re-run this same command to resume.
uv run python -m pad download_subset --fetch-files --files-out data/mirror_files.txt

# 3. build the manifest from that listing (no images on disk yet).
#    IMPORTANT: the mirror's folder structure only distinguishes `live/` vs
#    `spoof/`. To recover the TRUE 10-way attack type, download the official
#    annotation table and join it:
#      kaggle datasets download -d tungnguyentien/celeba-spoof-crop-1-9 \
#          -f 'CelebA_Spoof_crop_1_9/data_1.0_128/label.csv' -p data/labels
#    (label.csv is indexed by `split/subject/class/<img>`; column 40 = spoof
#    type 0-9, 41 = illumination, 42 = environment.)
uv run python -m pad make_crawl --from-file-list data/mirror_files.txt \
    --out data/crawl.csv --labels data/labels/label.csv

# 4. stratified, identity-exclusive subset (per spoof-type equal) + balance report
uv run python -m pad make_splits --config configs/base.yaml
cat data/subsets/balance_report.md     # verify RESULT: PASS

# 5. download exactly those images + their _BB.txt bboxes (official layout),
#    then re-links subset CSVs and patches x1..y2 in them
uv run python -m pad download_subset --subset-dir data/subsets \
    --out-dir data/subset --official --workers 8

# 6. (optional) dry check that every sampled path exists in the mirror
uv run python -m pad download_subset ... --check-only
```

> Tip: the full listing is ~1.25M entries (about 45 min at the API's hard 200/page
> cap). It's **resume-safe** — run it in the background, and if it's interrupted
> just re-run the same `--fetch-files` command; it continues from the saved token.
- The official-layout mirror stores labels in the **folder structure**
  (`Data/<split>/<subject>/<class>/<img>.png`) and bboxes in sibling
  `<img>_BB.txt` (`x y w h [conf]`) — `make_crawl --from-file-list` parses the
  paths, and `download_subset --official` fetches + patches the bboxes.
- After `download_subset`, don't re-run `make_splits` (it would re-point
  `image_path` back to the empty mirror root); re-run `download_subset` to
  re-link instead.

**Alternative (whole-archive) flow:** if a mirror packs everything into one
zip (no per-file access), use `bash scripts/download_data.sh` to download +
unzip once, then `make_crawl --root data/raw/celeba-spoof --layout official`
+ `make_splits` from the on-disk mirror.

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
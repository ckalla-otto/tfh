# PAD on CelebA-Spoof — Concise Report

> **STATUS: DRAFT / SCAFFOLD.** Every `[TBD]` block below is a placeholder to be
> filled with real values once the data download and training run are complete.
> The structure and headings are final and match the task's required content.

## 1. Dataset and subset used

**Dataset.** CelebA-Spoof is a large-scale face **Presentation Attack Detection
(PAD)** dataset of ~500k images across **10 classes**: live (bona-fide) plus 9
attack types — photo, poster, A4-print, face mask, upper-body mask, region mask,
PC-screen replay, phone replay, and 3D mask. Each sample carries a face bounding
box, plus environment and illumination labels. We read it from the public Google
Drive mirror (`1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z`, 74 × 1 GB split zips) and/or
the Kaggle mirror (`attentionlayer241/celeba-spoof-for-face-antispoofing`).

**Subset.** The repo trains on a **stratified, identity-exclusive subset**,
*sampled without needing the full archive on disk*:

- `[TBD: fill actual]` total images sampled (target ≈ 24k; smoke = 5k).
- **10 classes, equal per-class counts** per split, capped by the scarcest class
  (typically 3D-mask).
- **Raw 70 / 15 / 15** train / val / test split that is **subject-disjoint**
  (no identity leakage between splits).
- Within each attack type, **environment and illumination** are balanced
  proportionally as secondary strata.
- Per-class counts: `[TBD: paste from data/subsets/balance_report.md]`
- Images on disk: `[TBD: n/total expected]`, source: `[Drive | Kaggle mirror]`.

## 2. Method / hypothesis

**Hypothesis.** 2D attacks (printed/screen) are separable from live faces by
*texture + geometry*: they are flat (no depth relief) and carry high-frequency
print-grain / moiré / border artifacts, while live faces (and physical 3D masks)
have real anatomical depth relief. A model that combines **(a)** a strong
frozen-ish DINOv2 backbone, **(b)** a depth-regression head supervised to expect
flat planes on flat 2D attacks and real pseudo-depth on live/3D classes, and
**(c)** a hand-tuned high-frequency texture branch, should reach a low
APCER/BPCER/ACER with limited data.

**Model.** Two-stream network on a shared DINOv2-Base backbone, extended face
crop (bbox ×1.3) at 224 px + luminance high-frequency map, three heads:
binary live/spoof (BCE), depth (Smooth-L1, face-masked), and 10-way spoof-type
(CE). Loss = `BCE + λ_d·D(depth) + γ_hf·BCE(HF) + λ_t·CE(spoof_type)`.

**Results and conclusion.** `[TBD: after training — report val/test ACER,
APCER, BPCER, AUC + per-class breakdown from results/<run>/ and
test_metrics.json. State what worked / what didn't.]`

## 3. Exact commands (prepare → train → evaluate)

> `uv` manages the env (`pyproject.toml`). All CLIs use Google Fire: flags are
> keyword args. Commands assume a CUDA host for the real run.

### 3.1 Prepare the subset (metadata-only, no full archive needed)

```bash
# credentials (gitignored)
cp .env.example .env        # KAGGLE_USERNAME, KAGGLE_KEY, PAD_DATASET_SLUG

# (a) mirror file listing + crawl manifest (Kaggle mirror flow)
uv run python -m pad download_subset --fetch-files \
    --files-out data/mirror_files.txt --page-size 200
uv run python -m pad make_crawl --from-file-list data/mirror_files.txt \
    --out data/crawl.csv

# (b) OR crawl a locally-unzipped official layout (Drive mirror)
#     unzip -o data/CelebA-Spoof-zips/CelebA_Spoof.zip.001 -d data/raw/celeba-spoof # repeat for all
uv run python -m pad make_crawl --root data/raw/celeba-spoof \
    --layout official --labels path/to/label.csv --out data/crawl.csv

# stratified, identity-exclusive subset CSVs (equal per class, subject-disjoint)
uv run python -m pad make_splits --config configs/base.yaml

# pseudo-depth cache for ESTIMATED classes (base.yaml: depth.enabled=true)
uv run python -m pad depth_targets --config configs/base.yaml --splits "train val test"
```

### 3.2 Train

```bash
uv run python -m pad train --config configs/base.yaml --run-name experiment
# checkpoint: results/experiment/best.pt ; eval written to results/experiment/test_metrics.json
```

### 3.3 Evaluate

```bash
uv run python -m pad evaluate --config configs/base.yaml \
    --ckpt results/experiment/best.pt --split test
```

### 3.4 Predict a single image (model artifact usage)

```bash
uv run python -m pad predict --image_path path/to/img.jpg \
    --ckpt results/experiment/best.pt --config configs/base.yaml
```

## 4. Trained model artifact

- Path: `results/experiment/best.pt` `[TBD: commit/publish location]`
- Input: 224×224 extended face crop (+ optional HF map computed on the fly).
- Output: `P(live)`, `P(spoof)`, predicted `spoof_type`, decision at 0.5 (no TTA)
  or TTA-averaged during eval.
- Evaluation surface: ISO-ish APCER / BPCER / ACER @ BPCER≈1%, AUC, per-spoof-type
  error rates, confusion matrix, per-class hard-sample reports.

## 5. Rough breakdown of time spent

`[TBD: fill approximate hours, e.g.]`

| Area | Hours (approx.) | Notes |
|---|---|---|
| Data plumbing (crawl, split, subset download, bbox patch) | ... | incl. Kaggle API pagination/resume + Drive quota workarounds |
| Model + losses + depth targets | ... | |
| Training + eval + tuning (T4 CUDA) | ... | |
| Reports / docs / validation | ... | |

## 6. What I would do next

- `[TBD]` Full-dataset / data-efficiency sweep (subset 5k / 15k / 40k).
- `[TBD]` Loss-balance vs. depth/hf/type weighting; MSE-vs-Smooth-L1 ablation.
- `[TBD]` Tighter vs. extended crop; per-type vs. all-estimated depth strategy.
- `[TBD]` Leave-one-attack-out generalization test; TTA beyond flip.
- `[TBD]` rPPG / motion cues to close the 3D-mask gap (current model treats 3D
  classes via HF + spoof-type semantics, not depth).

## 7. Assumptions & limitations

- Pseudo-depth labels (Depth-Anything-V2-Small) are noisy → handled with
  Smooth-L1 + per-image confidence weighting; depth is **not** the discriminator
  for 3D-mask attacks.
- Secondary strata (environment, illumination) are balanced but not used as a
  discriminative signal; illumination values rely on `label.csv` join.
- Equal per-class sampling caps the budget on the scarcest class (3D-mask), so we
  intentionally train on a subset rather than the full imbalanced archive.
- Drive mirror is quota/rate-limited (per-request, IP-scoped); final source for
  the run is `[Drive | Kaggle subset flow]`.

## 8. Disclosure of external code, documentation, and AI tools

- **Libraries / code:** PyTorch, torchvision, timm (DINOv2-Base),
  HuggingFace Transformers (Depth-Anything-V2-Small-hf),
  `kaggle` CLI /Kaggle API, python-dotenv, Fire, pandas, numpy, OpenCV,
  InsightFace (SCRFD, buffalo_l) for face-detection in `predict`.
- **Datasets/mirrors:** CelebA-Spoof (Google Drive mirror `1OW_1baw...` +
  Kaggle mirror `attentionlayer241/...`), official annotation table `label.csv`.
- **Documentation evaluated:** `[TBD: cite any docs/guides consulted]`
- **AI assisted-development tools used:** `[TBD]` — e.g. an AI coding agent
  (Claude) assisted with architecture/implementation/download-tooling/debugging;
  specific detail below.

---
*Generated on `[date]` by `[author]`. This is a fill-in scaffold — results
sections are placeholders to be completed after the training run.*
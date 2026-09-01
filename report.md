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

**Model (baseline — depth OFF).** Baseline network: shared DINOv2-Base backbone,
extended face crop (bbox ×1.3) at 224 px + luminance high-frequency map, heads:
binary live/spoof (BCE), 10-way spoof-type (CE), and HF auxiliary (BCE).
Loss = `BCE + γ_hf·BCE(HF) + λ_t·CE(spoof_type)`. Depth is **off by default**
(`model.use_depth_head: false`).

**Depth head — optional add-on (evaluable experiment).** Adding
`model.use_depth_head: true` attaches a `DepthHead` on the DINOv2 patch tokens
→ 112×112 depth map, supervised with Smooth-L1 on the face-masked region: live +
physical-3D mask classes regress Depth-Anything pseudo-depth, the six flat 2D
classes regress a flat zero plane, with per-image confidence weighting and a
trivial-solution guard. See §2.1 for the evaluation protocol.

**Results and conclusion.** `[TBD: after training — report val/test ACER,
APCER, BPCER, AUC + per-class breakdown from results/<run>/ and
test_metrics.json. State what worked / what didn't.]`

### 2.1 Depth-head add-on evaluation protocol

Purpose: quantify whether the **depth head adds anything** over the simple
baseline on the *same* subset, split, and seed.

**Enable it:**
```bash
# configs/base.yaml: model.use_depth_head: true   (default is false)
uv run python -m pad depth_targets --config configs/base.yaml --splits "train val test"   # pseudo-depth for live+3D classes
uv run python -m pad train --config configs/base.yaml --run-name with_depth
```
(loss term becomes `+ λ_d·D(depth)`. `pad depth_targets` is only needed when
depth is enabled.)

**Runs to compare** (same data, splits, seed):

| run | config | collected |
|---|---|---|
| baseline | `use_depth_head: false` | ACER/APCER/BPCER/AUC, per-spoof-type, confusion, hard-samples |
| +depth | `use_depth_head: true` | same metrics + depth diagnostics |

**Depth diagnostics to report** (per run, `results/<run>/per_type.csv` +
`train.log`): per-class face-rect depth variance (structure expected on
live/3D, ~flat on 2D), trained MSE/var on estimated vs flat groups, and the
depth-guard `var_estimated / var_flat` ratio.

**Hypotheses & known risks:**
- Expected gain on 2D attacks (photo/poster/A4/pc/phone/region) via flat-vs-relief
  depth; uncertain whether the sparse 20k subset + pretrained DINOv2 already
  captures that signal from texture alone.
- **3D-mask classes are NOT depth-discriminated** — they get real pseudo-depth
  (treated like live for the depth head); their detection rests on HF +
  spoof-type semantics. So depth should NOT be expected to move 3D-mask ACER.
- Ablations if the add-on looks promising: MSE vs Smooth-L1, λ_d sweep
  {0.1, 1, 10}, per-type vs all-estimated vs all-flat depth targets, `depth_res`.

## 3. Exact commands (prepare → train → evaluate)

> `uv` manages the env (`pyproject.toml`). All CLIs use Google Fire: flags are
> keyword args. Commands assume a CUDA host for the real run.

### 3.1 Prepare the subset (self-contained under `data/`)

```bash
# credentials (gitignored)
cp .env.example .env        # KAGGLE_USERNAME, KAGGLE_KEY

# (a) fetch + verify the official annotations (one time per machine)
bash scripts/fetch_annotations.sh

# (b) stratified, identity-exclusive subset CSVs from the on-disk mirror
uv run python -m pad prepare --data-root /path/to/CelebA_Spoof \
    --labels data/labels/label.csv --config configs/base.yaml

# (c) organize the images into data/dataset/{split}/{class}/ (copies)
uv run python -m pad export --subsets-dir data/subsets --out-dir data/dataset

# (d) pseudo-depth cache -- ONLY for the depth-head add-on (model.use_depth_head: true)
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
    --ckpt results/experiment/best.pt --config configs/base.yaml     # depth-off baseline config
# for the depth add-on use a config with model.use_depth_head: true so the model matches
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

- `[TBD]` **Evaluate the depth add-on**: baseline (`use_depth_head: false`) vs
  +depth on the same subset/split/seed; report ACER/APCER/BPCER/AUC + per-class
  and depth diagnostics (see §2.1). Then ablations if promising:
  - Loss-balance: λ_d sweep {0.1, 1, 10}; γ_hf; λ_t.
  - Depth loss type: MSE vs Smooth-L1; per-type vs all-estimated vs all-flat depth
    target strategy; `depth_res`.
- `[TBD]` Tighter vs. extended crop; data-efficiency sweep (5k / 15k / 20k / full).
- `[TBD]` Leave-one-attack-out generalization test; TTA beyond flip.
- `[TBD]` rPPG / motion cues to close the 3D-mask gap (current model treats 3D
  classes via HF + spoof-type semantics, not depth).

## 7. Assumptions & limitations

- Pseudo-depth labels (Depth-Anything-V2-Small) are noisy → handled with
  Smooth-L1 + per-image confidence weighting; depth is **optional** (off by
  default) and **not** the discriminator for 3D-mask attacks (they get real
  pseudo-depth; HF + spoof-type semantics carry them).
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
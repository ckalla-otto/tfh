# Architecture

Two-stream network with a **shared DINOv2 backbone**:

```
Input A: extended face crop 224x224x3     Input B: high-frequency map 224x224x1
         (bbox x1.3, clamped)                    (luminance - GaussianBlur)
               |                                          |
               v                                          v
┌─────────────────────────────────────┐    ┌────────────────────────────┐
│  DINOv2 ViT-B (patch14, pretrained) │    │  HF-CNN branch (scratch)    │
│  .  CLS token        -> v_cls (D)   │    │  conv32 -> conv64 s2 ->     │
│  .  patch tokens     -> 14x14 (D)   │    │  conv128 s2 -> GAP -> 768   │
└──────────────────┬──────────────────┘    └──────────────┬─────────────┘
                   |                                     |
                   |                                      +-> HF binary head (BCE, gamma)
                   |                                      |
                   +---------[ v_cls ; v_hf ] 2D --------+
                                |  fusion MLP (2D->D, ReLU, Dropout)
                                +-----------> binary head      -> P(live)     (BCE)
                                +-----------> spoof-type head   -> 10 classes  (CE)
   patch tokens (14x14xD) ---> depth head (1x1 conv -> bilinear x8 -> 3x3 convs)
                                -> depth map 112x112        (MSE / Smooth-L1, face-masked)
```

## Loss

```
L = BCE(binary) + lambda_d * D(depth, face-masked) + gamma_hf * BCE(hf) + lambda_t * CE(spoof_type)
```

The depth term is **OPTIONAL**: with `model.use_depth_head: false` (default) the
model has **no depth head** and the loss is simply
`BCE(binary) + gamma_hf * BCE(hf) + lambda_t * CE(spoof_type)`. Turn depth on as
an experiment with `model.use_depth_head: true` (and run `pad depth_targets` first).

- `D` is **Smooth-L1 (Huber beta=0.1)** by default (MSE ablatable), **confidence-weighted**
  per-image (`w = clamp(var_i / median_var, 0.3, 2.0)` on the estimated samples; flats weight 1).
- Depth targets per class (when depth is enabled):
  | group | classes | target |
  |---|---|---|
  | estimated | live, face_mask, upper_body_mask, 3d_mask | Depth-Anything pseudo-depth (face rect, [0,1]) |
  | flat | photo, poster, a4, region_mask, pc_pad, phone | constant zero (face rect) |
- Sequence: `Smooth-L1 > MSE` for noisy pseudo-depth targets; MSE kept as an ablation.
- The depth head consumes the DINOv2 **patch tokens**; when disabled those tokens
  are simply not used by any head.
- **Evaluation protocol** for the depth add-on (baseline vs +depth, depth
  diagnostics, expected 3D-mask caveats) is in `report.md §2.1`.

## Training (GPU, CUDA)

- AMP fp16 (`torch.autocast` + GradScaler), batch 20 (10 classes x 2) @224 px.
- AdamW: backbone 2e-5 / heads 1e-4; cosine + 2-epoch warmup; 40 epochs default.
- Augmentation: flip, rotate ±5°, scale ±5%, color-jitter, **grayscale p=0.5** (before HF extraction).
- Depth trivial-solution guard: val var(estimated)/var(flat) monitored; abort on collapse.
- Early stopping on val ACER; best checkpoint saved.

## Data splits

- Stratified, identity-exclusive: per-spoof-type equal counts, subject-disjoint
  train/val/test (70/15/15), secondary strata (environment, illumination) balanced.
- Balanced batches: `StratifiedBatchSampler` (n_per class per batch).

## Evaluation

- APCER / BPCER / ACER / AUC / TPR@FPR at BPCER ~ 1% (threshold from val).
- Per-spoof-type error rates + depth-variance and HF-energy diagnostics.
- Spoof-type head confusion matrix; per-class hard-sample report (nearest-to-boundary).
- TTA (horizontal flip averaging) at eval.
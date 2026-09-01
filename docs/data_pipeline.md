# Data pipeline reference

Detailed spec of how the dataset is built and what every artifact means.
Companion to `docs/ONBOARDING.md` (read that first for the commands).

## Inputs

1. **Raw mirror on disk** (any location, configure `data.root`):
   ```
   <root>/Data/{train,test}/<subject_id>/{live,spoof}/<img>.png|jpg
   <root>/Data/{train,test}/<subject_id>/{live,spoof}/<img>_BB.txt
   ```
   - The official folders `train/` and `test/` are **pooled** by `prepare` (they
     are arbitrary download parts; subject ids do not overlap between them).
   - `<img>_BB.txt` holds the face bbox: `x y w h [conf]` (top-left + size).
   - Images without a bbox are skipped (cannot be face-cropped for PAD).

2. **Official annotations** `data/labels/label.csv`:
   - Fetch + verify with one command (commit-checksum checked):
     `bash scripts/fetch_annotations.sh`
   - Source: Kaggle `tungnguyentien/celeba-spoof-crop-1-9` →
     `CelebA_Spoof_crop_1_9/data_1.0_128/label.csv` (~60 MB, no images).
   - Row index = relative path `split/subject/class/<img>` (matches the mirror).
   - Column meaning (official 44-column vector): `[0:40]` face attributes,
     `[40]` spoof type 0-9, `[41]` illumination, `[42]` environment.
   - Images whose label isn't in the CSV (or with an invalid spoof type) are
     dropped unless `--include-unknown` (then kept as 999).
   - `pad prepare` verifies `label.csv` against the committed
     `data/labels/label.csv.sha256` check; mismatch → hard error telling you to
     re-run `bash scripts/fetch_annotations.sh`.

## `pad prepare`

1. Walks the pooled mirror, reads each bbox, joins `label.csv`.
2. Builds the combined manifest (`data/crawl.csv`, 238k images on the current
   mirror) — includes every usable annotated image.
3. Stratified, **identity-exclusive** 70/15/15 subset with **equal per-spoof-type
   counts** (secondary strata: environment, illumination).
4. Writes `data/subsets/{train,val,test}.csv` + `data/subsets/balance_report.md`
   (fails with `RESULT: FAIL` if any invariant is violated — equal classes,
   no subject overlap, all 10 types present, no dup paths).

### 20,000 default subset

| split | total | per spoof type |
|---|---|---|
| train | 14,000 | 1,400 |
| val | 3,000 | 300 |
| test | 3,000 | 300 |

Change with `data.subset.budget_total` (e.g. `configs/exp_smoke.yaml` uses 5,000
through a separate `subsets_dir`).

## CSV schema (the metadata contract)

Each row of `data/subsets/{train,val,test}.csv`:

| column | type | meaning |
|---|---|---|
| `image_id` | str | flattened unique id from the source path |
| `image_path` | str | path to the actual image (after `export`: under `data/dataset`) |
| `rel_path` | str | path relative to `data/dataset` |
| `subject_id` | str | identity (used for disjoint splits) |
| `split` | str | `train` / `val` / `test` |
| `spoof_type` | int 0–9 | official attack order (0=live) |
| `is_live` | int 0/1 | 1 if spoof_type==0 |
| `environment` | int | 0..2 from label.csv |
| `illumination` | int | 0..4 from label.csv |
| `x1,y1,x2,y2` | float | face bbox (absolute pixels) |
| `crop_x1..crop_y2` | int | extended-crop box (bbox ×1.3) used by the dataset |
| `face_x1..face_y2` | float | copy of x1..y2 (dataset reads these for depth masking) |

### Who consumes these CSVs

- `pad train` → `train.load_splits()` → `data.build_loaders`
- `pad evaluate` → reads the split CSV for that split
- `pad depth_targets` → reads rows, opens `image_path`, crops, caches depth

## `pad export`

- Reads `data/subsets/{train,val,test}.csv`, **copies** every image into
  `data/dataset/{split}/{spoof_order_name}/<basename>` (real files; default
  `link-mode=copy`, also `hardlink`/`symlink`).
- Collision-safe: if two *different* source images would share a basename in one
  class folder, appends `__2`, `__3`, ….
- Rewrites the CSVs' `image_path` / `rel_path` to the new location **in place**,
  so downstream steps are fully self-contained under `data/`.
- Prints per-class counts and validates all source files exist first.

Spoof order → folder name (from `split.IDX_TO_CLASS`):
`0 live, 1 photo, 2 poster, 3 a4, 4 face_mask, 5 upper_body_mask,
6 region_mask, 7 pc_pad, 8 phone, 9 3d_mask`.

## Depth cache (`pad depth_targets`)

- Only for **ESTIMATED** classes: `live, face_mask, upper_body_mask, 3d_mask`.
- Forward Depth-Anything-V2-Small (`-hf` transformers checkpoint) on the
  extended crop; normalize relative to the face region; cache as
  `data/caches/depth_v2_small/{split}/{image_id}.npz`
  `{depth: (224,224) [0,1], mask: (224,224)}`.
- FLAT classes (photo..phone) get a **flat zero** target synthesized at load
  time — no cache entry needed.

## Important gotchas

- Don't point `prepare` at `data/dataset` as the data root — it would try to
  re-crawl the organized tree. `data.root` should point at the original mirror.
- After `export`, the CSVs carry paths into `data/dataset`; re-running `prepare`
  regenerates them pointing back at the mirror (re-running `export` fixes it).
- `data/` (incl. `data/dataset`) is gitignored: it is a build artifact, not
  source.
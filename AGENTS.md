# Overview
This is a repo for single image liveness detection (face presentation attack
detection, PAD) using samples from https://github.com/ZhangYuanhan-AI/CelebA-Spoof

The canonical pipeline is:
`pad prepare` (build stratified train/val/test CSVs from the on-disk mirror +
`data/labels/label.csv`) → `pad export` (copy images into `data/dataset/{split}/{class}/`)
→ `pad depth_targets` (optional pseudo-depth cache) → `pad train` → `pad evaluate`
→ `pad predict`.

# Project-specific conventions
* All CLIs are Google Fire: `uv run python -m pad <cmd> --flag value`.
  `pad --help` lists them (prepare, export, train, evaluate, depth_targets, predict).
* The dataset is self-contained under `data/`:
  - `data/subsets/{train,val,test}.csv` = metadata truth (bbox, spoof_type,
    environment, illumination); `image_path` points at the images.
  - `data/dataset/{train,val,test}/{spoof_name}/<img>` = the actual images (copies).
* `data/` and `results/` are gitignored build artifacts; never commit them.
* New agents should start from `docs/ONBOARDING.md`; data-layer details in
  `docs/data_pipeline.md`, model details in `docs/architecture.md`.

# Python rules
* Use uv as a package manager
* Use ty as a typechecker
* use pytest for unit testing
* use ruff for linting
* use fire for arparsing
* use pydantic for structured input/output
* use torch for ml training scripts

## 4. Agent Workflow Rules
Always run `uv run ruff format` before concluding a task
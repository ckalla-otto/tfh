"""Offline pseudo-depth target generation for ESTIMATED classes.

For live + physical-3D classes (face_mask, upper_body_mask, 3d_mask) we run a
SOTA monocular depth estimator (default Depth-Anything-V2-Small via HF
Transformers) on the extended crop, extract the face-region map, normalize it,
and cache it as `{cache_dir}/{split}/{image_id}.npz`.

FLAT classes need no estimation: the dataloader synthesizes a flat plane.

Usage:
  python -m pad.depth_targets --config configs/exp_smoke.yaml --splits train val test
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .data import make_extended_crop_bbox
from .split import ESTIMATED_CLASSES, FLAT_CLASSES, IDX_TO_CLASS

CACHE_RES = 224


def _build_estimator(kind: str, model_id: str):
    """Return `estimator(pil_rgb) -> np.ndarray (1, H, W) in [0,1]`."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    mdl = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    if dtype == torch.float16:
        mdl.half()

    def _predict(pil_rgb: Image.Image) -> np.ndarray:
        inputs = proc(images=pil_rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            depth = mdl(**inputs).predicted_depth  # (1, H, W)
        depth = depth.float().cpu().numpy()       # (1, H, W) in [0, 1]
        return depth

    return _predict


def _process_row(row, estimator, cfg: dict, out_dir: Path) -> None:
    key = str(row["image_id"])
    out_file = out_dir / f"{key}.npz"
    if out_file.exists():
        return
    image_path = Path(row["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image for {key}: {image_path}")

    margin = float(cfg["data"]["crop"]["margin_factor"])
    min_side = int(cfg["data"]["crop"].get("min_side", 64))
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    c = make_extended_crop_bbox(
        w, h,
        float(row["face_x1"]), float(row["face_y1"]),
        float(row["face_x2"]), float(row["face_y2"]),
        margin, min_side,
    )
    crop = img.crop(c)

    depth = estimator(crop)                 # (1, H_est, W_est)
    depth_r = np.array(
        Image.fromarray((np.clip(depth[0], 0, 1) * 255).astype(np.uint8)).resize(
            (CACHE_RES, CACHE_RES), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0

    # face rectangle on the 224 grid
    mask = np.zeros((CACHE_RES, CACHE_RES), dtype=np.float32)
    cw, ch = crop.size
    fx0 = min(max(int(round(row["face_x1"] - c[0])), 0), cw - 1)
    fy0 = min(max(int(round(row["face_y1"] - c[1])), 0), ch - 1)
    fx2 = min(max(int(round(row["face_x2"] - c[0])), 0), cw)
    fy2 = min(max(int(round(row["face_y2"] - c[1])), 0), ch)
    if fx2 > fx0 and fy2 > fy0:
        sx = CACHE_RES / cw
        sy = CACHE_RES / ch
        mask[int(fy0 * sy):int(fy2 * sy), int(fx0 * sx):int(fx2 * sx)] = 1.0

    # normalize relative to the face region only
    norm = depth_r * mask
    face_max = float(norm[mask > 0].max()) if mask.sum() > 0 else 0.0
    if face_max > 1e-6:
        norm = norm / face_max
    norm = np.clip(norm, 0.0, 1.0) * mask

    np.savez_compressed(out_file, depth=norm, mask=mask)


def main(
    config: str = "configs/exp_smoke.yaml",
    splits: str = "train val test",
    subset_dir: str = None,
) -> None:
    """Generate offline pseudo-depth targets. Fire CLI: `python -m pad.depth_targets`."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src on path
    from pad.utils import load_config, get_logger
    from pad.split import ESTIMATED_CLASSES, IDX_TO_CLASS

    cfg = load_config(config)
    subsets_dir = Path(subset_dir or cfg["data"]["subsets_dir"])
    depth_cfg = cfg["depth"]
    cache_dir = Path(depth_cfg["cache_dir"])
    logger = get_logger(log_file=str(cache_dir / "_gen.log"))

    # ensure subset CSVs exist
    for split in splits.split():
        f = subsets_dir / f"{split}.csv"
        if not f.exists():
            raise FileNotFoundError(
                f"{f} missing. Run the split step first (train.py --rebuild-splits)."
            )

    logger.info(
        "Building estimator %s (%s)", depth_cfg["estimator"], depth_cfg["dav2_model_id"]
    )
    estimator = _build_estimator(depth_cfg["estimator"], depth_cfg["dav2_model_id"])
    total_done = 0
    for split in splits.split():
        out_dir = cache_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = pd.read_csv(subsets_dir / f"{split}.csv")
        done = 0
        for _, row in rows.iterrows():
            cls = IDX_TO_CLASS[int(row["spoof_type"])]
            if cls not in ESTIMATED_CLASSES:
                continue
            _process_row(row, estimator, cfg, out_dir)
            done += 1
        total_done += done
        logger.info("[%s] estimated-cache entries processed=%d", split, done)
    logger.info("FINISH depth targets: processed=%d cache entries", total_done)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
"""Single-image spoof/live prediction from a trained checkpoint.

Usage:
  uv run python -m pad predict --image_path path/to/img.jpg \
      --ckpt results/run/best.pt --config configs/base.yaml
  uv run python -m pad predict --image_path img.jpg --ckpt best.pt \
      --bbox "10 20 200 220"     # optional face bbox "x1 y1 x2 y2"
  uv run python -m pad predict --image_path img.jpg --ckpt best.pt --no-tta

Scoring convention:
  live_prob    = P(bona-fide face)  (sigmoid of the binary head, with optional
                 horizontal-flip TTA averaging -> final decision)
  spoof_prob   = 1 - live_prob
  spoof_type   = argmax of the 10-way spoof-type head (dataset index 0-9)

If no --bbox is given the full image is center-cropped/resized to the model
input size. The model was TRAINED on extended face crops; for best results pass
a face bounding box (or crop the face region first).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as F

from .split import IDX_TO_CLASS
from .utils import get_logger, load_config, resolve_device


def _prepare_crop(img: Image.Image, bbox, size: int, margin: float) -> Image.Image:
    """Return a squared image ready for the model (extended-crop semantics).

    If `bbox` (x1,y1,x2,y2) is given, expands it by `margin` (default 1.3) like
    the training dataset; otherwise uses the full image bounds (center crop).
    """
    w, h = img.size
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ew, eh = bw * margin, bh * margin
        box = (
            int(max(0, cx - ew / 2)),
            int(max(0, cy - eh / 2)),
            int(min(w, cx + ew / 2)),
            int(min(h, cy + eh / 2)),
        )
        crop = img.crop(box)
    else:
        crop = img
    if crop.size != (size, size):
        crop = F.center_crop(crop, min(crop.size))
        crop = crop.resize((size, size), Image.BILINEAR)
    return crop


def _hf_map(img_t: torch.Tensor, sigma: float = 3.0, kernel: int = 7) -> torch.Tensor:
    """Luminance high-frequency map (matches training augmentation)."""
    gray = F.rgb_to_grayscale(img_t, num_output_channels=1)
    blurred = F.gaussian_blur(gray, kernel_size=[kernel, kernel], sigma=sigma)
    return gray - blurred


@torch.no_grad()
def _predict_one(model, img_t: torch.Tensor, hf_t: torch.Tensor, use_flip: bool) -> dict:
    """Single-batch forward with optional flip TTA -> P(live), depth, spoof_type."""
    model.eval()
    out = model(img_t, hf_t)
    p_live = torch.sigmoid(out["binary"]).float().cpu()
    dep = out["depth"].float().cpu()
    if use_flip:
        out_f = model(torch.flip(img_t, dims=[3]), torch.flip(hf_t, dims=[3]))
        p_live = 0.5 * (p_live + torch.sigmoid(out_f["binary"]).float().cpu())
        dep = 0.5 * (dep + out_f["depth"].float().cpu())
    stype = int(out["spoof_type"].argmax(dim=1).item()) if "spoof_type" in out else None
    return {"live_prob": float(p_live.item()), "depth": dep.numpy()[0, 0], "spoof_type": stype}


def predict_image(
    image_path: str,
    ckpt: str,
    config: str = "configs/base.yaml",
    bbox: str = None,
    no_tta: bool = False,
    device: str = None,
    auto_face: bool = True,
) -> dict:
    """Load the model + checkpoint and predict a single image.

    bbox:        optional "x1 y1 x2 y2" face box. If omitted and auto_face=True,
                 InsightFace detects the face automatically.
    Returns dict with: live_prob, spoof_prob, spoof_type, spoof_type_name,
    decision ("live"/"spoof"), bbox, margin, img_size, auto_face.
    """
    logger = get_logger()
    cfg = load_config(config)
    dev = resolve_device(device)
    logger.info("device=%s", dev)

    from .model import build_model

    model = build_model(cfg)
    ck = torch.load(ckpt, map_location="cpu")
    model.load_state_dict({k: v for k, v in ck["model"].items() if k in model.state_dict()})
    model.to(dev)
    model.eval()
    logger.info("loaded checkpoint: %s", ckpt)

    img = Image.open(image_path).convert("RGB")
    size = int(cfg["data"]["crop"]["size"])
    margin = float(cfg["data"]["crop"].get("margin_factor", 1.3))
    n_bbox = tuple(map(float, bbox.split())) if bbox else None
    used_auto_face = False
    if n_bbox is None and auto_face:
        from .face import detect_face_bbox

        n_bbox = detect_face_bbox(img)
        used_auto_face = n_bbox is not None
        if n_bbox is None:
            logger.warning("no face detected by InsightFace; falling back to full-image center crop")
    crop = _prepare_crop(img, n_bbox, size, margin)
    tta_cfg = cfg.get("eval", {}).get("tta", {})
    use_flip = (not no_tta) and bool(tta_cfg.get("enabled", True)) and "flip" in tta_cfg.get(
        "transforms", ["flip"]
    )

    img_t = F.to_tensor(crop).unsqueeze(0).to(dev)
    hf_t = _hf_map(img_t).to(dev)
    res = _predict_one(model, img_t, hf_t, use_flip)

    st_name = IDX_TO_CLASS[res["spoof_type"]] if res["spoof_type"] is not None else None
    decision = "live" if res["live_prob"] >= 0.5 else "spoof"
    out = {
        "live_prob": res["live_prob"],
        "spoof_prob": 1.0 - res["live_prob"],
        "spoof_type": res["spoof_type"],
        "spoof_type_name": st_name,
        "decision": decision,
        "bbox": n_bbox,
        "margin": margin,
        "img_size": (crop.size[0], crop.size[1]),
        "tta_flip": use_flip,
        "auto_face": used_auto_face,
    }
    logger.info(
        "P(live)=%.4f P(spoof)=%.4f | type=%s | decision=%s",
        out["live_prob"], out["spoof_prob"], st_name, decision,
    )
    return out


def main(
    image_path: str,
    ckpt: str,
    config: str = "configs/base.yaml",
    bbox: str = None,
    no_tta: bool = False,
    device: str = None,
    auto_face: bool = True,
) -> None:
    """Predict a single image from the CLI (Fire). Prints live/spoof + probs."""
    res = predict_image(
        image_path=image_path, ckpt=ckpt, config=config,
        bbox=bbox, no_tta=no_tta, device=device, auto_face=auto_face,
    )
    print(f"image            : {image_path}")
    print(f"live probability : {res['live_prob']:.4f}")
    print(f"spoof probability: {res['spoof_prob']:.4f}")
    st_name = res["spoof_type_name"]
    print(f"spoof type       : {st_name} (idx {res['spoof_type']})" if res["spoof_type"] is not None
          else "spoof type       : n/a")
    print(f"decision         : {res['decision']}")
    if res["bbox"] is not None:
        kind = "auto-detected" if res["auto_face"] else "given"
        print(f"face bbox [{kind}]  : {res['bbox']}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
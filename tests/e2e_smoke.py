"""End-to-end smoke test (CPU): dataset -> model -> loss -> train step -> eval.

Creates tiny synthetic images so nothing about the real Kaggle mirror is required.
Run:  PYTHONPATH=src ./.venv/bin/python tests/e2e_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pad.split import build_subset, ESTIMATED_CLASSES, IDX_TO_CLASS
import pandas as pd


def synthetic_crawl(tmp: Path, n_subjects=12) -> pd.DataFrame:
    rows = []
    idx = 0
    (tmp / "imgs").mkdir(parents=True, exist_ok=True)
    for s in range(n_subjects):
        subj = f"subj_{s}"
        for cls in range(10):
            for im in range(8):  # 4 env * 2
                arr = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
                f = tmp / "imgs" / f"{idx:05d}.jpg"
                Image.fromarray(arr).save(f)
                rows.append({
                    "image_id": f"{idx:05d}",
                    "image_path": str(f),
                    "subject_id": subj,
                    "spoof_type": cls,
                    "is_live": int(cls == 0),
                    "environment": im % 2,
                    "illumination": (im * 7) % 4,
                    "x1": 30, "y1": 40, "x2": 120, "y2": 150,
                })
                idx += 1
    return pd.DataFrame(rows)


def main() -> None:
    from torch.utils.data import DataLoader
    from pad.data import PADDataset, StratifiedBatchSampler, build_loaders, pad_collate
    from pad.model import build_model
    from pad.losses import PADLoss
    from pad.inference import predict_epoch
    from pad.evaluate import evaluate_split_and_report
    from pad.utils import load_config, resolve_device, get_logger

    logger = get_logger()
    tmp = Path("/tmp/pad_smoke")
    if tmp.exists():
        import shutil; shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    cfg = load_config("configs/base.yaml", root=".")
    cfg["data"]["crawl_meta"] = str(tmp / "crawl.csv")
    cfg["data"]["subsets_dir"] = str(tmp / "subsets")
    cfg["model"]["backbone"] = "vit_small_patch14_dinov2"  # fast CPU smoke
    cfg["data"]["subset"]["budget_total"] = 480
    cfg["depth"]["enabled"] = False  # skip pseudo-depth cache for the smoke
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = 20
    cfg["train"]["workers"] = 0
    cfg["out_dir"] = str(tmp / "results")

    crawl = synthetic_crawl(tmp)
    crawl.to_csv(cfg["data"]["crawl_meta"], index=False)

    from pad.train import ensure_splits, add_crop_columns
    splits_df = ensure_splits(cfg, logger, rebuild=True)
    loaders, _ = build_loaders(cfg, splits_df, depth_cache=None, seed=42)
    for name, dl in loaders.items():
        logger.info("%s loader: %d samples, %d batches", name, len(dl.dataset), len(dl))
        for batch in dl:
            for k in ("img", "hf", "depth", "depth_mask", "label", "spoof_type"):
                assert batch[k].ndim >= 2
            break

    device = resolve_device()
    model = build_model(cfg).to(device)
    loss_fn = PADLoss(cfg)
    logger.info("model params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    # forward + loss on one batch
    model.train()
    batch = next(iter(loaders["train"]))
    img, hf = batch["img"].to(device), batch["hf"].to(device)
    tg = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    pred = model(img, hf)
    losses = loss_fn(pred, tg)
    logger.info("losses: total=%.4f bce=%.4f depth=%.4f hf=%.4f type=%.4f",
                losses["total"].item(), losses["bce"].item(), losses["depth"].item(),
                losses["hf_bce"].item(), losses["type_ce"].item())
    assert torch.isfinite(losses["total"]), "non-finite loss"

    # one optimization step
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses["total"].backward()
    optim.step(); optim.zero_grad()
    logger.info("single optim step OK")

    # eval pipeline on the test loader (TTA off for speed)
    model.eval()
    out_dir = Path(cfg["out_dir"]) / "smoke_eval"
    result = evaluate_split_and_report(model, loaders["test"], cfg, device,
                                       out_dir, enable_tta=False)
    print("METRICS:", {k: round(v, 4) for k, v in result["metrics"].items()
                       if isinstance(v, float)})
    assert (out_dir / "per_type.csv").exists()
    assert (out_dir / "hard_samples.csv").exists()
    assert (out_dir / "hard_samples_report.md").exists()
    logger.info("E2E SMOKE OK — all artifacts written")


if __name__ == "__main__":
    main()
"""Training entry point (CLI).

Usage:
  python -m pad.train --config configs/exp_smoke.yaml --run-name smoke

Pipeline:
  1. load config (+ merge includes), set seed, resolve device
  2. build/refresh stratified identity-exclusive subset CSVs (--rebuild-splits)
  3. build train/val/test loaders (per-class-balanced training batches)
  4. train with AMP, cosine schedule, depth trivial-guard, early stopping
  5. evaluate test split with the ISO metrics via evaluate.py

When the depth cache has not been generated yet, set depth.enabled=false.
"""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import build_loaders, make_extended_crop_bbox
from .inference import predict_epoch
from .losses import PADLoss
from .model import build_model
from .split import ESTIMATED_CLASSES, IDX_TO_CLASS, build_subset, read_crawl
from .utils import (
    compute_pad_metrics,
    find_threshold,
    get_logger,
    load_config,
    resolve_device,
    save_json,
    set_seed,
)


def add_crop_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add extended-crop + face bbox columns required by the dataset."""
    margin = float(cfg["data"]["crop"]["margin_factor"])
    min_side = int(cfg["data"]["crop"].get("min_side", 64))
    mode = str(cfg["data"]["crop"]["mode"])
    if "crop_x1" in df.columns:
        return df
    from PIL import Image

    crop_x1, crop_y1, crop_x2, crop_y2 = [], [], [], []
    for _, r in df.iterrows():
        try:
            with Image.open(r["image_path"]) as im:
                w, h = im.size
        except Exception:
            # image not downloaded yet (metadata-only crawl path): use a synthetic
            # size derived from the bbox. These crop columns are informational —
            # the dataset recomputes real crops at load time from the face bbox.
            w = max(int(float(r["x2"])), int(float(r["y2"])), 1) + 10
            h = w
        if mode == "tight_bbox":
            c = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
        else:
            c = make_extended_crop_bbox(
                w,
                h,
                float(r["x1"]),
                float(r["y1"]),
                float(r["x2"]),
                float(r["y2"]),
                margin,
                min_side,
            )
        crop_x1.append(c[0])
        crop_y1.append(c[1])
        crop_x2.append(c[2])
        crop_y2.append(c[3])
    df["crop_x1"], df["crop_y1"] = crop_x1, crop_y1
    df["crop_x2"], df["crop_y2"] = crop_x2, crop_y2
    df["face_x1"], df["face_y1"] = df["x1"], df["y1"]
    df["face_x2"], df["face_y2"] = df["x2"], df["y2"]
    return df


def make_splits(config: str = "configs/exp_smoke.yaml", crawl: str = None) -> None:
    """Build only the stratified subset CSVs + balance report (Fire CLI).

    Uses the crawl manifest (data.crawl_meta, or --crawl to override) WITHOUT
    needing images on disk, so you can sample the subset and then download only
    those images.
    """
    cfg = load_config(config)
    if crawl:
        cfg["data"]["crawl_meta"] = crawl
    cfg["out_dir"] = str(Path(cfg["out_dir"]))
    set_seed(cfg["data"]["subset"]["seed"])
    logger = get_logger()
    splits = ensure_splits(cfg, logger, rebuild=True)
    logger.info("subsets ready under %s", cfg["data"]["subsets_dir"])
    for s, df in splits.items():
        logger.info("%s: %d images", s, len(df))


def ensure_splits(cfg: dict, logger, rebuild: bool) -> dict:
    """Build subset CSVs if missing (or forced) and return split DataFrames."""
    subsets_dir = Path(cfg["data"]["subsets_dir"])
    subsets_dir.mkdir(parents=True, exist_ok=True)
    needed = [subsets_dir / f"{s}.csv" for s in ("train", "val", "test")]
    if rebuild or not all(f.exists() for f in needed):
        budget = cfg["data"]["subset"]["budget_total"]
        logger.info("Building stratified subset (budget=%d)...", budget)
        crawl = read_crawl(
            cfg["data"]["crawl_meta"], cfg["data"].get("layout", "kaggle_csv")
        )
        split_result = build_subset(
            crawl,
            budget_total=budget,
            split_fracs=tuple(cfg["data"]["subset"]["split"]),
            seed=cfg["data"]["subset"]["seed"],
            secondary=cfg["data"]["subset"].get(
                "secondary", ["environment", "illumination"]
            ),
        )
        for name in ("train", "val", "test"):
            df = add_crop_columns(split_result.splits[name], cfg)
            df.to_csv(subsets_dir / f"{name}.csv", index=False)
        (subsets_dir / "balance_report.md").write_text(split_result.report)
        logger.info("Subset report -> %s/balance_report.md", subsets_dir)
        logger.info(
            "Split sizes: %s",
            {s: len(split_result.splits[s]) for s in ("train", "val", "test")},
        )
        if "RESULT: FAIL" in split_result.report:
            raise RuntimeError("Subset verification FAILED — refusing to train.")
    return {s: pd.read_csv(subsets_dir / f"{s}.csv") for s in ("train", "val", "test")}


def depth_guard_stats(pred: dict, cfg: dict) -> dict:
    """Trivial-solution guard: variance on estimated vs flat groups (val)."""
    dg = cfg.get("depth_guard", {})
    est = np.array(
        [IDX_TO_CLASS[int(t)] in ESTIMATED_CLASSES for t in pred["spoof_type"]]
    )
    var_est = pred["depth_var"][est]
    var_flat = pred["depth_var"][~est]
    stats = {
        "var_estimated": float(var_est.mean()) if len(var_est) else float("nan"),
        "var_flat": float(var_flat.mean()) if len(var_flat) else float("nan"),
        "min_live_variance": float(dg.get("min_live_variance", 0.03)),
        "min_ratio": float(dg.get("min_ratio", 5.0)),
    }
    stats["ratio"] = (
        stats["var_estimated"] / max(stats["var_flat"], 1e-9)
        if np.isfinite(stats["var_estimated"])
        else float("nan")
    )
    stats["violated"] = bool(
        (
            np.isfinite(stats["var_estimated"])
            and stats["var_estimated"] < stats["min_live_variance"]
        )
        or (np.isfinite(stats["ratio"]) and stats["ratio"] < stats["min_ratio"])
    )
    return stats


def eval_val(model, val_loader, cfg, device):
    """Compute val metrics + guard stats (no TTA during training)."""
    pred = predict_epoch(model, val_loader, cfg, device, enable_tta=False)
    thr = find_threshold(
        pred["scores_live"], pred["labels"], cfg["eval"].get("bpcer_target", 0.01)
    )
    met = compute_pad_metrics(pred["scores_live"], pred["labels"], thr)
    met["threshold"] = thr
    guard = depth_guard_stats(pred, cfg)
    res = {f"val/{k}": v for k, v in met.items()}
    res.update({f"guard/{k}": v for k, v in guard.items() if k != "violated"})
    res["guard/violated"] = guard["violated"]
    return res, guard


def run_training(
    cfg: dict,
    run_name: str = "run",
    rebuild_splits: bool = False,
    epochs: int = None,
    device: str = None,
) -> None:
    set_seed(cfg["data"]["subset"]["seed"])
    out_dir = Path(cfg["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_file=str(out_dir / "train.log"))
    device = resolve_device(device)
    logger.info("Device: %s", device)

    splits = ensure_splits(cfg, logger, rebuild=rebuild_splits)
    depth_enabled = cfg["depth"].get("enabled", True)
    depth_cache = cfg["depth"].get("cache_dir") if depth_enabled else None
    loaders, samplers = build_loaders(
        cfg, splits, depth_cache=depth_cache, seed=cfg["data"]["subset"]["seed"]
    )
    logger.info(
        "Data: train=%d val=%d test=%d",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        len(loaders["test"].dataset),
    )

    model = build_model(cfg).to(device)
    loss_fn = PADLoss(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %.1fM", n_params / 1e6)

    backbone_ids = set(id(p) for p in model.backbone.parameters())
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optim = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": float(cfg["train"]["lr_backbone"]),
            },
            {"params": head_params, "lr": float(cfg["train"]["lr_head"])},
        ],
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(epochs or cfg["train"]["epochs"])
    warmup = int(cfg["train"].get("warmup_epochs", 2))

    def lr_lambda(ep: int) -> float:
        if ep < warmup:
            return 0.1 + 0.9 * ep / max(warmup, 1)
        p = (ep - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    amp_enabled = bool(cfg["train"].get("amp", True)) and device in ("cuda", "mps")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device == "cuda")
    accum = int(cfg["train"].get("grad_accum", 1))
    best_ckpt = out_dir / "best.pt"
    last_ckpt = out_dir / "last.pt"

    best_acer, best_epoch = float("inf"), -1
    best_threshold = 0.5
    guard_patience = int(cfg.get("depth_guard", {}).get("patience", 5))
    guard_streak = 0
    early_patience = int(cfg["train"].get("early_stop_patience", 10))
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        if samplers["train"] is not None:
            samplers["train"].set_epoch(epoch)
        t0 = time.time()
        keys = ("total", "bce", "depth", "hf_bce", "type_ce")
        agg = {k: 0.0 for k in keys}
        for it, batch in enumerate(loaders["train"]):
            img = batch["img"].to(device)
            hf_in = batch["hf"].to(device)
            tg = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            with torch.autocast(
                device_type="cuda" if device == "cuda" else "cpu", enabled=amp_enabled
            ):
                pred = model(img, hf_in)
                losses = loss_fn(pred, tg)
                loss = losses["total"] / accum
            if device == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (it + 1) % accum == 0 or it == len(loaders["train"]) - 1:
                if device == "cuda":
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
            for k in keys:
                agg[k] += float(losses[k].item())
            log_every = cfg["train"].get("log_every", 20)
            if log_every and (it + 1) % log_every == 0:
                logger.info(
                    "ep %d it %d/%d total=%.4f bce=%.4f depth=%.4f hf=%.4f type=%.4f lr=%.1e",
                    epoch,
                    it + 1,
                    len(loaders["train"]),
                    losses["total"].item(),
                    losses["bce"].item(),
                    losses["depth"].item(),
                    losses["hf_bce"].item(),
                    losses["type_ce"].item(),
                    optim.param_groups[0]["lr"],
                )
        sched.step()

        val_res, guard = eval_val(model, loaders["val"], cfg, device)
        acer = val_res["val/ACER"]
        history.append({"epoch": epoch, **val_res})
        logger.info(
            "ep %d | val ACER=%.4f APCER=%.4f BPCER=%.4f AUC=%.4f | "
            "guard var_est=%.4f var_flat=%.4f ratio=%.1f | %.1fs",
            epoch,
            acer,
            val_res["val/APCER"],
            val_res["val/BPCER"],
            val_res["val/AUC"],
            guard["var_estimated"],
            guard["var_flat"],
            guard["ratio"],
            time.time() - t0,
        )

        torch.save({"model": model.state_dict(), "epoch": epoch}, last_ckpt)
        if acer < best_acer:
            best_acer, best_epoch = acer, epoch
            best_threshold = val_res["val/threshold"]
            shutil.copy(last_ckpt, best_ckpt)
            no_improve = 0
            logger.info("  * new best val ACER %.4f (ep %d)", acer, epoch)
        else:
            no_improve += 1

        if guard["violated"]:
            guard_streak += 1
            logger.warning(
                "Depth-guard VIOLATION streak %d/%d: var_est=%.4f ratio=%.2f "
                "(min_var=%.3f min_ratio=%.1f).",
                guard_streak,
                guard_patience,
                guard["var_estimated"],
                guard["ratio"],
                guard["min_live_variance"],
                guard["min_ratio"],
            )
            if guard_streak >= guard_patience:
                raise RuntimeError(
                    "ABORT: depth head trivial collapse detected (see warnings above)."
                )
        else:
            guard_streak = 0

        if no_improve >= early_patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    logger.info("Loading best checkpoint (ep %d, val ACER %.4f)", best_epoch, best_acer)
    ck = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ck["model"])

    from .evaluate import evaluate_split_and_report

    result = evaluate_split_and_report(
        model,
        loaders["test"],
        cfg,
        device,
        out_dir,
        enable_tta=True,
        threshold=best_threshold,
    )
    result["best_val"] = {"acer": best_acer, "epoch": best_epoch}
    save_json(result, str(out_dir / "test_metrics.json"))
    save_json(history, str(out_dir / "history.json"))
    logger.info("Final test result -> %s/test_metrics.json", out_dir)


def main(
    config: str = "configs/exp_smoke.yaml",
    run_name: str = "run",
    rebuild_splits: bool = False,
    epochs: int = None,
    device: str = None,
    crawl: str = None,
) -> None:
    """Train the PAD model. Fire CLI: `python -m pad.train [flags]`."""
    cfg = load_config(config)
    if crawl:
        cfg["data"]["crawl_meta"] = crawl
    cfg["out_dir"] = str(Path(cfg["out_dir"]))
    run_training(
        cfg,
        run_name=run_name,
        rebuild_splits=rebuild_splits,
        epochs=epochs,
        device=device,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)

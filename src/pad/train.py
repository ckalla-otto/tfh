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

from .data import build_loaders
from .inference import predict_epoch
from .losses import PADLoss
from .model import build_model
from .split import ESTIMATED_CLASSES, IDX_TO_CLASS
from .utils import (
    compute_pad_metrics,
    find_threshold,
    get_logger,
    load_config,
    resolve_device,
    save_json,
    set_seed,
)


def load_splits(cfg: dict) -> dict:
    """Load the train/val/test subset CSVs produced by `pad prepare`.

    Raises if they are missing — run `uv run python -m pad prepare ...` first.
    """
    subsets_dir = Path(cfg["data"]["subsets_dir"])
    missing = [
        s for s in ("train", "val", "test") if not (subsets_dir / f"{s}.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing subset CSVs for {missing} under {subsets_dir}. "
            "Run `uv run python -m pad prepare --data-root <your-data> "
            "--labels data/labels/label.csv --config configs/base.yaml` first."
        )
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
    epochs: int = None,
    device: str = None,
) -> None:
    set_seed(cfg["data"]["subset"]["seed"])
    out_dir = Path(cfg["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_file=str(out_dir / "train.log"))
    device = resolve_device(device)
    logger.info("Device: %s", device)

    splits = load_splits(cfg)
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
    epochs: int = None,
    device: str = None,
) -> None:
    """Train the PAD model. Fire CLI: `python -m pad train [flags]`."""
    cfg = load_config(config)
    cfg["out_dir"] = str(Path(cfg["out_dir"]))
    run_training(cfg, run_name=run_name, epochs=epochs, device=device)


if __name__ == "__main__":
    import fire

    fire.Fire(main)

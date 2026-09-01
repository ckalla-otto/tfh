"""Inference utilities shared by training (val) and evaluation (test).

Scoring convention: `scores_live` = sigmoid(binary logit) = P(bona-fide face).
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import functional as F

from .data import hf_energy


@torch.no_grad()
def predict_epoch(
    model: torch.nn.Module,
    loader,
    cfg: dict,
    device: str,
    enable_tta: bool = False,
) -> dict:
    """Collect per-sample predictions across an entire split.

    Returns dict with arrays:
      scores_live (N,), labels (N,), spoof_type (N,), env (N,), illum (N,),
      keys (list[str]), depth_var (N,), hf_energy (N,), spoof_pred (N,)
    """
    model.eval()
    tta_cfg = cfg.get("eval", {}).get("tta", {})
    use_flip = (
        enable_tta
        and bool(tta_cfg.get("enabled", True))
        and "flip" in tta_cfg.get("transforms", ["flip"])
    )

    scores, labels, stypes, envs, illum, keys = ([] for _ in range(6))
    depth_vars, hf_ens, spoof_pred, paths = [], [], [], []
    depth_res = int(cfg.get("model", {}).get("depth_res", 112))
    use_depth = bool(cfg.get("model", {}).get("use_depth_head", False))

    def _single(img, hf):
        out = model(img, hf)
        p = torch.sigmoid(out["binary"]).squeeze(1)
        # depth is optional; when disabled return a flat (zero) map so metrics
        # that read depth_var still produce sensible (flat) values
        dp = (
            out["depth"][:, 0]
            if use_depth
            else torch.zeros((p.shape[0], depth_res, depth_res), device=p.device)
        )
        sp = None
        if "spoof_type" in out:
            sp = out["spoof_type"].argmax(dim=1)
        return p, dp, sp

    for batch in loader:
        img = batch["img"].to(device)
        hf_in = batch["hf"].to(device)
        p, dp, sp = _single(img, hf_in)
        if use_flip:
            p2, dp2, _ = _single(torch.flip(img, dims=[3]), torch.flip(hf_in, dims=[3]))
            p = 0.5 * (p + p2)
            dp = 0.5 * (dp + dp2)

        mask = batch["depth_mask"].to(device).float()  # (B, res, res)
        cnt = mask.sum(dim=(1, 2)).clamp(min=1.0)
        mean = (dp * mask).sum(dim=(1, 2)) / cnt
        var = ((dp - mean.unsqueeze(1).unsqueeze(2)) ** 2 * mask).sum(dim=(1, 2)) / cnt

        scores.append(p.float().cpu().numpy())
        labels.append(batch["label"].squeeze(1).numpy().astype(np.int64))
        stypes.append(batch["spoof_type"].squeeze(1).numpy())
        envs.append(batch["environment"].squeeze(1).numpy())
        illum.append(batch["illumination"].squeeze(1).numpy())
        keys.extend(batch["keys"])
        depth_vars.append(var.float().cpu().numpy())
        hf_ens.append(hf_energy(hf_in).float().cpu().numpy())
        spoof_pred.append(
            sp.cpu().numpy()
            if sp is not None
            else np.full(p.shape[0], -1, dtype=np.int64)
        )
        paths.extend(batch["path"])

    return {
        "scores_live": np.concatenate(scores),
        "labels": np.concatenate(labels),
        "spoof_type": np.concatenate(stypes),
        "env": np.concatenate(envs),
        "illum": np.concatenate(illum),
        "keys": keys,
        "path": paths,
        "depth_var": np.concatenate(depth_vars),
        "hf_energy": np.concatenate(hf_ens),
        "spoof_pred": np.concatenate(spoof_pred),
    }

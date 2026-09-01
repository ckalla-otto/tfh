"""Combined loss: binary BCE + masked depth (MSE/Smooth-L1, confidence-weighted)
+ HF-branch BCE + spoof-type CE.

L = BCE(binary) + lambda_d * depth  + gamma_hf * BCE(hf) + lambda_t * CE(spoof_type)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .split import SPOOF_TYPES


class PADLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg["loss"]
        self.depth_loss = str(lc.get("depth_loss", "smooth_l1"))
        self.huber_beta = float(lc.get("huber_beta", 0.1))
        self.lambda_depth = float(lc.get("lambda_depth", 1.0))
        self.gamma_hf = float(lc.get("gamma_hf", 0.1))
        self.lambda_type = float(lc.get("lambda_type", 0.3))
        self.conf_weight = bool(lc.get("conf_weight", True))
        self.w_pos = float(lc.get("pos_weight_live", 1.0))

    # ------------------------------------------------------------------
    def _depth_term(self, pred: torch.Tensor, batch: dict) -> torch.Tensor:
        """Masked depth loss with per-image confidence weighting.

        pred:  (B,1,res,res); target/mask passed as (B,res,res) -> unsqueezed.
        """
        target = batch["depth"].unsqueeze(1)
        mask = batch["depth_mask"].unsqueeze(1)

        if self.depth_loss == "mse":
            err = (pred - target) ** 2
        else:
            err = F.smooth_l1_loss(pred, target, reduction="none", beta=self.huber_beta)

        cnt = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        per_sample = (err * mask).sum(dim=(1, 2, 3)) / cnt  # (B,)

        w = torch.ones_like(per_sample)
        if self.conf_weight:
            est = batch["is_estimated"].squeeze(1) > 0.5
            if est.any():
                # masked variance per estimated image: modulates pseudo-label reliability
                m = mask[est]
                t = target[est] * m
                c = cnt[est]
                mean = t.sum(dim=(1, 2, 3)) / c
                center = mean.view(-1, 1, 1, 1)
                var = ((t - center) ** 2 * m).sum(dim=(1, 2, 3)) / c
                med = var.median().clamp(min=1e-6)
                w = w.clone()
                w[est] = (var / med).clamp(0.3, 2.0)
        return (per_sample * w).mean()

    def forward(self, pred: dict, batch: dict) -> dict:
        target = batch["label"].squeeze(1)
        bce = F.binary_cross_entropy_with_logits(
            pred["binary"].squeeze(1),
            target,
            pos_weight=torch.tensor(self.w_pos, device=pred["binary"].device),
        )

        depth_ret = self._depth_term(pred["depth"], batch)
        d_loss = depth_ret

        hf_bce = torch.zeros_like(bce)
        if pred.get("hf_binary") is not None:
            hf_bce = F.binary_cross_entropy_with_logits(
                pred["hf_binary"].squeeze(1), target
            )

        type_ce = torch.zeros_like(bce)
        if pred.get("spoof_type") is not None:
            type_ce = F.cross_entropy(
                pred["spoof_type"], batch["spoof_type"].squeeze(1)
            )

        total = (
            bce
            + self.lambda_depth * d_loss
            + self.gamma_hf * hf_bce
            + self.lambda_type * type_ce
        )
        return {
            "total": total,
            "bce": bce.detach(),
            "depth": d_loss.detach(),
            "hf_bce": hf_bce.detach(),
            "type_ce": type_ce.detach(),
        }

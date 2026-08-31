"""Two-stream PAD model: shared DINOv2 backbone + HF branch + fused heads.

Forward-path summary (locked in the plan):
  extended crop 224^2                 HF map 224^2 (luminance)
        | DINOv2ViT                       | HF-CNN branch
        +-- CLS token (D,)                +-- projection to (D,)
        +-- patch tokens (H/14, W/14, D)
                 |
                 +-- depth head (bilinear upsample) -> depth map (depth_res, depth_res)
        [CLS ; HF-emb] (2D,) -> fusion MLP
                 +-- binary head (1 logit)   -- BCE
                 +-- spoof-type head (10)    -- CE
        HF-emb alone -> hf binary head       -- BCE (aux, gamma)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .utils import DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD

SPOOF_TYPE_N = 10  # Live + 9 attack types


class HFBranch(nn.Module):
    """Lightweight high-frequency texture encoder (RGB -> border/grain cues)."""

    def __init__(self, out_dim: int, channels=(32, 64, 128)):
        super().__init__()
        c0, c1, c2 = channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=3, padding=1), nn.BatchNorm2d(c0), nn.ReLU(inplace=True),
            nn.Conv2d(c0, c1, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(c2, out_dim))
        self.head = nn.Linear(out_dim, 1)  # auxiliary BCE head on the HF embed

    def forward(self, hf: torch.Tensor):
        x = self.net(hf)            # (B, 128, 1, 1)
        emb = self.proj(x)          # (B, D)
        logit = self.head(emb)      # (B, 1)
        return emb, logit


class DepthHead(nn.Module):
    """Patch-token -> depth map regression head (bilinear upsampling)."""

    def __init__(self, in_dim: int, depth_res: int = 112):
        super().__init__()
        self.depth_res = depth_res
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, 128, kernel_size=1), nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        x = self.proj(patch_tokens)
        x = F.interpolate(
            x, size=(self.depth_res, self.depth_res),
            mode="bilinear", align_corners=False,
        )
        return self.conv(x)  # (B, 1, depth_res, depth_res)


class PADModel(nn.Module):
    """Shared-DINOv2 PAD model with binary + depth + spoof-type + HF heads."""

    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        self.backbone = timm.create_model(
            mcfg["backbone"],
            pretrained=bool(mcfg.get("pretrained", True)),
            img_size=int(mcfg.get("img_size", 224)),  # DINOv2 defaults to 518; we use 224
            num_classes=0,
        )
        self.embed_dim = getattr(self.backbone, "embed_dim", 768)
        D = self.embed_dim

        self.use_hf = bool(mcfg.get("use_hf_branch", True))
        self.use_spoof_type = bool(mcfg.get("use_spoof_type_head", True))
        hf_cfg = mcfg.get("hf", {})
        self.depth_res = int(mcfg.get("depth_res", 112))

        if self.use_hf:
            self.hf_branch = HFBranch(D, tuple(hf_cfg.get("channels", [32, 64, 128])))
        fusion_in = D + (D if self.use_hf else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, D),
            nn.ReLU(inplace=True),
            nn.Dropout(float(mcfg.get("dropout", 0.1))),
        )
        self.head_binary = nn.Linear(D, 1)
        if self.use_spoof_type:
            self.head_spoof = nn.Linear(D, SPOOF_TYPE_N)

        self.depth_head = DepthHead(D, self.depth_res)

        # registrable tensors (ImageNet stats)
        self.register_buffer(
            "mean", torch.tensor(DEFAULT_IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(DEFAULT_IMAGENET_STD).view(1, 3, 1, 1)
        )

    def _norm(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean.to(img.device)) / self.std.to(img.device)

    def forward(
        self, img: torch.Tensor, hf: torch.Tensor = None
    ) -> dict:
        """All heads in a single pass.

        img: (B,3,H,W) in [0,1]; hf: (B,1,H,W) high-frequency map.
        Returns dict with keys: binary, spoof_type(opt), depth, hf_binary(opt).
        """
        x = self._norm(img)
        feats = self.backbone.forward_features(x)  # (B, 1+N, D), post-norm
        cls = feats[:, 0]                          # (B, D)
        patch = feats[:, 1:]                       # (B, N, D)

        if self.use_hf:
            hf_emb, hf_logit = self.hf_branch(hf)
            fused = self.fusion(torch.cat([cls, hf_emb], dim=-1))
        else:
            hf_logit = None
            fused = self.fusion(cls)

        out = {
            "binary": self.head_binary(fused),  # (B, 1)
            "depth": self._depth_from_patch(patch),
            "hf_binary": hf_logit,
        }
        if self.use_spoof_type:
            out["spoof_type"] = self.head_spoof(fused)
        return out

    def _depth_from_patch(self, patch: torch.Tensor) -> torch.Tensor:
        B, N, D = patch.shape
        side = int(round(N ** 0.5))
        tokens = patch.permute(0, 2, 1).reshape(B, D, side, side)
        return self.depth_head(tokens)  # (B, 1, depth_res, depth_res)


def build_model(cfg: dict) -> PADModel:
    return PADModel(cfg)
"""Dataset, augmentation, high-frequency maps, and stratified samplers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import functional as F

from .split import FLAT_CLASSES, ESTIMATED_CLASSES, IDX_TO_CLASS


def make_extended_crop_bbox(
    w: int,
    h: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    margin: float,
    min_side: int,
) -> Tuple[int, int, int, int]:
    """Expand the face bbox to a centered crop of (margin*w, margin*h), clamped."""
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ew, eh = bw * margin, bh * margin
    cx1 = int(max(0.0, cx - ew / 2.0))
    cy1 = int(max(0.0, cy - eh / 2.0))
    cx2 = int(min(w, cx + ew / 2.0))
    cy2 = int(min(h, cy + eh / 2.0))
    if cx2 - cx1 < min_side or cy2 - cy1 < min_side:
        cxm, cym = (cx1 + cx2) // 2, (cy1 + cy2) // 2
        cx1 = max(0, cxm - min_side // 2)
        cx2 = min(w, cxm + min_side // 2)
        cy1 = max(0, cym - min_side // 2)
        cy2 = min(h, cym + min_side // 2)
    return cx1, cy1, cx2, cy2


def high_frequency_map(
    img: torch.Tensor, sigma: float = 3.0, kernel: int = 7
) -> torch.Tensor:
    """1-channel luminance HF map: gray - gaussian_blur(gray).

    Input (B,3,H,W) in [0,1]; output (B,1,H,W), may be negative. Removing the
    low-pass leaves print-grain / moiré / device-edge cues.
    """
    gray = F.rgb_to_grayscale(img, num_output_channels=1)
    blurred = F.gaussian_blur(gray, kernel_size=[kernel, kernel], sigma=sigma)
    return gray - blurred


def hf_energy(hf_map: torch.Tensor) -> torch.Tensor:
    """Per-sample log HF-band energy (used as a diagnostic in evaluation)."""
    return torch.log1p(hf_map.pow(2).mean(dim=(1, 2, 3)))


class PADDataset(Dataset):
    """Anti-spoofing dataset on top of the stratified subset CSVs.

    Returned dict keys:
      img          (3,H,W) in [0,1]
      hf           (1,H,W) luminance high-frequency map
      depth        (H,W) in [0,1]; face region carries the target, else 0
      depth_mask   (H,W) 0/1 (face rect in crop coordinates)
      label        (1,) int   1=live, 0=spoof
      spoof_type   (1,) int   dataset index 0-9
      environment  (1,) int
      illumination (1,) int
      is_estimated (1,) int   depth target estimated (vs flat)
      key          str        unique image id
    """

    def __init__(
        self,
        split_df,
        cfg: dict,
        split: str,
        augment: bool = False,
        depth_cache: Optional[str] = None,
    ):
        self.cfg = cfg
        self.split = split
        self.augment = augment
        self.size = int(cfg["data"]["crop"]["size"])
        self.mode = str(cfg["data"]["crop"]["mode"])
        self.margin = float(cfg["data"]["crop"]["margin_factor"])
        self.min_side = int(cfg["data"]["crop"].get("min_side", 64))
        self.depth_res = int(cfg.get("model", {}).get("depth_res", 112))
        self.flat_value = float(cfg["depth"].get("flat_value", 0.0))
        self.depth_cache = depth_cache
        self.aug_cfg = cfg["data"]["aug"]
        hf_cfg = cfg["model"].get("hf", {})
        self.hf_sigma = float(hf_cfg.get("blur_sigma", 3.0))
        self.hf_kernel = int(hf_cfg.get("blur_kernel", 7))

        self.rows = split_df.reset_index(drop=True)
        self._paths = list(self.rows["image_path"])
        self._keys = list(self.rows["image_id"].astype(str))
        self._stype = list(self.rows["spoof_type"].astype(int))
        self._env = list(self.rows["environment"].astype(int))
        self._illum = list(self.rows["illumination"].astype(int))
        self._crops = list(
            zip(
                self.rows["crop_x1"].astype(int),
                self.rows["crop_y1"].astype(int),
                self.rows["crop_x2"].astype(int),
                self.rows["crop_y2"].astype(int),
            )
        )
        self._faces = list(
            zip(
                self.rows["face_x1"].astype(int),
                self.rows["face_y1"].astype(int),
                self.rows["face_x2"].astype(int),
                self.rows["face_y2"].astype(int),
            )
        )
        self._geom: Dict[int, Tuple[Image.Image, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _crop_geometry(self, idx: int) -> Tuple[Image.Image, np.ndarray]:
        """Open the image once, return the extended crop + face-rect mask."""
        if idx in self._geom:
            return self._geom[idx]
        path = self._paths[idx]
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing image: {path}")
        img = Image.open(path).convert("RGB")
        w, h = img.size
        fx1, fy1, fx2, fy2 = (float(v) for v in self._faces[idx])
        if self.mode == "tight_bbox":
            c = (int(fx1), int(fy1), int(fx2), int(fy2))
        else:  # extended_bbox
            c = make_extended_crop_bbox(
                w, h, fx1, fy1, fx2, fy2, self.margin, self.min_side
            )
        crop = img.crop(c)
        cw, ch = crop.size
        fmask = np.zeros((ch, cw), dtype=np.float32)
        fx0 = min(max(int(round(fx1 - c[0])), 0), cw - 1)
        fy0 = min(max(int(round(fy1 - c[1])), 0), ch - 1)
        fx2r = min(max(int(round(fx2 - c[0])), 0), cw)
        fy2r = min(max(int(round(fy2 - c[1])), 0), ch)
        if fx2r > fx0 and fy2r > fy0:
            fmask[fy0:fy2r, fx0:fx2r] = 1.0
        self._geom[idx] = (crop, fmask)
        return crop, fmask

    def _depth_target(self, idx: int) -> Optional[np.ndarray]:
        """Load the cached pseudo-depth map for estimated classes (else None)."""
        if self.depth_cache is None:
            return None
        cache_file = Path(self.depth_cache) / self.split / f"{self._keys[idx]}.npz"
        if cache_file.exists():
            with np.load(cache_file) as d:
                return d["depth"].astype(np.float32)
        class_name = IDX_TO_CLASS[self._stype[idx]]
        if class_name not in FLAT_CLASSES:
            raise FileNotFoundError(
                f"Missing depth cache for {cache_file} (class {class_name}); "
                "run depth_targets.py first."
            )
        return None

    def __getitem__(self, idx: int) -> dict:
        crop, fmask = self._crop_geometry(idx)
        size = self.size

        # ----- geometric transforms: applied to BOTH image and face mask -----
        scale = 1.0
        angle = 0.0
        if self.augment:
            sr = self.aug_cfg.get("scale_range", 0.0)
            scale = 1.0 + random.uniform(-sr, sr)
            crop = F.center_crop(
                F.resize(crop, (int(size * scale), int(size * scale))), size
            )
            if self.aug_cfg.get("rotate_deg", 0.0) > 0:
                angle = random.uniform(
                    -self.aug_cfg["rotate_deg"], self.aug_cfg["rotate_deg"]
                )
                crop = F.rotate(crop, angle, interpolation=F.InterpolationMode.BILINEAR)
        elif crop.size != (size, size):
            crop = F.resize(crop, (size, size))

        fmask_t = torch.from_numpy(fmask.astype(np.float32)).unsqueeze(0)
        if self.augment:
            if scale != 1.0:
                fmask_t = F.center_crop(
                    F.resize(fmask_t, (int(size * scale), int(size * scale))), size
                )
            if angle != 0.0:
                fmask_t = F.rotate(
                    fmask_t, angle, interpolation=F.InterpolationMode.NEAREST
                )
        else:
            fmask_t = F.resize(fmask_t, (size, size))

        img = F.to_tensor(crop)  # [0, 1] float

        # ----- photometric transforms (image only) -----
        flip_applied = False
        if self.augment:
            if random.random() < self.aug_cfg.get("grayscale_prob", 0.0):
                img = F.rgb_to_grayscale(img, num_output_channels=3)
            if random.random() < self.aug_cfg.get("blur_prob", 0.0):
                img = F.gaussian_blur(
                    img, kernel_size=[5, 5], sigma=self.aug_cfg.get("blur_sigma", 2.0)
                )
            j = self.aug_cfg.get("color_jitter", 0.0)
            if j > 0:
                img = F.adjust_brightness(img, 1.0 + random.uniform(-j, j))
                img = F.adjust_contrast(img, 1.0 + random.uniform(-j, j))
                img = F.adjust_saturation(img, 1.0 + random.uniform(-j, j))
        if self.aug_cfg.get("flip", False) and random.random() < 0.5:
            img = F.hflip(img)
            fmask_t = F.hflip(fmask_t)
            flip_applied = True

        # ----- HF map computed AFTER augmentation (streams stay consistent) -----
        hf = high_frequency_map(img.unsqueeze(0), self.hf_sigma, self.hf_kernel)[0]

        # ----- depth target + face mask on the depth grid -----
        stype = self._stype[idx]
        depth_full = self._depth_target(idx)
        if depth_full is None:
            depth_full = np.full((size, size), self.flat_value, dtype=np.float32)
        else:
            depth_full = (
                np.array(
                    Image.fromarray(
                        (np.clip(depth_full, 0, 1) * 255).astype(np.uint8)
                    ).resize((size, size), Image.BILINEAR)
                ).astype(np.float32)
                / 255.0
            )
        depth_t = torch.from_numpy(depth_full)
        # align depth with the geometric transforms used on the image
        if self.augment:
            if scale != 1.0:
                depth_t = F.center_crop(
                    F.resize(
                        depth_t.unsqueeze(0), (int(size * scale), int(size * scale))
                    ),
                    size,
                )[0]
            if angle != 0.0:
                depth_t = F.rotate(
                    depth_t.unsqueeze(0),
                    angle,
                    interpolation=F.InterpolationMode.NEAREST,
                )[0]
        if flip_applied:
            depth_t = F.hflip(depth_t.unsqueeze(0))[0]
        # mask out surroundings at full resolution, then downsample both to depth grid
        depth_t = depth_t * fmask_t[0]
        depth_grid = F.resize(depth_t.unsqueeze(0), (self.depth_res, self.depth_res))[0]
        mask_grid = F.resize(fmask_t, (self.depth_res, self.depth_res))[0]

        is_est = int(IDX_TO_CLASS[stype] in ESTIMATED_CLASSES)
        env, illum = self._env[idx], self._illum[idx]
        return {
            "img": img,
            "hf": hf,
            "depth": depth_grid,
            "depth_mask": mask_grid,
            "label": torch.tensor([1.0 if stype == 0 else 0.0]),
            "spoof_type": torch.tensor([stype], dtype=torch.long),
            "environment": torch.tensor([env], dtype=torch.long),
            "illumination": torch.tensor([illum], dtype=torch.long),
            "is_estimated": torch.tensor([float(is_est)]),
            "key": self._keys[idx],
            "path": self._paths[idx],
        }


class StratifiedBatchSampler(Sampler):
    """Per-class-balanced batches.

    Each batch holds `n_per` images from each of the 10 spoof-type classes
    (incl. live) -> rare attack types are seen at a constant rate every step.
    """

    def __init__(self, class_ids: List[int], batch_size: int, seed: int = 0):
        self.n_classes = len(set(class_ids))
        self.n_per = max(1, batch_size // self.n_classes)
        self.pools: List[List[int]] = [
            [i for i, c in enumerate(class_ids) if c == cls]
            for cls in range(max(class_ids) + 1)
        ]
        self.pools = [p for p in self.pools if p]
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + 1000 * self.epoch)
        pools = [list(p) for p in self.pools]
        for p in pools:
            rng.shuffle(p)
        n_batches = min(len(p) // self.n_per for p in pools)
        for b in range(n_batches):
            batch = []
            for p in pools:
                batch.extend(p[b * self.n_per : (b + 1) * self.n_per])
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return min(len(p) // self.n_per for p in self.pools)


def pad_collate(batch: List[dict]) -> dict:
    """Default collate that stacks the tensor keys and keeps `key` as a list."""
    out: dict = {"keys": [b["key"] for b in batch]}
    for k in (
        "img",
        "hf",
        "depth",
        "depth_mask",
        "label",
        "spoof_type",
        "environment",
        "illumination",
        "is_estimated",
    ):
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["path"] = [b["path"] for b in batch]
    return out


def build_loaders(
    cfg: dict,
    splits: Dict[str, object],
    depth_cache: Optional[str] = None,
    seed: int = 0,
    batch_size: Optional[int] = None,
):
    """Create train/val/test DataLoaders from the split DataFrames."""
    from torch.utils.data import DataLoader

    bs = batch_size or int(cfg["train"]["batch_size"])
    workers = int(cfg["train"].get("workers", 4))
    loaders = {}
    samplers = {}
    for name in splits.keys():
        ds = PADDataset(
            splits[name],
            cfg,
            split=name,
            augment=(name == "train"),
            depth_cache=depth_cache,
        )
        if name == "train":
            sam = StratifiedBatchSampler(ds._stype, bs, seed=seed)
            loaders[name] = DataLoader(
                ds,
                batch_sampler=sam,
                num_workers=workers,
                collate_fn=pad_collate,
                pin_memory=False,
            )
        else:
            sam = None
            loaders[name] = DataLoader(
                ds,
                batch_size=bs,
                shuffle=False,
                num_workers=workers,
                collate_fn=pad_collate,
                pin_memory=False,
            )
        samplers[name] = sam
    return loaders, samplers

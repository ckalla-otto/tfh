"""Prepare train/val/test split CSVs directly from the on-disk CelebA-Spoof data.

Replaces the previous Kaggle-crawl flow. Requires the mirror downloaded to disk
(layout: `Data/{train,test}/<subject>/{live,spoof}/<img>.png|jpg` each with a
sibling `<img>_BB.txt` bbox) and the official annotation table `label.csv`
(from `tungnguyentien/celeba-spoof-crop-1-9`) at `data/labels/`.

Steps:
  1. Walk all images under `Data/` (both official `train/` and `test/` folders
     are pooled; the folder names are arbitrary download parts and subject IDs
     do not overlap, so pooling is identity-safe).
  2. Read each image's bbox from its `<img>_BB.txt` (`x y w h [conf]`).
  3. Join `label.csv` (indexed by `split/subject/class/<img>`) for the true
     10-way spoof_type + illumination + environment.
  4. Build a full manifest then a stratified, identity-exclusive 70/15/15 subset
     with equal per-spoof-type counts.
  5. Write `data/subsets/{train,val,test}.csv` + `balance_report.md`.

Usage (Fire):
  uv run python -m pad prepare --data-root /Users/christiankalla/Downloads/CelebA_Spoof \
      --labels data/labels/label.csv --config configs/base.yaml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .split import _ensure_columns, build_subset
from .utils import get_logger, load_config, set_seed

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _sanitize_id(rel: str) -> str:
    """Flatten a relative path into a cache-friendly unique id."""
    return rel.replace("/", "_").replace("\\", "_")


def _read_bb(img_path: Path) -> Optional[tuple]:
    """Read sibling `<img>_BB.txt` bbox. Format: `x y w h [conf]`.

    Returns (x1, y1, x2, y2) or None if absent/malformed.
    """
    bb = img_path.with_name(img_path.stem + "_BB.txt")
    if not bb.exists():
        return None
    try:
        toks = bb.read_text().strip().split()
        if len(toks) < 4:
            return None
        x, y, w, h = (float(v) for v in toks[:4])
        if w <= 0 or h <= 0:
            return None
        return (x, y, x + w, y + h)
    except Exception:
        return None


def build_manifest(
    data_root: str,
    labels_csv: str = "data/labels/label.csv",
    out_csv: str = "data/crawl.csv",
    include_unknown: bool = False,
) -> pd.DataFrame:
    """Walk the on-disk mirror + join labels -> full manifest DataFrame."""
    data_root = Path(data_root)
    if not (data_root / "Data").is_dir():
        raise FileNotFoundError(
            f"expected {data_root / 'Data'} (layout: Data/{{train,test}}/<subject>/<class>/<img>)"
        )

    labels = None
    if labels_csv and Path(labels_csv).exists():
        lab = pd.read_csv(labels_csv)
        first = lab.columns[0]
        lab = lab.set_index(first)
        lab = lab[~lab.index.duplicated(keep="first")]
        labels = lab

    records = []
    for split in ("train", "test"):
        split_dir = data_root / "Data" / split
        if not split_dir.is_dir():
            continue
        for subj in sorted(os.listdir(split_dir)):
            subj_dir = split_dir / subj
            if not subj_dir.is_dir():
                continue
            for cls in sorted(os.listdir(subj_dir)):
                cls_dir = subj_dir / cls
                if not cls_dir.is_dir():
                    continue
                for fn in sorted(os.listdir(cls_dir)):
                    if Path(fn).suffix.lower() not in IMAGE_EXTS:
                        continue
                    img = cls_dir / fn
                    rel = f"{split}/{subj}/{cls}/{fn}"
                    bb = _read_bb(img)
                    if bb is None:
                        continue  # no bbox -> cannot crop for PAD
                    d = {
                        "image_id": _sanitize_id(rel),
                        "image_path": str(img),
                        "rel_path": rel,
                        "subject_id": str(subj),
                        "split": "",
                        "spoof_type": 999,
                        "is_live": 0,
                        "environment": 0,
                        "illumination": 0,
                        "x1": bb[0],
                        "y1": bb[1],
                        "x2": bb[2],
                        "y2": bb[3],
                    }
                    if labels is not None and rel in labels.index:
                        row = labels.loc[rel]
                        cols = list(labels.columns)
                        st = int(row[cols[40]]) if len(cols) > 40 else 999
                        if st not in range(10):
                            if not include_unknown:
                                continue
                            st = 999
                        d["spoof_type"] = st
                        d["environment"] = int(row[cols[42]]) if len(cols) > 42 else 0
                        d["illumination"] = int(row[cols[41]]) if len(cols) > 41 else 0
                        d["is_live"] = 1 if st == 0 else 0
                    elif not include_unknown:
                        continue  # no label -> drop
                    records.append(d)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No valid annotated images found under {data_root}")
    df = df.drop_duplicates(subset=["image_id"])
    _ensure_columns(df)
    df["spoof_type"] = df["spoof_type"].astype(int)
    df = df.sort_values("image_id").reset_index(drop=True)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def _add_crop_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add extended-crop + face bbox columns expected by the dataset."""
    from PIL import Image

    from .data import make_extended_crop_bbox

    margin = float(cfg["data"]["crop"]["margin_factor"])
    min_side = int(cfg["data"]["crop"].get("min_side", 64))
    crop_x1, crop_y1, crop_x2, crop_y2 = [], [], [], []
    for _, r in df.iterrows():
        try:
            with Image.open(r["image_path"]) as im:
                w, h = im.size
        except Exception:
            w = max(int(float(r["x2"])), int(float(r["y2"])), 1) + 10
            h = w
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


def main(
    data_root: str = "data/raw/celeba-spoof",
    labels: str = "data/labels/label.csv",
    config: str = "configs/base.yaml",
    out_manifest: str = "data/crawl.csv",
    subsets_dir: str = "data/subsets",
    include_unknown: bool = False,
    export_dir: str = None,
) -> None:
    """Prepare train/val/test split CSVs from the on-disk mirror (Fire CLI)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    cfg = load_config(config) if config else {}
    if not cfg:
        cfg = {
            "data": {
                "crop": {"margin_factor": 1.3, "min_side": 64},
                "subset": {
                    "budget_total": 20000,
                    "split": [0.70, 0.15, 0.15],
                    "seed": 42,
                    "secondary": ["environment", "illumination"],
                },
            }
        }
    set_seed(cfg["data"]["subset"]["seed"])
    logger = get_logger()

    manifest = build_manifest(
        data_root=data_root,
        labels_csv=labels,
        out_csv=out_manifest,
        include_unknown=include_unknown,
    )
    logger.info(
        "manifest: %d images, per-type=%s",
        len(manifest),
        manifest["spoof_type"].value_counts().sort_index().to_dict(),
    )

    split_result = build_subset(
        manifest,
        budget_total=cfg["data"]["subset"]["budget_total"],
        split_fracs=tuple(cfg["data"]["subset"]["split"]),
        seed=cfg["data"]["subset"]["seed"],
        secondary=cfg["data"]["subset"].get(
            "secondary", ["environment", "illumination"]
        ),
    )
    out_dir = Path(subsets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        df = _add_crop_columns(split_result.splits[name], cfg)
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "balance_report.md").write_text(split_result.report)

    logger.info("subset report -> %s/balance_report.md", out_dir)
    logger.info(
        "sizes: %s",
        {s: len(split_result.splits[s]) for s in ("train", "val", "test")},
    )
    if "RESULT: FAIL" in split_result.report:
        raise RuntimeError("subset verification FAILED - see balance_report.md")

    if export_dir:
        logger.info("exporting dataset -> %s", export_dir)
        export(subsets_dir=subsets_dir, out_dir=export_dir, link_mode="copy")


def _export_one(
    src: Path,
    dst_dir: Path,
    seen: set,
    mode: str = "copy",
) -> Path:
    """Copy one image into `dst_dir` with collision-safe naming.

    Keeps the original basename unless a different source already occupies the
    same name in `dst_dir`, in which case it appends a numeric suffix.
    Returns the destination path.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if src.resolve() == dst.resolve():
        # already in place (e.g. re-running export after CSVs were re-pointed)
        seen.add(str(dst))
        return dst
    if dst.exists() or str(dst) in seen:
        base, suffix = dst.stem, dst.suffix
        i = 2
        while True:
            cand = dst_dir / f"{base}__{i}{suffix}"
            i += 1
            if not cand.exists() and str(cand) not in seen:
                dst = cand
                break
    if mode == "copy":
        import shutil

        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            import shutil

            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            dst.symlink_to(src)
        except OSError:
            import shutil

            shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown --link-mode {mode!r}; use copy|hardlink|symlink")
    seen.add(str(dst))
    return dst


def export(
    subsets_dir: str = "data/subsets",
    out_dir: str = "data/dataset",
    link_mode: str = "copy",
) -> None:
    """Assemble a self-contained organized dataset under `out_dir`.

    Reads `data/subsets/{train,val,test}.csv` and copies every image into a
    class-folder layout `{out_dir}/{split}/{spoof_name}/<img>` (real files by
    default, `--link-mode` can trade space for copying), then rewrites the CSVs'
    `image_path` / `rel_path` to point at the new files so downstream steps
    (depth_targets, train, evaluate) are fully self-contained under `data/`.

    Usage (Fire):
      uv run python -m pad export --subsets-dir data/subsets --out-dir data/dataset
    """
    from .split import IDX_TO_CLASS

    subsets_dir = Path(subsets_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        csv = subsets_dir / f"{split}.csv"
        if not csv.exists():
            print(f"  skip {split}: {csv} missing")
            continue
        df = pd.read_csv(csv)
        # validate all sources exist before writing anything
        for _, r in df.iterrows():
            src = Path(r["image_path"])
            if not src.exists():
                raise FileNotFoundError(
                    f"missing source image for {r['image_id']}: {src}"
                )
        print(f"  {split:5s} {len(df):6d} images -> {out_dir / split}/{{class}}/")

        out_paths = []
        seen = set()
        for _, r in df.iterrows():
            st = int(r["spoof_type"])
            cls = IDX_TO_CLASS[st] if st in IDX_TO_CLASS else "unknown"
            src = Path(r["image_path"])
            dst = _export_one(src, out_dir / split / cls, seen, link_mode)
            out_paths.append(str(dst))
        df["image_path"] = out_paths
        df["rel_path"] = [str(Path(p).relative_to(out_dir)) for p in out_paths]
        df.to_csv(csv, index=False)

        counts = df.groupby("spoof_type")["rel_path"].count()
        detail = ", ".join(f"{IDX_TO_CLASS[int(k)]}={v}" for k, v in counts.items())
        print(f"    per-class: {detail}")
        print(f"  wrote {csv} (image_path -> {out_dir}) <{link_mode}>")

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)

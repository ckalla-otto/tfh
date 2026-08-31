"""Crawling the CelebA-Spoof mirror into a normalized manifest (`crawl.csv`).

The "crawl" = walking the downloaded mirror and mapping every image file to its
annotations (spoof type / live, face bbox, subject, environment, illumination).

Two mirror layouts are auto-detected:

  * kaggle_csv  - a `train.csv` / `test.csv` (possibly `val.csv`) at the mirror
                  root with a relative-path column like `image` and label cols.
  * celeba_json - official layout: one `<img>.json` beside (or in `Json/`) every
                  image, with `image_id`, `x1..y2`, `live`, `spoof_type`, ...

Images with no matching metadata are skipped and counted (should be ~0 in any
complete mirror).

Usage (Fire):
  uv run python -m pad make_crawl --root data/raw/celeba-spoof --out data/crawl.csv
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .split import _ensure_columns

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Kaggle mirrors almost always ship one of these metadata tables at the root.
META_CSV_NAMES = ("train.csv", "test.csv", "val.csv", "metadata.csv")
_IMAGE_COL = "image"


def _sanitize_id(s: str) -> str:
    """Flatten a possibly nested image_id so cache filenames stay flat."""
    return s.replace("/", "_").replace("\\", "_")


def _subject_from_rel(rel: str) -> str:
    """Derive a subject id from the relative image path.

    Official layout: <s0>/<s1>/<s2>/<img>.jpg  -> 's0/s1/s2' (4+ components).
    Shallower Kaggle mirror paths (3 components) fall back to the first two
    components so nested subjects don't collapse into one group.
    """
    parts = rel.split("/")
    if len(parts) >= 4:
        return "/".join(parts[:3])
    if len(parts) == 3:
        return "/".join(parts[:2])
    return parts[0] if parts else rel


def _int0(v) -> int:
    try:
        return 0 if v is None else int(v)
    except (TypeError, ValueError):
        return 0


def _load_root_csvs(root: Path) -> Optional[pd.DataFrame]:
    """Concatenate any metadata CSVs at the mirror root (image-path indexed)."""
    frames = []
    aliases = ("image", "path", "file", "id", "filename")
    for name in META_CSV_NAMES:
        f = root / name
        if not f.exists():
            continue
        tables = pd.read_csv(f)
        cands = {str(c).strip().lower(): c for c in tables.columns}
        img_col = next((cands[a] for a in aliases if a in cands), None)
        if img_col is None:
            continue
        tables = tables.rename(columns={img_col: _IMAGE_COL})
        frames.append(tables)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df[_IMAGE_COL] = (
        df[_IMAGE_COL]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.replace(r"^\./", "", regex=True)
    )
    return df


def _lookup_json(img: Path) -> Optional[dict]:
    """Find the official per-image JSON (same stem next to image or in Json/)."""
    candidates = [
        img.with_suffix(".json"),
        img.parent / "Json" / f"{img.stem}.json",
        img.parent.parent / "Json" / f"{img.stem}.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                with open(c) as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def build_crawl(
    root: str,
    out_csv: Optional[str] = None,
    include_unknown: bool = False,
    from_metadata: bool = False,
) -> pd.DataFrame:
    """Build the normalized manifest.

    By default walks `root` (images + CSV/JSON metadata) like before.
    With `from_metadata=True` it only reads the root-level metadata CSVs
    (train/test/val/metadata.csv) so a full image download is NOT required yet —
    this is what enables downloading only the sampled subset afterwards.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"mirror root not found: {root}")

    meta = _load_root_csvs(root)
    meta_by_rel: Dict[str, dict] = {}
    meta_by_base: Dict[str, list] = {}
    if meta is not None:
        for _, row in meta.iterrows():
            rel = os.path.normpath(str(row[_IMAGE_COL])).replace("\\", "/")
            meta_by_rel[rel] = row.to_dict()
        for rel, row in meta_by_rel.items():
            meta_by_base.setdefault(os.path.basename(rel), []).append(row)

    def _make_record(rel: str, d: dict) -> dict:
        image_id = _sanitize_id(str(d.get("image_id") or os.path.splitext(rel)[0]))
        st = d.get("spoof_type")
        live = d.get("live", d.get("is_live"))
        try:
            st = None if st is None else int(st)
        except (TypeError, ValueError):
            st = None
        try:
            live = None if live is None else int(live)
        except (TypeError, ValueError):
            live = None
        if live == 1:
            st = 0  # live always maps to spoof-type 0
        if st is None:
            st = 0 if live == 1 else 999

        def _bbox(k: str) -> float:
            v = d.get(k)
            try:
                return float(str(v)) if v is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        return {
            "image_id": image_id,
            "image_path": str(root / rel),
            "rel_path": rel,
            "subject_id": _subject_from_rel(rel),
            "spoof_type": int(st),
            "is_live": int(live) if live is not None else int(st == 0),
            "environment": _int0(d.get("environment", d.get("env"))),
            "illumination": _int0(d.get("illumination", d.get("illum"))),
            "x1": _bbox("x1"),
            "y1": _bbox("y1"),
            "x2": _bbox("x2"),
            "y2": _bbox("y2"),
        }

    # ---- metadata-only crawl: no images on disk needed ----
    if from_metadata:
        if meta is None:
            raise RuntimeError(
                f"no metadata CSV (one of {META_CSV_NAMES}) under {root}; "
                "download the metadata files first (kaggle datasets download -f ...)"
            )
        meta_records = []
        for rel, meta_row in sorted(meta_by_rel.items()):
            meta_records.append(_make_record(rel, dict(meta_row)))
        df = pd.DataFrame(meta_records)
        df = df.drop_duplicates(subset=["image_id"])
        df = df.sort_values("image_id").reset_index(drop=True)
        _ensure_columns(df)
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
        _log_crawl(df, 0, 0)
        return df

    records = []
    n_skip, n_json = 0, 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            img = Path(dirpath) / fn
            rel = os.path.relpath(img, root).replace("\\", "/")

            d: dict = {}
            meta_row = meta_by_rel.get(rel)
            if meta_row is None:
                by_base = meta_by_base.get(os.path.basename(rel))
                if by_base:
                    meta_row = by_base[0]  # ambiguous duplicate -> take first
            if meta_row is not None:
                d = dict(meta_row)
            else:
                j = _lookup_json(img)
                if j is None:
                    if not include_unknown:
                        n_skip += 1
                        continue
                    d = {"spoof_type": 999, "live": None}
                else:
                    d = dict(j)
                    n_json += 1

            records.append(_make_record(rel, d))

    if not records:
        raise RuntimeError(f"No annotated images found under {root}")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["image_id"])
    _ensure_columns(df)
    df = df.sort_values("image_id").reset_index(drop=True)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    _log_crawl(df, n_skip, n_json)
    return df


def _log_crawl(df: pd.DataFrame, n_skip: int, n_json: int) -> None:
    from .split import IDX_TO_CLASS

    counts = df["spoof_type"].value_counts().to_dict()
    rows = {IDX_TO_CLASS[i]: counts.get(i, 0) for i in range(len(IDX_TO_CLASS))}
    total = sum(rows.values()) or 1
    print(f"  crawled {len(df)} images (JSON-sourced metadata: {n_json}; "
          f"skipped-no-meta: {n_skip})")
    print("  per-class:", ", ".join(f"{k}={v} ({100*v/total:.1f}%)" for k, v in rows.items()))
    n_unknown = int(df["spoof_type"].eq(999).sum())
    if n_unknown:
        print(f"  WARNING: {n_unknown} images have unknown spoof type (kept as 999).")


def main(
    root: str = "data/raw/celeba-spoof",
    out: str = "data/crawl.csv",
    include_unknown: bool = False,
    from_metadata: bool = False,
) -> None:
    """Crawl the mirror and write the normalized manifest CSV (Fire CLI).

    Use `--from-metadata true` to build the manifest ONLY from the root-level
    metadata CSVs (train/test/val/metadata.csv) without needing the images on
    disk — needed before downloading just the sampled subset.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    _ = build_crawl(
        root, out_csv=out, include_unknown=include_unknown, from_metadata=from_metadata
    )
    print(f"crawl manifest written -> {out}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
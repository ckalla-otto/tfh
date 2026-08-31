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
# Kaggle mirrors almost always ship one of these metadata tables at the root;
# `spoof.csv` / `data.csv` are the single-table names used by several mirrors.
META_CSV_NAMES = ("train.csv", "test.csv", "val.csv", "metadata.csv", "spoof.csv", "data.csv")
_IMAGE_COL = "image"

# Official Celeba-Spoof folder-name -> spoof-type index (0 = live).
# The Data/ layout is: Data/<split>/<subject_id>/<spoof_class>/<img>.png
# NOTE: several Kaggle mirrors collapse ALL attacks into a single `spoof`/
# folder (no per-attack-type folders and no JSON annotations). For those,
# `spoof` maps to 999 (unknown attack type, but definitely a spoof: is_live=0)
# so the images are kept, NOT dropped.
_SPOOF_FOLDER_TO_IDX = {
    "live": 0,
    "photo": 1,
    "poster": 2,
    "a4": 3,
    "facemask": 4,
    "face_mask": 4,
    "upperbody": 5,
    "upper_body_mask": 5,
    "regionmask": 6,
    "region_mask": 6,
    "pc": 7,
    "pc_pad": 7,
    "phone": 8,
    "3d": 9,
    "3d_mask": 9,
    "3dprint": 9,
    # mirrors that only distinguish live/spoof:
    "spoof": 999,
    "fake": 999,
}


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


def _lookup_bb(img: Path) -> Optional[tuple]:
    """Read the sibling `<img>_BB.txt` bbox if present.

    Format is `x y w h [conf]` (top-left + width/height) per CelebA-Spoof, so we
    convert to absolute (x1, y1, x2, y2).
    """
    bb_file = img.with_name(f"{img.stem}_BB.txt")
    if not bb_file.exists():
        alt = img.parent / f"{img.stem}_BB.txt"
        bb_file = alt if alt.exists() else bb_file
    if not bb_file.exists():
        return None
    try:
        with open(bb_file) as f:
            toks = f.read().strip().split()
        if len(toks) < 4:
            return None
        x, y, w, h = (float(v) for v in toks[:4])
        return (x, y, x + w, y + h)
    except Exception:
        return None


def _parse_official_rel(rel: str) -> Optional[dict]:
    """Parse an official-layout relative path (path- or file-list driven).

    Handles `Data/<split>/<subject>/<spoofclass>/<img>.png` both when the
    mirror root is `Data` itself and when an archive prefix precedes it
    (e.g. `CelebA_Spoof_/CelebA_Spoof/Data/test/10001/live/496120.png`).
    Ignores `_BB.txt` companions. Returns None for non-image/unparseable rels.
    """
    ext = os.path.splitext(rel)[1].lower()
    if ext not in IMAGE_EXTS:
        return None
    parts = rel.split("/")
    base = None
    for i, p in enumerate(parts):
        if p.lower() == "data":
            base = i + 1
            break
    if base is None:
        # no explicit Data/ segment: split is the first component
        if len(parts) < 4:
            return None
        split, subject, spoof_cls = parts[0], parts[1], parts[2]
    else:
        # need split/<subject>/<class> after Data/ and an image after that
        if len(parts) < base + 4:
            return None
        split = parts[base]
        subject = parts[base + 1]
        spoof_cls = parts[base + 2]
    idx = _SPOOF_FOLDER_TO_IDX.get(spoof_cls.lower())
    return {
        "rel": rel,
        "split": split,
        "subject": subject,
        "spoof_cls": spoof_cls,
        "spoof_type": idx,
    }


def _official_walk(root: Path, include_unknown: bool):
    """Walk the official `Data/<split>/<subject_id>/<spoof_class>/<img>` layout.

    Yields (rel_path, d) where d carries bbox (from _BB.txt) + inferred labels.
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            img = Path(dirpath) / fn
            rel = os.path.relpath(img, root).replace("\\", "/")
            parsed = _parse_official_rel(rel)
            if parsed is None:
                continue
            idx = parsed["spoof_type"]
            if idx is None and not include_unknown:
                continue
            bb = _lookup_bb(img)
            if bb is None:
                bb = (0.0, 0.0, 0.0, 0.0)
            d = {
                "spoof_type": idx if idx is not None else 999,
                "live": (1 if idx == 0 else 0) if idx is not None else None,
                "x1": bb[0], "y1": bb[1], "x2": bb[2], "y2": bb[3],
                "environment": 0, "illumination": 0,
            }
            # preserve the split + subject so the manifest is usable downstream
            d["_split"] = parsed["split"]
            d["_subject"] = parsed["subject"]
            yield rel, d


def build_crawl_from_file_list(
    file_list: str,
    out_csv: Optional[str] = None,
    include_unknown: bool = False,
    labels_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Build the manifest from a `kaggle datasets files` listing (paths only).

    Each line is a relative path in the official layout. No images or metadata
    need to be on disk — bboxes are filled in later (download_subset patches
    them from the per-image `_BB.txt` companions).

    `labels_csv` (optional): path to the official encodings table (e.g. the
    `label.csv` from `tungnguyentien/celeba-spoof-crop-1-9`) with rows indexed
    by the relative path (`split/subject/class/img.png`) and official columns
    40=spoof_type, 41=illumination, 42=environment, 43=live. When provided, the
    true 10-way spoof type + environment/illumination are filled from it; images
    that fail to match keep their folder-derived label (999 for unknown attacks).
    """
    lines = [ln.strip() for ln in open(file_list) if ln.strip()]
    records = []

    # label lookup: index = relative path after Data/, cols = official vector
    lab_rows = idx40 = idx41 = idx42 = None
    if labels_csv:
        lab_rows = pd.read_csv(labels_csv, index_col=0)
        idx40 = lab_rows[lab_rows.columns[40]] if len(lab_rows.columns) > 40 else None
        idx41 = lab_rows[lab_rows.columns[41]] if len(lab_rows.columns) > 41 else None
        idx42 = lab_rows[lab_rows.columns[42]] if len(lab_rows.columns) > 42 else None

    n_joined = 0
    for ln in lines:
        # strip size/date columns (the CLI table) if present
        rel = ln.split()[0]
        parsed = _parse_official_rel(rel)
        if parsed is None:
            continue
        idx = parsed["spoof_type"]
        if idx is None and not include_unknown:
            continue

        # canonical relative path = split/subject/class/img (== label.csv index)
        canon = "/".join(rel.split("/")[-4:]) if "/Data/" in rel else rel

        st, illum, env = idx if idx is not None else 999, 0, 0
        if lab_rows is not None and canon in lab_rows.index:
            st = int(idx40.loc[canon]) if idx40 is not None else st
            illum = int(idx41.loc[canon]) if idx41 is not None else illum
            env = int(idx42.loc[canon]) if idx42 is not None else env
            n_joined += 1

        records.append(
            {
                "image_id": _sanitize_id(os.path.splitext(rel)[0]),
                "image_path": "",  # not on disk yet; relink later
                "rel_path": rel,
                "subject_id": parsed["subject"],
                "split": parsed["split"],
                "spoof_type": st,
                "is_live": int(st == 0),
                "environment": env,
                "illumination": illum,
                "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0,
            }
        )
    if not records:
        raise RuntimeError(
            f"no image paths parsed from file list; expected official layout "
            f"(Data/<split>/<subject>/<class>/<img>) — got {file_list}"
        )
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["image_id"])
    df = df.sort_values("image_id").reset_index(drop=True)
    _ensure_columns(df)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    _log_file_list_crawl(df, labels_provided=labels_csv is not None)
    if labels_csv:
        n_unmatched = int(df["spoof_type"].eq(999).sum())
        print(f"  label join: {n_joined}/{len(df)} images matched {labels_csv}; "
              f"{n_unmatched} kept as 999 (unknown attack type)")
    return df


def _log_file_list_crawl(df: pd.DataFrame, labels_provided: bool = False) -> None:
    from .split import IDX_TO_CLASS

    counts = df["spoof_type"].value_counts().to_dict()
    rows = {IDX_TO_CLASS[i]: counts.get(i, 0) for i in range(len(IDX_TO_CLASS))}
    total = sum(rows.values()) or 1
    n_unknown = int(counts.get(999, 0))
    frac_unknown = n_unknown / max(len(df), 1)
    print(f"  crawled {len(df)} images from file list")
    print("  per-class:", ", ".join(f"{k}={v} ({100*v/total:.1f}%)" for k, v in rows.items()))
    if frac_unknown > (0.1 if labels_provided else 0.0):
        print(
            f"  WARNING: {n_unknown} images ({100*frac_unknown:.0f}%) have "
            "UNKNOWN attack type (folder `spoof/`)."
            + ("" if labels_provided else
               " This mirror only distinguishes live vs spoof — the 10-way spoof "
               "type is NOT available from the file listing; pass --labels <csv> "
               "to join the official annotations.")
        )
    elif n_unknown and labels_provided:
        print(f"  note: {n_unknown} ({100*frac_unknown:.1f}%) images kept as 999 "
              "(unknown attack type; not in the joined label table).")


def build_crawl(
    root: str,
    out_csv: Optional[str] = None,
    include_unknown: bool = False,
    from_metadata: bool = False,
    layout: str = "auto",
) -> pd.DataFrame:
    """Build the normalized manifest.

    `layout`:
      * auto        - walk images + CSV/JSON metadata (default; detects CSV)
      * official    - walk the official Data/<split>/<subject>/<class>/ layout,
                      reading per-image `<img>_BB.txt` bboxes
      * kaggle_csv  - metadata-only crawl from root-level CSVs (see from_metadata)
    `from_metadata=True` only reads root-level CSVs (no images needed).
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

        # subject: prefer the official path's explicit subject component
        subject = d.get("_subject") or _subject_from_rel(rel)

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
            "subject_id": str(subject),
            "split": d.get("_split", ""),
            "spoof_type": int(st),
            "is_live": int(live) if live is not None else int(st == 0),
            "environment": _int0(d.get("environment", d.get("env"))),
            "illumination": _int0(d.get("illumination", d.get("illum"))),
            "x1": _bbox("x1"),
            "y1": _bbox("y1"),
            "x2": _bbox("x2"),
            "y2": _bbox("y2"),
        }

    # ---- official-layout crawl: Data/<split>/<subject>/<class>/<img> + _BB.txt ----
    if layout == "official":
        records = [
            _make_record(rel, d)
            for rel, d in _official_walk(root, include_unknown)
        ]
        if not records:
            raise RuntimeError(f"no images found in official layout under {root}")
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=["image_id"])
        df = df.sort_values("image_id").reset_index(drop=True)
        _ensure_columns(df)
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
        _log_crawl(df, 0, 0)
        return df

    if layout not in ("auto", "kaggle_csv", "celeba_json"):
        raise ValueError(f"unknown layout: {layout}")

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
    layout: str = "auto",
    from_file_list: str = None,
    labels: str = None,
) -> None:
    """Crawl the mirror and write the normalized manifest CSV (Fire CLI).

    `--layout official` walks the official Data/<split>/<subject>/<class>/ layout
    (uses per-image `<img>_BB.txt` bboxes and the folder name for the spoof type).
    `--from-metadata true` builds the manifest only from root-level CSVs.
    `--from-file-list <file>` builds the manifest from a `kaggle datasets files`
    listing (no images on disk) — bboxes get patched later by download_subset.
    `--labels <csv>` joins the official annotation table (label.csv from
    tungnguyentien/celeba-spoof-crop-1-9) so the TRUE 10-way spoof_type,
    illumination and environment are filled from it.
    """
    import sys

    from .utils import load_env

    load_env()  # ensure KAGGLE_* creds are in os.environ (from .env via dotenv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    if from_file_list:
        _ = build_crawl_from_file_list(
            from_file_list, out_csv=out, include_unknown=include_unknown,
            labels_csv=labels,
        )
    else:
        _ = build_crawl(
            root, out_csv=out, include_unknown=include_unknown,
            from_metadata=from_metadata, layout=layout,
        )
    print(f"crawl manifest written -> {out}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
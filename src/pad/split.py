"""Stratified, identity-exclusive subset builder for Celeba-Spoof.

Design (locked in the plan):
  * 10 equally-sampled classes: Live + the 9 attack types.
  * Subject-disjoint train/val/test (no identity leakage across splits).
  * Within each spoof type, secondary strata (environment, illumination)
    are sampled proportionally.
  * Exact equal per-class counts per split, capped by the smallest class pool
    (e.g. 3D-Mask) so equality never silently breaks.

This module avoids torch on purpose -> unit-testable without a heavy stack.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Official Celeba-Spoof spoof-type index -> name (index order fixed by the dataset).
SPOOF_TYPES: List[str] = [
    "live",
    "photo",
    "poster",
    "a4",
    "face_mask",
    "upper_body_mask",
    "region_mask",
    "pc_pad",
    "phone",
    "3d_mask",
]

# Classes that get estimated (pseudo-)depth targets (live + physical-3D classes).
ESTIMATED_CLASSES: set = {"live", "face_mask", "upper_body_mask", "3d_mask"}
# Classes that get a flat depth target.
FLAT_CLASSES: set = set(SPOOF_TYPES) - ESTIMATED_CLASSES

# Spoof-type index (0-9) keeps the dataset's numbering.
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(SPOOF_TYPES)}
IDX_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(SPOOF_TYPES)}


@dataclass
class SplitResult:
    """Committable artifacts of the split step."""

    splits: Dict[str, pd.DataFrame]  # "train" | "val" | "test"
    report: str  # markdown balance report
    per_class_counts: pd.DataFrame


# Canonical column name -> accepted synonyms found in the mirror CSV / JSON.
_COLUMN_SYNONYMS = {
    "image_path": ["path", "image", "image_path", "file", "name", "img"],
    "subject_id": ["subject", "subject_id", "subj", "person", "id"],
    "spoof_type": ["spoof_type", "type", "attack_type", "cls", "label_type"],
    "is_live": ["live", "is_live", "bona_fide"],
    "environment": ["environment", "env", "env_id"],
    "illumination": ["illumination", "illum", "light", "illum_id"],
    "x1": ["x1", "x0", "left"],
    "y1": ["y1", "y0", "top"],
    "x2": ["x2", "x1", "right"],
    "y2": ["y2", "y1", "bottom"],
    "image_id": ["image_id", "img_id", "uuid"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename (lower-cased, stripped) columns to canonical names."""
    rename = {}
    for col in df.columns:
        low = str(col).strip().lower()
        for canon, aliases in _COLUMN_SYNONYMS.items():
            if low in aliases and canon not in rename.values():
                rename[col] = canon
                break
    return df.rename(columns=rename)


def _ensure_columns(df: pd.DataFrame) -> None:
    required = ["image_path", "subject_id", "spoof_type", "x1", "y1", "x2", "y2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in the crawl manifest. "
            f"Available: {list(df.columns)}. "
            "Adjust column names in _COLUMN_SYNONYMS for your mirror."
        )
    if df["image_path"].astype(str).replace("", pd.NA).dropna().duplicated().any():
        raise ValueError("Duplicate image_path entries in crawl manifest.")
    if df["subject_id"].isna().any():
        raise ValueError("NaN values in subject_id column.")


def read_crawl(path: Union[str, Path], layout: str = "kaggle_csv") -> pd.DataFrame:
    """Load + normalize the full crawl manifest (all images of the mirror)."""
    path = Path(path)
    if layout == "kaggle_csv":
        if path.suffix == ".json":
            df = pd.read_json(path)
        else:
            df = pd.read_csv(path)
    elif layout == "celeba_json":
        df = _read_celeba_json_mirror(path)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    df = _normalize_columns(df)
    _ensure_columns(df)
    df["spoof_type"] = df["spoof_type"].astype(int)
    if "is_live" not in df.columns:
        df["is_live"] = (df["spoof_type"] == 0).astype(int)
    else:
        df["is_live"] = df["is_live"].astype(int)
    if "environment" not in df.columns:
        df["environment"] = 0  # dummy (unknown) if the mirror lacks it
    if "illumination" not in df.columns:
        df["illumination"] = 0
    if "image_id" not in df.columns:
        df["image_id"] = df.index.astype(str)

    bb = df[["x1", "y1", "x2", "y2"]].astype(float)
    if (bb["x2"] <= bb["x1"]).any() or (bb["y2"] <= bb["y1"]).any():
        raise ValueError("Invalid face bbox in crawl manifest (x2<=x1 or y2<y1).")
    return df


def _read_celeba_json_mirror(root: Path) -> pd.DataFrame:
    """Walk the official Celeba-Spoof folder/json layout.

    Expected: <root>/<subject>/.../<img>.jpg plus .../Json/<img>.json.
    Prefer the Kaggle CSV layout whenever in doubt.
    """
    records = []
    for meta_f in sorted(root.rglob("**/*.json")):
        try:
            with open(meta_f, "r") as f:
                d = json.load(f)
        except Exception:
            continue
        img_id = str(d.get("image_id") or meta_f.stem)
        parts = str(meta_f.relative_to(root)).replace("\\", "/").split("/")
        subj = "/".join(parts[:3]) if len(parts) >= 4 else img_id
        records.append(
            {
                "image_path": str(root / f"{img_id}.jpg"),
                "image_id": img_id,
                "subject_id": subj,
                "spoof_type": int(d.get("spoof_type", 10)),
                "environment": int(d.get("env", 0)),
                "illumination": int(d.get("illum", 0)),
                "x1": float(d["x1"]),
                "y1": float(d["y1"]),
                "x2": float(d["x2"]),
                "y2": float(d["y2"]),
            }
        )
    if not records:
        raise FileNotFoundError(f"No JSON annotations under {root}")
    return pd.DataFrame(records)


def _split_subjects(
    subject_ids: np.ndarray, fracs: Tuple[float, float, float], seed: int
) -> Dict[str, List[str]]:
    """Deterministic subject partition (no identity overlap across splits)."""
    rng = random.Random(seed)
    ids = [str(s) for s in subject_ids]
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(n * fracs[0])
    n_va = int(n * fracs[1])
    return {
        "train": ids[:n_tr],
        "val": ids[n_tr : n_tr + n_va],
        "test": ids[n_tr + n_va :],
    }


def _stratified_sample(
    pool: pd.DataFrame, k: int, keys: List[str], seed: int
) -> pd.DataFrame:
    """Sample exactly k rows from `pool`, stratified proportionally on `keys`."""
    rng = np.random.RandomState(seed)
    if len(pool) <= k:
        return pool.sample(frac=1.0, random_state=rng)

    groups = {key: gb for key, gb in pool.groupby(keys, sort=True)}
    alloc = {}
    for key, gb in groups.items():
        alloc[key] = int(k * len(gb) / len(pool))

    parts = []
    for key, gb in groups.items():
        if alloc[key] > 0:
            parts.append(gb.sample(n=alloc[key], random_state=rng))
    assigned = pd.concat(parts) if parts else pool.iloc[0:0]

    taken = set(assigned.index)
    if len(assigned) < k:
        rest = pool.loc[~pool.index.isin(taken)]
        extra = rest.sample(n=k - len(assigned), random_state=rng)
        assigned = pd.concat([assigned, extra])
    elif len(assigned) > k:
        counts = assigned.groupby(keys).size()
        target = {key: int(k * cnt / len(assigned)) for key, cnt in counts.items()}
        pieces, rem = [], k
        for key, gb in assigned.groupby(keys):
            m = min(len(gb), target.get(key, 0))
            if rem - m < 0:
                m = max(rem, 0)
            if m > 0:
                pieces.append(gb.sample(n=m, random_state=rng))
                rem -= m
        if rem > 0:
            have = set(pd.concat(pieces).index)
            leftovers = assigned.loc[~assigned.index.isin(have)]
            pieces.append(leftovers.sample(n=min(rem, len(leftovers)), random_state=rng))
        assigned = pd.concat(pieces)

    if len(assigned) != k:
        assigned = pool.sample(n=k, random_state=rng)
    return assigned.sample(frac=1.0, random_state=rng)


def build_subset(
    crawl: pd.DataFrame,
    budget_total: int,
    split_fracs: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    secondary: Optional[List[str]] = None,
) -> SplitResult:
    """Build the equal-per-spoof-type, identity-exclusive subset.

    Per-class target per split = floor(budget_total * frac / n_classes), capped
    by the smallest class pool in that split so equality holds exactly.
    """
    secondary = secondary or ["environment", "illumination"]
    n_cls = len(SPOOF_TYPES)
    for cls in range(n_cls):
        if not (crawl["spoof_type"] == cls).any():
            raise ValueError(
                f"Missing spoof type {IDX_TO_CLASS[cls]} in crawl -> cannot stratify"
            )

    subj_splits = _split_subjects(
        np.unique(crawl["subject_id"].to_numpy()), split_fracs, seed
    )
    split_names = ["train", "val", "test"]
    splits: Dict[str, pd.DataFrame] = {}
    per_cls: Dict[str, List[int]] = {}

    for split_name in split_names:
        subjs = set(subj_splits[split_name])
        part = crawl[crawl["subject_id"].astype(str).isin(subjs)]
        frac = split_fracs[split_names.index(split_name)]
        k = max(1, int(budget_total * frac) // n_cls)

        class_pools = {cls: part[part["spoof_type"] == cls] for cls in range(n_cls)}
        for cls, pool in class_pools.items():
            if len(pool) == 0:
                raise ValueError(
                    f"No images for class {IDX_TO_CLASS[cls]} in split {split_name}"
                )

        # Equal cap driven by the scarcest class (typically 3D-Mask).
        k_eff = min(k, min(len(pool) for pool in class_pools.values()))

        pieces = []
        for cls, pool in class_pools.items():
            pieces.append(
                _stratified_sample(
                    pool, k_eff, secondary, seed + split_names.index(split_name)
                )
            )
        split_df = pd.concat(pieces)
        split_df["split"] = split_name
        splits[split_name] = split_df
        per_cls[split_name] = [
            int(len(split_df[split_df["spoof_type"] == c])) for c in range(n_cls)
        ]

    report, counts_df = build_report(splits)
    return SplitResult(splits=splits, report=report, per_class_counts=counts_df)


def build_report(splits: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    """Verify stratification invariants and render a markdown report.

    Fails loudly (via `problems`) on any invariant violation.
    """
    n_cls = len(SPOOF_TYPES)
    problems: List[str] = []
    per_cls: Dict[str, List[int]] = {
        s: [int(len(splits[s][splits[s]["spoof_type"] == c])) for c in range(n_cls)]
        for s in splits
    }

    for s in splits:
        if len(splits[s]) == 0:
            problems.append(f"{s}: empty split")
        if len(set(per_cls[s])) != 1:
            problems.append(f"{s}: unequal per-class counts {per_cls[s]}")
        if set(splits[s]["spoof_type"].unique()) != set(range(n_cls)):
            problems.append(f"{s}: does not contain all {n_cls} spoof types")

    names = list(splits.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(splits[a]["subject_id"]) & set(splits[b]["subject_id"])
            if overlap:
                problems.append(f"subject overlap {a}<->{b}: n={len(overlap)}")

    for s in splits:
        # ignore empty image_path rows (file-list crawl paths not yet on disk)
        nonempty = splits[s]["image_path"].astype(str).replace("", pd.NA).dropna()
        if nonempty.duplicated().any():
            problems.append(f"{s}: duplicate image paths")

    header = ["split"] + [IDX_TO_CLASS[i] for i in range(n_cls)]
    table = [[s] + [str(c) for c in per_cls[s]] for s in splits]
    counts_df = pd.DataFrame(table, columns=header)

    lines = [
        "# Stratification balance report",
        "",
        "Per-class counts (must be identical within a split):",
        "",
    ]
    try:
        lines.append(counts_df.to_markdown(index=False))
    except ImportError:
        lines.append(counts_df.to_string(index=False))
    lines.append("")
    lines.append("Checks:")
    if problems:
        lines += [f"- FAIL: {p}" for p in problems]
        lines += ["", "RESULT: FAIL"]
    else:
        lines += [
            "- OK: equal per-spoof-type counts per split",
            "- OK: zero subject overlap across splits",
            "- OK: all spoof types present in every split",
            "- OK: no duplicate image paths",
            "",
            "RESULT: PASS",
        ]
    return "\n".join(lines), counts_df
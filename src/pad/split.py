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

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def _ensure_columns(df: pd.DataFrame) -> None:
    required = ["image_path", "subject_id", "spoof_type", "x1", "y1", "x2", "y2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in the manifest. "
            f"Available: {list(df.columns)}. "
        )
    nonempty = df["image_path"].astype(str).replace("", pd.NA).dropna()
    if nonempty.duplicated().any():
        raise ValueError("Duplicate image_path entries in manifest.")
    if df["subject_id"].isna().any():
        raise ValueError("NaN values in subject_id column.")


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
            pieces.append(
                leftovers.sample(n=min(rem, len(leftovers)), random_state=rng)
            )
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

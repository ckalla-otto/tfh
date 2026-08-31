"""Unit tests for the stratified identity-exclusive split logic (split.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pad.split import (
    SPOOF_TYPES,
    build_subset,
    build_report,
    read_crawl,
)


def _synthetic_crawl(n_classes=10, n_subjects=60, images_per_subject_per_class=3) -> pd.DataFrame:
    """Small synthetic crawl mirroring the Celeba-Spoof annotation layout.

    Each subject owns images across ALL classes (like the real dataset), so a
    subject-disjoint split can still cover every spoof type in every split.
    """
    rows = []
    idx = 0
    for s in range(n_subjects):
        subj = f"subj_{s}"
        for cls in range(n_classes):
            for im in range(images_per_subject_per_class):
                env = im % 2
                illum = (im * 7) % 5
                rows.append(
                    {
                        "image_id": f"{idx:05d}",
                        "image_path": f"/data/x/{cls}/{subj}/{im}.jpg",
                        "subject_id": subj,
                        "spoof_type": cls,
                        "is_live": int(cls == 0),
                        "environment": env,
                        "illumination": illum,
                        "x1": 10, "y1": 20, "x2": 110, "y2": 140,
                    }
                )
                idx += 1
    return pd.DataFrame(rows)


def test_build_subset_equal_counts_and_disjoint_subjects():
    crawl = _synthetic_crawl()
    res = build_subset(crawl, budget_total=300, seed=42)
    assert set(res.splits) == {"train", "val", "test"}
    for name, df in res.splits.items():
        counts = df.groupby("spoof_type").size()
        assert len(counts) == len(SPOOF_TYPES)
        # equality is exact within a split
        assert counts.nunique() == 1, f"{name}: unequal per-class counts {counts}"
        assert set(df["spoof_type"]) == set(range(len(SPOOF_TYPES)))
    # subject disjointness
    subj_sets = {s: set(df["subject_id"]) for s, df in res.splits.items()}
    for a in subj_sets:
        for b in subj_sets:
            if a != b:
                assert not (subj_sets[a] & subj_sets[b]), f"overlap {a}<->{b}"


def test_build_subset_deterministic_same_seed():
    crawl = _synthetic_crawl()
    r1 = build_subset(crawl, budget_total=300, seed=7)
    r2 = build_subset(crawl, budget_total=300, seed=7)
    for name in ("train", "val", "test"):
        assert list(r1.splits[name]["image_id"]) == list(r2.splits[name]["image_id"])


def test_build_subset_different_seed_differs():
    crawl = _synthetic_crawl()
    r1 = build_subset(crawl, budget_total=300, seed=1)
    r2 = build_subset(crawl, budget_total=300, seed=2)
    assert list(r1.splits["train"]["image_id"]) != list(r2.splits["train"]["image_id"])


def test_build_subset_secondary_strata_present():
    crawl = _synthetic_crawl()
    res = build_subset(crawl, budget_total=300, seed=1)
    for name, df in res.splits.items():
        # every spoof type should see more than one environment in this synthetic data
        env_counts = df.groupby(["spoof_type", "environment"]).size()
        for cls in range(len(SPOOF_TYPES)):
            sub = env_counts[cls]
            assert len(sub) >= 1
    assert set(res.per_class_counts.columns) == {"split"} | set(SPOOF_TYPES)


def test_report_passes():
    crawl = _synthetic_crawl()
    res = build_subset(crawl, budget_total=300, seed=1)
    assert "RESULT: PASS" in res.report


def test_build_report_detects_overlap():
    crawl = _synthetic_crawl()
    res = build_subset(crawl, budget_total=300, seed=1)
    # inject a subject overlap between train and test
    bad = {s: df.copy() for s, df in res.splits.items()}
    bad["test"].loc[bad["test"].index[0], "subject_id"] = bad["train"].iloc[0]["subject_id"]
    report, _ = build_report(bad)
    assert "subject overlap" in report
    assert "RESULT: FAIL" in report


def test_read_crawl_normalization():
    df = pd.DataFrame(
        {
            "Path": ["/a.jpg", "/b.jpg"],
            "Subject": ["s0", "s1"],
            "Type": [0, 2],
            "X1": [10, 0], "Y1": [20, 0], "X2": [110, 30], "Y2": [140, 40],
        }
    )
    df.to_csv("/tmp/_test_crawl.csv", index=False)
    out = read_crawl("/tmp/_test_crawl.csv", "kaggle_csv")
    assert "image_path" in out.columns
    assert "subject_id" in out.columns
    assert out.loc[0, "is_live"] == 1
    assert out.loc[1, "is_live"] == 0


def test_budget_below_min_pool_caps_equality():
    # only 10 subjects -> few images per class pool; budget of 900 cannot be met,
    # but per-class equality within each split must still hold.
    crawl = _synthetic_crawl(n_classes=10, n_subjects=10, images_per_subject_per_class=4)
    res = build_subset(crawl, budget_total=900, seed=3)
    for name, df in res.splits.items():
        assert df.groupby("spoof_type").size().nunique() == 1
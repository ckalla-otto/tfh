"""Shared helpers: config loading, seeding, device resolution, PAD metrics."""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml


def load_config(path: str, root: str = ".") -> dict:
    """Load a YAML config, honoring `include: [base.yaml]` overlay chains."""
    path = Path(root) / path if not Path(path).is_absolute() else Path(path)
    cfg: dict = {}
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    for inc in data.pop("include", []):
        sub = load_config(str(inc), root=str(path.parent))
        cfg = _deep_merge(cfg, sub)
    return _deep_merge(cfg, data)


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(force_cpu: bool = False) -> str:
    try:
        import torch

        if force_cpu:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_logger(
    name: str = "pad",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


DEFAULT_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_env(path: Optional[str] = None, override: bool = False) -> str:
    """Load a local `.env` into os.environ via python-dotenv (idempotent).

    Searches from the current directory upward for `.env` by default; set
    `PAD_ENV` to point at an explicit file. Also normalizes the legacy
    `KAGGLE_API_KEY` alias to `KAGGLE_KEY` after loading.
    """
    import os
    from pathlib import Path

    if path is None:
        path = os.environ.get("PAD_ENV", ".env")
    try:
        from dotenv import load_dotenv

        if path == ".env":
            load_dotenv(override=override)  # walks CWD parents
        else:
            load_dotenv(dotenv_path=Path(path), override=override)
    except Exception:
        pass  # dotenv is optional; env vars still work if exported manually
    if not os.environ.get("KAGGLE_KEY") and os.environ.get("KAGGLE_API_KEY"):
        os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_KEY"]

    # guard against stray backticks/quotes from messy .env copy-paste
    for var in ("PAD_DATASET_SLUG", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_KEY"):
        if os.environ.get(var):
            os.environ[var] = str(os.environ[var]).strip().strip("`").strip()
    return os.environ.get("KAGGLE_KEY", "")


# ---------------------------------------------------------------------------
# PAD metrics (ISO/IEC 30107-3 style). Scores are P(live); label 1 = live.
# ---------------------------------------------------------------------------
def compute_pad_metrics(
    scores_live: np.ndarray, labels_live: np.ndarray, threshold: float
) -> Dict[str, float]:
    """APCER/BPCER/ACER/AUC/TPR@FPR given a P(bona-fide) score."""
    from sklearn.metrics import roc_auc_score, roc_curve

    labels_live = labels_live.astype(int)
    scores_live = np.asarray(scores_live, dtype=np.float64)
    pred_live = scores_live > threshold

    live = labels_live == 1
    spoof = labels_live == 0
    apcer = float(np.mean(pred_live[spoof])) if spoof.sum() > 0 else float("nan")
    bpcer = float(np.mean(~pred_live[live])) if live.sum() > 0 else float("nan")
    acer = 0.5 * (apcer + bpcer)

    auc = float(roc_auc_score(labels_live, scores_live))
    fpr, tpr, _ = roc_curve(labels_live, scores_live)
    tpr_at = 0.0
    for fp, tp in zip(fpr, tpr):
        if fp <= 0.01:
            tpr_at = tp
        else:
            break
    return {
        "threshold": float(threshold),
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
        "AUC": auc,
        "HTER": acer,
        "TPR@FPR0.01": float(tpr_at),
        "n_live": int(live.sum()),
        "n_spoof": int(spoof.sum()),
    }


def find_threshold(
    scores_live: np.ndarray, labels_live: np.ndarray, bpcer_target: float = 0.01
) -> float:
    """Operating threshold such that val-BPCER is just under `bpcer_target`."""
    labels_live = labels_live.astype(int)
    scores_live = np.asarray(scores_live, dtype=np.float64)
    uniq = np.unique(scores_live)
    if len(uniq) == 1:
        return float(uniq[0])
    best_t, best_err = uniq[0], float("inf")
    for t in uniq:
        pred_live = scores_live > t
        live = labels_live == 1
        bpcer = float(np.mean(~pred_live[live])) if live.sum() > 0 else 0.0
        err = abs(bpcer - bpcer_target)
        if err < best_err:
            best_err, best_t = err, t
    return float(best_t)


def bootstrap_acer_ci(
    scores_live: np.ndarray,
    labels_live: np.ndarray,
    threshold: float,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Percentile bootstrap CI on ACER (single-class resamples dropped)."""
    rng = np.random.RandomState(0)
    n = len(scores_live)
    acers = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        val = compute_pad_metrics(scores_live[idx], labels_live[idx], threshold)["ACER"]
        if np.isfinite(val):
            acers.append(val)
    if not acers:
        return float("nan"), float("nan")
    lo = np.percentile(acers, (1 - ci) / 2 * 100)
    hi = np.percentile(acers, (1 + ci) / 2 * 100)
    return float(lo), float(hi)

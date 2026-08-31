"""Evaluation: ISO 30107-3 metrics, per-spoof-type breakdown, confusion matrix,
and per-class hard-sample reports (both CLI and library entry points).

  python -m pad.evaluate --config configs/exp_smoke.yaml \
      --ckpt results/run/best.pt --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .inference import predict_epoch
from .split import IDX_TO_CLASS, SPOOF_TYPES
from .utils import (
    bootstrap_acer_ci,
    compute_pad_metrics,
    find_threshold,
    get_logger,
    load_config,
    resolve_device,
    save_json,
)


def per_type_df(pred: dict, threshold: float) -> pd.DataFrame:
    """Per-spoof-type breakdown incl. depth-variance and HF-energy diagnostics."""
    rows = []
    for cls in range(len(SPOOF_TYPES)):
        m = pred["spoof_type"] == cls
        if m.sum() == 0:
            continue
        scores = pred["scores_live"][m]
        if cls == 0:  # live
            error = float(np.mean(scores <= threshold))  # BPCER contribution
        else:
            error = float(np.mean(scores > threshold))   # APCER contribution
        rows.append(
            {
                "spoof_type": IDX_TO_CLASS[cls],
                "n": int(m.sum()),
                "error_rate": error,
                "depth_var": float(np.mean(pred["depth_var"][m])),
                "hf_energy": float(np.mean(pred["hf_energy"][m])),
                "mean_score_live": float(np.mean(scores)),
            }
        )
    return pd.DataFrame(rows)


def _hard_samples(pred: dict, threshold: float, top_k: int) -> pd.DataFrame:
    """Per-class top-k samples closest to the decision boundary."""
    df = pd.DataFrame(
        {
            "image_id": pred["keys"],
            "spoof_type": pred["spoof_type"],
            "true_label": pred["labels"],
            "score_live": pred["scores_live"],
            "depth_var": pred["depth_var"],
            "hf_energy": pred["hf_energy"],
            "env": pred["env"],
            "illum": pred["illum"],
            "spoof_pred": pred["spoof_pred"],
        }
    )
    df["boundary_dist"] = (df["score_live"] - threshold).abs()
    out = []
    for cls in range(len(SPOOF_TYPES)):
        part = df[df["spoof_type"] == cls].nsmallest(top_k, "boundary_dist")
        part = part.copy()
        part["spoof_type_name"] = IDX_TO_CLASS[cls]
        out.append(part)
    return pd.concat(out, ignore_index=True)


def hard_samples_markdown(hard: pd.DataFrame, threshold: float) -> str:
    lines = [
        f"# Per-class hard samples (boundary = P(live) = {threshold:.4f})",
        "",
        "`boundary_dist` is |score - threshold|; the smallest value per class is",
        "the hardest-to-decide sample.",
        "",
    ]
    for cls in range(len(SPOOF_TYPES)):
        part = hard[hard["spoof_type"] == cls].sort_values("boundary_dist")
        lines.append(f"## {IDX_TO_CLASS[cls]} ({len(part)})")
        if len(part):
            show = part[["image_id", "true_label", "score_live",
                         "boundary_dist", "spoof_pred", "env", "illum"]]
            try:
                lines.append(show.to_markdown(index=False))
            except ImportError:
                lines.append(show.to_string(index=False))
        lines.append("")
    return "\n".join(lines)


def make_confusion(pred: dict, out_dir: Path) -> None:
    """Confusion matrix of the spoof-type head (true vs predicted)."""
    if pred["spoof_pred"].max() < 0:
        return
    from sklearn.metrics import confusion_matrix
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(
        pred["spoof_type"], pred["spoof_pred"], labels=list(range(len(SPOOF_TYPES)))
    )
    np.save(out_dir / "confusion.npy", cm)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(SPOOF_TYPES)))
    ax.set_xticklabels(SPOOF_TYPES, rotation=45, ha="right")
    ax.set_yticks(range(len(SPOOF_TYPES)))
    ax.set_yticklabels(SPOOF_TYPES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    for i in range(len(SPOOF_TYPES)):
        for j in range(len(SPOOF_TYPES)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion.png", dpi=150)
    plt.close(fig)


def evaluate_split_and_report(
    model,
    loader,
    cfg: dict,
    device: str,
    out_dir: Path,
    enable_tta: bool = True,
    threshold: float = None,
) -> dict:
    """Run the full ISO-metrics evaluation on a split and write all artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_file=str(out_dir / "eval.log"))

    pred = predict_epoch(model, loader, cfg, device, enable_tta=enable_tta)
    if threshold is None:
        threshold = find_threshold(
            pred["scores_live"], pred["labels"], cfg["eval"].get("bpcer_target", 0.01)
        )
    met = compute_pad_metrics(pred["scores_live"], pred["labels"], threshold)
    lo, hi = bootstrap_acer_ci(pred["scores_live"], pred["labels"], threshold)
    met["ACER_ci95"] = [lo, hi]

    per = per_type_df(pred, threshold)
    per.to_csv(out_dir / "per_type.csv", index=False)

    make_confusion(pred, out_dir)

    hs = _hard_samples(pred, threshold, int(cfg["eval"]["hard_samples"]["top_k"]))
    hs.to_csv(out_dir / "hard_samples.csv", index=False)
    (out_dir / "hard_samples_report.md").write_text(
        hard_samples_markdown(hs, threshold)
    )

    result = {
        "metrics": met,
        "per_type": per.to_dict(orient="records"),
        "n": int(len(pred["labels"])),
    }
    logger.info(
        "split eval | ACER=%.4f APCER=%.4f BPCER=%.4f AUC=%.4f thr=%.4f",
        met["ACER"], met["APCER"], met["BPCER"], met["AUC"], threshold,
    )
    logger.info(
        "artifacts -> %s/{per_type.csv, hard_samples*.csv|md, confusion.*}", out_dir
    )
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/exp_smoke.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    dev = resolve_device(args.device)
    out = Path(args.out or (Path(cfg["out_dir"]) / "eval"))

    import pandas as pd

    from .data import build_loaders

    splits = {
        args.split: pd.read_csv(Path(cfg["data"]["subsets_dir"]) / f"{args.split}.csv")
    }
    depth_cache = cfg["depth"].get("cache_dir") if cfg["depth"].get("enabled", True) else None
    loaders, _ = build_loaders(cfg, splits, depth_cache=depth_cache, seed=0)

    from .model import build_model

    logger = get_logger()
    model = build_model(cfg)
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(
        {k: v for k, v in ck["model"].items() if k in model.state_dict()}
    )
    model.to(dev)
    logger.info("Loaded checkpoint: %s", args.ckpt)

    result = evaluate_split_and_report(
        model, loaders[args.split], cfg, dev, out, enable_tta=not args.no_tta
    )
    save_json(result, str(out / "metrics.json"))
    print(f"metrics.json written -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
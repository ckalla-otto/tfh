"""Evaluation: ISO 30107-3 metrics, per-spoof-type breakdown, confusion matrix,
and per-class hard-sample reports (both CLI and library entry points).

  python -m pad.evaluate --config configs/exp_smoke.yaml \
      --ckpt results/run/best.pt --split test
"""

from __future__ import annotations

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
            error = float(np.mean(scores > threshold))  # APCER contribution
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
    """Per-class top-k hard samples: MISCLASSIFIED first, then near-boundary.

    A sample is "hard" when it is wrongly classified:
      - spoof class (true=0): misclassified iff score_live > threshold (pred "live")
      - live class   (true=1): misclassified iff score_live <= threshold (pred "spoof")
    Misclassified samples are ranked by how *confidently wrong* they are
    (largest |score_live - threshold| first). If a class has fewer than `top_k`
    misclassified samples, it is padded with the *nearest-to-boundary* correctly
    classified samples (smallest |score_live - threshold|), which are flagged
    `is_error=False` so the report/visuals can distinguish them.
    """
    df = pd.DataFrame(
        {
            "image_id": pred["keys"],
            "image_path": pred["path"],
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
        part = df[df["spoof_type"] == cls]
        is_live = cls == 0
        # misclassified: predicted live/surviving when it should not, or vice-versa.
        err_mask = (part["score_live"] > threshold) != is_live
        err = part[err_mask]
        ok = part[~err_mask]
        # Most confidently wrong first (largest separation from the threshold).
        err = err.sort_values("boundary_dist", ascending=False)
        # Correctly-classified padding: nearest to the boundary (most ambiguous).
        ok = ok.sort_values("boundary_dist", ascending=True)
        n_err = min(len(err), top_k)
        pad_n = max(0, top_k - n_err)
        chosen = []
        if n_err:
            chosen.append(err.head(n_err))
        if pad_n and len(ok):
            chosen.append(ok.head(pad_n))
        if not chosen:
            continue
        part = pd.concat(chosen, ignore_index=False)
        part = part.copy()
        part["spoof_type_name"] = IDX_TO_CLASS[cls]
        # misclassified flag = the original error mask, recomputed on the sub-index.
        part["is_error"] = (part["score_live"] > threshold) != is_live
        out.append(part)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def hard_samples_markdown(hard: pd.DataFrame, threshold: float) -> str:
    lines = [
        f"# Per-class hard samples (boundary = P(live) = {threshold:.4f})",
        "",
        "Within each class, rows are ordered by **how confidently wrong** the model",
        "was: for spoof classes the misclassified samples with the **highest** P(live)",
        "come first (most confidently mistaken for live); for the `live` class the",
        "misclassified samples with the **lowest** P(live) come first (most",
        "confidently mistaken for spoof). `is_error=1` marks a sample the model got",
        "wrong (misclassified); `is_error=0` rows are correctly-classified samples",
        "padded in when a class has fewer than `top_k` errors (nearest to the",
        "boundary). `boundary_dist = |score_live - threshold|`.",
        "",
    ]
    for cls in range(len(SPOOF_TYPES)):
        part = hard[hard["spoof_type"] == cls]
        if len(part) == 0:
            continue
        # errors (is_error=1) first, then padding; each block by confidence-of-error.
        err = part[part["is_error"]].sort_values("boundary_dist", ascending=False)
        ok = part[~part["is_error"]].sort_values("boundary_dist", ascending=True)
        part = pd.concat([err, ok], ignore_index=False)
        lines.append(f"## {IDX_TO_CLASS[cls]} ({len(part)})")
        show = part[
            [
                "image_id",
                "true_label",
                "score_live",
                "is_error",
                "boundary_dist",
                "spoof_pred",
                "env",
                "illum",
            ]
        ]
        try:
            lines.append(show.to_markdown(index=False))
        except ImportError:
            lines.append(show.to_string(index=False))
        lines.append(
            f"![{IDX_TO_CLASS[cls]} top hard samples](hard_visuals/{IDX_TO_CLASS[cls]}.png)"
        )
        lines.append("")
    return "\n".join(lines)


def make_hard_visuals(
    hard: pd.DataFrame, out_dir: Path, top_n: int = 5, thresh_size: int = 224
) -> None:
    """Render per-class grids of the top-N hardest images with their P(live) score.

    For each spoof class the most-deceptive samples (highest P(live)) are shown
    first; for the live class the lowest P(live) (most mistaken for spoof) come
    first. Each tile is annotated with its P(live) and predicted/true class, and
    the grids are written under ``out_dir/hard_visuals/<class>.png``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    vis_dir = out_dir / "hard_visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)

    ncols = min(top_n, 5)
    nrows = 1
    # 2 rows only if we ever exceed 5; top_n is capped at 5 so single row.
    for cls in range(len(SPOOF_TYPES)):
        part = hard[hard["spoof_type"] == cls]
        if len(part) == 0:
            continue
        # match the markdown order: misclassified (most confidently wrong) first,
        # then correctly-classified padding (nearest to the boundary).
        err = part[part["is_error"]].sort_values("boundary_dist", ascending=False)
        ok = part[~part["is_error"]].sort_values("boundary_dist", ascending=True)
        part = pd.concat([err, ok], ignore_index=False).head(top_n)

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows))
        axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
        for ax, (_, row) in zip(axes.flat, part.iterrows()):
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                img = img.resize((thresh_size, thresh_size), Image.BILINEAR)
                ax.imshow(img)
            except OSError:
                ax.text(
                    0.5,
                    0.5,
                    "missing",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            color = "lime" if row["true_label"] == 1 else "crimson"
            tag = "ERROR" if row["is_error"] else "borderline"
            ax.set_title(
                f"P(live)={row['score_live']:.3f} [{tag}]  true={'live' if row['true_label'] == 1 else 'spoof'} "
                f"pred={IDX_TO_CLASS[int(row['spoof_pred'])] if row['spoof_pred'] >= 0 else 'n/a'}",
                fontsize=9,
                color=color,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.flat[len(part) :]:
            ax.axis("off")
        fig.suptitle(f"{IDX_TO_CLASS[cls]} — top {len(part)} hardest", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(vis_dir / f"{IDX_TO_CLASS[cls]}.png", dpi=150)
        plt.close(fig)


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
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_dir / "confusion.png", dpi=150)
    plt.close(fig)


def make_det_curve(pred: dict, out_dir: Path, threshold: float, auc: float) -> None:
    """DET curve: APCER (spoof→live misclass) vs BPCER (live→spoof misclass).

    Sweeps the P(live) threshold over every observed score value and plots the
    two error rates against each other on a log scale (ISO 30107-3 convention),
    marking the operating point used elsewhere. Also writes the raw curve
    points and the underlying per-sample scores/labels so the plot can be
    regenerated or extended later.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.asarray(pred["scores_live"], dtype=np.float64)
    labels = np.asarray(pred["labels"], dtype=int)
    live = labels == 1
    spoof = labels == 0

    # On a DET plot the x-axis is BPCER (live rejected as spoof) and the y-axis
    # is APCER (spoof accepted as live). At each candidate threshold t, a sample
    # is declared "live" iff score > t.
    n_live = int(live.sum())
    n_spoof = int(spoof.sum())
    uniq = np.unique(scores)
    n_live_ok = (scores[live][:, None] > uniq[None, :]).sum(axis=0)
    n_spoof_wrong = (scores[spoof][:, None] > uniq[None, :]).sum(axis=0)
    bpcer = np.where(n_live > 0, 1.0 - n_live_ok / max(n_live, 1), np.nan)
    apcer = np.where(n_spoof > 0, n_spoof_wrong / max(n_spoof, 1), np.nan)
    fin = np.isfinite(bpcer) & np.isfinite(apcer)
    bpcer, apcer, uniq = bpcer[fin], apcer[fin], uniq[fin]

    # Protect log-scale axes against exact 0 error rates.
    eps = 1e-3
    bp_plot = np.clip(bpcer, eps, 1.0)
    ap_plot = np.clip(apcer, eps, 1.0)
    op_idx = int(np.argmin(np.abs(uniq - threshold)))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(bp_plot, ap_plot, "-o", markersize=3, lw=1.5, label="DET curve")
    ax.plot(
        [bp_plot[op_idx]],
        [ap_plot[op_idx]],
        "rs",
        ms=8,
        label=f"operating point (BPCER={bpcer[op_idx]:.3f}, APCER={apcer[op_idx]:.3f})",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(eps, 1.0)
    ax.set_ylim(eps, 1.0)
    ax.set_xlabel("BPCER (live rejected as spoof)")
    ax.set_ylabel("APCER (spoof accepted as live)")
    ax.set_title(f"DET curve  |  AUC={auc:.4f}  (n_live={n_live}, n_spoof={n_spoof})")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "det_curve.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(
        {
            "threshold": uniq,
            "BPCER": bpcer,
            "APCER": apcer,
        }
    ).to_csv(out_dir / "det_curve.csv", index=False)
    np.savez(
        out_dir / "pred.npz",
        scores_live=scores,
        labels=labels,
        spoof_type=pred["spoof_type"],
        env=pred["env"],
        illum=pred["illum"],
        depth_var=pred["depth_var"],
        hf_energy=pred["hf_energy"],
        threshold=threshold,
    )


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

    make_det_curve(pred, out_dir, threshold, met["AUC"])

    hs = _hard_samples(pred, threshold, int(cfg["eval"]["hard_samples"]["top_k"]))
    hs.to_csv(out_dir / "hard_samples.csv", index=False)
    make_hard_visuals(hs, out_dir, top_n=5)
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
        met["ACER"],
        met["APCER"],
        met["BPCER"],
        met["AUC"],
        threshold,
    )
    logger.info(
        "artifacts -> %s/{per_type.csv, hard_samples*.csv|md, confusion.*, det_curve.*, pred.npz}",
        out_dir,
    )
    return result


def main(
    config: str = "configs/exp_smoke.yaml",
    ckpt: str = None,
    split: str = "test",
    out: str = None,
    device: str = None,
    no_tta: bool = False,
) -> None:
    """Evaluate a trained checkpoint. Fire CLI: `python -m pad.evaluate [flags]`."""
    if not ckpt:
        raise ValueError("`--ckpt <path>` is required (path to a best.pt / last.pt).")

    cfg = load_config(config)
    dev = resolve_device(device)
    eval_out = Path(out or (Path(cfg["out_dir"]) / "eval"))

    import pandas as pd

    from .data import build_loaders

    splits = {split: pd.read_csv(Path(cfg["data"]["subsets_dir"]) / f"{split}.csv")}
    use_depth = bool(cfg.get("model", {}).get("use_depth_head", False))
    depth_enabled = bool(cfg["depth"].get("enabled", True))
    depth_cache = (
        cfg["depth"].get("cache_dir") if (use_depth and depth_enabled) else None
    )
    loaders, _ = build_loaders(cfg, splits, depth_cache=depth_cache, seed=0)

    from .model import build_model

    logger = get_logger()
    model = build_model(cfg)
    ck = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(
        {k: v for k, v in ck["model"].items() if k in model.state_dict()}
    )
    model.to(dev)
    logger.info("Loaded checkpoint: %s", ckpt)

    result = evaluate_split_and_report(
        model, loaders[split], cfg, dev, eval_out, enable_tta=not no_tta
    )
    save_json(result, str(eval_out / "metrics.json"))
    print(f"metrics.json written -> {eval_out}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)

"""Download ONLY the sampled subset images from a Kaggle mirror.

Pre-conditions (see README "crawling the dataset"):
  1. The mirror root-level metadata CSVs (train/test/val/metadata.csv) were
     downloaded (kaggle datasets download -f ...) and
       `python -m pad make_crawl --from-metadata --root <root> --out data/crawl.csv`
     was run.
  2. `python -m pad make_splits --config ...` produced data/subsets/{train,val,test}.csv
     with a `rel_path` column (paths relative to the dataset archive).

This command then fetches exactly those image paths via the Kaggle CLI:
    kaggle datasets download -d <slug> -f <rel_path> -p <out_dir>
parallelized with a thread pool, skipping already-present files (resume-safe),
and verifying each payload actually landed.

NOTE: requires the mirror to expose per-file access (`kaggle datasets files`
shows individual images). If a mirror bundles everything into a single zip,
per-image selection is not possible via the Kaggle API — run the regular
download (`bash scripts/download_data.sh`) once instead.

Usage (Fire):
  uv run python -m pad download_subset --subset-dir data/subsets \
      --out-dir data/images --slug <owner>/<dataset> --workers 8
  uv run python -m pad download_subset ... --check-only   # diff vs dataset file list
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


def _kaggle_cmd() -> List[str]:
    """Locate the `kaggle` console script (venv-aware)."""
    exe = Path(sys.executable).parent / "kaggle"
    return [str(exe)] if exe.exists() else ["kaggle"]


def _download_one(slug: str, rel: str, out_dir: Path) -> Tuple[str, bool, str]:
    target = out_dir / rel
    if target.exists() and target.stat().st_size > 0:
        return (rel, True, "exists")
    cmd = _kaggle_cmd() + [
        "datasets", "download", "-d", slug, "-f", rel, "-p", str(out_dir), "-q",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600,
                       env=os.environ.copy())
    except Exception as e:  # noqa: BLE001
        return (rel, False, f"kaggle call failed: {e}")
    # CLI sometimes writes the file at out_dir/<basename> instead of nesting
    where = target
    if not (where.exists() and where.stat().st_size > 0):
        base = out_dir / os.path.basename(rel)
        if base.exists() and base.stat().st_size > 0:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                base.replace(target)
                where = target
            except OSError:
                where = base
    if where.exists() and where.stat().st_size > 0:
        return (rel, True, "downloaded")
    return (rel, False, "file missing after download")


def _collect_rels(subset_dir: Path) -> set:
    import pandas as pd

    rels: set = set()
    for split in ("train", "val", "test"):
        f = subset_dir / f"{split}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "rel_path" in df.columns:
            rels.update(df["rel_path"].dropna().astype(str))
        else:
            print(f"  NOTE: {f} lacks rel_path column — re-run with a make_crawl "
                  f"generated subset (metadata-only crawl adds it).")
    return rels


def main(
    subset_dir: str = "data/subsets",
    out_dir: str = "data/images",
    slug: str = None,
    workers: int = 8,
    check_only: bool = False,
) -> None:
    """Download (or just verify) the sampled subset images (Fire CLI)."""
    subset_dir = Path(subset_dir)
    out_dir = Path(out_dir)
    if not slug:
        raise SystemExit(
            "--slug <owner>/<dataset> is required (from .env: PAD_DATASET_SLUG)"
        )
    rels = _collect_rels(subset_dir)
    if not rels:
        raise SystemExit(
            f"no rel_path entries under {subset_dir}; run `make_crawl --from-metadata` "
            "and `make_splits` first"
        )
    print(f"  subset images to fetch: {len(rels)}")

    if check_only:
        cmd = _kaggle_cmd() + ["datasets", "files", "-d", slug]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        names = set()
        for ln in out.stdout.splitlines():
            ln = ln.strip()
            if ln and not ln.startswith(("name", "----", "files")):
                names.add(ln.split()[0])
        missing = sorted(r for r in rels if r not in names)
        print(f"  dataset file-list entries: {len(names)}")
        print(f"  subset paths present in mirror: {len(rels) - len(missing)}/{len(rels)}")
        for m in missing[:20]:
            print("   MISSING in mirror:", m)
        if missing:
            raise SystemExit(1)
        print("  OK: every sampled path exists in the mirror — safe to download.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futs = {
            pool.submit(_download_one, slug, rel, out_dir): rel for rel in sorted(rels)
        }
        for fut in as_completed(futs):
            rel, good, msg = fut.result()
            if good:
                ok += 1
            else:
                fail += 1
                print(f"  FAIL {rel}: {msg}")
    print(f"download done: ok={ok} fail={fail}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
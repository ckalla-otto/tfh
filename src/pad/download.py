"""Download ONLY the sampled subset images from a Kaggle mirror via per-file API.

Full subset-only flow (no full-archive download):

  1. Dump the mirror's file listing (paths only, cheap):
       uv run python -m pad download_subset --slug <owner>/<dataset> \
           --fetch-files --files-out data/mirror_files.txt --page-size 1000
  2. Build the manifest from that listing (no images on disk):
       uv run python -m pad make_crawl --from-file-list data/mirror_files.txt \
           --out data/crawl.csv
  3. Stratified identity-exclusive subset:
       uv run python -m pad make_splits --config configs/base.yaml
  4. Download exactly those images (+ official-layout `_BB.txt` bboxes):
       uv run python -m pad download_subset --subset-dir data/subsets \
           --out-dir data/sample --slug <owner>/<dataset> --official

`kaggle datasets download -d <slug> -f <rel_path>` is used under the hood,
parallelized and resume-safe. For the official layout (`--official`), each
image's sibling `<img>_BB.txt` is fetched and the bboxes are patched into the
subset CSVs afterwards.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

_TOKEN_RE = re.compile(r"Next Page Token = (\S+)")


def _kaggle_cmd() -> List[str]:
    """Locate the `kaggle` console script (venv-aware)."""
    exe = Path(sys.executable).parent / "kaggle"
    return [str(exe)] if exe.exists() else ["kaggle"]


def fetch_dataset_files(
    slug: str,
    page_size: int = 1000,
    max_pages: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Page through `kaggle datasets files` and return [(name, size), ...].

    Pages until no `Next Page Token` (or `max_pages` reached). Only the first
    whitespace token per data row is kept (the file path).
    """
    token = None
    items: List[Tuple[str, str]] = []
    pages = 0
    while max_pages is None or pages < max_pages:
        cmd = _kaggle_cmd() + ["datasets", "files", "-d", slug, "--page-size", str(page_size)]
        if token:
            cmd += ["--page-token", token]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        text = out.stdout
        m = _TOKEN_RE.search(text)
        token = m.group(1) if m else None
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith(("name", "----", "Next Page Token")):
                continue
            toks = ln.split()
            items.append((toks[0], toks[1] if len(toks) > 1 else ""))
        pages += 1
        if not token:
            break
        print(f"  ... listed {len(items)} files (page {pages})", flush=True)
    return items


def _place_downloaded(out_dir: Path, rel: str) -> bool:
    """Locate the payload for `rel` after a `kaggle -f` download.

    The Kaggle CLI is inconsistent: it may write the file at `out_dir/<rel>`,
    `out_dir/<basename>`, or `out_dir/<basename>.zip` (single-file downloads are
    sometimes wrapped in a zip). This resolves/copies it into `out_dir/<rel>`.
    """
    import zipfile

    target = out_dir / rel
    if target.exists() and target.stat().st_size > 0:
        return True
    base = out_dir / os.path.basename(rel)
    if base.exists() and base.stat().st_size > 0:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            base.replace(target)
            return True
        except OSError:
            return False

    # zip candidates (single-file endpoint) — basename.zip or full-path.zip
    for zpath in (base.with_suffix(base.suffix + ".zip"), base.with_name(os.path.basename(rel) + ".zip")):
        if not (zpath.exists() and zpath.stat().st_size > 0):
            continue
        try:
            with zipfile.ZipFile(zpath) as zf:
                want = os.path.basename(rel)
                names = zf.namelist()
                match = next((n for n in names if os.path.basename(n) == want), None)
                if match is None:
                    continue
                zf.extract(match, out_dir)   # lands at out_dir/<match>
                tmp = out_dir / match
                if tmp.exists() and tmp.stat().st_size > 0:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target != tmp:
                        try:
                            tmp.replace(target)
                        except OSError:
                            return False
                    zpath.unlink(missing_ok=True)
                    return True
        except Exception:
            continue
    return False


def _download_one(
    slug: str, rel: str, out_dir: Path, official: bool = False
) -> Tuple[str, bool, str]:
    """Download a single file (plus its `_BB.txt` companion in official mode)."""
    target = out_dir / rel
    if target.exists() and target.stat().st_size > 0:
        _maybe_fetch_bb(slug, rel, out_dir)
        return (rel, True, "exists")

    cmd = _kaggle_cmd() + [
        "datasets", "download", "-d", slug, "-f", rel, "-p", str(out_dir), "-q",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600,
                       env=os.environ.copy())
    except Exception as e:  # noqa: BLE001
        return (rel, False, f"kaggle call failed: {e}")

    if _place_downloaded(out_dir, rel):
        if official:
            _maybe_fetch_bb(slug, rel, out_dir)
        return (rel, True, "downloaded")
    return (rel, False, "file missing after download")


def _maybe_fetch_bb(slug: str, rel: str, out_dir: Path) -> None:
    """Fetch the sibling `<img>_BB.txt` (official layout) if not already present."""
    bb_rel = f"{os.path.splitext(rel)[0]}_BB.txt"
    if _place_downloaded(out_dir, bb_rel):
        return
    cmd = _kaggle_cmd() + [
        "datasets", "download", "-d", slug, "-f", bb_rel, "-p", str(out_dir), "-q",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120,
                       env=os.environ.copy())
    except Exception:
        return
    _place_downloaded(out_dir, bb_rel)


def _patch_bboxes_from_bb_txt(subset_dir: Path, out_dir: Path) -> None:
    """Fill x1..y2 in the subset CSVs from downloaded `<img>_BB.txt` files.

    `_BB.txt` format is `x y w h [conf]` (top-left + width/height).
    """
    import pandas as pd

    for split in ("train", "val", "test"):
        f = subset_dir / f"{split}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "rel_path" not in df.columns:
            continue
        x1s, y1s, x2s, y2s = [], [], [], []
        n_bb = 0
        for rel in df["rel_path"]:
            bb_rel = f"{os.path.splitext(rel)[0]}_BB.txt"
            bb_file = out_dir / bb_rel
            if bb_file.exists():
                try:
                    toks = bb_file.read_text().strip().split()
                    if len(toks) >= 4:
                        x, y, w, h = (float(v) for v in toks[:4])
                        x1s.append(x); y1s.append(y)
                        x2s.append(x + w); y2s.append(y + h)
                        n_bb += 1
                        continue
                except Exception:
                    pass
            x1s.append(0.0); y1s.append(0.0); x2s.append(0.0); y2s.append(0.0)
        df["x1"], df["y1"], df["x2"], df["y2"] = x1s, y1s, x2s, y2s
        # the dataset reads face_x*/face_y* columns; keep them in sync
        if "face_x1" in df.columns:
            df["face_x1"], df["face_y1"] = df["x1"], df["y1"]
            df["face_x2"], df["face_y2"] = df["x2"], df["y2"]
        df.to_csv(f, index=False)
        print(f"  patched bboxes for {split}.csv ({n_bb}/{len(df)} from _BB.txt)")


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


def _relink_subset_paths(subset_dir: Path, out_dir: Path) -> None:
    """Re-point each subset CSV's image_path column at the download dir."""
    import pandas as pd

    for split in ("train", "val", "test"):
        f = subset_dir / f"{split}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "rel_path" not in df.columns:
            continue
        df["image_path"] = df["rel_path"].map(lambda r: str(out_dir / r))
        df.to_csv(f, index=False)
        print(f"  re-linked {split}.csv image_path -> {out_dir}")


def main(
    subset_dir: str = "data/subsets",
    out_dir: str = "data/images",
    slug: str = None,
    workers: int = 8,
    check_only: bool = False,
    official: bool = False,
    fetch_files: bool = False,
    files_out: str = "data/mirror_files.txt",
    page_size: int = 1000,
    max_pages: int = None,
) -> None:
    """Download (or just verify) the sampled subset images (Fire CLI)."""
    subset_dir = Path(subset_dir)
    out_dir = Path(out_dir)

    if not slug:
        raise SystemExit(
            "--slug <owner>/<dataset> is required (from .env: PAD_DATASET_SLUG)"
        )

    # --- mode 1: just dump the mirror's file listing (paths only) -----------
    if fetch_files:
        print(f"listing files of {slug} (page-size {page_size}) ...")
        items = fetch_dataset_files(slug, page_size=page_size, max_pages=max_pages)
        files_out = Path(files_out)
        files_out.parent.mkdir(parents=True, exist_ok=True)
        with open(files_out, "w") as f:
            for name, size in items:
                f.write(f"{name}\t{size}\n")
        print(f"listed {len(items)} files -> {files_out}")
        print("next: make_crawl --from-file-list --out data/crawl.csv")
        return

    rels = _collect_rels(subset_dir)
    if not rels:
        raise SystemExit(
            f"no rel_path entries under {subset_dir}; run `make_crawl --from-metadata` "
            "(or `--from-file-list`) and `make_splits` first"
        )
    print(f"  subset images to fetch: {len(rels)}")

    if check_only:
        names = {n for n, _s in fetch_dataset_files(slug, page_size=page_size)}
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
            pool.submit(_download_one, slug, rel, out_dir, official): rel
            for rel in sorted(rels)
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
    _relink_subset_paths(subset_dir, out_dir)
    if official:
        _patch_bboxes_from_bb_txt(subset_dir, out_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
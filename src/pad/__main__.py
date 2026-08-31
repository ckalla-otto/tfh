"""Package-level CLI gateway.

With PYTHONPATH=src (or after `pip install -e .`), you can run:

    python -m pad train   --config configs/exp_smoke.yaml --run-name smoke
    python -m pad evaluate --ckpt results/smoke/best.pt --split test
    python -m pad depth_targets --config configs/exp_smoke.yaml

All subcommands delegate to the Fire-based `main()` of the corresponding module.
"""
from __future__ import annotations

import fire

from . import crawl, depth_targets, evaluate, train

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train.main,
            "evaluate": evaluate.main,
            "depth_targets": depth_targets.main,
            "make_crawl": crawl.main,
        }
    )
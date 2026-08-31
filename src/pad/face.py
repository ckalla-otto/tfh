"""Face detection via InsightFace (deepinsight/insightface) — SCRFD/RetinaFace.

Used by `predict.py` as the automatic face detector when no face bbox is
provided (explicit `--bbox` always wins). Lazy imports keep the training
pipeline (which uses the dataset's own x1..y2 annotations) free of the
insightface/onnxruntime dependency.

Usage:
  from pad.face import detect_face_bbox
  box = detect_face_bbox(pil_rgb)          # (x1, y1, x2, y2) in absolute px
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

_app = None  # lazy InsightFace FaceAnalysis singleton


def _available_providers() -> list:
    try:
        import onnxruntime

        return list(onnxruntime.get_available_providers())
    except Exception:
        return []


def _get_app(min_conf: float = 0.5, det_size: Tuple[int, int] = (640, 640)):
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis

        use = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in _available_providers()
        ] or ["CPUExecutionProvider"]
        # detection only -> loads the SCRFD detector (buffalo_l), not recognition.
        _app = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection"], providers=use
        )
        ctx = 0 if "CUDAExecutionProvider" in use else -1
        _app.prepare(ctx_id=ctx, det_thresh=min_conf, det_size=det_size)
    return _app


def detect_face_bbox(
    img: Image.Image,
    min_conf: float = 0.5,
) -> Optional[Tuple[float, float, float, float]]:
    """Detect the largest face and return an absolute bbox (x1, y1, x2, y2).

    Returns None if InsightFace is unavailable or no face is found.
    """
    try:
        app = _get_app(min_conf=min_conf)
    except Exception:
        return None
    rgb = np.asarray(img.convert("RGB"))
    try:
        faces = app.get(rgb)
    except Exception:
        return None
    if not faces:
        return None
    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    bb = faces[0].bbox  # [x1, y1, x2, y2] floats in absolute pixels
    x1, y1, x2, y2 = (float(v) for v in bb[:4])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)
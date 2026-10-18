"""2× and 4× upscaling via HAT-S (Apache-2.0, ArturWoz/HAT-Sentinel-onnx)."""
from __future__ import annotations

import numpy as np
from lce.engine import run_tiled

_MODEL_IDS = {2: "hat-s-x2", 4: "hat-s-x4"}
_TILE = 256
_OVERLAP = 16


def _infer(session, feed: dict) -> np.ndarray:
    """Unified inference for ONNX InferenceSession or OpenVINO compiled model."""
    try:
        return session.run(None, feed)[0]
    except AttributeError:
        result = session(feed)
        return result[session.output(0)]


def run(img_chw_f32: np.ndarray, cache, scale: int = 4) -> np.ndarray:
    """
    Upscale img_chw_f32 [C,H,W] by scale (2 or 4).
    Uses run_tiled from lce.engine with 256×256 tiles, overlap 16.
    """
    if scale not in _MODEL_IDS:
        raise ValueError(f"scale must be 2 or 4, got {scale}")

    from lce.models.registry import ensure
    path = ensure(_MODEL_IDS[scale])
    session = cache.load(path)

    def _feed(tile_chw: np.ndarray, sess) -> np.ndarray:
        out = _infer(sess, {"input": tile_chw[np.newaxis]})
        return out[0]  # [C, tile_h*scale, tile_w*scale]

    return run_tiled(session, img_chw_f32, _TILE, _TILE, _OVERLAP, scale, _feed)


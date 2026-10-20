"""2× and 4× upscaling via HAT (ArturWoz/HAT-Sentinel-onnx, Sentinel-2 fine-tune)."""
from __future__ import annotations

import numpy as np
from lce.engine import input_name, run_tiled, static_hw

_MODEL_IDS = {2: "hat-sentinel-x2", 4: "hat-sentinel-x4"}
_DEFAULT_TILE = 256   # used only when the model has dynamic spatial dims
_DEFAULT_OVERLAP = 16


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
    Tiles through the model with run_tiled. The registered HAT models take a
    fixed 64×64 input, so the tile size is read from the model; a dynamic-shape
    model would use 256×256 tiles.
    """
    if scale not in _MODEL_IDS:
        raise ValueError(f"scale must be 2 or 4, got {scale}")

    from lce.models.registry import ensure
    path = ensure(_MODEL_IDS[scale])
    session = cache.load(path)
    name = input_name(session)
    tile_h, tile_w = static_hw(session) or (_DEFAULT_TILE, _DEFAULT_TILE)
    overlap = min(_DEFAULT_OVERLAP, tile_h // 4, tile_w // 4)

    def _feed(tile_chw: np.ndarray, sess) -> np.ndarray:
        out = _infer(sess, {name: tile_chw[np.newaxis]})
        return out[0]  # [C, tile_h*scale, tile_w*scale]

    return run_tiled(session, img_chw_f32, tile_h, tile_w, overlap, scale, _feed)

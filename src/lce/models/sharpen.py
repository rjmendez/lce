"""Lens-blur/motion-blur deblurring via NAFNet int8 (MIT, opencv/deblurring_nafnet)."""
from __future__ import annotations

import numpy as np

_MODEL_IDS = {False: "nafnet-int8", True: "nafnet-fp32"}
_TILE = 720
_OVERLAP = 32


def _infer(session, feed: dict) -> np.ndarray:
    """Unified inference for ONNX InferenceSession or OpenVINO compiled model."""
    try:
        return session.run(None, feed)[0]
    except AttributeError:
        result = session(feed)
        return result[session.output(0)]


def _pad_to_multiple(img: np.ndarray, multiple: int = 8):
    """Pad [C,H,W] to dimensions divisible by *multiple*. Returns (padded, (H_orig, W_orig))."""
    C, H, W = img.shape
    pH = (multiple - H % multiple) % multiple
    pW = (multiple - W % multiple) % multiple
    if pH or pW:
        img = np.pad(img, ((0, 0), (0, pH), (0, pW)), mode="reflect")
    return img, (H, W)


def run(img_chw_f32: np.ndarray, cache, strength: float = 1.0, quality: bool = False) -> np.ndarray:
    """
    Deblur image. strength in [0,1] controls alpha blend with original.
    quality=True uses fp32 model; quality=False (default) uses int8.
    """
    from lce.models.registry import ensure
    path = ensure(_MODEL_IDS[quality])
    session = cache.load(path)

    _, H, W = img_chw_f32.shape
    if H > 1440 or W > 1920:
        from lce.engine import run_tiled

        def _feed(tile_chw: np.ndarray, sess) -> np.ndarray:
            padded, (oh, ow) = _pad_to_multiple(tile_chw)
            out = _infer(sess, {"input": padded[np.newaxis]})[0]
            return out[:, :oh, :ow]

        deblurred = run_tiled(session, img_chw_f32, _TILE, _TILE, _OVERLAP, 1, _feed)
    else:
        padded, (H_orig, W_orig) = _pad_to_multiple(img_chw_f32)
        out = _infer(session, {"input": padded[np.newaxis]})[0]
        deblurred = out[:, :H_orig, :W_orig]

    return np.clip(strength * deblurred + (1.0 - strength) * img_chw_f32, 0.0, 1.0)


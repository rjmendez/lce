"""
lce.router — DA-CLIP degradation-aware route selection.

Runs a lightweight CLIP-based degradation classifier on a centre crop
and returns a route string that the enhance pipeline uses to select models.

Routes:
  skip          — image looks clean; no enhancement needed
  sharpen       — moderate lens/motion blur detected
  sharpen_strong— heavy blur / compression artifacts
  upscale       — low resolution, upscaling is the primary need
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

_CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _preprocess(img_chw_f32: np.ndarray) -> np.ndarray:
    """Resize-crop to 224×224 and apply CLIP normalisation. Returns [1,224,224,3]."""
    from PIL import Image
    hwc = (img_chw_f32.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(hwc).resize((224, 224), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr - _CLIP_MEAN) / _CLIP_STD
    return arr[np.newaxis]             # [1, 224, 224, 3]


def _route_from_norm(degrad_norm: float) -> str:
    if degrad_norm < 0.30:
        return "skip"
    if degrad_norm < 0.60:
        return "sharpen"
    return "sharpen_strong"


def route(img_chw_f32: np.ndarray, cache, model_id: str = "daclip") -> str:
    """Return a route string for *img_chw_f32*.

    Falls back to "sharpen" if the DA-CLIP model is unavailable.
    """
    from lce.models.registry import model_path
    p = model_path(model_id)
    if p is None:
        return "sharpen"          # safe default

    session = cache.load(p)
    patch = _preprocess(img_chw_f32)  # [1,224,224,3]
    # Run inference (ONNX or OV)
    try:
        outputs = session.run(None, {"image": patch})
        vecs = outputs[0]             # [2, 512]
    except Exception:
        try:
            result = session({"image": patch})
            vecs = next(iter(result.values()))
        except Exception:
            return "sharpen"

    degrad_vec = vecs[1]              # row 1 = degradation embedding
    norm = float(np.linalg.norm(degrad_vec))
    return _route_from_norm(norm)

"""Full auto-enhance pipeline: triage → route → process."""
from __future__ import annotations

import numpy as np


def run(img_chw_f32: np.ndarray, cache, mode: str = "auto") -> np.ndarray:
    """
    mode="auto"           : triage then apply recommended pipeline
    mode="sharpen"        : always sharpen
    mode="upscale"        : always upscale
    mode="sharpen_upscale": sharpen then upscale
    """
    from lce.models import sharpen, upscale, triage

    if mode == "sharpen":
        return sharpen.run(img_chw_f32, cache)
    if mode == "upscale":
        return upscale.run(img_chw_f32, cache)
    if mode == "sharpen_upscale":
        sharpened = sharpen.run(img_chw_f32, cache)
        return upscale.run(sharpened, cache)

    # mode == "auto"
    route = triage.run(img_chw_f32, cache)
    if route == "skip":
        return img_chw_f32
    if route == "sharpen":
        return sharpen.run(img_chw_f32, cache)
    if route == "sharpen_strong":
        return sharpen.run(img_chw_f32, cache, strength=1.0)
    if route == "upscale":
        return upscale.run(img_chw_f32, cache)
    return img_chw_f32


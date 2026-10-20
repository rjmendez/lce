"""Full auto-enhance pipeline: triage → route → process."""
from __future__ import annotations

import numpy as np

_MODES = ("auto", "sharpen", "upscale", "sharpen_upscale")
_ROUTES = ("skip", "sharpen", "sharpen_strong", "upscale")


def run_with_route(
    img_chw_f32: np.ndarray, cache, mode: str = "auto", scale: int = 4
) -> tuple[np.ndarray, str]:
    """Like :func:`run` but also returns the route that was applied:
    'skip' | 'sharpen' | 'sharpen_strong' | 'upscale' | 'sharpen_upscale'.

    Raises ValueError for an unknown *mode* or an unknown triage route.
    """
    from lce.models import sharpen, upscale, triage

    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    if mode == "sharpen":
        return sharpen.run(img_chw_f32, cache), "sharpen"
    if mode == "upscale":
        return upscale.run(img_chw_f32, cache, scale=scale), "upscale"
    if mode == "sharpen_upscale":
        sharpened = sharpen.run(img_chw_f32, cache)
        return upscale.run(sharpened, cache, scale=scale), "sharpen_upscale"

    # mode == "auto"
    route = triage.run(img_chw_f32, cache)
    if route == "skip":
        return img_chw_f32, route
    if route == "sharpen":
        return sharpen.run(img_chw_f32, cache), route
    if route == "sharpen_strong":
        return sharpen.run(img_chw_f32, cache, strength=1.0), route
    if route == "upscale":
        return upscale.run(img_chw_f32, cache, scale=scale), route
    raise ValueError(f"triage returned unknown route {route!r}; expected one of {_ROUTES}")


def run(img_chw_f32: np.ndarray, cache, mode: str = "auto", scale: int = 4) -> np.ndarray:
    """
    mode="auto"           : triage then apply recommended pipeline
    mode="sharpen"        : always sharpen
    mode="upscale"        : always upscale by *scale* (2 or 4)
    mode="sharpen_upscale": sharpen then upscale by *scale*

    Note: auto routes any image with min(H, W) < 800 to upscale, whatever its
    blur score, and the result is not guaranteed to meet any export size.
    """
    return run_with_route(img_chw_f32, cache, mode, scale)[0]

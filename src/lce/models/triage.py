"""Image quality routing via BRISQUE (piq, MIT) + resolution check."""
from __future__ import annotations

import numpy as np

_SKIP_THRESHOLD = 20.0
_SHARPEN_THRESHOLD = 50.0
_UPSCALE_MIN_DIM = 800


def run(img_chw_f32: np.ndarray, cache=None) -> str:
    """
    Returns route: 'skip' | 'sharpen' | 'sharpen_strong' | 'upscale'.

    *cache* is accepted for API consistency but unused (BRISQUE is pure math).
    """
    import torch
    from piq import brisque

    _, H, W = img_chw_f32.shape

    tensor = torch.from_numpy(img_chw_f32[np.newaxis])
    score = float(brisque(tensor, data_range=1.0))

    if score < _SKIP_THRESHOLD:
        route = "skip"
    elif score < _SHARPEN_THRESHOLD:
        route = "sharpen"
    else:
        route = "sharpen_strong"

    if min(H, W) < _UPSCALE_MIN_DIM:
        route = "upscale"

    return route


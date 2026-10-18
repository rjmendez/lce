"""
lce.router — degradation-aware route selection.

Delegates to lce.models.triage for BRISQUE-based quality routing.

Routes:
  skip          — image looks clean; no enhancement needed
  sharpen       — moderate lens/motion blur detected
  sharpen_strong— heavy blur / compression artifacts
  upscale       — low resolution, upscaling is the primary need
"""
from __future__ import annotations

import numpy as np


def route(img_chw_f32: np.ndarray, cache=None, model_id: str = "daclip") -> str:
    """Return a route string for *img_chw_f32*.

    Delegates to :func:`lce.models.triage.run` for BRISQUE-based routing.
    The *model_id* parameter is retained for API compatibility but is unused.
    """
    from lce.models.triage import run as triage_run
    return triage_run(img_chw_f32, cache)


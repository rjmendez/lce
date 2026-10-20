"""
lce.router — route selection.

Delegates to lce.models.triage (BRISQUE no-reference quality score plus a
resolution check). Not used by the MCP server, which calls triage directly.

Routes:
  skip          — BRISQUE score < 20; no enhancement needed
  sharpen       — score 20-50
  sharpen_strong— score >= 50
  upscale       — min(H, W) < 800, regardless of score
"""
from __future__ import annotations

import numpy as np


def route(img_chw_f32: np.ndarray, cache=None) -> str:
    """Return a route string for *img_chw_f32* (see :func:`lce.models.triage.run`)."""
    from lce.models.triage import run as triage_run
    return triage_run(img_chw_f32, cache)

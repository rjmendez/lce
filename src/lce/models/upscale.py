"""2x and 4x upscaling via HAT. Default at 2x is a local-only HAT-S + LoRA
fine-tune on real stock photos (measured best on ordinary photos); default at
4x is still ArturWoz/HAT-Sentinel-onnx, a Sentinel-2 satellite-imagery
fine-tune (no validated real-photo 4x model exists). variant="sentinel" forces
the satellite model explicitly at either scale. If the scale=2 default's
local-only weights are absent (e.g. a fresh install before weights are
synced), run() falls back to the satellite model instead of hard-failing --
but always via warnings.warn(), never silently; see run()'s docstring."""
from __future__ import annotations

import warnings

import numpy as np
from lce.engine import input_name, run_tiled, static_hw

_MODEL_IDS = {2: "hat-stock-lora-x2", 4: "hat-sentinel-x4"}
# Alternate upscalers selected by variant instead of just scale. "stock" at
# scale=2 is a HAT-S + LoRA fine-tune on real stock nature photos (local-only,
# see models.toml); it is also the scale=2 default above (measured better on
# ordinary photos than the Sentinel-2 satellite fine-tune). "stock" at scale=4
# is the plain official HAT-S checkpoint, no fine-tuning -- a LoRA attempt at
# 4x failed its gate, so this is zero-shot only -- and it is NOT the scale=4
# default, since (unlike the x2 case) it has no real-photo head-to-head
# measurement against the satellite model, only a vs-bicubic one. "sentinel"
# gives the satellite-imagery HAT its own explicit key at both scales, so
# variant="sentinel" always means the satellite model regardless of what the
# scale's default currently is.
_VARIANT_MODEL_IDS = {
    ("sentinel", 2): "hat-sentinel-x2",
    ("sentinel", 4): "hat-sentinel-x4",
    ("stock", 2): "hat-stock-lora-x2",
    ("stock", 4): "hat-stock-x4",
}
_DEFAULT_TILE = 256   # used only when the model has dynamic spatial dims
_DEFAULT_OVERLAP = 16


def recommended_scale(w: int, h: int, max_mp: float = 100.0) -> int | None:
    """The largest registered scale (see _MODEL_IDS) whose output -- w*scale
    by h*scale -- stays at or under *max_mp* megapixels; None if even the
    smallest registered scale would exceed it.

    A platform's megapixel ceiling (e.g. Adobe Stock's 100MP) is easy to
    blow past: scale multiplies each linear dimension, so scale=4 multiplies
    the pixel COUNT by 16x, not 4x. This picks the scale for you instead of
    assuming the largest one is always safe.
    """
    mp = w * h / 1e6
    for scale in sorted(_MODEL_IDS, reverse=True):
        if mp * (scale ** 2) <= max_mp:
            return scale
    return None


def _infer(session, feed: dict) -> np.ndarray:
    """Unified inference for ONNX InferenceSession or OpenVINO compiled model."""
    try:
        return session.run(None, feed)[0]
    except AttributeError:
        result = session(feed)
        return result[session.output(0)]


def run(
    img_chw_f32: np.ndarray,
    cache,
    scale: int = 4,
    variant: str | None = None,
    warn: list[str] | None = None,
) -> np.ndarray:
    """
    Upscale img_chw_f32 [C,H,W] by scale (2 or 4).
    Tiles through the model with run_tiled. The registered HAT models take a
    fixed 64×64 input, so the tile size is read from the model; a dynamic-shape
    model would use 256×256 tiles.

    variant: None (default) selects, per scale, the model measured best on
    ordinary photos: at scale=2 that is the HAT-S + LoRA fine-tune on real
    stock nature photos ("stock" below, local-only -- see models.toml); at
    scale=4 it is still the Sentinel-2 satellite-imagery HAT, because no
    *real-photo-measured* 4x model exists yet -- a HAT-S 4x LoRA fine-tune
    attempt failed its quality gate (best checkpoint was step 0, no
    improvement over zero-shot) and was never shipped. "sentinel" forces
    the Sentinel-2 satellite-imagery HAT explicitly, at either scale, even
    though it is no longer the scale=2 default. "stock" forces the HAT-S +
    LoRA fine-tune explicitly at scale=2, or, at scale=4, the official
    HAT-S checkpoint with no fine-tuning at all (only measured against
    bicubic, not head-to-head against the satellite model on real photos
    -- so it is offered as an explicit variant, not the scale=4 default).

    If the DEFAULT (variant=None) model's weights are missing -- e.g. a
    fresh install of this machine-specific local-only model before weights
    are synced over -- this falls back to the satellite model for that
    scale instead of hard-failing, but never silently: it always calls
    warnings.warn(), and if *warn* (a list) is passed, it also appends the
    message there for a caller (e.g. the MCP server) to surface directly in
    its response. An EXPLICIT variant request (e.g. variant="stock") never
    falls back -- registry.ensure()'s FileNotFoundError propagates unchanged,
    since a caller who named a specific model should get a loud, specific
    error, not a substitution they didn't ask for.
    """
    from lce.models.registry import ensure

    if variant is None:
        if scale not in _MODEL_IDS:
            raise ValueError(f"scale must be 2 or 4, got {scale}")
        model_id = _MODEL_IDS[scale]
        try:
            path = ensure(model_id)
        except FileNotFoundError:
            fallback_id = _VARIANT_MODEL_IDS.get(("sentinel", scale))
            if fallback_id is None:
                raise
            msg = (
                f"default upscale model '{model_id}' is missing on this "
                f"machine; falling back to '{fallback_id}' (lower measured "
                f"quality -- run `lce-weights get {model_id}` or copy the "
                f"local file into place to get the better default back)."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            if warn is not None:
                warn.append(msg)
            model_id = fallback_id
            path = ensure(model_id)
    else:
        key = (variant, scale)
        if key not in _VARIANT_MODEL_IDS:
            raise ValueError(
                f"no {variant!r} upscale model for scale={scale}; available: "
                f"{sorted(_VARIANT_MODEL_IDS)}"
            )
        model_id = _VARIANT_MODEL_IDS[key]
        path = ensure(model_id)  # explicit request: no fallback, ever

    session = cache.load(path)
    name = input_name(session)
    tile_h, tile_w = static_hw(session) or (_DEFAULT_TILE, _DEFAULT_TILE)
    overlap = min(_DEFAULT_OVERLAP, tile_h // 4, tile_w // 4)

    def _feed(tile_chw: np.ndarray, sess) -> np.ndarray:
        out = _infer(sess, {name: tile_chw[np.newaxis]})
        return out[0]  # [C, tile_h*scale, tile_w*scale]

    return run_tiled(session, img_chw_f32, tile_h, tile_w, overlap, scale, _feed)

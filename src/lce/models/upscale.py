"""2x and 4x upscaling via HAT. Default at 2x is a local-only HAT-S + LoRA
fine-tune on real stock photos (measured best on ordinary photos); default at
4x is still ArturWoz/HAT-Sentinel-onnx, a Sentinel-2 satellite-imagery
fine-tune (no validated real-photo 4x model exists). variant="sentinel" forces
the satellite model explicitly at either scale."""
from __future__ import annotations

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


def _infer(session, feed: dict) -> np.ndarray:
    """Unified inference for ONNX InferenceSession or OpenVINO compiled model."""
    try:
        return session.run(None, feed)[0]
    except AttributeError:
        result = session(feed)
        return result[session.output(0)]


def run(img_chw_f32: np.ndarray, cache, scale: int = 4, variant: str | None = None) -> np.ndarray:
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
    There is no fallback: if a "stock" model's weights are absent,
    registry.ensure() raises FileNotFoundError (naming the model id and
    missing path) and that propagates unchanged -- a silent drop to a
    lower-quality model would be an unannounced degraded result.
    """
    from lce.models.registry import ensure

    if variant is None:
        if scale not in _MODEL_IDS:
            raise ValueError(f"scale must be 2 or 4, got {scale}")
        model_id = _MODEL_IDS[scale]
    else:
        key = (variant, scale)
        if key not in _VARIANT_MODEL_IDS:
            raise ValueError(
                f"no {variant!r} upscale model for scale={scale}; available: "
                f"{sorted(_VARIANT_MODEL_IDS)}"
            )
        model_id = _VARIANT_MODEL_IDS[key]

    path = ensure(model_id)
    session = cache.load(path)
    name = input_name(session)
    tile_h, tile_w = static_hw(session) or (_DEFAULT_TILE, _DEFAULT_TILE)
    overlap = min(_DEFAULT_OVERLAP, tile_h // 4, tile_w // 4)

    def _feed(tile_chw: np.ndarray, sess) -> np.ndarray:
        out = _infer(sess, {name: tile_chw[np.newaxis]})
        return out[0]  # [C, tile_h*scale, tile_w*scale]

    return run_tiled(session, img_chw_f32, tile_h, tile_w, overlap, scale, _feed)

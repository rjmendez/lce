"""Segmentation mask via MobileSAM (Acly/MobileSAM ONNX export)."""
from __future__ import annotations

import numpy as np

from lce.engine import input_name


_ENCODER_ID = "mobilesam-encoder"
_DECODER_ID = "mobilesam-decoder"
_LONG_SIDE = 1024


def _infer(session, feed: dict) -> list:
    """Return all outputs as a list for ONNX or OpenVINO sessions."""
    try:
        return session.run(None, feed)
    except AttributeError:
        result = session(feed)
        return [result[session.output(i)] for i in range(len(result))]


def _resize_hwc255(img_chw: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """float [0,1] CHW -> float32 HWC in [0,255], bicubic-resized."""
    from PIL import Image
    from lce.engine import to_uint8_hwc
    pil = Image.fromarray(to_uint8_hwc(img_chw)).resize((new_w, new_h), Image.BICUBIC)
    return np.asarray(pil, dtype=np.float32)


def run(
    img_chw_f32: np.ndarray,
    cache,
    points: list[tuple[float, float]] | None = None,
    labels: list[int] | None = None,
) -> np.ndarray:
    """
    Returns a mask [1,H,W] float32 (1 = selected, 0 = not) at the input size.

    The image is resized so its long side is 1024 (aspect preserved) and fed
    to the encoder as HWC float 0-255; the exported encoder does the SAM
    normalisation and padding itself. The single-mask decoder returns one
    mask for whatever points are passed as the prompt.

    points: (x, y) in normalised [0,1] fractions of the ORIGINAL image's
    width/height (not the resized/pixel frame -- this function handles that
    conversion). None (default) uses a uniform 3x3 grid of positive points
    over the whole image -- a generic prompt that selects whatever region the
    grid points agree on, not necessarily "the subject". Passing points lets a
    caller who knows roughly where the subject is get an actual subject mask
    instead.
    labels: 1 (foreground) or 0 (background) per point, same length as
    points; defaults to all-foreground (1) if omitted. Ignored if points is
    None (the default grid is always all-foreground, unchanged from before).
    """
    from lce.models.registry import ensure

    _, H, W = img_chw_f32.shape
    k = _LONG_SIDE / max(H, W)
    new_h, new_w = max(1, round(H * k)), max(1, round(W * k))

    enc_sess = cache.load(ensure(_ENCODER_ID))
    resized = _resize_hwc255(img_chw_f32, new_h, new_w)
    embeddings = _infer(enc_sess, {input_name(enc_sess): resized})[0]  # [1,256,64,64]

    if points is None:
        # Default prompt, in the resized-image pixel frame.
        pts = [[(gx + 0.5) / 3.0 * new_w, (gy + 0.5) / 3.0 * new_h]
               for gy in range(3) for gx in range(3)]
        point_labels = np.ones((1, 9), dtype=np.float32)
    else:
        if labels is not None and len(labels) != len(points):
            raise ValueError(
                f"labels must be the same length as points (got {len(labels)} "
                f"labels for {len(points)} points)"
            )
        pts = [[x * new_w, y * new_h] for x, y in points]
        point_labels = np.array(
            [labels if labels is not None else [1] * len(points)], dtype=np.float32
        )
    point_coords = np.array(pts, dtype=np.float32).reshape(1, len(pts), 2)

    dec_sess = cache.load(ensure(_DECODER_ID))
    outputs = _infer(dec_sess, {
        "image_embeddings": embeddings,
        "point_coords":     point_coords,
        "point_labels":     point_labels,
        "mask_input":       np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input":   np.zeros((1,), dtype=np.float32),
        "orig_im_size":     np.array([H, W], dtype=np.float32),
    })
    masks = outputs[0]  # [1, 1, H, W] for the single-mask decoder
    return (masks[0, :1] > 0.0).astype(np.float32)

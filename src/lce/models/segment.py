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


def run(img_chw_f32: np.ndarray, cache) -> np.ndarray:
    """
    Returns a mask [1,H,W] float32 (1 = selected, 0 = not) at the input size.

    The image is resized so its long side is 1024 (aspect preserved) and fed
    to the encoder as HWC float 0-255; the exported encoder does the SAM
    normalisation and padding itself. A uniform 3×3 grid of positive points
    over the image is the prompt, and the single-mask decoder returns one
    mask. This is a generic prompt: it selects whatever region the grid points
    agree on, not necessarily "the subject".
    """
    from lce.models.registry import ensure

    _, H, W = img_chw_f32.shape
    k = _LONG_SIDE / max(H, W)
    new_h, new_w = max(1, round(H * k)), max(1, round(W * k))

    enc_sess = cache.load(ensure(_ENCODER_ID))
    resized = _resize_hwc255(img_chw_f32, new_h, new_w)
    embeddings = _infer(enc_sess, {input_name(enc_sess): resized})[0]  # [1,256,64,64]

    # Prompt points are in the resized-image pixel frame.
    pts = [[(gx + 0.5) / 3.0 * new_w, (gy + 0.5) / 3.0 * new_h]
           for gy in range(3) for gx in range(3)]
    point_coords = np.array(pts, dtype=np.float32).reshape(1, 9, 2)
    point_labels = np.ones((1, 9), dtype=np.float32)

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

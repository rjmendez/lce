"""Subject segmentation via MobileSAM (Apache-2.0, qualcomm/MobileSam)."""
from __future__ import annotations

import numpy as np

_ENCODER_ID = "mobilesam-encoder"
_DECODER_ID = "mobilesam-decoder"
_IMG_SIZE = 1024
# ImageNet-derived normalization constants used by SAM
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32).reshape(3, 1, 1)


def _infer(session, feed: dict) -> list:
    """Return all outputs as a list for ONNX or OpenVINO sessions."""
    try:
        return session.run(None, feed)
    except AttributeError:
        result = session(feed)
        return [result[session.output(i)] for i in range(len(result))]


def _resize_chw(img: np.ndarray, h: int, w: int) -> np.ndarray:
    from PIL import Image
    hwc = (img.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(hwc).resize((w, h), Image.BICUBIC)
    return np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0


def run(img_chw_f32: np.ndarray, cache) -> np.ndarray:
    """
    Returns binary mask [1,H,W] float32 (1=subject, 0=background).
    Uses a 3×3 grid auto-prompt; returns the largest mask.
    """
    from lce.models.registry import ensure

    _, H, W = img_chw_f32.shape

    # --- Encode ---
    enc_path = ensure(_ENCODER_ID)
    enc_sess = cache.load(enc_path)
    resized = _resize_chw(img_chw_f32, _IMG_SIZE, _IMG_SIZE)
    enc_input = (resized * 255.0 - _MEAN) / _STD
    embeddings = _infer(enc_sess, {"input_image": enc_input[np.newaxis]})[0]  # [1,256,64,64]

    # --- Build 3×3 center-biased grid prompts ---
    pts = []
    for gy in range(3):
        for gx in range(3):
            px = (gx + 0.5) / 3.0 * _IMG_SIZE
            py = (gy + 0.5) / 3.0 * _IMG_SIZE
            pts.append([px, py])
    point_coords = np.array(pts, dtype=np.float32).reshape(1, 9, 2)
    point_labels = np.ones((1, 9), dtype=np.float32)

    # --- Decode ---
    dec_path = ensure(_DECODER_ID)
    dec_sess = cache.load(dec_path)
    has_mask_input = np.zeros((1,), dtype=np.float32)
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    orig_im_size = np.array([H, W], dtype=np.float32)

    outputs = _infer(dec_sess, {
        "image_embeddings": embeddings,
        "point_coords":     point_coords,
        "point_labels":     point_labels,
        "has_mask_input":   has_mask_input,
        "mask_input":       mask_input,
        "orig_im_size":     orig_im_size,
    })
    masks_raw = outputs[0]  # [1, N, H, W] or [1, 1, H, W]

    if masks_raw.ndim == 4 and masks_raw.shape[1] > 1:
        areas = (masks_raw > 0.0).sum(axis=(2, 3))
        best = int(np.argmax(areas[0]))
        mask = masks_raw[0, best : best + 1]
    else:
        mask = masks_raw[0]  # [1, H, W]

    return (mask > 0.0).astype(np.float32)


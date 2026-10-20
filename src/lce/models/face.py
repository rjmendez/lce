"""Face restoration via GFPGAN v1.4 (optional face pack). Not exposed by the MCP server."""
from __future__ import annotations

import numpy as np

from lce.engine import input_name

_MODEL_ID = "gfpgan-v14"
_SIZE = 512


def _infer(session, feed: dict) -> np.ndarray:
    """Unified inference for ONNX InferenceSession or OpenVINO compiled model."""
    try:
        return session.run(None, feed)[0]
    except AttributeError:
        result = session(feed)
        return result[session.output(0)]


def run(img_chw_f32: np.ndarray, cache) -> np.ndarray:
    """
    Restore a face image. Model must be downloaded separately:
      lce-weights get gfpgan-v14
    Raises RuntimeError with helpful message if model not downloaded.
    There is no face detection: the WHOLE image is resized to 512×512 (aspect
    ratio not preserved), run through GFPGAN, and resized back. Only meaningful
    for an already-cropped face; a full photo comes back distorted.
    Normalise to [-1,1] before inference, denormalise after.
    """
    from lce.models.registry import model_path
    p = model_path(_MODEL_ID)
    if p is None:
        raise RuntimeError(
            "GFPGAN v1.4 model not found. Download it with:\n"
            "  lce-weights get gfpgan-v14\n"
            "(~340 MB, optional face pack)"
        )

    from PIL import Image
    _, H, W = img_chw_f32.shape
    session = cache.load(p)

    hwc = (img_chw_f32.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    resized = (
        np.asarray(Image.fromarray(hwc).resize((_SIZE, _SIZE), Image.BICUBIC), dtype=np.float32)
        .transpose(2, 0, 1)
        / 255.0
    )
    normalized = resized * 2.0 - 1.0                               # [0,1] → [-1,1]
    out = _infer(session, {input_name(session): normalized[np.newaxis]})[0]    # [3,512,512] in [-1,1]
    restored = np.clip((out + 1.0) / 2.0, 0.0, 1.0)               # [-1,1] → [0,1]

    hwc_out = (restored.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    result = (
        np.asarray(Image.fromarray(hwc_out).resize((W, H), Image.BICUBIC), dtype=np.float32)
        .transpose(2, 0, 1)
        / 255.0
    )
    return result


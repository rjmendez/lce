"""
lce.engine — model-agnostic tiled inference engine.

Loads OpenVINO IR (.xml/.bin) or ONNX (.onnx) models, caches compiled
instances in memory (LRU), and runs large images through any model in
overlapping tiles with cosine-blended seams.

Intentionally model-agnostic: no Topaz-specific logic here.
The models/ package maps capabilities (upscale, sharpen, ...) to specific
open-weight files and feeds.

GPU configuration (RTX 2080 Ti = GPU.0, RTX 4070 Ti = GPU.1):
  ONNX models  -> CUDAExecutionProvider device_id=1 (4070 Ti)
  OpenVINO IR  -> GPU.1 via OpenCL (fallback; prefer ONNX/CUDA for NVIDIA)
"""
from __future__ import annotations

import os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np


def _ensure_cudnn() -> None:
    """Add the nvidia-cudnn-cu12 DLL directory so ORT can find cudnn64_9.dll."""
    try:
        import nvidia.cudnn as _cudnn
        cudnn_bin = Path(list(_cudnn.__path__)[0]) / "bin"
        if cudnn_bin.is_dir():
            os.add_dll_directory(str(cudnn_bin))
    except Exception:
        pass  # cuDNN not installed via pip; rely on system PATH

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

class ModelCache:
    """LRU in-memory cache of compiled inference-session objects.

    Supports both OpenVINO (compiled_model) and ONNX Runtime (InferenceSession).
    The runtime is chosen per-file based on extension:
      .xml  →  OpenVINO
      .onnx →  ONNX Runtime
    """

    def __init__(self, max_models: int = 4, device: str = "CPU"):
        """
        device applies to OpenVINO IR models only.
        ONNX models always use CUDAExecutionProvider (RTX 4070 Ti, device_id=1)
        when available, regardless of this setting.
        OpenVINO GPU plugin via OpenCL is slower than CPU on NVIDIA hardware.
        """
        """
        device: OpenVINO device string — "CPU", "GPU", "AUTO".
                ONNX Runtime always uses CPU/DirectML depending on
                installed execution providers.
        """
        self._max = max_models
        self._device = device
        self._lru: OrderedDict[str, object] = OrderedDict()

    def load(self, model_path: Path):
        """Return a compiled model / inference session for *model_path* (LRU cached)."""
        key = str(model_path)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]

        if len(self._lru) >= self._max:
            self._lru.popitem(last=False)

        session = _compile(model_path, self._device)
        self._lru[key] = session
        return session

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._lru.clear()


def _compile(model_path: Path, device: str):
    """Compile a model file to an inference-ready object.

    For ONNX: prefers CUDAExecutionProvider on device_id=1 (RTX 4070 Ti).
    cuDNN must be on PATH; if unavailable falls back to CPU.
    For OpenVINO IR: uses the passed device string (default GPU.1 = RTX 4070 Ti
    via OpenCL — use CUDA/TRT path via ONNX for best NVIDIA perf).
    """
    model_path = Path(model_path)
    if model_path.suffix == ".onnx":
        import onnxruntime as ort
        # Ensure cuDNN DLL is discoverable (installed via nvidia-cudnn-cu12)
        _ensure_cudnn()
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = [
                ("CUDAExecutionProvider", {"device_id": 1}),  # RTX 4070 Ti
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(str(model_path), providers=providers)
    else:
        import openvino as ov
        core = ov.Core()
        bin_ = model_path.with_suffix(".bin")
        m = core.read_model(
            str(model_path),
            str(bin_) if bin_.exists() and bin_.stat().st_size > 0 else "",
        )
        return core.compile_model(m, device)


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

def _cosine_window(n: int) -> np.ndarray:
    t = np.linspace(0, np.pi, n, dtype=np.float32)
    return (1.0 - np.cos(t)) / 2.0


def run_tiled(
    session,
    img_chw_f32: np.ndarray,
    tile_h: int,
    tile_w: int,
    overlap: int,
    scale: int,
    feed: Callable[[np.ndarray, object], np.ndarray],
) -> np.ndarray:
    """Tile *img_chw_f32* [C,H,W] through *session* with cosine blend.

    *feed(tile_chw, session)* → float32 [C, tile_h*scale, tile_w*scale].
    Returns float32 [C, H*scale, W*scale], values clamped to [0,1].
    """
    C, H, W = img_chw_f32.shape
    stride_h = max(tile_h - overlap, 1)
    stride_w = max(tile_w - overlap, 1)
    out_H, out_W = H * scale, W * scale

    accum  = np.zeros((C, out_H, out_W), np.float32)
    weight = np.zeros((1, out_H, out_W), np.float32)

    win2d = (
        _cosine_window(tile_h * scale)[:, np.newaxis] *
        _cosine_window(tile_w * scale)[np.newaxis, :]
    )

    pad_h = max(0, tile_h - H)
    pad_w = max(0, tile_w - W)
    if pad_h or pad_w:
        img_chw_f32 = np.pad(img_chw_f32, ((0,0),(0,pad_h),(0,pad_w)), mode="reflect")
    _, pH, pW = img_chw_f32.shape

    def _starts(total: int, tile: int, stride: int) -> list[int]:
        pts = list(range(0, total - tile + 1, stride))
        if not pts or pts[-1] + tile < total:
            pts.append(max(0, total - tile))
        return pts

    for y in _starts(pH, tile_h, stride_h):
        for x in _starts(pW, tile_w, stride_w):
            tile = img_chw_f32[:, y:y+tile_h, x:x+tile_w]
            out_tile = feed(tile, session)          # [C, th*s, tw*s]
            oy, ox = y * scale, x * scale
            oy1 = min(oy + tile_h * scale, out_H)
            ox1 = min(ox + tile_w * scale, out_W)
            rh, rw = oy1 - oy, ox1 - ox
            accum [:, oy:oy1, ox:ox1] += out_tile[:, :rh, :rw] * win2d[:rh, :rw]
            weight[:, oy:oy1, ox:ox1] += win2d[:rh, :rw]

    return np.clip(accum / np.maximum(weight, 1e-6), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load any image to float32 RGB CHW [0,1]."""
    from PIL import Image as _Im
    img = _Im.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0


def save_image(img_chw_f32: np.ndarray, path: Path, quality: int = 97) -> None:
    """Save float32 RGB CHW to JPEG/PNG depending on extension."""
    from PIL import Image as _Im
    hwc = (img_chw_f32.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = _Im.fromarray(hwc)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        pil.save(str(path), "JPEG", quality=quality, subsampling=0)
    else:
        pil.save(str(path))

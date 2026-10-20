"""
lce.engine — model-agnostic tiled inference engine.

Loads OpenVINO IR (.xml/.bin) or ONNX (.onnx) models, caches compiled
instances in memory (LRU), and runs large images through any model in
overlapping tiles with cosine-blended seams.

Model-agnostic: the models/ package maps capabilities (upscale, sharpen, ...)
to specific open-weight files and feeds.

Device configuration:
  ONNX models  -> CUDAExecutionProvider when onnxruntime offers it, on the
                  device id in $LCE_CUDA_DEVICE_ID (default 0); if the CUDA
                  session cannot be created the model falls back to CPU and a
                  warning is logged. LCE_ONNX_CPU_ONLY=1 forces CPU. The providers each session really got are
                  recorded in ModelCache.providers (shown by the health tool).
  OpenVINO IR  -> the ``device`` string passed to ModelCache.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("lce.engine")


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
        device: OpenVINO device string ("CPU", "GPU", "AUTO"); applies to
                OpenVINO IR models only. ONNX models use the providers
                described in the module docstring.
        """
        self._max = max_models
        self._device = device
        self._lru: OrderedDict[str, object] = OrderedDict()
        #: model path -> execution providers the session actually got (ONNX only)
        self.providers: dict[str, list[str]] = {}

    def load(self, model_path: Path):
        """Return a compiled model / inference session for *model_path* (LRU cached)."""
        key = str(model_path)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]

        if len(self._lru) >= self._max:
            old, _ = self._lru.popitem(last=False)
            self.providers.pop(old, None)

        session = _compile(model_path, self._device)
        self._lru[key] = session
        get_providers = getattr(session, "get_providers", None)
        if get_providers is not None:
            self.providers[key] = list(get_providers())
        return session

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._lru.clear()
        self.providers.clear()


def _cuda_device_id() -> int:
    raw = os.environ.get("LCE_CUDA_DEVICE_ID", "0")
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"LCE_CUDA_DEVICE_ID must be an integer, got {raw!r}") from None


def _compile(model_path: Path, device: str):
    """Compile a model file to an inference-ready object.

    ONNX: CUDAExecutionProvider on $LCE_CUDA_DEVICE_ID (default 0) when
    onnxruntime offers it, with CPUExecutionProvider as the fallback. If the
    CUDA session cannot be created (missing cuDNN, wrong device id) the model is
    loaded on CPU and a warning is logged; inspect ``session.get_providers()``
    (or ModelCache.providers) to see what was really used.
    Set LCE_ONNX_CPU_ONLY=1 to skip CUDA entirely (a broken cuDNN install can
    abort the process at the first inference instead of raising).
    OpenVINO IR: the passed *device* string.
    """
    model_path = Path(model_path)
    if model_path.suffix == ".onnx":
        import onnxruntime as ort
        _ensure_cudnn()
        if (os.environ.get("LCE_ONNX_CPU_ONLY") != "1"
                and "CUDAExecutionProvider" in ort.get_available_providers()):
            try:
                return ort.InferenceSession(str(model_path), providers=[
                    ("CUDAExecutionProvider", {"device_id": _cuda_device_id()}),
                    "CPUExecutionProvider",
                ])
            except Exception as exc:
                log.warning("CUDA session for %s failed (%s); falling back to CPU",
                            model_path.name, exc)
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
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

def input_name(session) -> str:
    """Name of the first model input (ONNX Runtime or OpenVINO)."""
    if hasattr(session, "get_inputs"):
        return session.get_inputs()[0].name
    return session.input(0).get_any_name()


def static_hw(session) -> tuple[int, int] | None:
    """(H, W) of the first input when the model fixes them, else None."""
    if not hasattr(session, "get_inputs"):
        return None
    shape = session.get_inputs()[0].shape
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        return shape[2], shape[3]
    return None


def pad_reflect(img: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Pad [C,H,W] at the bottom/right; reflect, or edge-replicate when the
    pad is too large for reflection (np.pad 'reflect' needs pad < size)."""
    if not (pad_h or pad_w):
        return img
    _, H, W = img.shape
    mode = "reflect" if pad_h < H and pad_w < W else "edge"
    return np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)


def _window_1d(n: int, ramp_lo: int, ramp_hi: int) -> np.ndarray:
    """Blend weights for one tile axis: strictly positive everywhere, ramping
    up over *ramp_lo* samples at the start and down over *ramp_hi* at the end.
    A ramp of 0 means the tile edge is an image edge: full weight."""
    w = np.ones(n, np.float32)

    def _ramp(r: int) -> np.ndarray:
        t = (np.arange(r, dtype=np.float32) + 0.5) / r * np.pi
        return (1.0 - np.cos(t)) / 2.0

    if ramp_lo:
        w[:ramp_lo] = _ramp(ramp_lo)
    if ramp_hi:
        w[n - ramp_hi:] = np.minimum(w[n - ramp_hi:], _ramp(ramp_hi)[::-1])
    return w


def run_tiled(
    session,
    img_chw_f32: np.ndarray,
    tile_h: int,
    tile_w: int,
    overlap: int,
    scale: int,
    feed: Callable[[np.ndarray, object], np.ndarray],
) -> np.ndarray:
    """Tile *img_chw_f32* [C,H,W] through *session* with cosine-blended seams.

    *feed(tile_chw, session)* -> float32 [C, tile_h*scale, tile_w*scale].
    Returns float32 [C, H*scale, W*scale], values clamped to [0,1].
    Blend weights ramp only where a neighbouring tile overlaps; the outer
    image border keeps full weight so it is not darkened.
    """
    C, H, W = img_chw_f32.shape
    stride_h = max(tile_h - overlap, 1)
    stride_w = max(tile_w - overlap, 1)
    out_H, out_W = H * scale, W * scale

    accum  = np.zeros((C, out_H, out_W), np.float32)
    weight = np.zeros((1, out_H, out_W), np.float32)

    img_chw_f32 = pad_reflect(img_chw_f32, max(0, tile_h - H), max(0, tile_w - W))
    _, pH, pW = img_chw_f32.shape

    def _starts(total: int, tile: int, stride: int) -> list[int]:
        pts = list(range(0, total - tile + 1, stride))
        if not pts or pts[-1] + tile < total:
            pts.append(max(0, total - tile))
        return pts

    ys = _starts(pH, tile_h, stride_h)
    xs = _starts(pW, tile_w, stride_w)
    ramp_h = min(overlap * scale, tile_h * scale // 2)
    ramp_w = min(overlap * scale, tile_w * scale // 2)

    for iy, y in enumerate(ys):
        wy = _window_1d(tile_h * scale, ramp_h if iy > 0 else 0,
                        ramp_h if iy < len(ys) - 1 else 0)
        for ix, x in enumerate(xs):
            wx = _window_1d(tile_w * scale, ramp_w if ix > 0 else 0,
                            ramp_w if ix < len(xs) - 1 else 0)
            win2d = wy[:, np.newaxis] * wx[np.newaxis, :]
            tile = img_chw_f32[:, y:y+tile_h, x:x+tile_w]
            out_tile = feed(tile, session)          # [C, th*s, tw*s]
            oy, ox = y * scale, x * scale
            oy1 = min(oy + tile_h * scale, out_H)
            ox1 = min(ox + tile_w * scale, out_W)
            rh, rw = oy1 - oy, ox1 - ox
            accum [:, oy:oy1, ox:ox1] += out_tile[:, :rh, :rw] * win2d[:rh, :rw]
            weight[:, oy:oy1, ox:ox1] += win2d[:rh, :rw]

    return np.clip(accum / weight, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_image_with_meta(path) -> tuple[np.ndarray, dict]:
    """Load an image (path or binary file object) to float32 RGB CHW [0,1]
    plus its EXIF / ICC bytes.

    The EXIF orientation is applied to the pixels (and removed from the
    returned EXIF), so rotated camera JPEGs are processed upright.
    Limitation: the image is converted to 8-bit RGB, so 16-bit sources are
    reduced to 8 bits here.
    """
    from PIL import Image as _Im, ImageOps as _Ops
    with _Im.open(path) as opened:
        img = _Ops.exif_transpose(opened)
        meta = {"exif": img.info.get("exif"), "icc_profile": img.info.get("icc_profile")}
        rgb = img.convert("RGB")
    return np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0, meta


def load_image(path: Path) -> np.ndarray:
    """Load any image to float32 RGB CHW [0,1] (see load_image_with_meta)."""
    return load_image_with_meta(path)[0]


def to_uint8_hwc(img_chw_f32: np.ndarray) -> np.ndarray:
    """float [0,1] CHW -> uint8 HWC, rounding to nearest (not truncating)."""
    return np.rint(img_chw_f32.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)


def save_image(
    img_chw_f32: np.ndarray,
    path: Path,
    quality: int = 95,
    exif: bytes | None = None,
    icc_profile: bytes | None = None,
) -> None:
    """Save float32 RGB CHW to JPEG/PNG depending on extension.

    JPEG is written at *quality* (default 95) with 4:4:4 subsampling; *exif*
    and *icc_profile* (from load_image_with_meta) are embedded when given.
    """
    from PIL import Image as _Im
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = _Im.fromarray(to_uint8_hwc(img_chw_f32))
    extra = {}
    if exif:
        extra["exif"] = exif
    if icc_profile:
        extra["icc_profile"] = icc_profile
    if path.suffix.lower() in (".jpg", ".jpeg"):
        pil.save(str(path), "JPEG", quality=quality, subsampling=0, **extra)
    else:
        pil.save(str(path), **extra)

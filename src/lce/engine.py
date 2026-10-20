"""
lce.engine — model-agnostic tiled inference engine.

Loads OpenVINO IR (.xml/.bin) or ONNX (.onnx) models, caches compiled
instances in memory (LRU), and runs large images through any model in
overlapping tiles with cosine-blended seams.

Model-agnostic: the models/ package maps capabilities (upscale, sharpen, ...)
to specific open-weight files and feeds.

GPU configuration:
  ONNX models  -> CUDAExecutionProvider on the card with the most free VRAM when the session is created
                  (LCE_CUDA_DEVICE=auto, the default), trying the cards best first and failing over when a
                  session cannot be created on one. An integer LCE_CUDA_DEVICE pins a card explicitly.
                  This module sets CUDA_DEVICE_ORDER=PCI_BUS_ID (before CUDA initialises) so ONNX Runtime's
                  device ids are nvidia-smi's indices, which is where the free-memory figures come from.
                  A session that does not end up on CUDA is reported (WARNING, counter in
                  degraded_counts(), shown by the server's health tool); LCE_REQUIRE_CUDA=1
                  turns that into an error.
  OpenVINO IR  -> the ``device`` given to ModelCache (CPU default).
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

# Must happen before anything initialises CUDA in this process: with the default (fastest-first) ordering ORT's device ids
# are NOT nvidia-smi's indices on a mixed-GPU box, and the free-memory ranking below reads nvidia-smi.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

log = logging.getLogger("lce.engine")

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"
CUDA_DEVICE_ENV = "LCE_CUDA_DEVICE"
MIN_FREE_ENV = "LCE_MIN_FREE_MB"
DEFAULT_MIN_FREE_MB = 1024
REQUIRE_CUDA_ENV = "LCE_REQUIRE_CUDA"
WARN_INTERVAL_S = 60.0

_COUNTS: Counter = Counter()
_LAST_WARN: dict = {}


def degraded_counts() -> dict:
    """Degraded ONNX outcomes in this process, keyed ``"<reason>"`` (shown by the server health tool)."""
    return dict(_COUNTS)


def _report(reason: str, message: str, clock=time.monotonic) -> None:
    """Count every degraded outcome; WARNING at most once per WARN_INTERVAL_S per reason."""
    _COUNTS[reason] += 1
    now = clock()
    last = _LAST_WARN.get(reason)
    if last is None or now - last >= WARN_INTERVAL_S:
        _LAST_WARN[reason] = now
        log.warning("%s (occurrence %d of this kind in this process)", message, _COUNTS[reason])


def _cuda_device_id(environ=None) -> Optional[int]:
    """The explicit CUDA device index from LCE_CUDA_DEVICE, or None for ``auto`` (the default: the card with the most
    free VRAM when the session is created)."""
    environ = os.environ if environ is None else environ
    raw = environ.get(CUDA_DEVICE_ENV)
    if raw is None or raw.strip() == "" or raw.strip().lower() == "auto":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{CUDA_DEVICE_ENV} must be 'auto' or an integer CUDA device index, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{CUDA_DEVICE_ENV} must be >= 0, got {value}")
    return value


def _gpu_free_memory() -> Optional[List[tuple]]:
    """``[(index, name, free_mb), ...]`` from nvidia-smi, read NOW, or None when it cannot be read (reported by the caller).
    Tests replace this function; nothing else shells out."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except Exception as exc:  # noqa: BLE001 - missing binary, driver hiccup, timeout
        log.debug("nvidia-smi free-memory query failed: %r", exc)
        return None
    cards = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            cards.append((int(parts[0]), parts[1], int(float(parts[2]))))
        except ValueError:
            continue                      # "[N/A]" and the like: that card is simply not a candidate
    return cards or None


def _cuda_candidates(environ=None, clock=time.monotonic) -> List[int]:
    """CUDA device ids to try, best first. An explicit LCE_CUDA_DEVICE is the only candidate. ``auto`` ranks the cards by
    free VRAM, those with LCE_MIN_FREE_MB (default 1024) or more first; when no card has that much, every card is still
    tried, most free first (and the shortfall is reported). When nvidia-smi cannot be read, device 0 is tried (reported)."""
    explicit = _cuda_device_id(environ)
    if explicit is not None:
        return [explicit]
    cards = _gpu_free_memory()
    if not cards:
        _report("gpu-query-failed", "lce: could not read free VRAM from nvidia-smi; trying CUDA device 0", clock)
        return [0]
    min_free = int((environ if environ is not None else os.environ).get(MIN_FREE_ENV) or DEFAULT_MIN_FREE_MB)
    ranked = sorted(cards, key=lambda c: c[2], reverse=True)
    if ranked[0][2] < min_free:
        _report("no-card-has-room",
                f"lce: no card has {min_free} MiB free ({', '.join(f'cuda:{i} {n} {f} MiB' for i, n, f in ranked)}); "
                f"trying the one with the most", clock)
        return [c[0] for c in ranked]
    return [c[0] for c in ranked if c[2] >= min_free]


def _open_onnx(ort, model_path, *, device_id=None, require_cuda=None, retries=2, backoff_s=0.5,
               sleep=time.sleep, clock=time.monotonic):
    """Create an ONNX Runtime session on CUDA when possible and say so when it is not.

    ort.preload_dlls() loads the CUDA/cuDNN DLLs from the nvidia-* wheels in the order ORT needs; a bare
    os.add_dll_directory() made session creation abort the process ("Cannot load symbol cudnnCreate").
    session.get_providers() is checked afterwards, because ORT falls back to CPU without raising.
    The card is chosen at creation time (see _cuda_candidates). Session-creation errors are retried with exponential
    backoff per card; when a card still fails, the next best card is tried; when all fail the error propagates.
    """
    if require_cuda is None:
        require_cuda = os.environ.get(REQUIRE_CUDA_ENV, "") not in ("", "0")
    available = list(ort.get_available_providers())
    if CUDA in available:
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as e:  # the provider check below decides whether this mattered
                _report("preload-dlls-failed", f"lce: ort.preload_dlls() raised {type(e).__name__}: {str(e)[:200]}", clock)
        candidates = [device_id] if device_id is not None else _cuda_candidates(clock=clock)
    else:
        _report("cuda-unavailable",
                f"lce: this onnxruntime offers only {available}; install onnxruntime-gpu to use the GPU", clock)
        candidates = [None]

    session, device_id = None, None
    for n, dev in enumerate(candidates):
        providers = [(CUDA, {"device_id": dev}), CPU] if dev is not None else [CPU]
        for attempt in range(retries + 1):
            try:
                session = ort.InferenceSession(str(model_path), providers=providers)
                break
            except Exception as e:
                if attempt >= retries:
                    if n < len(candidates) - 1:
                        _report("cuda-device-failed",
                                f"lce: could not create {Path(model_path).name} on cuda:{dev} "
                                f"({type(e).__name__}: {str(e)[:200]}); trying the next card", clock)
                        break
                    raise
                _report("session-create-retry",
                        f"lce: InferenceSession creation failed ({type(e).__name__}: {str(e)[:200]}); "
                        f"retry {attempt + 1} of {retries}", clock)
                sleep(backoff_s * (2 ** attempt))
        if session is not None:
            device_id = dev
            break

    got = list(session.get_providers())
    if not got or got[0] != CUDA:
        msg = (f"lce: CUDA was requested (device_id={device_id}) but {Path(model_path).name} runs on {got}; "
               f"expect ~10x slower inference (cuDNN/CUDA DLLs missing or CPU-only onnxruntime)")
        _report("cuda-not-obtained", msg, clock)
        if require_cuda:
            raise RuntimeError(msg)
    else:
        log.info("lce: %s runs on %s (device_id=%d)", Path(model_path).name, got, device_id)
    return session

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
        device: OpenVINO device string - "CPU", "GPU", "AUTO"; applies to OpenVINO IR models only.
        ONNX models use CUDAExecutionProvider on LCE_CUDA_DEVICE (default 0) regardless of this
        setting, and report it when they end up on CPU. OpenVINO GPU via OpenCL is slower than
        CPU on NVIDIA hardware.
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


def _compile(model_path: Path, device: str):
    """Compile a model file to an inference-ready object.

    For ONNX: CUDAExecutionProvider on LCE_CUDA_DEVICE (see _open_onnx); a CPU result is reported.
    For OpenVINO IR: uses the passed device string.
    """
    model_path = Path(model_path)
    if model_path.suffix == ".onnx":
        import onnxruntime as ort
        return _open_onnx(ort, model_path)
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

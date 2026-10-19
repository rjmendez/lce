"""
lce.engine — model-agnostic tiled inference engine.

Loads OpenVINO IR (.xml/.bin) or ONNX (.onnx) models, caches compiled
instances in memory (LRU), and runs large images through any model in
overlapping tiles with cosine-blended seams.

Intentionally model-agnostic: no Topaz-specific logic here.
The models/ package maps capabilities (upscale, sharpen, ...) to specific
open-weight files and feeds.

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

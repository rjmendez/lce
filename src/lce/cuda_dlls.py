"""Put CUDA/cuDNN DLL folders installed system-wide (not through pip) on the
Windows DLL search path, before onnxruntime loads its CUDA provider.

Separate from, and complementary to, ``ort.preload_dlls()`` -- which
``engine._open_onnx`` already calls automatically whenever a CUDA provider is
available. That loads the CUDA/cuDNN DLLs bundled in the nvidia-* pip wheels
(onnxruntime-gpu's own declared dependencies). It does nothing for a
cuDNN/CUDA Toolkit installed system-wide instead (e.g. via NVIDIA's own
installer, as opposed to ``pip install nvidia-cudnn-cu12`` or similar) --
that case is what this module is for.

Call ``add_cuda_dll_dirs()`` once, explicitly, before the first
``import onnxruntime`` in your process -- module import time is the natural
place. Do NOT call it reactively after onnxruntime is already loaded and its
own CUDA provider has started resolving symbols: mixing
``os.add_dll_directory()`` with an in-flight provider load has aborted the
process ("Cannot load symbol cudnnCreate") in earlier testing. Keep the two
mechanisms independent -- call this first, let ``ort.preload_dlls()`` run
afterwards inside ``_open_onnx`` as usual -- rather than interleaving them.
"""
from __future__ import annotations

import os
from pathlib import Path

_added: set[str] = set()

DEFAULT_DIRS = tuple(
    Path(p) for p in os.environ.get("LCE_CUDA_DLL_DIRS", os.pathsep.join([
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.13\bin\13.0",
    ])).split(os.pathsep) if p
)


def add_cuda_dll_dirs(dirs=DEFAULT_DIRS, *, add_dll_directory=getattr(os, "add_dll_directory", None),
                       environ=os.environ) -> list[Path]:
    """Add each existing folder in *dirs* to the DLL search path (both
    ``os.add_dll_directory`` and ``PATH`` -- cuDNN loads its sub-libraries
    through PATH, not the add_dll_directory list alone). Idempotent across
    repeated calls. Returns the folders that do not exist, so a caller can
    report or raise on them; requiring CUDA to actually be obtained is the
    caller's decision, not this function's."""
    missing = []
    for d in map(Path, dirs):
        if not d.is_dir():
            missing.append(d)
            continue
        key = os.path.normcase(str(d))
        if key in _added:
            continue
        if add_dll_directory is not None:
            add_dll_directory(str(d))
        environ["PATH"] = str(d) + os.pathsep + environ.get("PATH", "")
        _added.add(key)
    return missing

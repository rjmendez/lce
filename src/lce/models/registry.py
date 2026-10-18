"""
lce.models.registry — model registry and weight downloader.

Reads src/lce/weights/models.toml for the list of available model bundles.
Each bundle has:
  id          — unique identifier used in code
  capability  — upscale | sharpen | triage | segment | face
  url         — download URL (HuggingFace hub or GitHub release)
  sha256      — hex digest of the downloaded file
  format      — onnx | openvino
  size_mb     — approximate compressed download size
  license     — SPDX license expression
  description — human-readable description

run `lce-weights download` to fetch everything.
run `lce-weights list` to see what's available.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

_REGISTRY_FILE = Path(__file__).parent.parent / "weights" / "models.toml"
_DEFAULT_WEIGHTS_DIR = Path.home() / ".lce" / "weights"


def _registry() -> dict:
    if not _REGISTRY_FILE.exists():
        return {"model": []}
    with open(_REGISTRY_FILE, "rb") as f:
        return tomllib.load(f)


def available() -> list[dict]:
    return _registry().get("model", [])


def weights_dir() -> Path:
    import os
    d = Path(os.environ.get("LCE_WEIGHTS_DIR", str(_DEFAULT_WEIGHTS_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(model_id: str) -> Path | None:
    """Return local path to the model file, or None if not downloaded."""
    for m in available():
        if m["id"] == model_id:
            fname = Path(m["url"]).name
            p = weights_dir() / fname
            return p if p.exists() else None
    return None


def ensure(model_id: str) -> Path:
    """Return path to model, downloading if necessary."""
    p = model_path(model_id)
    if p is not None:
        return p
    for m in available():
        if m["id"] == model_id:
            return _download(m)
    raise KeyError(f"Unknown model: {model_id!r}. Run `lce-weights list`.")


def _download(entry: dict) -> Path:
    url   = entry["url"]
    sha   = entry.get("sha256", "")
    fname = Path(url).name
    dest  = weights_dir() / fname
    print(f"Downloading {entry['id']} ({entry.get('size_mb', '?')} MB) …", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()
    if sha:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest != sha:
            dest.unlink()
            raise ValueError(f"SHA-256 mismatch for {fname}: expected {sha}, got {digest}")
    return dest


def _progress(block_count: int, block_size: int, total: int) -> None:
    if total > 0:
        pct = min(100, block_count * block_size * 100 // total)
        print(f"\r  {pct}%", end="", flush=True)

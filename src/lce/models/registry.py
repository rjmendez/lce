"""
lce.models.registry — model registry and weight downloader.

Reads src/lce/weights/models.toml for the list of available model bundles.
Each bundle has:
  id          — unique identifier used in code
  capability  — upscale | sharpen | segment | face
  url         — https download URL
  sha256      — hex digest of the file (required; an entry without one is refused)
  size_bytes  — exact file size, checked whenever a cached file is reused
  format      — onnx
  license     — license of the weights (or "unspecified")
  description — human-readable description
  optional    — true for packs not needed by the core tools

An entry may give ``local_path`` instead of ``url``: a personal fine-tune kept
off any public host (not downloadable, not part of ``lce-weights download``),
resolved straight from that path with no network involved. ``sha256``/
``size_bytes`` are still recorded for the registry listing, but only
``size_bytes`` is checked on reuse (same as a downloaded file) -- there is no
download step to verify a fresh sha256 against.

A download goes to ``<name>.part``, its SHA-256 and size are verified, and only
then is it renamed to its final name, so an interrupted download is never
mistaken for a model.

run `lce-weights download` to fetch everything.
run `lce-weights list` to see what's available.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import tomllib

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
    d = Path(os.environ.get("LCE_WEIGHTS_DIR", str(_DEFAULT_WEIGHTS_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _entry(model_id: str) -> dict | None:
    for m in available():
        if m["id"] == model_id:
            return m
    return None


def _file_ok(path: Path, entry: dict) -> bool:
    """A cached file counts only if its size matches the registry."""
    size = entry.get("size_bytes")
    return path.is_file() and (size is None or path.stat().st_size == size)


def is_local_only(model_id: str) -> bool:
    """True if this entry has no public URL to download (see ``local_path`` above)."""
    m = _entry(model_id)
    return m is not None and "local_path" in m


def model_path(model_id: str) -> Path | None:
    """Return local path to the model file, or None if not (fully) downloaded."""
    m = _entry(model_id)
    if m is None:
        return None
    p = Path(m["local_path"]).expanduser() if "local_path" in m else weights_dir() / Path(m["url"]).name
    return p if _file_ok(p, m) else None


def ensure(model_id: str) -> Path:
    """Return path to model, downloading if necessary."""
    m = _entry(model_id)
    if m is None:
        raise KeyError(f"Unknown model: {model_id!r}. Run `lce-weights list`.")
    p = model_path(model_id)
    if p is not None:
        return p
    if "local_path" in m:
        raise FileNotFoundError(
            f"{model_id!r} is a local-only model with no public download: expected it at "
            f"{m['local_path']!r} but it is missing or the wrong size."
        )
    return _download(m)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(entry: dict) -> Path:
    url   = entry["url"]
    sha   = entry.get("sha256", "")
    if not sha:
        raise ValueError(f"Registry entry {entry['id']!r} has no sha256; refusing to download")
    if urlparse(url).scheme != "https":
        raise ValueError(f"Registry entry {entry['id']!r} url is not https: {url}")
    fname = Path(url).name
    dest  = weights_dir() / fname
    part  = dest.with_name(dest.name + ".part")
    print(f"Downloading {entry['id']} ({entry.get('size_bytes', '?')} bytes) …", flush=True)
    try:
        urllib.request.urlretrieve(url, part, reporthook=_progress)
        print()
        digest = _sha256_of(part)
        if digest != sha:
            raise ValueError(f"SHA-256 mismatch for {fname}: expected {sha}, got {digest}")
        size = entry.get("size_bytes")
        if size is not None and part.stat().st_size != size:
            raise ValueError(
                f"Size mismatch for {fname}: expected {size}, got {part.stat().st_size}")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    return dest


def _progress(block_count: int, block_size: int, total: int) -> None:
    if total > 0:
        pct = min(100, block_count * block_size * 100 // total)
        print(f"\r  {pct}%", end="", flush=True)

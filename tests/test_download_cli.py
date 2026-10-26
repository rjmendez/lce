"""Tests for lce.weights.download's CLI: local-only registry entries must be
skipped by `download` (nothing to fetch for them), not crash the whole run."""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from lce.models import registry
from lce.weights import download


@pytest.fixture
def reg_with_local_only(tmp_path, monkeypatch):
    wdir = tmp_path / "weights"
    monkeypatch.setenv("LCE_WEIGHTS_DIR", str(wdir))
    toml = tmp_path / "models.toml"
    monkeypatch.setattr(registry, "_REGISTRY_FILE", toml)
    local_path = str(tmp_path / "missing.onnx").replace("\\", "\\\\")
    toml.write_text(
        "[[model]]\n"
        'id = "remote"\ncapability = "sharpen"\nurl = "https://example.invalid/remote.onnx"\n'
        f'sha256 = "{"a" * 64}"\nsize_bytes = 10\nformat = "onnx"\nlicense = "MIT"\n'
        'description = "r"\noptional = false\n\n'
        "[[model]]\n"
        f'id = "local"\ncapability = "upscale"\nlocal_path = "{local_path}"\n'
        f'sha256 = "{"b" * 64}"\nsize_bytes = 10\nformat = "onnx"\nlicense = "unspecified"\n'
        'description = "l"\noptional = true\n',
        encoding="utf-8")
    return tmp_path


def test_download_skips_local_only_without_calling_ensure(reg_with_local_only, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(download, "ensure", lambda mid: calls.append(mid) or f"/fake/{mid}")
    monkeypatch.setattr(sys, "argv", ["lce-weights", "download"])
    download.main()
    out = capsys.readouterr().out
    assert calls == ["remote"]  # local-only entry never reaches ensure()
    assert "skipping local (local-only)" in out
    assert "MISSING (no public download)" in out
    assert "All models ready." in out


def test_download_reports_found_when_local_file_present(reg_with_local_only, monkeypatch, capsys):
    local_entry = next(m for m in registry.available() if m["id"] == "local")
    import pathlib
    p = pathlib.Path(local_entry["local_path"])
    p.write_bytes(b"0123456789")  # matches size_bytes = 10 in the fixture
    monkeypatch.setattr(download, "ensure", lambda mid: f"/fake/{mid}")
    monkeypatch.setattr(sys, "argv", ["lce-weights", "download"])
    download.main()
    out = capsys.readouterr().out
    assert "skipping local (local-only): found" in out

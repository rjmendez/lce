"""Tests for lce.models.registry: integrity checks and the shipped registry."""
from __future__ import annotations

import hashlib
import re
import types
import urllib.request

import pytest

from lce.models import registry

PAYLOAD = b"pretend-onnx-weights" * 50
GOOD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def reg(tmp_path, monkeypatch):
    """A temp registry + weights dir; returns a function to (re)write entries."""
    wdir = tmp_path / "weights"
    monkeypatch.setenv("LCE_WEIGHTS_DIR", str(wdir))
    toml = tmp_path / "models.toml"
    monkeypatch.setattr(registry, "_REGISTRY_FILE", toml)

    def write(sha=GOOD_SHA, size=len(PAYLOAD), url="https://example.invalid/m/tiny.onnx"):
        toml.write_text(
            '[[model]]\nid = "tiny"\ncapability = "sharpen"\n'
            f'url = "{url}"\nsha256 = "{sha}"\nsize_bytes = {size}\n'
            'format = "onnx"\nlicense = "MIT"\ndescription = "t"\noptional = false\n',
            encoding="utf-8")
    write()
    return types.SimpleNamespace(write=write, wdir=wdir)


class _Net:
    """Recording replacement for urllib.request.urlretrieve (the network)."""

    def __init__(self, payload=PAYLOAD, fail=False):
        self.payload, self.fail, self.urls = payload, fail, []

    def __call__(self, url, dest, reporthook=None):
        self.urls.append(url)
        with open(dest, "wb") as f:
            f.write(self.payload[: len(self.payload) // 2] if self.fail else self.payload)
        if self.fail:
            raise ConnectionError("link dropped")


class TestDownload:
    def test_matching_digest_installs_file_and_leaves_no_part(self, reg, monkeypatch):
        net = _Net()
        monkeypatch.setattr(urllib.request, "urlretrieve", net)
        p = registry.ensure("tiny")
        assert p == reg.wdir / "tiny.onnx"
        assert p.read_bytes() == PAYLOAD
        assert not (reg.wdir / "tiny.onnx.part").exists()
        assert net.urls == ["https://example.invalid/m/tiny.onnx"]

    def test_digest_mismatch_raises_and_installs_nothing(self, reg, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlretrieve", _Net(payload=PAYLOAD + b"tampered"))
        with pytest.raises(ValueError, match="SHA-256 mismatch for tiny.onnx"):
            registry.ensure("tiny")
        assert not (reg.wdir / "tiny.onnx").exists()
        assert not (reg.wdir / "tiny.onnx.part").exists()

    def test_empty_sha256_is_refused_before_any_network_use(self, reg, monkeypatch):
        reg.write(sha="")
        net = _Net()
        monkeypatch.setattr(urllib.request, "urlretrieve", net)
        with pytest.raises(ValueError, match="has no sha256"):
            registry.ensure("tiny")
        assert net.urls == []

    def test_non_https_url_is_refused(self, reg, monkeypatch):
        reg.write(url="http://example.invalid/m/tiny.onnx")
        net = _Net()
        monkeypatch.setattr(urllib.request, "urlretrieve", net)
        with pytest.raises(ValueError, match="url is not https"):
            registry.ensure("tiny")
        assert net.urls == []

    def test_interrupted_download_is_not_mistaken_for_the_model(self, reg, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlretrieve", _Net(fail=True))
        with pytest.raises(ConnectionError, match="link dropped"):
            registry.ensure("tiny")
        assert registry.model_path("tiny") is None
        assert not (reg.wdir / "tiny.onnx").exists()
        assert not (reg.wdir / "tiny.onnx.part").exists()
        # and a retry on a healthy link succeeds
        monkeypatch.setattr(urllib.request, "urlretrieve", _Net())
        assert registry.ensure("tiny").read_bytes() == PAYLOAD


class TestCachedFiles:
    def test_complete_cached_file_is_reused_without_network(self, reg, monkeypatch):
        reg.wdir.mkdir(parents=True, exist_ok=True)
        (reg.wdir / "tiny.onnx").write_bytes(PAYLOAD)
        net = _Net()
        monkeypatch.setattr(urllib.request, "urlretrieve", net)
        assert registry.ensure("tiny") == reg.wdir / "tiny.onnx"
        assert net.urls == []

    def test_truncated_cached_file_is_ignored_and_redownloaded(self, reg, monkeypatch):
        reg.wdir.mkdir(parents=True, exist_ok=True)
        (reg.wdir / "tiny.onnx").write_bytes(PAYLOAD[:10])
        assert registry.model_path("tiny") is None
        net = _Net()
        monkeypatch.setattr(urllib.request, "urlretrieve", net)
        assert registry.ensure("tiny").read_bytes() == PAYLOAD
        assert len(net.urls) == 1

    def test_unknown_model_raises_keyerror(self, reg):
        with pytest.raises(KeyError, match="Unknown model: 'nope'"):
            registry.ensure("nope")


class TestLocalPath:
    """A local_path entry (personal fine-tune, no public URL) resolves straight
    from disk, with no network involved -- see models.toml's local_path docs."""

    def _write_local(self, toml_path, local_file, sha=GOOD_SHA, size=len(PAYLOAD)):
        escaped = str(local_file).replace("\\", "\\\\")
        toml_path.write_text(
            '[[model]]\nid = "local-tiny"\ncapability = "upscale"\n'
            f'local_path = "{escaped}"\nsha256 = "{sha}"\nsize_bytes = {size}\n'
            'format = "onnx"\nlicense = "unspecified"\ndescription = "t"\noptional = true\n',
            encoding="utf-8")

    def test_present_file_of_the_right_size_resolves_without_network(self, tmp_path, monkeypatch):
        toml = tmp_path / "models.toml"
        monkeypatch.setattr(registry, "_REGISTRY_FILE", toml)
        local_file = tmp_path / "local.onnx"
        local_file.write_bytes(PAYLOAD)
        self._write_local(toml, local_file)
        net = _Net()
        monkeypatch.setattr(urllib.request, "urlretrieve", net)
        assert registry.is_local_only("local-tiny") is True
        assert registry.ensure("local-tiny") == local_file
        assert net.urls == []  # never touched the network

    def test_missing_file_raises_file_not_found_not_a_download_attempt(self, tmp_path, monkeypatch):
        toml = tmp_path / "models.toml"
        monkeypatch.setattr(registry, "_REGISTRY_FILE", toml)
        local_file = tmp_path / "missing.onnx"  # never created
        self._write_local(toml, local_file)
        net = _Net()
        monkeypatch.setattr(urllib.request, "urlretrieve", net)
        with pytest.raises(FileNotFoundError, match="local-tiny.*no public download"):
            registry.ensure("local-tiny")
        assert net.urls == []

    def test_wrong_size_file_is_treated_as_missing(self, tmp_path, monkeypatch):
        toml = tmp_path / "models.toml"
        monkeypatch.setattr(registry, "_REGISTRY_FILE", toml)
        local_file = tmp_path / "truncated.onnx"
        local_file.write_bytes(PAYLOAD[:10])
        self._write_local(toml, local_file)
        assert registry.model_path("local-tiny") is None
        with pytest.raises(FileNotFoundError):
            registry.ensure("local-tiny")

    def test_url_based_entry_is_not_local_only(self, reg):
        # positive twin: a normal downloadable entry reports is_local_only() == False
        assert registry.is_local_only("tiny") is False


class TestShippedRegistry:
    """Properties of the real models.toml (not a stub)."""

    def test_every_entry_pins_digest_size_and_https_url_or_local_path(self):
        entries = registry.available()
        assert [e["id"] for e in entries] == [
            "hat-sentinel-x4", "hat-sentinel-x2", "hat-stock-lora-x2", "hat-stock-x4",
            "nafnet-fp32", "mobilesam-encoder", "mobilesam-decoder", "gfpgan-v14"]
        for e in entries:
            assert re.fullmatch(r"[0-9a-f]{64}", e["sha256"]), e["id"]
            assert isinstance(e["size_bytes"], int) and e["size_bytes"] > 0, e["id"]
            if "local_path" in e:
                assert "url" not in e, e["id"]
            else:
                assert e["url"].startswith("https://huggingface.co/"), e["id"]

    def test_exactly_the_two_stock_entries_are_local_only(self):
        entries = registry.available()
        local_only = [e["id"] for e in entries if "local_path" in e]
        assert local_only == ["hat-stock-lora-x2", "hat-stock-x4"]

    def test_every_model_id_used_by_code_is_registered(self):
        from lce.models import face, segment, sharpen, upscale
        used = {sharpen._MODEL_ID, segment._ENCODER_ID, segment._DECODER_ID,
                face._MODEL_ID, *upscale._MODEL_IDS.values(), *upscale._VARIANT_MODEL_IDS.values()}
        assert used == {e["id"] for e in registry.available()}

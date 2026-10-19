"""lce.engine ONNX session creation: CUDA DLL preload, configured device, and a reported CPU fallback (EXT-3).

Everything runs against a fake ``onnxruntime`` module injected into ``sys.modules``; no GPU, no model file.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from lce import engine

CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"
LOGGER = "lce.engine"


class FakeSession:
    def __init__(self, got):
        self._got = list(got)

    def get_providers(self):
        return list(self._got)


class FakeOrt:
    def __init__(self, available=(CUDA, CPU), got=(CUDA, CPU), create_errors=(), preload_error=None, with_preload=True):
        self.available = list(available)
        self.got = list(got)
        self.create_errors = list(create_errors)
        self.preload_error = preload_error
        self.events = []
        self.session_calls = []
        if with_preload:
            self.preload_dlls = self._preload

    def _preload(self):
        self.events.append("preload")
        if self.preload_error is not None:
            raise self.preload_error

    def get_available_providers(self):
        return list(self.available)

    def InferenceSession(self, path, providers=None):
        self.events.append("session")
        self.session_calls.append((path, providers))
        if self.create_errors:
            raise self.create_errors.pop(0)
        return FakeSession(self.got)


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(engine, "_COUNTS", Counter())
    monkeypatch.setattr(engine, "_LAST_WARN", {})
    monkeypatch.delenv(engine.CUDA_DEVICE_ENV, raising=False)
    monkeypatch.delenv(engine.REQUIRE_CUDA_ENV, raising=False)


@pytest.fixture
def install_ort(monkeypatch):
    def _install(ort):
        monkeypatch.setitem(sys.modules, "onnxruntime", ort)
        return ort
    return _install


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.name == LOGGER]


class TestCompileOnnx:
    def test_preloads_dlls_then_requests_cuda_on_device_zero_and_never_uses_add_dll_directory(self, install_ort, monkeypatch, caplog):
        dll_dirs = []
        monkeypatch.setattr(os, "add_dll_directory", dll_dirs.append, raising=False)
        ort = install_ort(FakeOrt())
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            sess = engine._compile(Path("C:/models/daclip.onnx"), "CPU")
        assert sess.get_providers() == [CUDA, CPU]
        assert ort.events == ["preload", "session"]
        assert ort.session_calls == [(str(Path("C:/models/daclip.onnx")), [(CUDA, {"device_id": 0}), CPU])]
        assert dll_dirs == []
        assert _warnings(caplog) == []
        assert engine.degraded_counts() == {}

    def test_device_comes_from_the_environment(self, install_ort, monkeypatch):
        monkeypatch.setenv(engine.CUDA_DEVICE_ENV, "1")
        ort = install_ort(FakeOrt())
        engine._compile(Path("m.onnx"), "CPU")
        assert ort.session_calls[0][1] == [(CUDA, {"device_id": 1}), CPU]

    def test_cuda_not_obtained_is_reported_with_the_real_providers(self, install_ort, monkeypatch, caplog):
        monkeypatch.setenv(engine.CUDA_DEVICE_ENV, "1")
        install_ort(FakeOrt(got=[CPU]))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            sess = engine._compile(Path("m.onnx"), "CPU")
        assert sess.get_providers() == [CPU]
        assert _warnings(caplog) == [(
            "lce: CUDA was requested (device_id=1) but m.onnx runs on ['CPUExecutionProvider']; expect ~10x slower "
            "inference (cuDNN/CUDA DLLs missing or CPU-only onnxruntime) (occurrence 1 of this kind in this process)")]
        assert engine.degraded_counts() == {"cuda-not-obtained": 1}

    def test_require_cuda_env_turns_the_fallback_into_an_error(self, install_ort, monkeypatch):
        monkeypatch.setenv(engine.REQUIRE_CUDA_ENV, "1")
        install_ort(FakeOrt(got=[CPU]))
        with pytest.raises(RuntimeError, match=r"CUDA was requested \(device_id=0\) but m.onnx runs on \['CPUExecutionProvider'\]"):
            engine._compile(Path("m.onnx"), "CPU")

    def test_require_cuda_env_allows_a_cuda_session(self, install_ort, monkeypatch):
        monkeypatch.setenv(engine.REQUIRE_CUDA_ENV, "1")
        install_ort(FakeOrt())
        assert engine._compile(Path("m.onnx"), "CPU").get_providers() == [CUDA, CPU]

    def test_cpu_only_onnxruntime_uses_cpu_provider_without_preload_and_reports_twice(self, install_ort, caplog):
        ort = install_ort(FakeOrt(available=[CPU], got=[CPU]))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            engine._compile(Path("m.onnx"), "CPU")
        assert ort.events == ["session"]
        assert ort.session_calls[0][1] == [CPU]
        assert engine.degraded_counts() == {"cuda-unavailable": 1, "cuda-not-obtained": 1}
        assert _warnings(caplog)[0] == (
            "lce: this onnxruntime offers only ['CPUExecutionProvider']; install onnxruntime-gpu to use the GPU "
            "(occurrence 1 of this kind in this process)")

    def test_preload_failure_is_reported_and_creation_is_still_attempted(self, install_ort, caplog):
        ort = install_ort(FakeOrt(preload_error=OSError("no cudnn")))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            sess = engine._compile(Path("m.onnx"), "CPU")
        assert sess.get_providers() == [CUDA, CPU]
        assert ort.events == ["preload", "session"]
        assert _warnings(caplog) == ["lce: ort.preload_dlls() raised OSError: no cudnn (occurrence 1 of this kind in this process)"]

    def test_transient_creation_errors_are_retried_with_backoff_then_propagate(self):
        slept = []
        ort = FakeOrt(create_errors=[RuntimeError("busy"), RuntimeError("busy")])
        sess = engine._open_onnx(ort, Path("m.onnx"), sleep=slept.append)
        assert sess.get_providers() == [CUDA, CPU]
        assert slept == [0.5, 1.0]
        assert len(ort.session_calls) == 3
        assert engine.degraded_counts() == {"session-create-retry": 2}

        slept2 = []
        ort2 = FakeOrt(create_errors=[RuntimeError("oom")] * 3)
        with pytest.raises(RuntimeError, match="oom"):
            engine._open_onnx(ort2, Path("m.onnx"), sleep=slept2.append)
        assert len(ort2.session_calls) == 3
        assert slept2 == [0.5, 1.0]

    def test_warnings_are_rate_limited_but_every_occurrence_is_counted(self, caplog):
        now = [100.0]
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            for t in (100.0, 159.9, 160.0):
                now[0] = t
                engine._open_onnx(FakeOrt(got=[CPU]), Path("m.onnx"), clock=lambda: now[0])
        assert [m.rsplit("(", 1)[1] for m in _warnings(caplog)] == [
            "occurrence 1 of this kind in this process)", "occurrence 3 of this kind in this process)"]
        assert engine.degraded_counts() == {"cuda-not-obtained": 3}

    def test_openvino_models_do_not_touch_onnxruntime(self, monkeypatch):
        compiled = []
        core = SimpleNamespace(read_model=lambda xml, binp: ("model", xml, binp),
                               compile_model=lambda m, dev: compiled.append((m, dev)) or "compiled")
        monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core))
        monkeypatch.setitem(sys.modules, "onnxruntime", None)   # importing it would raise ImportError
        assert engine._compile(Path("m.xml"), "GPU") == "compiled"
        assert compiled == [(("model", "m.xml", ""), "GPU")]


class TestCudaDeviceId:
    def test_default_zero(self):
        assert engine._cuda_device_id({}) == 0

    def test_value(self):
        assert engine._cuda_device_id({"LCE_CUDA_DEVICE": "1"}) == 1

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="LCE_CUDA_DEVICE must be an integer CUDA device index, got 'x'"):
            engine._cuda_device_id({"LCE_CUDA_DEVICE": "x"})

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="LCE_CUDA_DEVICE must be >= 0, got -2"):
            engine._cuda_device_id({"LCE_CUDA_DEVICE": "-2"})

    def test_the_boundary_is_zero_inclusive_and_minus_one_exclusive(self):
        assert engine._cuda_device_id({"LCE_CUDA_DEVICE": "0"}) == 0          # card 0 is a legal choice, not a default only
        with pytest.raises(ValueError, match="LCE_CUDA_DEVICE must be >= 0, got -1"):
            engine._cuda_device_id({"LCE_CUDA_DEVICE": "-1"})


class TestHealthSurface:
    def test_health_tool_reports_onnx_degradation_counts(self, monkeypatch):
        from lce.server import LceHandler

        monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: SimpleNamespace(available_devices=["CPU"])))
        engine._open_onnx(FakeOrt(got=[CPU]), Path("m.onnx"))
        handler = LceHandler(SimpleNamespace(_lru={"/x/daclip.onnx": object()}), "CPU")
        result = handler._dispatch("health", {})
        assert json.loads(result["content"][0]["text"]) == {
            "status": "ok", "device": "CPU", "openvino_devices": ["CPU"], "loaded_models": ["daclip"],
            "onnx_degraded": {"cuda-not-obtained": 1}}

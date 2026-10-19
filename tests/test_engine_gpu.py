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
    monkeypatch.delenv(engine.MIN_FREE_ENV, raising=False)
    monkeypatch.delenv(engine.REQUIRE_CUDA_ENV, raising=False)
    # never shell out to the real nvidia-smi: two cards, card 0 with the room
    monkeypatch.setattr(engine, "_gpu_free_memory", lambda: [(0, "GPU0", 8000), (1, "GPU1", 3000)])


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
        with pytest.raises(RuntimeError, match="oom"):           # one explicit card: nothing to fail over to
            engine._open_onnx(ort2, Path("m.onnx"), device_id=0, sleep=slept2.append)
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
    def test_default_is_auto(self):
        assert engine._cuda_device_id({}) is None
        assert engine._cuda_device_id({"LCE_CUDA_DEVICE": ""}) is None
        assert engine._cuda_device_id({"LCE_CUDA_DEVICE": " AUTO "}) is None

    def test_value(self):
        assert engine._cuda_device_id({"LCE_CUDA_DEVICE": "1"}) == 1

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="LCE_CUDA_DEVICE must be 'auto' or an integer CUDA device index, got 'x'"):
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


# ---------------------------------------------------------------- the card is chosen by free VRAM at creation time

REAL_GPU_QUERY = engine._gpu_free_memory          # the autouse fixture replaces engine._gpu_free_memory; keep the real one


class TestAutoCardChoice:
    def cards(self, monkeypatch, cards):
        calls = []
        monkeypatch.setattr(engine, "_gpu_free_memory", lambda: calls.append(1) or cards)
        return calls

    def test_cards_are_ranked_by_free_vram_most_free_first(self, monkeypatch):
        self.cards(monkeypatch, [(0, "A", 3000), (1, "B", 8000)])
        assert engine._cuda_candidates({}) == [1, 0]

    def test_a_card_below_the_minimum_is_left_out_and_exactly_the_minimum_is_in(self, monkeypatch):
        self.cards(monkeypatch, [(0, "A", 500), (1, "B", 3000), (2, "C", 1024), (3, "D", 1023)])
        assert engine._cuda_candidates({}) == [1, 2]
        assert engine.degraded_counts() == {}

    def test_the_minimum_is_configurable(self, monkeypatch):
        self.cards(monkeypatch, [(0, "A", 3000), (1, "B", 8000)])
        assert engine._cuda_candidates({"LCE_MIN_FREE_MB": "5000"}) == [1]

    def test_no_card_with_room_still_tries_every_card_most_free_first_and_says_so(self, monkeypatch, caplog):
        self.cards(monkeypatch, [(0, "A", 300), (1, "B", 900)])
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert engine._cuda_candidates({}) == [1, 0]
        assert _warnings(caplog) == [
            "lce: no card has 1024 MiB free (cuda:1 B 900 MiB, cuda:0 A 300 MiB); trying the one with the most "
            "(occurrence 1 of this kind in this process)"]
        assert engine.degraded_counts() == {"no-card-has-room": 1}

    def test_unreadable_nvidia_smi_falls_back_to_device_zero_and_is_counted(self, monkeypatch):
        self.cards(monkeypatch, None)
        assert engine._cuda_candidates({}) == [0]
        assert engine.degraded_counts() == {"gpu-query-failed": 1}

    def test_an_explicit_card_is_the_only_candidate_and_nvidia_smi_is_not_asked(self, monkeypatch):
        calls = self.cards(monkeypatch, [(0, "A", 8000), (1, "B", 100)])
        assert engine._cuda_candidates({"LCE_CUDA_DEVICE": "1"}) == [1]
        assert calls == []

    def test_open_onnx_uses_the_card_with_the_most_room(self, install_ort, monkeypatch):
        self.cards(monkeypatch, [(0, "A", 3000), (1, "B", 8000)])
        ort = install_ort(FakeOrt())
        engine._open_onnx(ort, Path("m.onnx"))
        assert ort.session_calls[0][1] == [(CUDA, {"device_id": 1}), CPU]

    def test_a_card_that_cannot_create_the_session_fails_over_to_the_next(self, install_ort, monkeypatch, caplog):
        self.cards(monkeypatch, [(0, "A", 3000), (1, "B", 8000)])
        ort = FakeOrt(create_errors=[RuntimeError("cudaMalloc failed: out of memory")] * 3)       # all of card 1's attempts
        slept = []
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            sess = engine._open_onnx(ort, Path("m.onnx"), sleep=slept.append)
        assert sess.get_providers() == [CUDA, CPU]
        assert [c[1][0][1]["device_id"] for c in ort.session_calls] == [1, 1, 1, 0]
        assert slept == [0.5, 1.0]
        assert engine.degraded_counts() == {"session-create-retry": 2, "cuda-device-failed": 1}
        assert (
            "lce: could not create m.onnx on cuda:1 (RuntimeError: cudaMalloc failed: out of memory); trying the next card "
            "(occurrence 1 of this kind in this process)") in _warnings(caplog)

    def test_when_every_card_fails_the_last_error_propagates(self, install_ort, monkeypatch):
        self.cards(monkeypatch, [(0, "A", 3000), (1, "B", 8000)])
        ort = FakeOrt(create_errors=[RuntimeError("oom-1")] * 3 + [RuntimeError("oom-0")] * 3)
        with pytest.raises(RuntimeError, match="oom-0"):
            engine._open_onnx(ort, Path("m.onnx"), sleep=lambda s: None)
        assert [c[1][0][1]["device_id"] for c in ort.session_calls] == [1, 1, 1, 0, 0, 0]

    def test_a_cpu_only_onnxruntime_never_asks_nvidia_smi(self, install_ort, monkeypatch):
        calls = self.cards(monkeypatch, [(0, "A", 8000)])
        ort = install_ort(FakeOrt(available=[CPU], got=[CPU]))
        engine._open_onnx(ort, Path("m.onnx"))
        assert calls == [] and ort.session_calls[0][1] == [CPU]


class TestNvidiaSmiParsing:
    def run_with(self, monkeypatch, stdout=None, exc=None):
        def fake(cmd, **kw):
            assert cmd == ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"]
            if exc:
                raise exc
            return SimpleNamespace(stdout=stdout)
        monkeypatch.setattr(engine.subprocess, "run", fake)
        return REAL_GPU_QUERY()

    def test_parses_index_name_and_free_memory(self, monkeypatch):
        out = "0, NVIDIA GeForce RTX 2080 Ti, 7677\n1, NVIDIA GeForce RTX 4070 Ti, 2733\n"
        assert self.run_with(monkeypatch, out) == [(0, "NVIDIA GeForce RTX 2080 Ti", 7677), (1, "NVIDIA GeForce RTX 4070 Ti", 2733)]

    def test_crlf_output_and_a_not_available_row(self, monkeypatch):
        out = "0, NVIDIA GeForce RTX 2080 Ti, 7677\r\n1, Broken Card, [N/A]\r\n2, Short\r\n"
        assert self.run_with(monkeypatch, out) == [(0, "NVIDIA GeForce RTX 2080 Ti", 7677)]

    @pytest.mark.parametrize("exc", [FileNotFoundError("nvidia-smi"), RuntimeError("driver"),
                                     __import__("subprocess").TimeoutExpired("nvidia-smi", 10)])
    def test_a_failure_is_none_not_an_exception(self, monkeypatch, exc):
        assert self.run_with(monkeypatch, exc=exc) is None

    def test_no_usable_rows_is_none(self, monkeypatch):
        assert self.run_with(monkeypatch, "\n") is None


class TestDeviceOrder:
    def order(self, env_extra):
        env = {k: v for k, v in os.environ.items() if k != "CUDA_DEVICE_ORDER"}
        env.update(env_extra)
        env["PYTHONPATH"] = str(Path(engine.__file__).resolve().parent.parent)
        import subprocess
        r = subprocess.run([sys.executable, "-c", "import os, lce.engine; print(os.environ.get('CUDA_DEVICE_ORDER'))"],
                           capture_output=True, text=True, env=env, timeout=60)
        assert r.returncode == 0, r.stderr
        return r.stdout.strip()

    def test_importing_the_engine_makes_ort_ids_equal_nvidia_smi_indices(self):
        assert self.order({}) == "PCI_BUS_ID"

    def test_a_value_the_user_already_set_is_respected(self):
        assert self.order({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}) == "FASTEST_FIRST"

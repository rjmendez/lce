import pytest


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Tests that load real ONNX models run them on CPU: results must not depend on the GPU, and a broken CUDA/cuDNN install can abort
    the whole process at the first inference. onnxruntime is made to report only the CPU provider, so lce.engine._open_onnx takes its
    'CUDA unavailable' path. (The GPU provider logic itself is tested in test_engine_gpu.py against fakes.)"""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

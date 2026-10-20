import pytest


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Tests run ONNX on CPU: a broken CUDA/cuDNN install can abort the whole
    process at the first inference, and results must not depend on the GPU."""
    monkeypatch.setenv("LCE_ONNX_CPU_ONLY", "1")

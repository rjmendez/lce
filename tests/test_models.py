"""Hermetic tests for lce.models.* — no real model files required."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _img(C: int = 3, H: int = 64, W: int = 64, val: float = 0.5) -> np.ndarray:
    return np.full((C, H, W), val, dtype=np.float32)


def _onnx_session(*outputs: np.ndarray):
    """Fake ONNX InferenceSession whose .run() returns *outputs as a list."""
    sess = MagicMock(spec=["run"])
    sess.run.return_value = list(outputs)
    return sess


def _fake_cache(*sessions):
    cache = MagicMock()
    if len(sessions) == 1:
        cache.load.return_value = sessions[0]
    else:
        cache.load.side_effect = list(sessions)
    return cache


# ---------------------------------------------------------------------------
# upscale
# ---------------------------------------------------------------------------

class TestUpscale:
    """upscale.run() output shape must be exactly scale× the input."""

    def _run_mocked(self, scale: int, H: int = 32, W: int = 32):
        from lce.models import upscale

        input_img = _img(3, H, W)
        # The tile is 256×256; mock returns full tile-sized output at scale
        mock_out = np.full((1, 3, 256 * scale, 256 * scale), 0.5, dtype=np.float32)
        sess = _onnx_session(mock_out)
        cache = _fake_cache(sess)
        with patch("lce.models.registry.ensure", return_value=Path("/fake/hat.onnx")):
            result = upscale.run(input_img, cache, scale=scale)
        return result, H, W

    def test_4x_output_shape(self):
        result, H, W = self._run_mocked(4)
        assert result.shape == (3, H * 4, W * 4)

    def test_2x_output_shape(self):
        result, H, W = self._run_mocked(2)
        assert result.shape == (3, H * 2, W * 2)

    def test_invalid_scale_raises_value_error(self):
        from lce.models import upscale
        with pytest.raises(ValueError, match="scale must be 2 or 4"):
            upscale.run(_img(), MagicMock(), scale=3)

    def test_valid_scale_does_not_raise(self):
        # positive twin for the invalid-scale test
        result, _, _ = self._run_mocked(4)
        assert result is not None

    def test_4x_selects_hat_s_x4_model(self):
        from lce.models import upscale
        calls = []

        def _ensure(model_id: str):
            calls.append(model_id)
            return Path("/fake/hat.onnx")

        mock_out = np.zeros((1, 3, 256 * 4, 256 * 4), dtype=np.float32)
        cache = _fake_cache(_onnx_session(mock_out))
        with patch("lce.models.registry.ensure", side_effect=_ensure):
            upscale.run(_img(), cache, scale=4)
        assert calls == ["hat-s-x4"]

    def test_2x_selects_hat_s_x2_model(self):
        from lce.models import upscale
        calls = []

        def _ensure(model_id: str):
            calls.append(model_id)
            return Path("/fake/hat.onnx")

        mock_out = np.zeros((1, 3, 256 * 2, 256 * 2), dtype=np.float32)
        cache = _fake_cache(_onnx_session(mock_out))
        with patch("lce.models.registry.ensure", side_effect=_ensure):
            upscale.run(_img(), cache, scale=2)
        assert calls == ["hat-s-x2"]


# ---------------------------------------------------------------------------
# sharpen
# ---------------------------------------------------------------------------

class TestSharpen:
    """sharpen.run() alpha-blend: strength=0 → input; strength=1 → deblurred."""

    def _run_mocked(
        self,
        strength: float,
        input_val: float = 0.2,
        deblur_val: float = 0.8,
        H: int = 64,
        W: int = 64,
    ):
        from lce.models import sharpen

        input_img = _img(3, H, W, val=input_val)
        mock_out = np.full((1, 3, H, W), deblur_val, dtype=np.float32)
        sess = _onnx_session(mock_out)
        cache = _fake_cache(sess)
        with patch("lce.models.registry.ensure", return_value=Path("/fake/nafnet.onnx")):
            result = sharpen.run(input_img, cache, strength=strength)
        return result, input_img, mock_out[0]

    def test_strength_zero_returns_input(self):
        result, original, _ = self._run_mocked(0.0)
        np.testing.assert_allclose(result, original, atol=1e-6)

    def test_strength_one_returns_deblurred(self):
        result, _, deblurred = self._run_mocked(1.0)
        np.testing.assert_allclose(result, deblurred, atol=1e-6)

    def test_strength_half_blends_equally(self):
        result, original, deblurred = self._run_mocked(0.5, input_val=0.0, deblur_val=1.0)
        expected = np.full_like(result, 0.5)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_quality_true_selects_fp32_model(self):
        from lce.models import sharpen
        calls = []

        def _ensure(mid):
            calls.append(mid)
            return Path("/fake/nafnet.onnx")

        mock_out = np.full((1, 3, 64, 64), 0.5, dtype=np.float32)
        cache = _fake_cache(_onnx_session(mock_out))
        with patch("lce.models.registry.ensure", side_effect=_ensure):
            sharpen.run(_img(), cache, quality=True)
        assert calls == ["nafnet-fp32"]

    def test_quality_false_selects_int8_model(self):
        from lce.models import sharpen
        calls = []

        def _ensure(mid):
            calls.append(mid)
            return Path("/fake/nafnet.onnx")

        mock_out = np.full((1, 3, 64, 64), 0.5, dtype=np.float32)
        cache = _fake_cache(_onnx_session(mock_out))
        with patch("lce.models.registry.ensure", side_effect=_ensure):
            sharpen.run(_img(), cache, quality=False)
        assert calls == ["nafnet-int8"]


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------

class TestSegment:
    """segment.run() returns [1,H,W] float32 with only 0s and 1s."""

    def _run_mocked(self, mask_val: float = 1.0, H: int = 64, W: int = 64):
        from lce.models import segment

        input_img = _img(3, H, W)
        enc_out = np.zeros((1, 256, 64, 64), dtype=np.float32)
        mask_out = np.full((1, 1, H, W), mask_val, dtype=np.float32)
        enc_sess = _onnx_session(enc_out)
        dec_sess = _onnx_session(mask_out)
        cache = _fake_cache(enc_sess, dec_sess)
        with patch(
            "lce.models.registry.ensure",
            side_effect=[Path("/fake/enc.onnx"), Path("/fake/dec.onnx")],
        ):
            result = segment.run(input_img, cache)
        return result, H, W

    def test_output_shape(self):
        result, H, W = self._run_mocked()
        assert result.shape == (1, H, W)

    def test_output_dtype_float32(self):
        result, _, _ = self._run_mocked()
        assert result.dtype == np.float32

    def test_foreground_mask_all_ones(self):
        result, _, _ = self._run_mocked(mask_val=1.0)
        assert set(np.unique(result)) == {1.0}

    def test_background_mask_all_zeros(self):
        # Values below threshold 0.0 → background
        result, _, _ = self._run_mocked(mask_val=-1.0)
        assert set(np.unique(result)) == {0.0}

    def test_output_only_binary_values(self):
        # Mixed: above-threshold pixels → 1, below → 0
        from lce.models import segment
        H, W = 64, 64
        input_img = _img(3, H, W)
        enc_out = np.zeros((1, 256, 64, 64), dtype=np.float32)
        mask_out = np.zeros((1, 1, H, W), dtype=np.float32)
        mask_out[0, 0, :H // 2, :] = 1.0   # top half foreground
        mask_out[0, 0, H // 2:, :] = -1.0  # bottom half background
        enc_sess = _onnx_session(enc_out)
        dec_sess = _onnx_session(mask_out)
        cache = _fake_cache(enc_sess, dec_sess)
        with patch(
            "lce.models.registry.ensure",
            side_effect=[Path("/fake/enc.onnx"), Path("/fake/dec.onnx")],
        ):
            result = segment.run(input_img, cache)
        assert set(np.unique(result)).issubset({0.0, 1.0})
        assert result[0, 0, 0] == 1.0       # top half is foreground
        assert result[0, H - 1, 0] == 0.0   # bottom half is background


# ---------------------------------------------------------------------------
# triage
# ---------------------------------------------------------------------------

class TestTriage:
    """triage.run() routes based on BRISQUE score and resolution."""

    def _run_mocked(self, score: float, H: int = 1000, W: int = 1000) -> str:
        from lce.models import triage
        with patch("piq.brisque", return_value=score):
            return triage.run(_img(3, H, W))

    def test_score_10_returns_skip(self):
        assert self._run_mocked(10.0) == "skip"

    def test_score_35_returns_sharpen(self):
        assert self._run_mocked(35.0) == "sharpen"

    def test_score_75_returns_sharpen_strong(self):
        assert self._run_mocked(75.0) == "sharpen_strong"

    def test_small_image_returns_upscale(self):
        # min(400, 600) = 400 < 800
        assert self._run_mocked(35.0, H=400, W=600) == "upscale"

    def test_large_clean_image_skip(self):
        # positive twin: large + clean → skip
        assert self._run_mocked(10.0, H=1200, W=1800) == "skip"

    def test_boundary_score_20_is_sharpen(self):
        # 20 is not < 20, so falls in [20, 50) → sharpen
        assert self._run_mocked(20.0) == "sharpen"

    def test_boundary_score_50_is_sharpen_strong(self):
        # 50 is not < 50 → sharpen_strong
        assert self._run_mocked(50.0) == "sharpen_strong"

    def test_cache_argument_accepted(self):
        # cache=None must not raise
        from lce.models import triage
        with patch("piq.brisque", return_value=35.0):
            result = triage.run(_img(3, 1000, 1000), cache=None)
        assert result == "sharpen"


# ---------------------------------------------------------------------------
# enhance
# ---------------------------------------------------------------------------

class TestEnhance:
    """enhance.run() delegates to the right sub-module for each mode."""

    def test_mode_sharpen_calls_sharpen_run(self):
        from lce.models import enhance
        input_img = _img(3, 64, 64)
        expected = _img(3, 64, 64, val=0.9)
        with patch("lce.models.sharpen.run", return_value=expected) as mock_sh:
            result = enhance.run(input_img, MagicMock(), mode="sharpen")
        mock_sh.assert_called_once()
        np.testing.assert_array_equal(result, expected)

    def test_mode_upscale_calls_upscale_run(self):
        from lce.models import enhance
        input_img = _img(3, 64, 64)
        expected = _img(3, 256, 256, val=0.8)
        with patch("lce.models.upscale.run", return_value=expected) as mock_up:
            result = enhance.run(input_img, MagicMock(), mode="upscale")
        mock_up.assert_called_once()
        np.testing.assert_array_equal(result, expected)

    def test_mode_sharpen_upscale_chains_both(self):
        from lce.models import enhance
        input_img = _img(3, 64, 64)
        sharpened = _img(3, 64, 64, val=0.7)
        upscaled = _img(3, 256, 256, val=0.8)
        with patch("lce.models.sharpen.run", return_value=sharpened) as mock_sh, \
             patch("lce.models.upscale.run", return_value=upscaled) as mock_up:
            result = enhance.run(input_img, MagicMock(), mode="sharpen_upscale")
        mock_sh.assert_called_once()
        mock_up.assert_called_once()
        np.testing.assert_array_equal(result, upscaled)

    def test_mode_auto_skip_returns_input_unchanged(self):
        from lce.models import enhance
        input_img = _img(3, 1000, 1000, val=0.4)
        with patch("lce.models.triage.run", return_value="skip"):
            result = enhance.run(input_img, MagicMock(), mode="auto")
        np.testing.assert_array_equal(result, input_img)

    def test_mode_auto_sharpen_delegates_to_sharpen(self):
        from lce.models import enhance
        input_img = _img(3, 64, 64)
        enhanced = _img(3, 64, 64, val=0.9)
        with patch("lce.models.triage.run", return_value="sharpen"), \
             patch("lce.models.sharpen.run", return_value=enhanced) as mock_sh:
            result = enhance.run(input_img, MagicMock(), mode="auto")
        mock_sh.assert_called_once()
        np.testing.assert_array_equal(result, enhanced)

    def test_mode_auto_upscale_delegates_to_upscale(self):
        from lce.models import enhance
        input_img = _img(3, 64, 64)
        expected = _img(3, 256, 256, val=0.6)
        with patch("lce.models.triage.run", return_value="upscale"), \
             patch("lce.models.upscale.run", return_value=expected) as mock_up:
            result = enhance.run(input_img, MagicMock(), mode="auto")
        mock_up.assert_called_once()
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# face
# ---------------------------------------------------------------------------

class TestFace:
    """face.run() — RuntimeError if model absent; correct shape if present."""

    def test_raises_runtime_error_when_model_absent(self):
        from lce.models import face
        with patch("lce.models.registry.model_path", return_value=None):
            with pytest.raises(RuntimeError, match="GFPGAN v1.4 model not found"):
                face.run(_img(), MagicMock())

    def test_no_error_when_model_absent_message_is_descriptive(self):
        from lce.models import face
        with patch("lce.models.registry.model_path", return_value=None):
            with pytest.raises(RuntimeError, match="lce-weights get gfpgan-v14"):
                face.run(_img(), MagicMock())

    def test_output_shape_matches_input_when_model_present(self):
        from lce.models import face
        H, W = 48, 60
        input_img = _img(3, H, W, val=0.5)
        # Model returns [-1,1] output at 512×512
        mock_out = np.zeros((1, 3, 512, 512), dtype=np.float32)
        sess = _onnx_session(mock_out)
        cache = _fake_cache(sess)
        with patch("lce.models.registry.model_path", return_value=Path("/fake/gfpgan.onnx")):
            result = face.run(input_img, cache)
        assert result.shape == (3, H, W)

    def test_output_dtype_float32_when_model_present(self):
        from lce.models import face
        mock_out = np.zeros((1, 3, 512, 512), dtype=np.float32)
        sess = _onnx_session(mock_out)
        cache = _fake_cache(sess)
        with patch("lce.models.registry.model_path", return_value=Path("/fake/gfpgan.onnx")):
            result = face.run(_img(), cache)
        assert result.dtype == np.float32

    def test_output_values_in_unit_range(self):
        from lce.models import face
        # Zero output from model → (0+1)/2 = 0.5 → all pixels 0.5
        mock_out = np.zeros((1, 3, 512, 512), dtype=np.float32)
        sess = _onnx_session(mock_out)
        cache = _fake_cache(sess)
        with patch("lce.models.registry.model_path", return_value=Path("/fake/gfpgan.onnx")):
            result = face.run(_img(), cache)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

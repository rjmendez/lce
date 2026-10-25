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


class _Input:
    def __init__(self, name: str, shape: list):
        self.name = name
        self.shape = shape


class _FakeSession:
    """Test double for an ONNX InferenceSession: run() returns canned outputs and
    records every feed dict it was given (so tests can assert tensor names)."""

    def __init__(self, outputs, input_name="x", input_shape=("n", 3, "h", "w")):
        self._outputs = list(outputs)
        self._input = _Input(input_name, list(input_shape))
        self.feeds: list[dict] = []

    def get_inputs(self):
        return [self._input]

    def run(self, _names, feed):
        self.feeds.append(feed)
        return self._outputs


def _onnx_session(*outputs: np.ndarray, **kw):
    """Fake ONNX InferenceSession whose .run() returns *outputs as a list."""
    return _FakeSession(outputs, **kw)


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

    def test_4x_selects_hat_sentinel_x4_model(self):
        from lce.models import upscale
        calls = []

        def _ensure(model_id: str):
            calls.append(model_id)
            return Path("/fake/hat.onnx")

        mock_out = np.zeros((1, 3, 256 * 4, 256 * 4), dtype=np.float32)
        cache = _fake_cache(_onnx_session(mock_out))
        with patch("lce.models.registry.ensure", side_effect=_ensure):
            upscale.run(_img(), cache, scale=4)
        assert calls == ["hat-sentinel-x4"]

    def test_explicit_sentinel_variant_at_2x_selects_hat_sentinel_x2_model(self):
        # the explicit path: variant="sentinel" always means the satellite
        # model, independent of what the scale=2 default currently is.
        from lce.models import upscale
        calls = []

        def _ensure(model_id: str):
            calls.append(model_id)
            return Path("/fake/hat.onnx")

        mock_out = np.zeros((1, 3, 256 * 2, 256 * 2), dtype=np.float32)
        cache = _fake_cache(_onnx_session(mock_out))
        with patch("lce.models.registry.ensure", side_effect=_ensure):
            upscale.run(_img(), cache, scale=2, variant="sentinel")
        assert calls == ["hat-sentinel-x2"]

    def test_2x_default_selects_hat_stock_lora_x2_model(self):
        # positive twin: the new scale=2 default (variant=None) is the
        # stock LoRA fine-tune, not the satellite model above.
        from lce.models import upscale
        calls = []

        def _ensure(model_id: str):
            calls.append(model_id)
            return Path("/fake/hat.onnx")

        mock_out = np.zeros((1, 3, 256 * 2, 256 * 2), dtype=np.float32)
        cache = _fake_cache(_onnx_session(mock_out))
        with patch("lce.models.registry.ensure", side_effect=_ensure):
            upscale.run(_img(), cache, scale=2)
        assert calls == ["hat-stock-lora-x2"]

    def test_2x_default_with_stock_weights_missing_raises_not_a_fallback(self):
        # no silent fallback to the sentinel model when the stock weights
        # are absent: registry.ensure()'s FileNotFoundError propagates as-is.
        from lce.models import upscale

        def _ensure(model_id: str):
            if model_id == "hat-stock-lora-x2":
                raise FileNotFoundError(
                    "'hat-stock-lora-x2' is a local-only model with no public "
                    "download; expected it at /fake/missing/stock.onnx"
                )
            raise AssertionError(f"unexpected model id {model_id!r}")

        with patch("lce.models.registry.ensure", side_effect=_ensure):
            with pytest.raises(FileNotFoundError, match="hat-stock-lora-x2"):
                upscale.run(_img(), MagicMock(), scale=2)

    def test_stock_variant_selects_hat_stock_lora_x2_model(self):
        from lce.models import upscale
        calls = []

        def _ensure(model_id: str):
            calls.append(model_id)
            return Path("/fake/hat.onnx")

        mock_out = np.zeros((1, 3, 256 * 2, 256 * 2), dtype=np.float32)
        cache = _fake_cache(_onnx_session(mock_out))
        with patch("lce.models.registry.ensure", side_effect=_ensure):
            upscale.run(_img(), cache, scale=2, variant="stock")
        assert calls == ["hat-stock-lora-x2"]

    def test_stock_variant_at_scale_4_selects_hat_stock_x4_model(self):
        # "stock" at scale=4 is the plain zero-shot official checkpoint (no
        # LoRA, no gated fine-tune) -- a distinct model id from "stock" at
        # scale=2, and not the scale=4 default (see test_4x_selects_hat_sentinel_x4_model).
        from lce.models import upscale
        calls = []

        def _ensure(model_id: str):
            calls.append(model_id)
            return Path("/fake/hat.onnx")

        mock_out = np.zeros((1, 3, 256 * 4, 256 * 4), dtype=np.float32)
        cache = _fake_cache(_onnx_session(mock_out))
        with patch("lce.models.registry.ensure", side_effect=_ensure):
            upscale.run(_img(), cache, scale=4, variant="stock")
        assert calls == ["hat-stock-x4"]

    def test_unknown_variant_raises_value_error(self):
        from lce.models import upscale
        with pytest.raises(ValueError, match="no 'nope' upscale model for scale=2"):
            upscale.run(_img(), MagicMock(), scale=2, variant="nope")


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

    def test_selects_nafnet_fp32_model_and_feeds_its_input_name(self):
        from lce.models import sharpen
        calls = []

        def _ensure(mid):
            calls.append(mid)
            return Path("/fake/nafnet.onnx")

        mock_out = np.full((1, 3, 64, 64), 0.5, dtype=np.float32)
        sess = _onnx_session(mock_out, input_name="lq")
        with patch("lce.models.registry.ensure", side_effect=_ensure):
            sharpen.run(_img(), _fake_cache(sess))
        assert calls == ["nafnet-fp32"]
        assert list(sess.feeds[0]) == ["lq"]

    def test_strength_outside_unit_range_raises(self):
        from lce.models import sharpen
        with pytest.raises(ValueError, match=r"strength must be in \[0, 1\], got 1.5"):
            sharpen.run(_img(), MagicMock(), strength=1.5)

    def test_strength_inside_unit_range_runs(self):
        # positive twin for the out-of-range test
        result, _, _ = self._run_mocked(1.0)
        assert result.shape == (3, 64, 64)

    def test_small_image_is_padded_without_reflect_error(self):
        # 3x5 image: reflect-padding to a multiple of 8 needs pad < size and
        # would raise ValueError; the edge fallback must handle it.
        from lce.models import sharpen
        mock_out = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        cache = _fake_cache(_onnx_session(mock_out))
        with patch("lce.models.registry.ensure", return_value=Path("/fake/nafnet.onnx")):
            result = sharpen.run(_img(3, 3, 5, val=0.1), cache)
        np.testing.assert_allclose(result, np.full((3, 3, 5), 0.5), atol=1e-6)


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

    def _run_with_prompt(self, points=None, labels=None, H=64, W=64):
        from lce.models import segment
        input_img = _img(3, H, W)
        enc_out = np.zeros((1, 256, 64, 64), dtype=np.float32)
        mask_out = np.ones((1, 1, H, W), dtype=np.float32)
        enc_sess = _onnx_session(enc_out)
        dec_sess = _onnx_session(mask_out)
        cache = _fake_cache(enc_sess, dec_sess)
        with patch(
            "lce.models.registry.ensure",
            side_effect=[Path("/fake/enc.onnx"), Path("/fake/dec.onnx")],
        ):
            segment.run(input_img, cache, points=points, labels=labels)
        return dec_sess.feeds[0]

    def test_custom_point_is_scaled_into_the_resized_pixel_frame(self):
        # H=W=64 -> long side 1024 means k=16, so a center point (0.5,0.5)
        # lands at exactly (512, 512) in the resized pixel frame, and with
        # no labels given it defaults to foreground (1.0).
        feed = self._run_with_prompt(points=[(0.5, 0.5)])
        np.testing.assert_array_equal(feed["point_coords"], [[[512.0, 512.0]]])
        np.testing.assert_array_equal(feed["point_labels"], [[1.0]])

    def test_custom_labels_are_passed_through_exactly(self):
        feed = self._run_with_prompt(points=[(0.1, 0.2), (0.8, 0.9)], labels=[1, 0])
        np.testing.assert_allclose(
            feed["point_coords"], [[[0.1 * 1024, 0.2 * 1024], [0.8 * 1024, 0.9 * 1024]]]
        )
        np.testing.assert_array_equal(feed["point_labels"], [[1.0, 0.0]])

    def test_mismatched_labels_length_raises_value_error(self):
        from lce.models import segment
        with pytest.raises(ValueError, match="same length as points"):
            segment.run(_img(), MagicMock(), points=[(0.5, 0.5)], labels=[1, 0])

    def test_matching_labels_length_does_not_raise(self):
        # positive twin for the mismatched-length test above
        feed = self._run_with_prompt(points=[(0.5, 0.5), (0.1, 0.1)], labels=[1, 0])
        assert feed["point_coords"].shape == (1, 2, 2)


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


    def test_unknown_mode_raises(self):
        from lce.models import enhance
        with pytest.raises(ValueError, match="mode must be one of"):
            enhance.run(_img(), MagicMock(), mode="bogus")

    def test_unknown_triage_route_raises_instead_of_returning_input(self):
        from lce.models import enhance
        with patch("lce.models.triage.run", return_value="mystery"):
            with pytest.raises(ValueError, match="unknown route 'mystery'"):
                enhance.run(_img(), MagicMock(), mode="auto")

    def test_scale_is_passed_to_upscale(self):
        from lce.models import enhance
        with patch("lce.models.upscale.run", return_value=_img(3, 128, 128)) as mock_up:
            enhance.run(_img(), MagicMock(), mode="upscale", scale=2)
        assert mock_up.call_args.kwargs["scale"] == 2

    def test_run_with_route_reports_route(self):
        from lce.models import enhance
        with patch("lce.models.triage.run", return_value="skip"):
            _, route = enhance.run_with_route(_img(), MagicMock(), mode="auto")
        assert route == "skip"
        with patch("lce.models.sharpen.run", return_value=_img()):
            _, route = enhance.run_with_route(_img(), MagicMock(), mode="sharpen")
        assert route == "sharpen"


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

    def test_non_square_image_raises_value_error(self):
        # 2:1 aspect ratio (e.g. 64x128) looks nothing like a cropped face.
        from lce.models import face
        wide_img = _img(3, 64, 128)
        with patch("lce.models.registry.model_path", return_value=Path("/fake/gfpgan.onnx")):
            with pytest.raises(ValueError, match="Image does not look like a cropped face"):
                face.run(wide_img, MagicMock())

    def test_near_square_image_passes_guard_and_reaches_inference(self):
        # Positive twin: a near-square, face-crop-shaped image (ratio 1.25,
        # under the 1.4 threshold) must NOT raise, and must actually reach
        # the model (a recording stub that returns a fixed array, not one
        # that raises on call).
        from lce.models import face
        H, W = 48, 60   # 60/48 = 1.25 < 1.4
        mock_out = np.zeros((1, 3, 512, 512), dtype=np.float32)
        sess = _onnx_session(mock_out)
        cache = _fake_cache(sess)
        with patch("lce.models.registry.model_path", return_value=Path("/fake/gfpgan.onnx")):
            result = face.run(_img(3, H, W), cache)
        assert len(sess.feeds) == 1          # inference was actually called
        assert result.shape == (3, H, W)

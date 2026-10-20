"""Tests for lce.engine: tiling, image I/O, device handling."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from lce import engine


def _rand(C, H, W, seed=0):
    return np.random.default_rng(seed).random((C, H, W), dtype=np.float32)


def _identity_feed(scale):
    def feed(tile, _sess):
        return np.repeat(np.repeat(tile, scale, axis=1), scale, axis=2)
    return feed


class TestRunTiled:
    @pytest.mark.parametrize("scale", [1, 2, 4])
    @pytest.mark.parametrize("H,W", [(100, 150), (64, 64), (20, 30), (97, 131)])
    def test_identity_feed_reproduces_image_everywhere_including_borders(self, scale, H, W):
        img = _rand(3, H, W)
        out = engine.run_tiled(None, img, 64, 64, 8, scale, _identity_feed(scale))
        expected = np.repeat(np.repeat(img, scale, axis=1), scale, axis=2)
        assert out.shape == (3, H * scale, W * scale)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_outer_border_is_not_black(self):
        # regression: the blend window used to be exactly 0 at tile edges, so
        # the outermost row/column of every output was 0.
        img = np.full((3, 100, 150), 0.7, dtype=np.float32)
        out = engine.run_tiled(None, img, 64, 64, 8, 1, _identity_feed(1))
        np.testing.assert_allclose(out[:, 0, :], 0.7, atol=1e-6)
        np.testing.assert_allclose(out[:, :, -1], 0.7, atol=1e-6)

    def test_seam_blend_is_a_weighted_average_of_overlapping_tiles(self):
        # two tiles disagreeing (0.2 vs 0.8) must blend strictly between them
        # inside the overlap, and keep each tile's value outside it.
        img = np.zeros((1, 16, 24), dtype=np.float32)
        calls = {"n": 0}

        def feed(tile, _s):
            calls["n"] += 1
            return np.full_like(tile, 0.2 if calls["n"] == 1 else 0.8)

        out = engine.run_tiled(None, img, 16, 16, 8, 1, feed)
        assert calls["n"] == 2
        assert out[0, 0, 0] == pytest.approx(0.2, abs=1e-6)
        assert out[0, 0, 23] == pytest.approx(0.8, abs=1e-6)
        seam = out[0, 0, 8:16]
        assert (seam > 0.2).all() and (seam < 0.8).all()
        assert (np.diff(seam) > 0).all()


class TestPadReflect:
    def test_reflects_when_pad_fits(self):
        img = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
        out = engine.pad_reflect(img, 1, 2)
        assert out.shape == (1, 3, 5)
        np.testing.assert_array_equal(out[0, 2], [0, 1, 2, 1, 0])

    def test_edge_fallback_when_pad_not_smaller_than_size(self):
        img = np.arange(3, dtype=np.float32).reshape(1, 1, 3)
        out = engine.pad_reflect(img, 4, 0)
        assert out.shape == (1, 5, 3)
        np.testing.assert_array_equal(out[0, 4], [0, 1, 2])


class TestImageIO:
    def test_save_rounds_to_nearest_not_truncates(self):
        # 0.999 * 255 = 254.745 -> 255 (truncation would give 254)
        img = np.full((3, 2, 2), 0.999, dtype=np.float32)
        assert engine.to_uint8_hwc(img)[0, 0, 0] == 255

    def test_png_roundtrip_is_exact(self, tmp_path):
        arr = np.random.default_rng(1).integers(0, 256, (7, 9, 3), dtype=np.uint8)
        p = tmp_path / "a.png"
        Image.fromarray(arr).save(p)
        img = engine.load_image(p)
        out = tmp_path / "b.png"
        engine.save_image(img, out)
        np.testing.assert_array_equal(np.asarray(Image.open(out)), arr)

    def test_default_jpeg_quality_is_95(self, tmp_path):
        img = _rand(3, 64, 64)
        p = tmp_path / "q.jpg"
        engine.save_image(img, p)
        with Image.open(p) as im:
            q = im.quantization
        ref = io.BytesIO()
        Image.fromarray(engine.to_uint8_hwc(img)).save(ref, "JPEG", quality=95, subsampling=0)
        ref_q = Image.open(io.BytesIO(ref.getvalue())).quantization
        assert q == ref_q

    def test_exif_and_icc_survive_load_save(self, tmp_path):
        exif = Image.Exif()
        exif[0x010F] = "TestMake"
        icc = b"fake-icc-profile-bytes"
        src = tmp_path / "src.jpg"
        Image.fromarray(np.full((8, 8, 3), 100, np.uint8)).save(
            src, "JPEG", exif=exif.tobytes(), icc_profile=icc)
        img, meta = engine.load_image_with_meta(src)
        dst = tmp_path / "dst.jpg"
        engine.save_image(img, dst, exif=meta["exif"], icc_profile=meta["icc_profile"])
        with Image.open(dst) as im:
            assert im.info["icc_profile"] == icc
            assert im.getexif()[0x010F] == "TestMake"

    def test_save_without_meta_writes_none(self, tmp_path):
        # positive twin: nothing is invented when no metadata is passed
        dst = tmp_path / "plain.jpg"
        engine.save_image(_rand(3, 8, 8), dst)
        with Image.open(dst) as im:
            assert "icc_profile" not in im.info
            assert len(im.getexif()) == 0

    def test_exif_orientation_is_applied_to_pixels_and_removed(self, tmp_path):
        arr = np.zeros((4, 6, 3), np.uint8)   # H=4, W=6
        arr[0, 0] = (255, 0, 0)               # red top-left
        exif = Image.Exif()
        exif[0x0112] = 6                      # rotate 90 CW to display
        src = tmp_path / "rot.jpg"
        Image.fromarray(arr).save(src, "JPEG", quality=100, subsampling=0, exif=exif.tobytes())
        img, meta = engine.load_image_with_meta(src)
        assert img.shape == (3, 6, 4)         # now upright: H=6, W=4
        assert img[0, 0, 3] > 0.9             # red moved to top-right
        loaded = Image.Exif()
        loaded.load(meta["exif"] or b"")
        assert loaded.get(0x0112) in (None, 1)  # no second rotation downstream


class TestDevice:
    def test_cuda_device_id_default_is_zero(self, monkeypatch):
        monkeypatch.delenv("LCE_CUDA_DEVICE_ID", raising=False)
        assert engine._cuda_device_id() == 0

    def test_cuda_device_id_from_env(self, monkeypatch):
        monkeypatch.setenv("LCE_CUDA_DEVICE_ID", "1")
        assert engine._cuda_device_id() == 1

    def test_cuda_device_id_rejects_garbage(self, monkeypatch):
        monkeypatch.setenv("LCE_CUDA_DEVICE_ID", "gpu1")
        with pytest.raises(ValueError, match="LCE_CUDA_DEVICE_ID must be an integer, got 'gpu1'"):
            engine._cuda_device_id()

"""End-to-end tests of LceHandler (tools/list, tools/call) with tiny real ONNX
models generated on the fly, registered through a temp registry."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from unittest.mock import patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto as TP, helper, numpy_helper
from PIL import Image

from lce.engine import ModelCache
from lce.models import registry
from lce.server import LceHandler


# ---------------------------------------------------------------------------
# tiny ONNX models
# ---------------------------------------------------------------------------

def _save(model, path):
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())


def _model(graph):
    # ir_version 9: readable by every onnxruntime >= 1.17 (the declared minimum)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)


def _identity_model(input_name, path):
    """Dynamic-shape identity: out = in."""
    x = helper.make_tensor_value_info(input_name, TP.FLOAT, ["n", 3, "h", "w"])
    y = helper.make_tensor_value_info("output", TP.FLOAT, ["n", 3, "h", "w"])
    g = helper.make_graph([helper.make_node("Identity", [input_name], ["output"])], "id", [x], [y])
    _save(_model(g), path)


def _nearest_up_model(input_name, scale, path):
    """Static 64x64 input, nearest-neighbour upscale by *scale* (like the HAT files)."""
    x = helper.make_tensor_value_info(input_name, TP.FLOAT, [1, 3, 64, 64])
    y = helper.make_tensor_value_info("output", TP.FLOAT, [1, 3, 64 * scale, 64 * scale])
    scales = numpy_helper.from_array(np.array([1, 1, scale, scale], np.float32), "scales")
    node = helper.make_node("Resize", [input_name, "", "scales"], ["output"], mode="nearest")
    g = helper.make_graph([node], "up", [x], [y], initializer=[scales])
    _save(_model(g), path)


def _nearest_up_model_dynamic(input_name, scale, path):
    """Dynamic H/W input, nearest-neighbour upscale by *scale* (like the real
    stock-variant model, which has no fixed tile size)."""
    x = helper.make_tensor_value_info(input_name, TP.FLOAT, [1, 3, "h", "w"])
    y = helper.make_tensor_value_info("output", TP.FLOAT, [1, 3, "h2", "w2"])
    scales = numpy_helper.from_array(np.array([1, 1, scale, scale], np.float32), "scales")
    node = helper.make_node("Resize", [input_name, "", "scales"], ["output"], mode="nearest")
    g = helper.make_graph([node], "up_dyn", [x], [y], initializer=[scales])
    _save(_model(g), path)


def _sam_encoder_model(path):
    x = helper.make_tensor_value_info("input_image", TP.FLOAT, ["ih", "iw", 3])
    y = helper.make_tensor_value_info("image_embeddings", TP.FLOAT, [1, 256, 64, 64])
    const = numpy_helper.from_array(np.zeros((1, 256, 64, 64), np.float32), "emb")
    g = helper.make_graph([helper.make_node("Identity", ["emb"], ["image_embeddings"])],
                          "enc", [x], [y], initializer=[const])
    _save(_model(g), path)


def _sam_decoder_model(path):
    """masks = ones[1,1,H,W] with (H,W) read from the orig_im_size input."""
    ins = [
        helper.make_tensor_value_info("image_embeddings", TP.FLOAT, [1, 256, 64, 64]),
        helper.make_tensor_value_info("point_coords", TP.FLOAT, [1, "np", 2]),
        helper.make_tensor_value_info("point_labels", TP.FLOAT, [1, "np"]),
        helper.make_tensor_value_info("mask_input", TP.FLOAT, [1, 1, 256, 256]),
        helper.make_tensor_value_info("has_mask_input", TP.FLOAT, [1]),
        helper.make_tensor_value_info("orig_im_size", TP.FLOAT, [2]),
    ]
    out = helper.make_tensor_value_info("masks", TP.FLOAT, [1, 1, "h", "w"])
    head = numpy_helper.from_array(np.array([1, 1], np.int64), "head")
    nodes = [
        helper.make_node("Cast", ["orig_im_size"], ["hw"], to=TP.INT64),
        helper.make_node("Concat", ["head", "hw"], ["shape"], axis=0),
        helper.make_node("ConstantOfShape", ["shape"], ["masks"],
                         value=numpy_helper.from_array(np.array([1.0], np.float32))),
    ]
    g = helper.make_graph(nodes, "dec", ins, [out], initializer=[head])
    _save(_model(g), path)


MODELS = {  # registry id -> (file name, builder)
    "nafnet-fp32": ("nafnet.onnx", lambda p: _identity_model("lq", p)),
    "hat-sentinel-x2": ("hat2.onnx", lambda p: _nearest_up_model("onnx::Sub_0", 2, p)),
    "hat-sentinel-x4": ("hat4.onnx", lambda p: _nearest_up_model("onnx::Sub_0", 4, p)),
    "mobilesam-encoder": ("enc.onnx", _sam_encoder_model),
    "mobilesam-decoder": ("dec.onnx", _sam_decoder_model),
}


@pytest.fixture
def handler(tmp_path, monkeypatch):
    wdir = tmp_path / "weights"
    wdir.mkdir()
    monkeypatch.setenv("LCE_WEIGHTS_DIR", str(wdir))
    toml = tmp_path / "models.toml"
    monkeypatch.setattr(registry, "_REGISTRY_FILE", toml)
    blocks = []
    for mid, (fname, build) in MODELS.items():
        build(wdir / fname)
        data = (wdir / fname).read_bytes()
        blocks.append(
            f'[[model]]\nid = "{mid}"\ncapability = "x"\nurl = "https://example.invalid/{fname}"\n'
            f'sha256 = "{hashlib.sha256(data).hexdigest()}"\nsize_bytes = {len(data)}\n'
            'format = "onnx"\nlicense = "MIT"\ndescription = "tiny"\noptional = false\n')
    toml.write_text("\n".join(blocks), encoding="utf-8")
    return LceHandler(ModelCache(4, "CPU"), "CPU")


def _call(h, name, **args):
    return h.handle({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                     "params": {"name": name, "arguments": args}})


def _photo(path, H=50, W=70, fmt="PNG", **save_kw):
    arr = np.random.default_rng(3).integers(0, 256, (H, W, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, fmt, **save_kw)
    return arr


def _b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _item(resp, i=0):
    assert "error" not in resp, resp
    return resp["result"]["content"][i]


# ---------------------------------------------------------------------------

class TestToolsList:
    def test_lists_exactly_the_implemented_tools(self, handler):
        resp = handler.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = {t["name"]: t for t in resp["result"]["tools"]}
        assert sorted(tools) == [
            "enhance", "health", "restore_face", "segment", "sharpen", "triage", "upscale",
        ]
        assert tools["sharpen"]["inputSchema"]["properties"]["strength"]["default"] == 1.0
        assert tools["restore_face"]["inputSchema"]["required"] == ["confirm_cropped"]
        text = json.dumps(resp)
        assert "DA-CLIP" not in text and "Gigapixel" not in text


class TestSharpen:
    def test_identity_model_roundtrips_exactly_via_output_path(self, handler, tmp_path):
        src = tmp_path / "in.png"
        arr = _photo(src)
        out = tmp_path / "out.png"
        resp = _call(handler, "sharpen", path=str(src), output_path=str(out))
        assert _item(resp)["mimeType"] == "image/jpeg"
        np.testing.assert_array_equal(np.asarray(Image.open(out)), arr)

    def test_explicit_strength_zero_also_returns_input(self, handler, tmp_path):
        # positive twin: the same call with an explicit strength works
        src = tmp_path / "in.png"
        arr = _photo(src)
        out = tmp_path / "out.png"
        _item(_call(handler, "sharpen", path=str(src), strength=0.0, output_path=str(out)))
        np.testing.assert_array_equal(np.asarray(Image.open(out)), arr)

    def test_b64_input_and_jpeg_preview_decodes_to_input_size(self, handler):
        arr = np.random.default_rng(4).integers(0, 256, (30, 40, 3), dtype=np.uint8)
        item = _item(_call(handler, "sharpen", image_b64=_b64(arr)))
        preview = Image.open(io.BytesIO(base64.b64decode(item["data"])))
        assert preview.format == "JPEG" and preview.size == (40, 30)

    def test_missing_image_is_a_clean_error(self, handler):
        resp = _call(handler, "sharpen")
        assert resp["error"]["code"] == -32000
        assert resp["error"]["message"] == "Provide image_b64 or path"

    def test_exif_and_icc_are_carried_to_output_jpeg(self, handler, tmp_path):
        exif = Image.Exif()
        exif[0x010F] = "Maker"
        src = tmp_path / "in.jpg"
        _photo(src, fmt="JPEG", quality=95, exif=exif.tobytes(), icc_profile=b"icc-bytes")
        out = tmp_path / "out.jpg"
        _item(_call(handler, "sharpen", path=str(src), output_path=str(out)))
        with Image.open(out) as im:
            assert im.info["icc_profile"] == b"icc-bytes"
            assert im.getexif()[0x010F] == "Maker"


class TestUpscale:
    @pytest.mark.parametrize("scale", [2, 4])
    def test_nearest_model_gives_exact_pixels_incl_borders(self, handler, tmp_path, scale):
        # the shared `handler` fixture only registers the two sentinel models
        # (no stock weights); scale=2's default is now the stock model, so
        # request the satellite model explicitly at scale=2. scale=4 is
        # unaffected -- the satellite model is still its default.
        src = tmp_path / "in.png"
        arr = _photo(src, H=50, W=70)   # smaller than the model's 64 px tile in H
        out = tmp_path / "out.png"
        kwargs = {"variant": "sentinel"} if scale == 2 else {}
        _item(_call(handler, "upscale", path=str(src), scale=scale, output_path=str(out), **kwargs))
        expected = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)
        np.testing.assert_array_equal(np.asarray(Image.open(out)), expected)

    def test_bad_scale_is_a_clean_error(self, handler, tmp_path):
        src = tmp_path / "in.png"
        _photo(src)
        resp = _call(handler, "upscale", path=str(src), scale=3)
        assert resp["error"]["message"] == "scale must be 2 or 4, got 3"

    def test_explicit_sentinel_variant_behaves_like_the_default(self, handler, tmp_path):
        # explicit sentinel variant always selects the satellite model,
        # independent of the default (which is now the stock model at scale=2).
        src = tmp_path / "in.png"
        arr = _photo(src, H=50, W=70)
        out = tmp_path / "out.png"
        _item(_call(handler, "upscale", path=str(src), scale=2, variant="sentinel", output_path=str(out)))
        expected = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
        np.testing.assert_array_equal(np.asarray(Image.open(out)), expected)


class TestUpscaleStockVariant:
    """The "stock" variant is a local-only registry entry (local_path, not url) --
    exercised with its own registry so the shared `handler` fixture (every other
    test in this file) never has to know about it."""

    @pytest.fixture
    def handler_with_stock(self, tmp_path, monkeypatch):
        wdir = tmp_path / "weights"
        wdir.mkdir()
        monkeypatch.setenv("LCE_WEIGHTS_DIR", str(wdir))
        toml = tmp_path / "models.toml"
        monkeypatch.setattr(registry, "_REGISTRY_FILE", toml)

        blocks = []
        for mid, (fname, build) in MODELS.items():
            build(wdir / fname)
            data = (wdir / fname).read_bytes()
            blocks.append(
                f'[[model]]\nid = "{mid}"\ncapability = "x"\nurl = "https://example.invalid/{fname}"\n'
                f'sha256 = "{hashlib.sha256(data).hexdigest()}"\nsize_bytes = {len(data)}\n'
                'format = "onnx"\nlicense = "MIT"\ndescription = "tiny"\noptional = false\n')

        stock_path = wdir / "stock.onnx"
        _nearest_up_model_dynamic("input", 2, stock_path)
        stock_data = stock_path.read_bytes()
        escaped = str(stock_path).replace("\\", "\\\\")
        blocks.append(
            '[[model]]\nid = "hat-stock-lora-x2"\ncapability = "upscale"\n'
            f'local_path = "{escaped}"\n'
            f'sha256 = "{hashlib.sha256(stock_data).hexdigest()}"\nsize_bytes = {len(stock_data)}\n'
            'format = "onnx"\nlicense = "unspecified"\ndescription = "tiny local"\noptional = true\n')

        stock4_path = wdir / "stock4.onnx"
        _nearest_up_model_dynamic("input", 4, stock4_path)
        stock4_data = stock4_path.read_bytes()
        escaped4 = str(stock4_path).replace("\\", "\\\\")
        blocks.append(
            '[[model]]\nid = "hat-stock-x4"\ncapability = "upscale"\n'
            f'local_path = "{escaped4}"\n'
            f'sha256 = "{hashlib.sha256(stock4_data).hexdigest()}"\nsize_bytes = {len(stock4_data)}\n'
            'format = "onnx"\nlicense = "Apache-2.0"\ndescription = "tiny local"\noptional = true\n')
        toml.write_text("\n".join(blocks), encoding="utf-8")
        return LceHandler(ModelCache(4, "CPU"), "CPU")

    def test_stock_variant_scale_2_uses_the_local_only_model(self, handler_with_stock, tmp_path):
        src = tmp_path / "in.png"
        arr = _photo(src, H=50, W=70)
        out = tmp_path / "out.png"
        _item(_call(handler_with_stock, "upscale", path=str(src), scale=2, variant="stock", output_path=str(out)))
        expected = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
        np.testing.assert_array_equal(np.asarray(Image.open(out)), expected)

    def test_default_scale_2_uses_the_stock_model(self, handler_with_stock, tmp_path):
        # positive twin: the scale=2 default (no variant given) reaches the
        # same local-only LoRA model as the explicit variant="stock" above,
        # end-to-end through the server, not just inside upscale.run(). Both
        # the registered sentinel and stock fakes do an identical nearest-
        # neighbour upscale, so pixel output alone can't tell which model
        # ran -- the wrapped registry.ensure recording stub below (it still
        # calls through to the real ensure()) asserts the exact model id.
        src = tmp_path / "in.png"
        arr = _photo(src, H=50, W=70)
        out = tmp_path / "out.png"
        with patch("lce.models.registry.ensure", wraps=registry.ensure) as mock_ensure:
            _item(_call(handler_with_stock, "upscale", path=str(src), scale=2, output_path=str(out)))
        assert mock_ensure.call_args.args == ("hat-stock-lora-x2",)
        expected = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
        np.testing.assert_array_equal(np.asarray(Image.open(out)), expected)

    def test_stock_variant_scale_4_uses_the_zero_shot_local_only_model(self, handler_with_stock, tmp_path):
        # "stock" at scale=4 is a distinct, local-only model id (hat-stock-x4,
        # the plain zero-shot checkpoint, no LoRA) from "stock" at scale=2 --
        # and unlike scale=4's default, it is reachable only by this explicit
        # variant (see test_4x_selects_hat_sentinel_x4_model, unaffected).
        src = tmp_path / "in.png"
        arr = _photo(src, H=50, W=70)
        out = tmp_path / "out.png"
        _item(_call(handler_with_stock, "upscale", path=str(src), scale=4, variant="stock", output_path=str(out)))
        expected = np.repeat(np.repeat(arr, 4, axis=0), 4, axis=1)
        np.testing.assert_array_equal(np.asarray(Image.open(out)), expected)

    def test_missing_local_file_is_a_clean_error_not_a_crash(self, handler_with_stock, tmp_path, monkeypatch):
        import lce.models.registry as registry_mod
        stock_entry = next(m for m in registry_mod.available() if m["id"] == "hat-stock-lora-x2")
        import os
        os.remove(stock_entry["local_path"])
        src = tmp_path / "in.png"
        _photo(src)
        resp = _call(handler_with_stock, "upscale", path=str(src), scale=2, variant="stock")
        assert "hat-stock-lora-x2" in resp["error"]["message"]
        assert "no public download" in resp["error"]["message"]

    def test_missing_local_file_is_a_clean_error_not_a_crash_via_default(
        self, handler_with_stock, tmp_path, monkeypatch
    ):
        # twin of the above for the default (no variant kwarg) path: the
        # decided no-fallback behaviour holds through the server for the
        # default too, not just when "stock" is requested explicitly.
        import lce.models.registry as registry_mod
        stock_entry = next(m for m in registry_mod.available() if m["id"] == "hat-stock-lora-x2")
        import os
        os.remove(stock_entry["local_path"])
        src = tmp_path / "in.png"
        _photo(src)
        resp = _call(handler_with_stock, "upscale", path=str(src), scale=2)
        assert "hat-stock-lora-x2" in resp["error"]["message"]
        assert "no public download" in resp["error"]["message"]


class TestSegment:
    def test_returns_png_mask_at_input_size(self, handler, tmp_path):
        arr = np.random.default_rng(5).integers(0, 256, (37, 53, 3), dtype=np.uint8)
        out = tmp_path / "mask.png"
        item = _item(_call(handler, "segment", image_b64=_b64(arr), output_path=str(out)))
        assert item["mimeType"] == "image/png"
        for src in (io.BytesIO(base64.b64decode(item["data"])), out):
            mask = Image.open(src)
            assert mask.format == "PNG" and mask.mode == "L"
            assert mask.size == (53, 37)
            np.testing.assert_array_equal(np.asarray(mask), np.full((37, 53), 255, np.uint8))

    def test_points_and_labels_args_reach_segment_run(self, handler, tmp_path):
        arr = np.random.default_rng(5).integers(0, 256, (37, 53, 3), dtype=np.uint8)
        with patch(
            "lce.models.segment.run", return_value=np.ones((1, 37, 53), dtype=np.float32)
        ) as mock_run:
            _call(handler, "segment", image_b64=_b64(arr), points=[[0.1, 0.2], [0.8, 0.9]], labels=[1, 0])
        assert mock_run.call_args.kwargs["points"] == [(0.1, 0.2), (0.8, 0.9)]
        assert mock_run.call_args.kwargs["labels"] == [1, 0]

    def test_omitted_points_reaches_segment_run_as_none(self, handler, tmp_path):
        # positive twin: no points/labels in the MCP call -> both args are
        # None, not e.g. an empty list, so segment.run's own default-grid
        # branch (points is None) actually runs.
        arr = np.random.default_rng(5).integers(0, 256, (37, 53, 3), dtype=np.uint8)
        with patch(
            "lce.models.segment.run", return_value=np.ones((1, 37, 53), dtype=np.float32)
        ) as mock_run:
            _call(handler, "segment", image_b64=_b64(arr))
        assert mock_run.call_args.kwargs["points"] is None
        assert mock_run.call_args.kwargs["labels"] is None


class TestTriageAndEnhance:
    def test_triage_reports_route(self, handler, tmp_path):
        src = tmp_path / "in.png"
        _photo(src)
        with patch("lce.models.triage.run", return_value="sharpen_strong") as t:
            item = _item(_call(handler, "triage", path=str(src)))
        assert json.loads(item["text"]) == {"route": "sharpen_strong"}
        assert t.call_count == 1

    def test_enhance_auto_reports_route_and_applies_it(self, handler, tmp_path):
        src = tmp_path / "in.png"
        arr = _photo(src)
        out = tmp_path / "out.png"
        with patch("lce.models.triage.run", return_value="sharpen"):
            resp = _call(handler, "enhance", path=str(src), output_path=str(out))
        assert json.loads(_item(resp, 0)["text"]) == {"route": "sharpen"}
        assert _item(resp, 1)["type"] == "image"
        np.testing.assert_array_equal(np.asarray(Image.open(out)), arr)

    def test_enhance_upscale_uses_requested_scale(self, handler, tmp_path):
        src = tmp_path / "in.png"
        _photo(src, H=20, W=30)
        out = tmp_path / "out.png"
        resp = _call(handler, "enhance", path=str(src), mode="upscale", scale=4, output_path=str(out))
        assert json.loads(_item(resp, 0)["text"]) == {"route": "upscale"}
        assert Image.open(out).size == (120, 80)

    def test_enhance_unknown_mode_is_a_clean_error(self, handler, tmp_path):
        src = tmp_path / "in.png"
        _photo(src)
        resp = _call(handler, "enhance", path=str(src), mode="bogus")
        assert "mode must be one of" in resp["error"]["message"]


class TestRestoreFace:
    def test_missing_confirm_cropped_is_a_clean_error(self, handler, tmp_path):
        src = tmp_path / "in.png"
        _photo(src, H=64, W=64)
        resp = _call(handler, "restore_face", path=str(src))
        assert resp["error"]["message"] == (
            "Set confirm_cropped=true to confirm the image is already "
            "cropped to a single face."
        )

    def test_confirm_cropped_false_is_also_a_clean_error(self, handler, tmp_path):
        # positive-for-the-check twin: an explicit falsy value is refused too,
        # not just an absent key.
        src = tmp_path / "in.png"
        _photo(src, H=64, W=64)
        resp = _call(handler, "restore_face", path=str(src), confirm_cropped=False)
        assert resp["error"]["message"] == (
            "Set confirm_cropped=true to confirm the image is already "
            "cropped to a single face."
        )

    def test_confirm_cropped_true_with_square_image_dispatches_to_face_run(self, handler, tmp_path):
        from lce.engine import to_uint8_hwc

        src = tmp_path / "in.png"
        _photo(src, H=64, W=64)
        out = tmp_path / "out.png"
        expected = np.full((3, 64, 64), 0.5, dtype=np.float32)
        with patch("lce.models.face.run", return_value=expected) as mock_face:
            resp = _call(handler, "restore_face", path=str(src), confirm_cropped=True, output_path=str(out))
        assert mock_face.call_count == 1
        assert _item(resp)["mimeType"] == "image/jpeg"
        np.testing.assert_array_equal(np.asarray(Image.open(out)), to_uint8_hwc(expected))


class TestHealthAndDispatch:
    def test_health_reports_loaded_models_and_real_providers(self, handler, tmp_path):
        src = tmp_path / "in.png"
        _photo(src)
        _item(_call(handler, "sharpen", path=str(src)))
        info = json.loads(_item(_call(handler, "health"))["text"])
        assert info["status"] == "ok"
        assert info["loaded_models"] == ["nafnet"]
        assert info["providers"] == {"nafnet": ["CPUExecutionProvider"]}

    def test_unknown_tool(self, handler):
        resp = _call(handler, "nope")
        assert resp["error"] == {"code": -32000, "message": "Unknown tool: nope"}


class TestPathAllowlist:
    def test_inside_allowed_dir_works(self, handler, tmp_path):
        d = tmp_path / "ok"
        d.mkdir()
        src = d / "in.png"
        _photo(src)
        h = LceHandler(handler._cache, "CPU", [d])
        _item(_call(h, "sharpen", path=str(src), output_path=str(d / "out.png")))
        assert (d / "out.png").exists()

    def test_outside_allowed_dir_is_refused_for_path_and_output_path(self, handler, tmp_path):
        ok, other = tmp_path / "ok", tmp_path / "other"
        ok.mkdir()
        other.mkdir()
        _photo(other / "secret.png")
        _photo(ok / "in.png")
        h = LceHandler(handler._cache, "CPU", [ok])
        r1 = _call(h, "sharpen", path=str(other / "secret.png"))
        assert "outside the allowed directories" in r1["error"]["message"]
        r2 = _call(h, "sharpen", path=str(ok / "in.png"), output_path=str(other / "stolen.png"))
        assert "outside the allowed directories" in r2["error"]["message"]
        assert not (other / "stolen.png").exists()

    def test_dotdot_escape_is_refused(self, handler, tmp_path):
        ok = tmp_path / "ok"
        ok.mkdir()
        _photo(tmp_path / "x.png")
        h = LceHandler(handler._cache, "CPU", [ok])
        r = _call(h, "sharpen", path=str(ok / ".." / "x.png"))
        assert "outside the allowed directories" in r["error"]["message"]

    def test_empty_allowlist_refuses_paths_but_b64_still_works(self, handler, tmp_path):
        src = tmp_path / "in.png"
        arr = _photo(src)
        h = LceHandler(handler._cache, "CPU", [])
        assert "outside the allowed directories" in _call(h, "sharpen", path=str(src))["error"]["message"]
        _item(_call(h, "sharpen", image_b64=_b64(arr)))

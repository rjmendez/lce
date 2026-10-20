"""
lce.server — MCP JSON-RPC 2.0 server.

LightCraft connects here as a client and discovers tools:
  enhance   — auto pipeline (BRISQUE triage -> route -> process)
  sharpen   — NAFNet deblurring
  upscale   — 2x or 4x super-resolution
  segment   — grid-prompted MobileSAM mask
  triage    — BRISQUE quality / resolution routing
  health    — server status + loaded model list

Run:  python -m lce.server  [--transport http --host 127.0.0.1 --port 8091 --allow-dir DIR]
The default transport is stdio (--host/--port apply to http only). Over http
the path/output_path arguments are refused unless they lie inside an
--allow-dir directory; image_b64 always works.
Or:   lce-serve  (installed entry-point)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import base64
import io
from pathlib import Path
from typing import Any

log = logging.getLogger("lce.server")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def _tools() -> list[dict]:
    return [
        {
            "name": "health",
            "description": "Return server status, device, and loaded model list.",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "triage",
            "description": (
                "Route an image by BRISQUE score and resolution (needs the triage extra). "
                "Returns route: skip | sharpen | sharpen_strong | upscale."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_b64": {"type": "string", "description": "Base64-encoded JPEG/PNG"},
                    "path": {"type": "string", "description": "Absolute path to image file"},
                },
            },
        },
        {
            "name": "sharpen",
            "description": "Deblur with NAFNet. Returns a JPEG preview; set output_path for the full-quality file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_b64": {"type": "string"},
                    "path": {"type": "string"},
                    "strength": {
                        "type": "number",
                        "description": "Blend of deblurred result with the original, 0-1 (default 1.0)",
                        "default": 1.0,
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "output_path": {"type": "string", "description": "Write the full-quality result here (optional)"},
                },
            },
        },
        {
            "name": "upscale",
            "description": "2x or 4x super-resolution. Returns a JPEG preview; set output_path for the full-quality file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_b64": {"type": "string"},
                    "path": {"type": "string"},
                    "scale": {"type": "integer", "enum": [2, 4], "default": 2},
                    "output_path": {"type": "string", "description": "Write the full-quality result here (optional)"},
                },
            },
        },
        {
            "name": "segment",
            "description": "Grid-prompted MobileSAM mask (not a subject detector). Returns the mask as a base64 PNG (255 = selected).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_b64": {"type": "string"},
                    "path": {"type": "string"},
                    "output_path": {"type": "string", "description": "Write the mask PNG here (optional)"},
                },
            },
        },
        {
            "name": "enhance",
            "description": (
                "Auto-enhancement: BRISQUE triage -> sharpen/upscale. "
                "Returns a text item naming the route that ran, then a JPEG preview image; "
                "set output_path for the full-quality file."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_b64": {"type": "string"},
                    "path": {"type": "string"},
                    "output_path": {"type": "string", "description": "Write result here (optional)"},
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "sharpen", "upscale", "sharpen_upscale"],
                        "default": "auto",
                    },
                    "scale": {"type": "integer", "enum": [2, 4], "default": 2,
                              "description": "Upscale factor when the route upscales"},
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class LceHandler:
    def __init__(self, cache, device: str, allowed_dirs: list[Path] | None = None):
        """*allowed_dirs*: None = any path (stdio); a list = path/output_path
        must lie inside one of these directories (empty list = no paths)."""
        self._cache = cache
        self._device = device
        self._allowed = None if allowed_dirs is None else [Path(d).resolve() for d in allowed_dirs]

    def _check_path(self, raw: str) -> Path:
        p = Path(raw)
        if self._allowed is None:
            return p
        rp = p.resolve()
        for d in self._allowed:
            if rp == d or d in rp.parents:
                return p
        raise PermissionError(f"path {raw!r} is outside the allowed directories (--allow-dir)")

    def handle(self, req: dict) -> dict | None:
        method = req.get("method", "")
        rid = req.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": rid,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "lce", "version": "0.1.0"},
                },
            }

        if method == "notifications/initialized":
            return None

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": rid, "result": {"tools": _tools()}}

        if method == "tools/call":
            name = req["params"]["name"]
            args = req["params"].get("arguments", {})
            try:
                result = self._dispatch(name, args)
                return {"jsonrpc": "2.0", "id": rid, "result": result}
            except Exception as exc:
                log.exception("Tool %s failed", name)
                return {
                    "jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32000, "message": str(exc)},
                }

        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}}

    # ------------------------------------------------------------------
    def _dispatch(self, name: str, args: dict) -> dict:
        from lce.models import enhance, segment, sharpen, triage, upscale

        if name == "health":
            from lce.engine import degraded_counts

            info: dict[str, Any] = {
                "status": "ok",
                "device": self._device,
                "loaded_models": [Path(p).stem for p in self._cache._lru.keys()],
                "providers": {Path(p).stem: v for p, v in self._cache.providers.items()},
                "onnx_degraded": degraded_counts(),      # sessions that did not end up on CUDA (from main's GPU provider logic)
            }
            try:
                import openvino as ov
                info["openvino_devices"] = ov.Core().available_devices
            except Exception as exc:  # openvino is only needed for IR models
                info["openvino_devices"] = f"unavailable: {exc}"
            return _text_result(info)

        if name not in {t["name"] for t in _tools()}:
            raise ValueError(f"Unknown tool: {name}")

        img, meta = self._load_arg(args)
        out_path = args.get("output_path")
        if out_path:
            out_path = str(self._check_path(out_path))

        if name == "triage":
            return _text_result({"route": triage.run(img, self._cache)})

        if name == "sharpen":
            strength = float(args.get("strength", 1.0))
            out = sharpen.run(img, self._cache, strength=strength)
            return _image_result(out, out_path, meta)

        if name == "upscale":
            scale = int(args.get("scale", 2))
            out = upscale.run(img, self._cache, scale=scale)
            return _image_result(out, out_path, meta)

        if name == "segment":
            mask = segment.run(img, self._cache)
            return _mask_result(mask, out_path)

        if name == "enhance":
            mode = args.get("mode", "auto")
            scale = int(args.get("scale", 2))
            out, route = enhance.run_with_route(img, self._cache, mode=mode, scale=scale)
            res = _image_result(out, out_path, meta)
            res["content"].insert(0, {"type": "text", "text": json.dumps({"route": route})})
            return res

        raise ValueError(f"Unknown tool: {name}")

    def _load_arg(self, args: dict):
        """Return (img_chw_f32, meta) from image_b64 or path."""
        from lce.engine import load_image_with_meta
        if "image_b64" in args:
            return load_image_with_meta(io.BytesIO(base64.b64decode(args["image_b64"])))
        if "path" in args:
            return load_image_with_meta(self._check_path(args["path"]))
        raise ValueError("Provide image_b64 or path")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _text_result(obj: dict) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(obj)}]}


def _image_result(img_chw: "np.ndarray", output_path: str | None = None, meta: dict | None = None) -> dict:
    """MCP image reply (JPEG preview, q95). *output_path*, when given, gets the
    full-quality file via engine.save_image with EXIF/ICC carried from *meta*."""
    from PIL import Image
    from lce.engine import save_image, to_uint8_hwc
    meta = meta or {}
    if output_path:
        save_image(img_chw, Path(output_path), exif=meta.get("exif"),
                   icc_profile=meta.get("icc_profile"))
    buf = io.BytesIO()
    Image.fromarray(to_uint8_hwc(img_chw)).save(buf, format="JPEG", quality=95, subsampling=0)
    return {"content": [{"type": "image", "data": base64.b64encode(buf.getvalue()).decode(),
                         "mimeType": "image/jpeg"}]}


def _mask_result(mask: "np.ndarray", output_path: str | None = None) -> dict:
    """Mask [1,H,W] in {0,1} -> 8-bit grayscale PNG (255 = selected)."""
    import numpy as np
    from PIL import Image
    hw = (np.asarray(mask)[0] > 0.5).astype(np.uint8) * 255
    pil = Image.fromarray(hw, mode="L")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pil.save(output_path, format="PNG")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return {"content": [{"type": "image", "data": base64.b64encode(buf.getvalue()).decode(),
                         "mimeType": "image/png"}]}


# ---------------------------------------------------------------------------
# Stdio transport (LightCraft's native MCP protocol)
# ---------------------------------------------------------------------------

def serve_stdio(handler: LceHandler) -> None:
    log.info("lce MCP server ready on stdio")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handler.handle(req)
        if resp is not None:
            print(json.dumps(resp), flush=True)


# ---------------------------------------------------------------------------
# HTTP transport (for tools that prefer HTTP over stdio)
# ---------------------------------------------------------------------------

def serve_http(handler: LceHandler, host: str, port: int) -> None:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI(title="lce", version="0.1.0")

    @app.post("/mcp")
    async def mcp(req: Request):
        body = await req.json()
        resp = handler.handle(body)
        return JSONResponse(resp or {})

    @app.get("/health")
    def health():
        resp = handler.handle({"method": "tools/call", "id": 0, "params": {"name": "health", "arguments": {}}})
        if "error" in resp:
            return JSONResponse(resp["error"], status_code=500)
        return JSONResponse(json.loads(resp["result"]["content"][0]["text"]))

    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="lce MCP enhancement server")
    ap.add_argument("--host",   default="127.0.0.1")
    ap.add_argument("--port",   type=int, default=8091)
    ap.add_argument("--device", default="AUTO",
                    help="OpenVINO IR device: CPU | GPU | AUTO (default); ONNX models ignore it")
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                    help="stdio = LightCraft native; http = REST/HTTP")
    ap.add_argument("--allow-dir", action="append", default=[], metavar="DIR",
                    help="http only: directory that path/output_path may use (repeatable)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    from lce.engine import ModelCache
    cache = ModelCache(max_models=4, device=args.device)
    allowed = args.allow_dir if args.transport == "http" else None
    handler = LceHandler(cache, args.device, allowed)

    if args.transport == "http":
        serve_http(handler, args.host, args.port)
    else:
        serve_stdio(handler)


if __name__ == "__main__":
    main()


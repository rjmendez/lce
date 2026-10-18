"""
lce.server — MCP JSON-RPC 2.0 server.

LightCraft connects here as a client and discovers tools:
  enhance   — full auto-pipeline (triage → route → process)
  sharpen   — lens-blur/motion-blur sharpening
  upscale   — 2× or 4× Gigapixel-quality upscaling
  segment   — subject mask (SAM)
  triage    — degradation classification (DA-CLIP)
  health    — server status + loaded model list

Run:  python -m lce.server  [--host 0.0.0.0] [--port 8091] [--device AUTO]
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
                "Classify image degradation via DA-CLIP. "
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
            "description": "Lens-blur and motion-blur sharpening.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_b64": {"type": "string"},
                    "path": {"type": "string"},
                    "strength": {
                        "type": "number",
                        "description": "Blur strength 0–1 (default: auto via DA-CLIP)",
                        "default": -1,
                    },
                },
            },
        },
        {
            "name": "upscale",
            "description": "2× or 4× upscaling.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_b64": {"type": "string"},
                    "path": {"type": "string"},
                    "scale": {"type": "integer", "enum": [2, 4], "default": 2},
                },
            },
        },
        {
            "name": "segment",
            "description": "Subject segmentation mask using SAM. Returns mask as base64 PNG.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_b64": {"type": "string"},
                    "path": {"type": "string"},
                },
            },
        },
        {
            "name": "enhance",
            "description": (
                "Full auto-enhancement: triage → select model → process. "
                "Best-quality single-call operation for batch workflows."
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
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class LceHandler:
    def __init__(self, cache, device: str):
        self._cache = cache
        self._device = device

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
        from lce import models as _m

        if name == "health":
            import openvino as ov
            devs = ov.Core().available_devices
            loaded = list(self._cache._lru.keys())
            return {"content": [{"type": "text", "text": json.dumps({
                "status": "ok",
                "device": self._device,
                "openvino_devices": devs,
                "loaded_models": [Path(p).stem for p in loaded],
            })}]}

        img = _load_arg(args)

        if name == "triage":
            route = _m.triage.route(img, self._cache)
            return _text_result({"route": route})

        if name == "sharpen":
            strength = float(args.get("strength", -1))
            out = _m.sharpen.run(img, self._cache, strength=None if strength < 0 else strength)
            return _image_result(out, args.get("output_path"))

        if name == "upscale":
            scale = int(args.get("scale", 2))
            out = _m.upscale.run(img, self._cache, scale=scale)
            return _image_result(out, args.get("output_path"))

        if name == "segment":
            mask = _m.segment.run(img, self._cache)
            return _image_result(mask, args.get("output_path"), mode="P")

        if name == "enhance":
            mode = args.get("mode", "auto")
            out = _m.enhance.run(img, self._cache, mode=mode)
            return _image_result(out, args.get("output_path"))

        raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_arg(args: dict) -> "np.ndarray":  # noqa: F821
    import numpy as np
    from PIL import Image
    if "image_b64" in args:
        data = base64.b64decode(args["image_b64"])
        img = Image.open(io.BytesIO(data)).convert("RGB")
    elif "path" in args:
        img = Image.open(args["path"]).convert("RGB")
    else:
        raise ValueError("Provide image_b64 or path")
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _text_result(obj: dict) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(obj)}]}


def _image_result(img_chw: "np.ndarray", output_path: str | None = None, mode: str = "RGB") -> dict:
    from PIL import Image
    import numpy as np
    hwc = (img_chw.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(hwc, mode=mode)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pil.save(output_path, quality=97, subsampling=0)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG" if mode == "RGB" else "PNG", quality=97)
    return {"content": [{"type": "image", "data": base64.b64encode(buf.getvalue()).decode(), "mimeType": "image/jpeg"}]}


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
        return handler.handle({"method": "tools/call", "id": 0, "params": {"name": "health", "arguments": {}}})

    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="lce MCP enhancement server")
    ap.add_argument("--host",   default="127.0.0.1")
    ap.add_argument("--port",   type=int, default=8091)
    ap.add_argument("--device", default="AUTO",
                    help="OpenVINO device: CPU | GPU | AUTO (default)")
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                    help="stdio = LightCraft native; http = REST/HTTP")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    from lce.engine import ModelCache
    cache = ModelCache(max_models=4, device=args.device)
    handler = LceHandler(cache, args.device)

    if args.transport == "http":
        serve_http(handler, args.host, args.port)
    else:
        serve_stdio(handler)


if __name__ == "__main__":
    main()

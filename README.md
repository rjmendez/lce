# lce — LightCraft Enhancement Plugin

AI-powered photo enhancement for [LightCraft](https://lightcraft.app) using only open-source model weights.

**Scope: image enhancement only.** Sharpening, upscaling, a segmentation mask, and quality routing — nothing else. No workflow, no metadata, no submission pipelines.

**Capabilities** (what the code actually does)
- 2× / 4× upscaling with HAT ONNX models. The default registered weights are a Sentinel-2 satellite-imagery fine-tune of full HAT (fixed 64×64 input); quality on ordinary photos is not established, and the weights' license is unspecified on their model card. `variant: "stock"` (2× only) switches to a HAT-S + LoRA fine-tune on real stock nature photos instead -- measured on held-out photos to recover real detail (web strands, dewdrop highlights) that both bicubic and the satellite fine-tune miss. It's a **local-only** model: not publicly hosted, no `lce-weights download` fallback -- only present on the machine it was trained/exported on.
- Deblurring with NAFNet (fp32).
- Quality routing by BRISQUE score plus a resolution check (not DA-CLIP). Needs the `triage` extra (torch + piq).
- `segment`: a MobileSAM mask from a uniform 3×3 grid of positive points. It is a generic mask, not a subject detector, and no other tool uses it.
- `lce.models.face` (GFPGAN v1.4, optional pack) restores an already-cropped face. It has no face detection and is not exposed by the MCP server.

Image tools return a JPEG q95 preview over MCP; pass `output_path` to get the full-quality file (EXIF and ICC profile carried over from the input). Upscaling does not by itself meet any marketplace size or quality requirement.

**Architecture**

`lce` runs as a local MCP server. LightCraft connects to it and discovers enhancement tools as native operations — no export/reimport, no manual round-trips.

```
LightCraft ──MCP──▶ lce server ──▶ tiled inference engine ──▶ open model weights
               ◀──────────────────── enhanced image ◀──────────────────────────
```

## Quick start

```bash
pip install "lce[serve,triage]"
lce-weights download          # downloads ~800 MB of open weights (sha256-verified)
lce-serve                     # MCP server on stdio (what LightCraft expects)
lce-serve --transport http --port 8091 --allow-dir D:\photos   # HTTP; path/output_path limited to --allow-dir
```

In LightCraft → Preferences → Plugins → Add MCP Server → point at `lce-serve`.

## Models

Weights are open-source (MIT / Apache-2.0) except the HAT upscalers, whose license is unspecified on their model card.
Run `lce-weights list` to see what's available and their provenance.
Each entry in `src/lce/weights/models.toml` records its source URL, pinned sha256 and license --
except `hat-stock-lora-x2`, a local-only personal fine-tune with a `local_path` instead of a URL:
`lce-weights download` skips it (nothing to fetch), and it's only usable on a machine that already
has the file.

## What this is not

- Not a photo management tool
- Not a metadata / submission pipeline
- Not tied to any specific photography genre or workflow

## License

MIT. Model weights retain their own licenses — see the entries in `src/lce/weights/models.toml`.

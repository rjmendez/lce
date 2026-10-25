# lce — LightCraft Enhancement Plugin

AI-powered photo enhancement for [LightCraft](https://lightcraft.app) using only open-source model weights.

**Scope: image enhancement only.** Sharpening, upscaling, a segmentation mask, and quality routing — nothing else. No workflow, no metadata, no submission pipelines.

**Capabilities** (what the code actually does)
- 2× / 4× upscaling with HAT ONNX models. The 2× default is a HAT-S + LoRA fine-tune on real stock nature photos (**local-only**: not publicly hosted, no `lce-weights download` fallback -- only present on the machine it was trained/exported on), measured on held-out photos to recover real detail (web strands, dewdrop highlights) that both bicubic and the satellite fine-tune miss. The 4× default is still a Sentinel-2 satellite-imagery fine-tune of full HAT (fixed 64×64 input); quality on ordinary photos is not established there, and the weights' license is unspecified on their model card -- no validated real-photo 4× model exists yet (an attempted HAT-S 4× LoRA fine-tune failed its quality gate and was never shipped). `variant: "sentinel"` forces the satellite model explicitly at either scale, independent of the default.
- Deblurring with NAFNet (fp32).
- Quality routing by BRISQUE score plus a resolution check (not DA-CLIP). Needs the `triage` extra (torch + piq).
- `segment`: a MobileSAM mask. Pass `points` (normalised [0,1] coordinates, optionally with `labels` for background points) to prompt it near the actual subject; without `points` it falls back to a uniform 3×3 grid of positive points over the whole image, which is a generic mask, not a subject detector. No other tool uses it.
- `restore_face`: GFPGAN v1.4 face restoration (optional pack — needs `lce-weights get gfpgan-v14`). Manually invoked only, never auto-routed by `enhance`. There is no face detection, so the caller must set `confirm_cropped: true` to acknowledge the image is already cropped to a single face; the server also rejects images whose aspect ratio is far from square (>1.4:1) as a cheap sanity check. Neither `confirm_cropped` nor the aspect check is a real face check — a square crop with no face in it still passes — so the caller is responsible for actually cropping to the face first.

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

# lce — LightCraft Enhancement Plugin

AI-powered photo enhancement for [LightCraft](https://lightcraft.app) using only open-source model weights.

**Scope: image enhancement only.** Sharpening, upscaling, subject segmentation, and quality routing — nothing else. No workflow, no metadata, no submission pipelines.

**Capabilities**
- 2× / 4× upscaling
- Lens-blur and motion-blur sharpening
- Degradation-aware auto-routing via DA-CLIP
- Subject-aware processing with SAM segmentation
- Face enhancement

**Architecture**

`lce` runs as a local MCP server. LightCraft connects to it and discovers enhancement tools as native operations — no export/reimport, no manual round-trips.

```
LightCraft ──MCP──▶ lce server ──▶ tiled inference engine ──▶ open model weights
               ◀──────────────────── enhanced image ◀──────────────────────────
```

## Quick start

```bash
pip install lce[serve]
lce-weights download          # downloads ~600 MB of open weights
lce-serve --port 8091         # starts the MCP server (stdio by default for LightCraft)
```

In LightCraft → Preferences → Plugins → Add MCP Server → point at `lce-serve`.

## Models

All weights are open-source with permissive licenses (Apache 2.0 / MIT / BSD).
Run `lce-weights list` to see what's available and their provenance.
See [docs/models.md](docs/models.md) for details and benchmarks.

## What this is not

- Not a photo management tool
- Not a metadata / submission pipeline
- Not tied to any specific photography genre or workflow

## License

MIT. Model weights retain their own licenses — see [docs/models.md](docs/models.md).

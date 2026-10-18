# lce — LightCraft Enhancement Plugin

AI-powered photo enhancement for [LightCraft](https://lightcraft.app) using only open-source model weights.

**Capabilities**
- 2× / 4× upscaling (Real-ESRGAN quality)
- Lens-blur and motion-blur sharpening (NAFNet / Restormer)
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
lce-serve --port 8091         # starts the MCP server
```

In LightCraft → Preferences → Plugins → Add MCP Server → `http://localhost:8091`

## Models

All weights are open-source with permissive licenses (Apache 2.0 / MIT / BSD).
Run `lce-weights list` to see what's available and their sources.

## License

MIT. See [LICENSE](LICENSE).
Model weights are subject to their own licenses — see [docs/models.md](docs/models.md).

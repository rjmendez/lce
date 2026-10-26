# Examples

Scripts that exercise lce's real, shipped code (`lce.models.*`) against real
photos, to prove what the plugin actually does rather than what the eval
harness predicts. `sam3_zeroshot_mask_demo.py` is the one exception -- it
demonstrates a candidate technology (Meta's SAM3, zero-shot) that lce does
**not** currently use (lce's `segment` tool uses MobileSAM).

**Paths are the original author's personal machine layout.** Several
scripts hardcode things like a stock-photo source directory or a results
output directory (e.g. `F:\...\ready`, `D:\bio-results\...`) -- edit those
constants near the top of each script to point at your own photos and
output location before running them. Nothing here needs those exact paths
to exist; they're just where the author's own examples were generated from.

`_lce_gpu.py` / `_upscale_one.py` are small shared helpers, not standalone
examples on their own -- see their own docstrings for why they exist (a
real, measured GPU-vs-CPU dead end for lce's current ONNX models on the
author's machine, documented there in detail).

`sam3_zeroshot_mask_demo.py` additionally needs a local clone of
[facebookresearch/sam3](https://github.com/facebookresearch/sam3) (not
vendored here -- see that script's own license note) and its pretrained
checkpoint, neither of which this repo distributes.

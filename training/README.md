# Training

How the model weights lce actually ships (`hat-stock-lora-x2`, `hat-stock-x4`)
were produced, plus adjacent research/evaluation work. This is **not**
installed as part of the `lce` package and has its own, much heavier
dependency set (PyTorch, CUDA, training-only libraries) -- none of it is a
`lce` runtime dependency.

**Paths are the original author's personal machine layout** -- weight
cache directories, a stock-photo source directory, AWS S3 bucket names,
local venv locations. Edit the constants near the top of each script
before running them; nothing here needs those exact paths or that exact
AWS account to exist.

- `hat/` -- the HAT-S + LoRA fine-tune pipeline that produced
  `hat-stock-lora-x2` and the plain zero-shot export that produced
  `hat-stock-x4` (see `hat/hat_vendor/NOTICE.md` for HAT's own license/
  provenance). `hat/gallery_*_examples.py` are the original proof-of-concept
  demos for this model, predating (and using a different approach than)
  `examples/gallery_lce_features_*.py`.
- `sr/` -- a broader, earlier crop-upscale/raw-denoise training pipeline
  (TinySR2x, TinyRestore1x, NAFNet) that predates and goes beyond just the
  HAT-S LoRA fine-tune; not all of it maps to a model lce currently ships.
- `sam3_eval/` -- code and findings from comparing Meta's SAM3 against
  Topaz's own subject-mask feature on real photos. Scripts with `topaz` in
  the name depend on a `topaz_subject_mask_port.py` module that is private
  (bio-automation-specific, porting a third-party commercial tool's
  behavior) and was deliberately not brought over here -- those specific
  scripts won't run standalone from this repo. Generated result data
  (JSON pools, hundreds of mask PNGs, tensorboard logs) was deliberately
  left behind as regeneratable, not committed here -- only the code and
  the `REPORT.md`/`ANALYSIS.md` writeups.
- `cloud/` -- the AWS EC2 user-data launch scripts used to run the HAT-S
  LoRA training on a GPU instance.

See also `examples/sam3_zeroshot_mask_demo.py` for a from-scratch zero-shot
SAM3 demo (no fine-tuning, no dependency on the Topaz comparison code here).

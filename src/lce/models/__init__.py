"""
lce.models — open-weight model modules for LCE.

Each public module exposes a single entry point::

    run(img_chw_f32: np.ndarray, cache, **kwargs) -> np.ndarray

``img_chw_f32`` is a float32 ndarray with shape [C, H, W] in the range [0, 1].
``cache`` is a :class:`lce.engine.ModelCache` instance that handles model loading.
The return value is a float32 ndarray of the same rank (shape may differ for upscale).

Modules:
  upscale  -- 2x / 4x super-resolution via HAT (Sentinel-2 fine-tune, license unspecified)
  sharpen  -- motion/lens deblurring via NAFNet fp32 (MIT)
  segment  -- grid-prompted mask via MobileSAM
  triage   -- BRISQUE quality routing; no model file required (MIT)
  enhance  -- auto pipeline: triage -> route -> process
  face     -- GFPGAN v1.4 on a cropped face, optional pack; not exposed by the server

The submodules are not imported here (triage needs optional deps); import them
explicitly, e.g. ``from lce.models import sharpen``.
"""

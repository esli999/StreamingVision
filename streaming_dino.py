"""Shared dense DINOv2-S featurizer for the StreamingVision pipeline.

Factored out of ``render_demo.FeatureWorker`` so the offline calibration / debug
scripts (``sam_calibrate``, ``build_references``, ``diagnose_streaming``) can
compute DINO features at *exactly* the streaming settings without importing the
live demo — which loads the depth + flow models at import time and pins
``XLA_PYTHON_CLIENT_MEM_FRACTION=0.25`` (the offline scripts want 0.85).

The default geometry (``DINO_H x DINO_W`` -> ``DINO_GH x DINO_GW`` patches) is the
"denser featurizer" from Workstream A: DINOv2-S/14 run at 364x644 with position
embeddings interpolated, giving a 26x46 patch grid aspect-matched to the 640x360
working frame instead of the HF processor's 16x16 (224 center-crop).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch

DINO_CKPT = "facebook/dinov2-small"
DINO_PATCH = 14
DINO_H = 364                       # 26 patches tall (364 = 26 * 14)
DINO_W = 644                       # 46 patches wide (644 = 46 * 14)
DINO_GH = DINO_H // DINO_PATCH     # 26
DINO_GW = DINO_W // DINO_PATCH     # 46
N_DINO_PATCHES = DINO_GH * DINO_GW  # 1196
DINO_DIM = 384
# ImageNet normalization (matches the offline GenMatter++ DINO extractor).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dino(device: str = "cuda", dtype=torch.float16):
    """Load DINOv2-S in eval mode on ``device`` (no AutoImageProcessor — we
    normalize manually so we can pick the patch resolution)."""
    from transformers import AutoModel

    return AutoModel.from_pretrained(DINO_CKPT, torch_dtype=dtype).to(device).eval()


def dino_patches(model, rgb: np.ndarray, device: str = "cuda",
                 dino_h: int = DINO_H, dino_w: int = DINO_W,
                 ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Dense DINOv2-S patch features for one RGB frame.

    rgb : (H, W, 3) uint8 RGB.  Resized to ``(dino_h, dino_w)``, ImageNet-
          normalized, run through DINOv2-S with ``interpolate_pos_encoding`` so
          the patch grid is ``(dino_h // 14, dino_w // 14)`` rather than the
          processor's 16x16 224 crop.

    Returns ``(patches (gh*gw, 384) float32, (gh, gw))``.
    """
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    rgb_resized = cv2.resize(rgb, (dino_w, dino_h), interpolation=cv2.INTER_LINEAR)
    model_dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        x = torch.from_numpy(np.ascontiguousarray(rgb_resized)).to(device)
        x = x.permute(2, 0, 1)[None].float() / 255.0
        x = ((x - mean) / std).to(dtype=model_dtype)
        out = model(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[0]
        patches = out[1:].float().cpu().numpy()
    gh, gw = dino_h // DINO_PATCH, dino_w // DINO_PATCH
    return patches, (gh, gw)

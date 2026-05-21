"""SAM2 frame-0 pseudo-color mask export."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class SamFrame0Params:
    model: str = "sam2.1_l.pt"
    min_threshold: float = 0.15
    seed: int = 42
    frame_name: str = "00000.jpg"


@dataclass
class SamFrame0Result:
    output_path: Path
    elapsed_seconds: float
    file_size_bytes: int


def resolve_model_weights(model: str, weights_dir: Path) -> str:
    """Bare filenames use ``weights_dir/<name>``; download if missing."""
    from ultralytics.utils.downloads import attempt_download_asset

    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    p = Path(model).expanduser()
    if p.parent != Path("."):
        if not p.is_file():
            raise FileNotFoundError(f"Model weights not found: {p}")
        return str(p.resolve())
    dest = weights_dir / p.name
    if not dest.is_file():
        attempt_download_asset(str(dest), progress=True)
    return str(dest.resolve())


def combined_mask(masks_tensor, min_threshold: float, seed: int) -> np.ndarray:
    masks = masks_tensor
    n, h, w = int(masks.shape[0]), int(masks.shape[1]), int(masks.shape[2])
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    colors: list[np.ndarray] = []
    for _ in range(n):
        c = rng.integers(0, 255, size=3, dtype=np.int64)
        while np.all(c > 200):
            c = rng.integers(0, 255, size=3, dtype=np.int64)
        colors.append(c.astype(np.uint8))
    prio = np.zeros((h, w), dtype=np.float32)
    for i in range(n):
        m = masks[i].detach().cpu().numpy()
        upd = (m > min_threshold) & (m > prio)
        out[upd] = colors[i]
        prio[upd] = m[upd]
    return out


def run_sam_frame0(
    video_id: str,
    frame_path: Path,
    output_dir: Path,
    weights_dir: Path,
    params: SamFrame0Params,
    model: Optional[object] = None,
) -> SamFrame0Result:
    """Segment frame 0 and write ``{video_id}_SAM_frame0.png``."""
    from ultralytics import SAM

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}_SAM_frame0.png"

    if model is None:
        weights_path = resolve_model_weights(params.model, weights_dir)
        model = SAM(weights_path)

    t0 = time.perf_counter()
    r = model(str(frame_path), verbose=False)[0]
    if r.masks is None or len(r.masks.data) == 0:
        raise RuntimeError(f"SAM produced no masks for {frame_path}")

    rgb = combined_mask(r.masks.data, params.min_threshold, params.seed)
    cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    elapsed = time.perf_counter() - t0

    return SamFrame0Result(
        output_path=output_path,
        elapsed_seconds=elapsed,
        file_size_bytes=output_path.stat().st_size,
    )

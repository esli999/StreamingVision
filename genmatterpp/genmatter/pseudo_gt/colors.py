"""Consistent pseudo-color RGB views of uint16 track label maps."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def build_track_color_palette(max_track_id: int, seed: int = 42) -> dict[int, np.ndarray]:
    """One RGB color per global track id (0 .. max_track_id), fixed across frames."""
    rng = np.random.default_rng(seed)
    palette: dict[int, np.ndarray] = {}
    for tid in range(max_track_id + 1):
        c = rng.integers(0, 255, size=3, dtype=np.int64)
        while np.all(c > 200):
            c = rng.integers(0, 255, size=3, dtype=np.int64)
        palette[tid] = c.astype(np.uint8)
    return palette


def max_track_id_from_label_maps(label_maps: list[np.ndarray]) -> int:
    if not label_maps:
        return -1
    max_label = int(max(int(lab.max()) for lab in label_maps))
    return max(0, max_label - 1)


def label_map_to_rgb(
    label_map: np.ndarray,
    palette: dict[int, np.ndarray],
    *,
    background: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Convert uint16 labels (0=bg, track_id+1=instance) to RGB for visualization.

    Colors are keyed by global ``track_id`` so the same id is the same color in
    every frame when using a shared ``palette``.
    """
    lab = np.asarray(label_map, dtype=np.uint16)
    h, w = lab.shape[:2]
    out = np.full((h, w, 3), background, dtype=np.uint8)
    labels = np.unique(lab)
    for label_val in labels:
        if label_val == 0:
            continue
        tid = int(label_val) - 1
        color = palette.get(tid)
        if color is None:
            continue
        out[lab == label_val] = color
    return out


def colorize_label_dir(
    label_dir: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    pattern: str = "*.png",
) -> int:
    """
    Write RGB previews for all label PNGs in ``label_dir``.

    Returns number of frames written.
    """
    label_dir = Path(label_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(label_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No label PNGs in {label_dir}")

    label_maps = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in paths]
    for arr, p in zip(label_maps, paths, strict=True):
        if arr is None:
            raise FileNotFoundError(f"Could not read {p}")

    max_tid = max(int(lab.max()) for lab in label_maps) - 1
    palette = build_track_color_palette(max(0, max_tid), seed=seed)

    for path, lab in zip(paths, label_maps, strict=True):
        rgb = label_map_to_rgb(lab, palette)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / path.name), bgr)
    return len(paths)

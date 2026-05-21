"""Color utilities for Rerun visualization."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def distinct_palette(n: int, seed: int = 42) -> np.ndarray:
    """Return ``(n, 3)`` uint8 RGB colors as distinct as possible."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    if n <= 20:
        base = np.array(
            [
                [230, 25, 75],
                [60, 180, 75],
                [255, 225, 25],
                [0, 130, 200],
                [245, 130, 48],
                [145, 30, 180],
                [70, 240, 240],
                [240, 50, 230],
                [210, 245, 60],
                [250, 190, 190],
                [0, 128, 128],
                [230, 190, 255],
                [170, 110, 40],
                [255, 250, 200],
                [128, 0, 0],
                [170, 255, 195],
                [128, 128, 0],
                [255, 215, 180],
                [0, 0, 128],
                [128, 128, 128],
            ],
            dtype=np.uint8,
        )
        return base[:n]
    hues = np.linspace(0, 1, n, endpoint=False)
    rng.shuffle(hues)
    colors = np.zeros((n, 3), dtype=np.uint8)
    for i, h in enumerate(hues):
        colors[i] = _hsv_to_rgb(h, 0.75, 0.95)
    return colors


def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (np.array([r, g, b]) * 255).astype(np.uint8)


def features_to_rgb_pca(
    features: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """PCA ``(T, N, F)`` or ``(N, F)`` features to ``(..., 3)`` uint8 RGB."""
    x = np.asarray(features, dtype=np.float32)
    orig_shape = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] < 3 or flat.shape[1] < 3:
        out = np.zeros((*orig_shape, 3), dtype=np.uint8)
        out[..., :] = 128
        return out
    n_comp = min(3, flat.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    proj = pca.fit_transform(flat)
    if n_comp < 3:
        pad = np.zeros((proj.shape[0], 3 - n_comp), dtype=np.float32)
        proj = np.concatenate([proj, pad], axis=1)
    lo = proj.min(axis=0)
    hi = proj.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    rgb = ((proj - lo) / span * 255).clip(0, 255).astype(np.uint8)
    return rgb.reshape(*orig_shape, 3)


def point_colors_from_assignments(
    assignments: np.ndarray,
    n_blobs: int,
    palette: np.ndarray,
) -> np.ndarray:
    """Map per-pixel blob id to RGB; outlier / invalid ids are dark gray."""
    assign = assignments.astype(np.int64)
    colors = np.full((assign.shape[0], 3), 40, dtype=np.uint8)
    valid = (assign >= 0) & (assign < n_blobs)
    colors[valid] = palette[assign[valid] % len(palette)]
    return colors


def point_colors_from_hyperblob_assignments(
    blob_assignments: np.ndarray,
    hyperblob_assignments_per_blob: np.ndarray,
    n_blobs: int,
    n_hyperblobs: int,
    palette: np.ndarray,
) -> np.ndarray:
    """Map each pixel to its blob's hyperblob id; outliers / invalid ids are dark gray."""
    assign = blob_assignments.astype(np.int64)
    hb_per_blob = hyperblob_assignments_per_blob.astype(np.int64)
    colors = np.full((assign.shape[0], 3), 40, dtype=np.uint8)
    valid_blob = (assign >= 0) & (assign < n_blobs)
    if not np.any(valid_blob):
        return colors
    hb_ids = hb_per_blob[assign[valid_blob]]
    valid_hb = (hb_ids >= 0) & (hb_ids < n_hyperblobs)
    pixel_idx = np.where(valid_blob)[0]
    colors[pixel_idx[valid_hb]] = palette[hb_ids[valid_hb] % len(palette)]
    return colors


def blob_mean_rgb_colors(
    assignments: np.ndarray,
    point_colors: np.ndarray,
    n_blobs: int,
) -> np.ndarray:
    """Per-blob mean RGB from assigned datapoints (excludes outlier id ``n_blobs``)."""
    out = np.zeros((n_blobs, 3), dtype=np.uint8)
    for b in range(n_blobs):
        mask = assignments == b
        if np.any(mask):
            out[b] = point_colors[mask].mean(axis=0).astype(np.uint8)
        else:
            out[b] = 64
    return out


def hyperblob_covariances_from_blobs(
    hyperblob_assignments: np.ndarray,
    blob_covs: np.ndarray,
    n_hyperblobs: int,
) -> np.ndarray:
    """Mean blob covariance per hyperblob for ellipsoid display."""
    out = np.zeros((n_hyperblobs, 3, 3), dtype=np.float64)
    counts = np.zeros(n_hyperblobs, dtype=np.int32)
    for b in range(len(hyperblob_assignments)):
        hb = int(hyperblob_assignments[b])
        if 0 <= hb < n_hyperblobs:
            out[hb] += blob_covs[b]
            counts[hb] += 1
    for hb in range(n_hyperblobs):
        if counts[hb] > 0:
            out[hb] /= counts[hb]
        else:
            out[hb] = np.eye(3) * 1e-4
    return out

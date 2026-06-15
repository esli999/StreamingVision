"""Shared label/metric helpers for the self-supervised calibration + debug
scripts (``sam_calibrate``, ``build_references``, ``diagnose_streaming``).

Everything here is featurizer-independent numpy/sklearn; the dense DINO
featurizer lives in ``streaming_dino`` and the tracker glue in ``genmatter_rt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------- SAM / pseudo-GT label maps ----------------

def sam_png_to_instances(path) -> np.ndarray:
    """RGB pseudo-color SAM-frame-0 PNG -> ``(H, W)`` int instance ids.

    White ``(255, 255, 255)`` is background (0); every other unique color gets a
    sequential 1..K id.  Mirrors the RGB->label block in
    ``utils.make_hierarchical_kmeans_chm_with_SAM_segmentations`` but vectorized.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    flat = rgb.reshape(-1, 3)
    unique_colors = np.unique(flat, axis=0)
    out = np.zeros(flat.shape[0], dtype=np.int32)
    label = 0
    for color in unique_colors:
        if np.array_equal(color, [255, 255, 255]):
            continue  # background stays 0
        label += 1
        out[np.all(flat == color[None, :], axis=1)] = label
    return out.reshape(rgb.shape[:2])


def downsample_label_map(label_map: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Nearest-neighbor downsample an int ``(H, W)`` label map to ``(gh, gw)``."""
    return cv2.resize(label_map.astype(np.int32), (gw, gh),
                      interpolation=cv2.INTER_NEAREST).astype(np.int32)


def num_instances(label_map: np.ndarray) -> int:
    """Count of non-background (>0) instance ids in a label map."""
    u = np.unique(label_map)
    return int((u > 0).sum())


# ---------------- Clustering ----------------

def cluster_labels(feats: np.ndarray, k: int, method: str = "kmeans",
                   seed: int = 0) -> np.ndarray:
    """Cluster ``feats`` (N, D) into ~k groups.  ``method='hdbscan'`` falls back
    to KMeans if the optional ``hdbscan`` package is missing."""
    method = (method or "kmeans").lower()
    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore

            min_cluster = max(10, feats.shape[0] // max(k * 4, 1))
            lab = hdbscan.HDBSCAN(min_cluster_size=min_cluster).fit_predict(feats)
            # HDBSCAN labels noise as -1; fold it into one extra cluster id.
            if (lab < 0).any():
                lab = np.where(lab < 0, lab.max() + 1, lab)
            return lab.astype(np.int32)
        except Exception:
            pass  # fall through to KMeans
    from sklearn.cluster import KMeans

    k = max(2, int(k))
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(feats).astype(np.int32)


# ---------------- Feature-variance estimates (Step 2 diagnostic) ----------------

def within_label_variance(feats: np.ndarray, labels: np.ndarray) -> float:
    """Mean per-dim squared residual of ``feats`` around their per-label mean —
    the MLE within-cluster feature variance (a ``sigma_F`` estimate).  Numpy
    mirror of ``genmatter.tracking.dino.estimate_feature_sigmas_from_chm``."""
    labels = np.asarray(labels)
    uniq, idx = np.unique(labels, return_inverse=True)
    K, D = len(uniq), feats.shape[1]
    sums = np.zeros((K, D), dtype=np.float64)
    counts = np.zeros((K, 1), dtype=np.float64)
    np.add.at(sums, idx, feats)
    np.add.at(counts, idx, 1.0)
    means = sums / np.maximum(counts, 1.0)
    resid = feats - means[idx]
    return float(np.mean(resid ** 2))


# ---------------- Agreement metrics ----------------

def agreement(labels_a: np.ndarray, labels_b: np.ndarray,
              mask: Optional[np.ndarray] = None) -> dict:
    """ARI + NMI between two label vectors (optionally over a boolean mask)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    a = np.asarray(labels_a).reshape(-1)
    b = np.asarray(labels_b).reshape(-1)
    if mask is not None:
        mask = np.asarray(mask).reshape(-1)
        a, b = a[mask], b[mask]
    if a.size == 0:
        return {"ari": float("nan"), "nmi": float("nan"), "n": 0}
    return {
        "ari": float(adjusted_rand_score(a, b)),
        "nmi": float(normalized_mutual_info_score(a, b)),
        "n": int(a.size),
    }


def safe_silhouette(feats: np.ndarray, labels: np.ndarray) -> float:
    """silhouette_score guarded against the degenerate (1 cluster / all-distinct)
    cases that raise in sklearn."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    n_lab = len(np.unique(labels))
    if n_lab < 2 or n_lab >= len(labels):
        return float("nan")
    try:
        return float(silhouette_score(feats, labels))
    except Exception:
        return float("nan")


# ---------------- Visualization ----------------

def make_palette(n: int, seed: int = 0) -> np.ndarray:
    """Visually-distinct random BGR palette (matches genmatter_viz._build_palette)."""
    rng = np.random.default_rng(seed)
    hsv = np.zeros((max(n, 1), 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = rng.integers(0, 180, max(n, 1)).astype(np.uint8)
    hsv[:, 0, 1] = rng.integers(140, 255, max(n, 1)).astype(np.uint8)
    hsv[:, 0, 2] = rng.integers(140, 255, max(n, 1)).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(max(n, 1), 3)


def colorize_labels(label_grid: np.ndarray, palette: np.ndarray,
                    bg_id: Optional[int] = 0,
                    bg_color: Tuple[int, int, int] = (40, 40, 40)) -> np.ndarray:
    """Map an int ``(gh, gw)`` label grid to a BGR image via ``palette`` (mod
    len).  Cells equal to ``bg_id`` (if not None) are painted ``bg_color``."""
    g = np.asarray(label_grid)
    out = palette[np.mod(g, palette.shape[0])].astype(np.uint8)
    if bg_id is not None:
        out[g == bg_id] = np.array(bg_color, dtype=np.uint8)
    return out


def upscale_grid_bgr(grid_bgr: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest-upscale a small BGR label image to ``out_hw`` for montages."""
    h, w = out_hw
    return cv2.resize(grid_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

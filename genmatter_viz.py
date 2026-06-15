"""Visualization library for the StreamingVision GenMatter demo.

This is the RENDERER half of the pipeline; the tracker lives in
``genmatter_rt.py``.  Every function here draws the tracker's REAL state
(blob means/covariances, hyperblob clusters, per-datapoint assignments) — it
never re-fits or mutates anything, so swapping or tuning the visualization can
never change tracking results.  The dependency is strictly one-directional:
``genmatter_viz`` imports a handful of constants + ``_depth_to_Z`` from
``genmatter_rt``; ``genmatter_rt`` never imports ``genmatter_viz`` (the only
exception is a local import inside its ``__main__`` render smoke-test).

Palette-size invariant: ``compute_blob_color_lut`` and
``render_matter_label_grids`` default the blob-id range to the 128-entry
``BLOB_PALETTE``; callers that run with more blobs MUST pass ``num_blobs`` so
ids >= 128 are not silently relabeled as blob 127 (a one-colour smear).
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional, Tuple

from genmatter_rt import (
    DEFAULT_INTRINSICS, Intrinsics, NUM_BLOBS, NUM_HYPERBLOBS, N_KEEP, STRIDE, _depth_to_Z,
)

# ---------------- 64-entry BGR palette ----------------

def _build_palette(n: int = NUM_BLOBS, seed: int = 0) -> np.ndarray:
    """Visually-distinct random BGR palette, matches GenMatter++ DAVIS viz style
    (random per-blob color)."""
    rng = np.random.default_rng(seed)
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = (rng.integers(0, 180, n)).astype(np.uint8)
    hsv[:, 0, 1] = rng.integers(140, 255, n).astype(np.uint8)
    hsv[:, 0, 2] = rng.integers(140, 255, n).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(n, 3)
    return bgr


# Sized to the larger calibrated blob budget (128) so the matter-tile renders
# distinct colors for every blob under the stronger inference strategy; the live
# demo at NUM_BLOBS=64 just uses the first 64 entries. render_matter_tile reads
# the length dynamically, so this is safe to enlarge.
BLOB_PALETTE = _build_palette(max(NUM_BLOBS, 128), seed=0)
# Sized generously (not just NUM_HYPERBLOBS=4) so the SAM-frame-0 semantic-init
# path — which yields one hyperblob per SAM instance plus k-means hyperblobs for
# unsegmented regions, typically 15-40 total — still renders distinct colors.
HYPERBLOB_PALETTE = _build_palette(max(NUM_HYPERBLOBS, 64), seed=42)
# Hyperblob 0 is NOT background under the semantic-seed init: when
# make_hierarchical_kmeans_chm_with_SAM_segmentations seeds clusters it assigns
# the segmented OBJECT instances first (hyperblob_idx 0, 1, ...) and the
# background k-means hyperblobs AFTER them, so id 0 is typically the primary
# tracked object. A muted-gray id 0 would paint that tracked object gray and bury
# it in the background, so we give id 0 a vivid, saturated color instead (so the
# seeded object visually pops).
HYPERBLOB_PALETTE[0] = np.array([60, 220, 60], dtype=np.uint8)   # vivid green (BGR)


def _build_vivid_fg_colors(n: int) -> np.ndarray:
    """``n`` maximally-distinct VIVID/saturated BGR colors for the foreground
    (seeded-object) hyperblobs in the CLUSTER tiles.

    Hue is spread on the golden-ratio sequence (so even a handful of instances
    land far apart on the wheel) at full saturation/value, so each tracked
    object reads as a punchy, high-chroma color that pops against the muted
    background. Id 0 is pinned to the canonical vivid green so the primary
    tracked object keeps a consistent identity color.
    """
    if n <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    # Golden-angle hue walk in OpenCV's 0..179 hue space, max S/V.
    hues = (np.arange(n) * (179.0 * 0.61803398875)) % 180.0
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = hues.astype(np.uint8)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(n, 3)
    bgr[0] = (60, 220, 60)                           # keep the canonical vivid green for id 0
    return bgr


def _build_muted_bg_colors(n: int, seed: int = 42) -> np.ndarray:
    """``n`` DESATURATED, low-contrast muted BGR tones for the background
    (k-means) hyperblobs in the CLUSTER tiles.

    Low saturation + mid-low value keeps the many background clusters legible
    as distinct regions WITHOUT competing with the vivid foreground — they read
    as a quiet palette of grays / earth tones so the eye snaps to the object.
    """
    if n <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = rng.integers(0, 180, n).astype(np.uint8)      # any hue ...
    hsv[:, 0, 1] = rng.integers(18, 55, n).astype(np.uint8)      # ... but barely saturated
    hsv[:, 0, 2] = rng.integers(95, 165, n).astype(np.uint8)     # mid-low value (muted)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(n, 3)


def build_cluster_palette(num_fg: int, n_total: int = None) -> np.ndarray:
    """CLUSTER-tile palette: VIVID foreground (ids ``< num_fg``) + MUTED
    background (ids ``>= num_fg``), so the seeded object POPS against a
    desaturated background.

    ``num_fg`` is the number of seeded OBJECT-instance hyperblobs — under
    ``make_hierarchical_kmeans_chm_with_SAM_segmentations`` these are exactly
    the low ids ``0 .. num_fg-1`` (the segmented instances are assigned first,
    background k-means hyperblobs after). Sized to ``n_total`` (defaults to the
    rainbow ``HYPERBLOB_PALETTE`` length) so the ``palette[label]`` lookup in
    ``render_matter_tile`` never indexes out of range.
    """
    if n_total is None:
        n_total = HYPERBLOB_PALETTE.shape[0]
    num_fg = int(max(0, min(num_fg, n_total)))
    pal = np.empty((n_total, 3), dtype=np.uint8)
    pal[:num_fg] = _build_vivid_fg_colors(num_fg)
    pal[num_fg:] = _build_muted_bg_colors(n_total - num_fg)
    return pal


# Sentinel index marking outlier datapoints in the rendered palette.  The
# Gibbs sampler uses index == num_blobs for outliers; we clip those before
# the palette lookup and color them with this entry instead.
# Outliers (Gibbs couldn't assign a datapoint to any cluster) get a SOLID, light,
# neutral gray — perceptually distinct from the saturated cluster palettes and
# clearly readable as "unassigned", instead of a muddy darkened tint.
OUTLIER_BGR = np.array([205, 205, 210], dtype=np.uint8)


# ---------------- Visualization ----------------

def project_3d_to_2d(means_Nx3: np.ndarray, intr: Intrinsics = DEFAULT_INTRINSICS,
                      ) -> np.ndarray:
    """Perspective project ``(N, 3)`` 3D means in camera frame to ``(N, 2)``
    pixel coordinates ``(u, v)``.  Points with Z <= 0 are projected to NaN."""
    X = means_Nx3[:, 0]
    Y = means_Nx3[:, 1]
    Z = means_Nx3[:, 2]
    safe_Z = np.where(Z > 1e-3, Z, np.nan)
    u = intr.fx * X / safe_Z + intr.cx
    v = intr.fy * Y / safe_Z + intr.cy
    return np.stack([u, v], axis=1)


def covariance_to_ellipse_2d(cov_3x3: np.ndarray, mean_3: np.ndarray,
                              intr: Intrinsics = DEFAULT_INTRINSICS,
                              sigma_scale: float = 1.5,
                              max_half: float = 120.0) -> Tuple[Tuple[float, float], float]:
    """Approximate 2D image-plane ellipse for a 3D Gaussian, using first-order
    perspective linearization at ``mean_3``.  Returns ``((axis_a, axis_b),
    angle_deg)`` suitable for ``cv2.ellipse``.

    The Jacobian of the perspective projection x = (fx X + cx Z) / Z,
    y = (fy Y + cy Z) / Z evaluated at the mean is:

        J = [[fx/Z,  0,   -fx X / Z^2],
             [ 0,  fy/Z,  -fy Y / Z^2]]

    Then 2D cov = J Σ_3 J.T; we take its eigenvalues and clip the axes so the
    ellipse stays within the tile.  This is only accurate when Σ is small
    relative to Z (true for blobs, less true for whole-scene hyperblobs).
    """
    X, Y, Z = float(mean_3[0]), float(mean_3[1]), float(mean_3[2])
    if Z <= 1e-3:
        return (1.0, 1.0), 0.0
    Z2 = Z * Z
    J = np.array(
        [[intr.fx / Z, 0.0, -intr.fx * X / Z2],
         [0.0, intr.fy / Z, -intr.fy * Y / Z2]],
        dtype=np.float64,
    )
    cov2d = J @ cov_3x3.astype(np.float64) @ J.T
    cov2d = 0.5 * (cov2d + cov2d.T)
    eigvals, eigvecs = np.linalg.eigh(cov2d)
    eigvals = np.clip(eigvals, 1e-6, None)
    half_a, half_b = sigma_scale * np.sqrt(eigvals[1]), sigma_scale * np.sqrt(eigvals[0])
    half_a = float(min(half_a, max_half))
    half_b = float(min(half_b, max_half))
    # Principal axis (largest eigenvalue) is the second column after eigh's ascending sort.
    angle_rad = float(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    return (max(half_a, 1.0), max(half_b, 1.0)), float(np.degrees(angle_rad))


def render_centroid_tile(means_3d: np.ndarray, covs_3d: np.ndarray,
                          palette: np.ndarray,
                          base_bgr: np.ndarray,
                          intr: Intrinsics = DEFAULT_INTRINSICS,
                          sigma_scale: float = 1.5,
                          alpha: float = 0.55) -> np.ndarray:
    """Render blob (or hyperblob) Gaussian ellipses overlaid on a darkened
    copy of the RGB frame.

    means_3d  : (N, 3) blob means in camera coords.
    covs_3d   : (N, 3, 3) blob covariance matrices.
    palette   : (N, 3) BGR uint8 — one color per Gaussian.
    base_bgr  : (H, W, 3) BGR uint8 — the source RGB frame to overlay onto.
    """
    H, W = base_bgr.shape[:2]
    # Darken the base so colored ellipses pop.
    canvas = (base_bgr.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
    uv = project_3d_to_2d(means_3d, intr)
    overlay = np.zeros_like(canvas)
    mask = np.zeros((H, W), dtype=np.uint8)
    n = means_3d.shape[0]
    for i in range(n):
        u, v = uv[i]
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        ui, vi = int(round(u)), int(round(v))
        if ui < -200 or ui > W + 200 or vi < -200 or vi > H + 200:
            continue
        axes, angle_deg = covariance_to_ellipse_2d(
            covs_3d[i], means_3d[i], intr, sigma_scale=sigma_scale,
            max_half=min(W, H) * 0.45,
        )
        color = (int(palette[i, 0]), int(palette[i, 1]), int(palette[i, 2]))
        cv2.ellipse(overlay, (ui, vi), (int(axes[0]), int(axes[1])),
                    angle_deg, 0, 360, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.ellipse(mask, (ui, vi), (int(axes[0]), int(axes[1])),
                    angle_deg, 0, 360, 255, thickness=-1, lineType=cv2.LINE_AA)
        # Also draw a 2px outline for definition when ellipses overlap.
        cv2.ellipse(canvas, (ui, vi), (int(axes[0]), int(axes[1])),
                    angle_deg, 0, 360, color, thickness=2, lineType=cv2.LINE_AA)
    m = (mask[..., None].astype(np.float32) / 255.0) * alpha
    blended = (canvas.astype(np.float32) * (1 - m) + overlay.astype(np.float32) * m)
    return blended.clip(0, 255).astype(np.uint8)


def hyperblob_palette_per_blob(state) -> np.ndarray:
    """For each blob, return the hyperblob color (so blob ellipses can be
    drawn in hyperblob colors if desired)."""
    hb = np.asarray(state.blobs_state.hyperblob_assignments)
    return HYPERBLOB_PALETTE[np.clip(hb, 0, HYPERBLOB_PALETTE.shape[0] - 1)]


def _nn_fill_grid(grid: np.ndarray) -> np.ndarray:
    """Fill the ``< 0`` (unknown) cells of an int ``(gh, gw)`` grid with the
    value of their nearest known cell, via cv2's labelled distance transform.
    All-unknown grids fall back to zeros."""
    unknown = (grid < 0)
    if not unknown.any():
        return grid
    if (~unknown).sum() == 0:
        return np.zeros_like(grid)
    # cv2.distanceTransformWithLabels: pass a mask where the unknown cells are
    # 255 (positive) and known cells are 0.  It returns, for each cell, the
    # 1-based ordinal label of the nearest 0-cell in raster-scan order over the
    # known cells.
    known_mask = (~unknown).astype(np.uint8) * 255
    _, labels = cv2.distanceTransformWithLabels(
        255 - known_mask, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    known_vals = grid[~unknown]
    out = grid.copy()
    out[unknown] = known_vals[labels[unknown] - 1]
    return out


def _edge_aware_upsample(label_grid: np.ndarray, h: int, w: int,
                         rgb_guide: np.ndarray) -> np.ndarray:
    """Edge-aware upsample of a coarse ``(gh, gw)`` integer ``label_grid`` to
    ``(h, w)``, snapping label boundaries to the edges of ``rgb_guide`` WITHOUT
    changing any underlying assignment (output labels are always one of the
    coarse input labels — this is a relabel-free joint upsample).

    Cheap joint/guided nearest-neighbour: each coarse cell carries a
    representative color (the area-downsampled guide). For every full-res pixel
    we consider the 2x2 block of coarse cells straddling it and pick the cell
    whose representative color is closest (squared L2) to that pixel's actual
    guide color. The label boundary therefore lands on the RGB edge between two
    cells instead of the blocky cell border — the stride-8 ``cv2`` nearest
    upsample's coarse staircase becomes a crisp, image-aligned contour.

    All gathers are C-speed ``cv2.resize`` (INTER_NEAREST) of the coarse grids —
    the per-cell color map and the four 2x2 neighbour candidates are built by
    NEAREST-upsampling rolled copies of the coarse arrays, and the color test
    runs on a single luma channel — so the whole pass is a few ms at 640x360
    (no slow ``np.ix_`` fancy-index gathers, no per-pixel Python loop).
    """
    gh, gw = label_grid.shape
    lab_c = label_grid.astype(np.int32)
    cell_rgb = cv2.resize(rgb_guide, (gw, gh), interpolation=cv2.INTER_AREA)
    # Single luma channel for the color test (cuts the per-candidate distance to
    # 1 channel; boundaries snap on intensity edges, which is what reads as the
    # cluster contour). int16 to hold signed differences.
    cell_y = (0.114 * cell_rgb[..., 0] + 0.587 * cell_rgb[..., 1]
              + 0.299 * cell_rgb[..., 2]).astype(np.int16)
    guide = rgb_guide if rgb_guide.shape[:2] == (h, w) else \
        cv2.resize(rgb_guide, (w, h), interpolation=cv2.INTER_AREA)
    guide_y = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY).astype(np.int16)  # C-fast luma

    def _up(arr):
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)

    best_lab = None
    best_d = None
    # 2x2 neighbour candidates: the coarse grid and its ±1-cell rolls. Each
    # NEAREST-upsamples to full res for ~0.04 ms; we then pick, per pixel, the
    # candidate whose cell luma is closest to the pixel's luma.
    for dy in (0, 1):
        for dx in (0, 1):
            lab_s = np.roll(np.roll(lab_c, -dy, axis=0), -dx, axis=1)
            celly_s = np.roll(np.roll(cell_y, -dy, axis=0), -dx, axis=1)
            cand_lab = _up(lab_s)
            cand_y = _up(celly_s).astype(np.int16)
            d = np.abs(guide_y - cand_y)                       # luma distance (h, w)
            if best_lab is None:
                best_lab, best_d = cand_lab, d
            else:
                take = d < best_d
                best_lab = np.where(take, cand_lab, best_lab)
                best_d = np.where(take, d, best_d)
    return best_lab.astype(np.int32)


def labels_to_filled_grid(labels: np.ndarray, indices: np.ndarray,
                          gh: int, gw: int, drop_outliers: bool = True) -> np.ndarray:
    """Sparse -> dense label adapter (shares ``render_matter_tile``'s NN-fill).

    Scatter per-datapoint integer ``labels`` (``-1`` = outlier) onto the
    ``(gh, gw)`` grid at ``indices``, then NN-fill the unset / outlier cells
    from the nearest non-outlier neighbour.  Returns an int ``(gh, gw)`` label
    grid (labels, not BGR) so the debug harness can score it against reference
    grids without re-deriving the fill.
    """
    grid = np.full((gh * gw,), -1, dtype=np.int32)
    lab = np.asarray(labels)
    keep = (lab >= 0) if drop_outliers else np.ones(lab.shape[0], dtype=bool)
    grid[np.asarray(indices)[keep]] = lab[keep].astype(np.int32)
    return _nn_fill_grid(grid.reshape(gh, gw))


_CLUSTER_PALETTE_CACHE: dict = {}


def _cluster_palette_for(num_fg: int, n_total: int) -> np.ndarray:
    """Memoized vivid-fg / muted-bg CLUSTER palette (``build_cluster_palette``)
    keyed on ``(num_fg, n_total)`` so render_matter_tile doesn't rebuild it
    every frame (deterministic, so caching is bit-exact)."""
    key = (int(num_fg), int(n_total))
    pal = _CLUSTER_PALETTE_CACHE.get(key)
    if pal is None:
        pal = build_cluster_palette(num_fg, n_total)
        _CLUSTER_PALETTE_CACHE[key] = pal
    return pal


def _avg_rgb_by_label(label_map: np.ndarray, rgb_guide: np.ndarray,
                      h: int, w: int) -> np.ndarray:
    """Paint each pixel with the MEAN BGR of its label's region — i.e. the
    average colour of the particle assigned to it. ``label_map`` is the (h, w)
    integer label image; ``rgb_guide`` is the source BGR frame (resized to
    (h, w) if needed). Vectorized per-label mean via bincount."""
    guide = rgb_guide if rgb_guide.shape[:2] == (h, w) else \
        cv2.resize(rgb_guide, (w, h), interpolation=cv2.INTER_AREA)
    lab = np.asarray(label_map).reshape(-1).astype(np.int64)
    rgb = guide.reshape(-1, 3).astype(np.float32)
    n = int(lab.max()) + 1 if lab.size else 1
    counts = np.maximum(np.bincount(lab, minlength=n).astype(np.float32), 1.0)
    means = np.empty((n, 3), dtype=np.float32)
    for c in range(3):
        means[:, c] = np.bincount(lab, weights=rgb[:, c], minlength=n) / counts
    return means[np.asarray(label_map)].clip(0, 255).astype(np.uint8)


def compute_blob_color_lut(blob_assignments: np.ndarray, indices: np.ndarray,
                           rgb_guide: np.ndarray, h: int = 360, w: int = 640,
                           stride: int = STRIDE,
                           num_blobs: Optional[int] = None) -> np.ndarray:
    """Per-blob mean BGR at this frame's datapoints — the LOCKED 'colour of each
    particle'. Compute it ONCE (frame 1) and thread it back into
    ``render_matter_tile(blob_color_lut=...)`` so each particle keeps a fixed
    colour across frames (no per-frame colour drift / flicker), even as it moves.

    Returns a ``(num_blobs, 3)`` uint8 LUT indexed by blob id (``num_blobs``
    defaults to ``BLOB_PALETTE.shape[0]`` — bit-exact for existing callers; PASS
    it when running MORE blobs than the palette, otherwise ids past the LUT clip
    into the last bin and every extra particle inherits one smeared colour).
    Blobs with no datapoints this frame fall back to the global mean colour. The
    grid cell each datapoint occupies (``indices``) is coloured from
    ``rgb_guide`` downsampled to the stride grid."""
    gh, gw = h // stride, w // stride
    guide = rgb_guide if rgb_guide.shape[:2] == (h, w) else \
        cv2.resize(rgb_guide, (w, h), interpolation=cv2.INTER_AREA)
    cell_rgb = cv2.resize(guide, (gw, gh), interpolation=cv2.INTER_AREA
                          ).reshape(-1, 3).astype(np.float32)
    lab = np.asarray(blob_assignments).reshape(-1)
    idx = np.asarray(indices).reshape(-1)
    n = BLOB_PALETTE.shape[0] if num_blobs is None else max(int(num_blobs), 1)
    lut = np.zeros((n, 3), dtype=np.float32)
    valid = lab >= 0
    if not valid.any():
        return lut.astype(np.uint8)
    lab_v = np.clip(lab[valid], 0, n - 1).astype(np.int64)
    rgb_v = cell_rgb[idx[valid]]
    counts = np.bincount(lab_v, minlength=n).astype(np.float32)
    for c in range(3):
        lut[:, c] = np.bincount(lab_v, weights=rgb_v[:, c], minlength=n)
    nz = counts > 0
    lut[nz] /= counts[nz, None]
    lut[~nz] = rgb_v.mean(axis=0)
    return lut.clip(0, 255).astype(np.uint8)


def compute_blob_motion_lut(blob_vel_means: np.ndarray,
                            ref_mag: Optional[float] = None,
                            clip_pct: float = 95.0,
                            subtract_median: bool = True,
                            ) -> Tuple[np.ndarray, float]:
    """Per-blob MOTION colour (Middlebury optical-flow-wheel convention): each
    particle's average 3-D velocity ``blob_vel_means[i]`` -> a BGR colour whose
    HUE encodes the in-image-plane motion DIRECTION and whose SATURATION encodes
    the motion MAGNITUDE. Still particles read near-WHITE (S~0); differentially
    moving particles read as a vivid direction-coloured hue — so the tile is a
    motion segmentation: which bits of matter move, and which way.

    ``blob_vel_means`` is ``(L, 3)`` (the per-blob velocity off
    ``state.blobs_state.blob_vel_means``). Returns ``((L, 3) uint8 BGR LUT,
    frame_ref)`` where ``frame_ref`` is THIS frame's ``clip_pct`` percentile of
    (relative) speed.

    ``subtract_median`` (default True) removes the robust median velocity of the
    moving blobs first — EGO-MOTION COMPENSATION. Without it a panning camera
    makes EVERY particle (background included) share one saturated hue (the whole
    tile reads as a single colour); subtracting the dominant background velocity
    sends the background to ~white and lets the object's RELATIVE motion pop. The
    background is assumed to be the majority of moving blobs (true for DAVIS-style
    single-object clips); set False for the raw absolute-velocity wheel.

    ``ref_mag`` is the (relative) speed that maps to full saturation. Pass a value
    (e.g. an EMA of ``frame_ref`` frozen after the first moving frame) so the
    colours are STABLE / comparable across frames — a particle that speeds up gets
    more saturated rather than the whole frame re-normalising. When None (or ~0)
    the per-frame ``frame_ref`` is used (self-normalised), which is the right
    behaviour on frame 0 / loop seams where flow is zero and everything is white.
    """
    v = np.asarray(blob_vel_means, dtype=np.float32)
    if v.ndim != 2 or v.shape[1] < 2:
        return np.full((v.shape[0], 3), 255, np.uint8), 0.0
    L = v.shape[0]
    active = np.linalg.norm(v, axis=1) > 1e-6
    if subtract_median and int(active.sum()) >= 8:
        v = v - np.median(v[active], axis=0)             # ego-motion compensation
    mag = np.linalg.norm(v, axis=1)                       # relative 3-D speed
    nz = mag[mag > 1e-6]
    frame_ref = float(np.percentile(nz, clip_pct)) if nz.size else 0.0
    scale = float(ref_mag) if (ref_mag is not None and ref_mag > 1e-9) else max(frame_ref, 1e-6)
    ang = np.arctan2(v[:, 1], v[:, 0])                    # image-plane direction
    hue = ((ang + np.pi) / (2.0 * np.pi) * 179.0).astype(np.float32)      # OpenCV H in [0,179]
    sat = np.clip(mag / scale, 0.0, 1.0) * 255.0
    hsv = np.stack([hue, sat, np.full(L, 255.0, np.float32)], axis=1
                   ).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]
    return bgr.astype(np.uint8), frame_ref


def compute_blob_feature_lut(blob_features: np.ndarray,
                             basis: Optional[dict] = None,
                             ) -> Tuple[np.ndarray, dict]:
    """Per-blob FEATURE colour: PCA-project each particle's mean DINO feature
    ``blob_features[i]`` (``(L, D)`` off ``state.blobs_state.blob_features``) to a
    BGR colour, so particles with similar APPEARANCE get similar colours — an
    appearance/semantic colouring of the matter (does the cluster's particles
    share a look?). Returns ``((L, 3) uint8 BGR LUT, basis)``.

    ``basis`` FREEZES the PCA-to-RGB mapping (components + per-axis min/max) at
    the first call, so colours stay STABLE across frames; a fresh per-frame PCA
    would flip component signs / reorder axes and flicker. Pass the returned
    basis back in on every later frame (mirrors the frozen pc_focal trick). Uses
    the same project->minmax->255 recipe as ``viz.colors.features_to_rgb_pca``,
    then flips RGB->BGR for the BGR render pipeline.
    """
    f = np.asarray(blob_features, dtype=np.float32)
    if f.ndim != 2 or f.shape[0] < 3 or f.shape[1] < 3:
        return np.full((f.shape[0], 3), 128, np.uint8), (basis or {"degenerate": True})
    if basis is None or basis.get("degenerate"):
        from sklearn.decomposition import PCA
        n_comp = min(3, f.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        proj = pca.fit_transform(f)
        if n_comp < 3:
            proj = np.concatenate([proj, np.zeros((proj.shape[0], 3 - n_comp), np.float32)], axis=1)
        lo = proj.min(axis=0)
        hi = proj.max(axis=0)
        basis = {"pca": pca, "lo": lo, "hi": hi, "n_comp": n_comp}
    else:
        proj = basis["pca"].transform(f)
        if basis["n_comp"] < 3:
            proj = np.concatenate([proj, np.zeros((proj.shape[0], 3 - basis["n_comp"]), np.float32)], axis=1)
        lo, hi = basis["lo"], basis["hi"]
    span = np.maximum(hi - lo, 1e-8)
    rgb = ((proj - lo) / span * 255.0).clip(0, 255).astype(np.uint8)     # (L, 3) RGB
    return np.ascontiguousarray(rgb[:, ::-1]), basis                     # -> BGR


def render_blob_gaussian_tile(blob_means: np.ndarray, blob_covs: np.ndarray,
                              blob_assignments: np.ndarray, color_lut: np.ndarray,
                              intr: Intrinsics = DEFAULT_INTRINSICS,
                              h: int = 360, w: int = 640, *,
                              base_bgr: Optional[np.ndarray] = None, dim: float = 0.32,
                              ksigma: float = 2.5, thickness: int = 1,
                              fill_alpha: float = 0.0, floor: float = 1.0,
                              max_half: Optional[float] = None) -> np.ndarray:
    """Render the tracker's ACTUAL particle Gaussians as 2-D image-plane ellipses,
    colored by ``color_lut`` (per-blob average BGR, e.g. ``compute_blob_color_lut``).

    FAITHFUL by construction: it draws the model's OWN latent ``blob_means`` (L,3)
    and ``blob_covs`` (L,3,3) — NOT a per-frame re-fit of the datapoints — so the
    visualization cannot drift away from what the tracker believes. Each occupied
    blob (>=1 assigned datapoint) is projected with the ROBUST marginal in-plane
    covariance at the mean depth:

        u,v   = (f*X/Z + cx, f*Y/Z + cy)
        Cov_px = (f/Z)^2 * blob_covs[:2,:2]            (+ floor*I)

    Dropping the dU/dZ depth-coupling term of the full perspective Jacobian (which
    ``covariance_to_ellipse_2d`` keeps) is what removes the brittleness — large
    depth variance no longer blows the ellipse up. Ellipses are drawn largest-first
    over a dimmed copy of ``base_bgr`` (so the scene shows faintly under the
    particles), as crisp outlines (``thickness``) with an optional faint ``fill_alpha``.

    Returns a (h, w, 3) uint8 BGR tile. Pure/additive — no tracker state is mutated
    and ``render_matter_tile`` is untouched.
    """
    if max_half is None:
        max_half = 0.33 * max(h, w)
    if base_bgr is None:
        canvas = np.full((h, w, 3), (18, 18, 26), np.uint8)
    else:
        bg = base_bgr if base_bgr.shape[:2] == (h, w) else cv2.resize(base_bgr, (w, h))
        canvas = (bg.astype(np.float32) * dim).clip(0, 255).astype(np.uint8)
    bm = np.asarray(blob_means, dtype=np.float64)
    bc = np.asarray(blob_covs, dtype=np.float64)
    lut = np.asarray(color_lut)
    ba = np.asarray(blob_assignments)
    f, cx, cy = float(intr.fx), float(intr.cx), float(intr.cy)
    counts = np.bincount(ba[ba >= 0], minlength=bm.shape[0])
    ell = []  # (area, center, axes, angle, color)
    for b in np.where(counts > 0)[0]:
        X, Y, Z = float(bm[b, 0]), float(bm[b, 1]), float(bm[b, 2])
        if Z <= 1e-3:
            continue
        u, v = f * X / Z + cx, f * Y / Z + cy
        cov2 = (f / Z) ** 2 * bc[b, :2, :2] + np.eye(2) * floor
        evals, evecs = np.linalg.eigh(cov2)
        evals = np.clip(evals, 1e-3, None)
        maj = min(ksigma * float(np.sqrt(evals[1])), max_half)
        mnr = min(ksigma * float(np.sqrt(evals[0])), max_half)
        ang = float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])))
        color = tuple(int(c) for c in lut[b]) if b < lut.shape[0] else (200, 200, 200)
        ell.append((maj * mnr, (int(round(u)), int(round(v))),
                    (max(1, int(maj)), max(1, int(mnr))), ang, color))
    ell.sort(key=lambda e: -e[0])                     # largest first (small on top)
    if fill_alpha > 0.0:
        overlay = canvas.copy()
        for _, ctr, axes, ang, color in ell:
            cv2.ellipse(overlay, ctr, axes, ang, 0, 360, color, -1, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, fill_alpha, canvas, 1.0 - fill_alpha, 0.0)
    for _, ctr, axes, ang, color in ell:
        cv2.ellipse(canvas, ctr, axes, ang, 0, 360, color, thickness, cv2.LINE_AA)
    return canvas


def render_matter_label_grids(blob_assignments: np.ndarray,
                              hyperblob_per_dp: np.ndarray,
                              indices: np.ndarray,
                              h: int = 360, w: int = 640,
                              stride: int = STRIDE,
                              rgb_guide: Optional[np.ndarray] = None,
                              num_blobs: Optional[int] = None,
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the upsampled per-pixel LABEL grids that ``render_matter_tile``
    colors, WITHOUT applying any palette/LUT — so a caller that needs the SAME
    geometry under several colorings (e.g. particles by feature AND particles by
    motion AND clusters) pays the expensive NN-fill + edge-aware upsample ONCE
    instead of once per coloring.

    Returns ``(blob_full, hyper_full, outlier_full)``: two ``(h, w)`` int32 label
    grids (blob ids clamped to ``[0, num_blobs)`` — ``num_blobs`` defaults to
    ``BLOB_PALETTE.shape[0]``, bit-exact for existing callers; PASS it when
    running MORE blobs than the palette, otherwise every id past the clamp is
    silently RELABELED as the last in-range blob and painted with ITS color (a
    large one-color smear over the reconstruction); hyperblob ids clamped to
    ``[0, HYPERBLOB_PALETTE)``) and a ``(h, w)`` bool outlier mask. Colour them
    with ``palette[grid]`` / ``lut[grid]`` then paint ``OUTLIER_BGR`` where
    ``outlier_full`` — this reproduces ``render_matter_tile``'s output BIT-FOR-BIT.
    This is a pure ADD; ``render_matter_tile`` itself is unchanged (the live path
    is byte-identical).
    """
    gh, gw = h // stride, w // stride
    n_blobs = BLOB_PALETTE.shape[0] if num_blobs is None else max(int(num_blobs), 1)
    n_hypers = HYPERBLOB_PALETTE.shape[0]

    is_outlier_dp = (blob_assignments < 0) | (hyperblob_per_dp < 0)
    blob_safe = np.minimum(np.maximum(blob_assignments, 0), n_blobs - 1)
    hyper_safe = np.minimum(np.maximum(hyperblob_per_dp, 0), n_hypers - 1)

    UNFILLED = -1
    blob_grid = np.full((gh * gw,), UNFILLED, dtype=np.int32)
    hyper_grid = np.full((gh * gw,), UNFILLED, dtype=np.int32)
    outlier_grid = np.zeros((gh * gw,), dtype=bool)

    inlier_mask = ~is_outlier_dp
    blob_grid[indices[inlier_mask]] = blob_safe[inlier_mask]
    hyper_grid[indices[inlier_mask]] = hyper_safe[inlier_mask]
    outlier_grid[indices[is_outlier_dp]] = True

    blob_grid = _nn_fill_grid(blob_grid.reshape(gh, gw))
    hyper_grid = _nn_fill_grid(hyper_grid.reshape(gh, gw))
    outlier_grid = outlier_grid.reshape(gh, gw)

    if rgb_guide is not None:
        blob_full = _edge_aware_upsample(blob_grid.astype(np.int32), h, w, rgb_guide)
        hyper_full = _edge_aware_upsample(hyper_grid.astype(np.int32), h, w, rgb_guide)
    else:
        blob_full = cv2.resize(blob_grid.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        hyper_full = cv2.resize(hyper_grid.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
    outlier_full = cv2.resize(outlier_grid.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
    return blob_full, hyper_full, outlier_full


def render_matter_tile(blob_assignments: np.ndarray,
                        hyperblob_per_dp: np.ndarray,
                        indices: np.ndarray,
                        h: int = 360, w: int = 640,
                        stride: int = STRIDE,
                        rgb_guide: Optional[np.ndarray] = None,
                        num_fg_hyperblobs: Optional[int] = None,
                        particle_avg_rgb: bool = False,
                        blob_color_lut: Optional[np.ndarray] = None,
                        num_blobs: Optional[int] = None,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Project the per-datapoint blob (and hyperblob) labels back into a
    (h, w, 3) BGR image, with each datapoint covering a stride x stride block
    and unassigned grid cells filled by NN from neighbours.

    `blob_assignments` and `hyperblob_per_dp` may use -1 to mark outliers.
    Outlier cells are NOT propagated through the NN-fill — they're held back
    so empty cells take their color from the nearest *non-outlier* neighbour;
    after the fill the actual outlier cells are re-painted as a 50/50 alpha
    blend with the surrounding NN value (so an outlier reads as a darkened
    tint, not a solid gray block that visually dominates its neighbourhood).
    Cells that have no datapoint at all (e.g. the ~19 % of grid cells the
    subsampler skipped) are filled by NN from their nearest *non-outlier*
    neighbour.

    ``rgb_guide`` (the BGR frame, any size): when supplied, the coarse stride-8
    label grids are upsampled EDGE-AWARE (``_edge_aware_upsample``) so cluster /
    particle boundaries snap to the image edges instead of the blocky stride-8
    staircase. Assignments are unchanged — only the upsample boundary moves.
    None falls back to the original ``cv2`` nearest upsample.

    ``num_fg_hyperblobs``: number of seeded OBJECT-instance hyperblobs (the low
    ids 0..num_fg-1). When supplied, the CLUSTER tile uses the vivid-fg /
    muted-bg palette (``build_cluster_palette``) so the tracked object pops; the
    PARTICLE tile always keeps the full rainbow ``BLOB_PALETTE``. None falls
    back to the rainbow ``HYPERBLOB_PALETTE`` for clusters too.

    ``num_blobs``: the blob-id count. Defaults to ``BLOB_PALETTE``'s length
    (128); callers running with MORE blobs must pass the real count so ids
    >= 128 survive instead of being clamped onto blob 127 (the palette-size
    smear — see module docstring). Bit-exact for the 128-blob default.

    Returns (blob_bgr, hyperblob_bgr).  Both are uint8 (h, w, 3) BGR.
    """
    gh, gw = h // stride, w // stride

    blob_palette = BLOB_PALETTE
    # Cluster (hyperblob) tile: vivid-fg / muted-bg palette when we know the
    # foreground instance count; otherwise the legacy rainbow palette. Both are
    # sized to HYPERBLOB_PALETTE's length so the label lookup never overruns.
    if num_fg_hyperblobs is not None:
        hyper_palette = _cluster_palette_for(num_fg_hyperblobs, HYPERBLOB_PALETTE.shape[0])
    else:
        hyper_palette = HYPERBLOB_PALETTE
    # Blob-id range: default to BLOB_PALETTE's length; trust num_blobs when the
    # caller runs with more blobs so high ids are not collapsed onto blob 127.
    n_blobs = blob_palette.shape[0] if num_blobs is None else max(int(num_blobs), 1)
    n_hypers = hyper_palette.shape[0]

    # Datapoints with assignment < 0 are real Gibbs outliers.  We keep them
    # out of the NN-fill source set so unfilled grid cells can't inherit
    # outlier-gray transitively (which would amplify a few real outliers
    # into wide gray patches via the cv2 distance transform).
    is_outlier_dp = (blob_assignments < 0) | (hyperblob_per_dp < 0)
    blob_safe = np.minimum(np.maximum(blob_assignments, 0), n_blobs - 1)
    hyper_safe = np.minimum(np.maximum(hyperblob_per_dp, 0), n_hypers - 1)

    UNFILLED = -1
    blob_grid = np.full((gh * gw,), UNFILLED, dtype=np.int32)
    hyper_grid = np.full((gh * gw,), UNFILLED, dtype=np.int32)
    outlier_grid = np.zeros((gh * gw,), dtype=bool)

    inlier_mask = ~is_outlier_dp
    inlier_indices = indices[inlier_mask]
    outlier_indices = indices[is_outlier_dp]
    blob_grid[inlier_indices] = blob_safe[inlier_mask]
    hyper_grid[inlier_indices] = hyper_safe[inlier_mask]
    outlier_grid[outlier_indices] = True

    blob_grid = blob_grid.reshape(gh, gw)
    hyper_grid = hyper_grid.reshape(gh, gw)
    outlier_grid = outlier_grid.reshape(gh, gw)

    blob_grid = _nn_fill_grid(blob_grid)
    hyper_grid = _nn_fill_grid(hyper_grid)

    if rgb_guide is not None:
        # Edge-aware joint upsample: snap label boundaries to RGB edges so the
        # stride-8 staircase becomes crisp. Assignments are preserved exactly
        # (output is always one of the coarse labels).
        blob_full = _edge_aware_upsample(blob_grid.astype(np.int32), h, w, rgb_guide)
        hyper_full = _edge_aware_upsample(hyper_grid.astype(np.int32), h, w, rgb_guide)
    else:
        blob_full = cv2.resize(blob_grid.astype(np.int32), (w, h),
                                interpolation=cv2.INTER_NEAREST)
        hyper_full = cv2.resize(hyper_grid.astype(np.int32), (w, h),
                                 interpolation=cv2.INTER_NEAREST)
    outlier_full = cv2.resize(outlier_grid.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)

    # PARTICLE tile colouring, in precedence order:
    #  1. blob_color_lut (LOCKED frame-1 colours, from compute_blob_color_lut):
    #     each blob id keeps its fixed mean BGR across frames — no colour drift.
    #  2. particle_avg_rgb (+ rgb_guide): per-frame mean BGR of each particle's
    #     region (true-colour posterization), recomputed every frame.
    #  3. default: the rainbow BLOB_PALETTE keyed by id.
    if blob_color_lut is not None:
        lut = np.asarray(blob_color_lut)
        blob_bgr = lut[np.clip(blob_full, 0, lut.shape[0] - 1)].astype(np.uint8)
    elif particle_avg_rgb and rgb_guide is not None:
        blob_bgr = _avg_rgb_by_label(blob_full, rgb_guide, h, w)
    else:
        # Rainbow fallback: clip into the 128-entry palette (no-op for <=128
        # blobs; > 128 reuse colours rather than indexing out of range).
        blob_bgr = blob_palette[np.clip(blob_full, 0, blob_palette.shape[0] - 1)].astype(np.uint8)
    hyper_bgr = hyper_palette[hyper_full].astype(np.uint8)
    if outlier_full.any():
        # SOLID outlier color (not a 50/50 darken) so outliers read clearly.
        blob_bgr[outlier_full] = OUTLIER_BGR
        hyper_bgr[outlier_full] = OUTLIER_BGR
    return blob_bgr, hyper_bgr


# Default 3D-view rotation for the ROW3 point-cloud tiles. A SMALL ~3° yaw
# (no pitch) keeps the view nearly FULL-FRONTAL with the camera — just enough
# parallax to read depth without the off-kilter skew of a 3/4 orbit. (At exactly
# 0° a pinhole reprojection cancels depth and collapses to the flat 2D image, so
# a couple degrees is the floor for any 3D to be visible.) Pitch=0 keeps the
# scene level and vertically centered.
POINTCLOUD_YAW_DEG = 1.5
POINTCLOUD_PITCH_DEG = 0.0


def _build_pointcloud_projection(
    depth_hxw: np.ndarray,
    intr: Intrinsics = DEFAULT_INTRINSICS,
    yaw_deg: float = POINTCLOUD_YAW_DEG,
    pitch_deg: float = POINTCLOUD_PITCH_DEG,
    point_subsample: int = 2,
    out_hw: Tuple[int, int] = (360, 960),
    focal_length: Optional[float] = None,
    prev_centroid: Optional[np.ndarray] = None,
    centroid_alpha: float = 1.0,
    depth_lo: Optional[float] = None,
    depth_hi: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Subsample, unproject, rotate, project, cull, depth-sort.  Returns
    ``(u, v, ys, xs, focal_length)`` — the int32 pixel coordinates in the
    output canvas plus the source-frame pixel coordinates each surviving point
    came from (in painter far→near order), and the focal length used for
    projection (either the caller-supplied value, or the value the auto-fit
    chose if ``focal_length`` was None).

    Pass ``focal_length`` once you have a stable value to FREEZE the framing:
    re-computing the autofit every frame makes the cloud appear to drift even
    when the underlying 3D geometry is stable, because the 98 th-percentile
    XY extents jitter with depth normalization noise.  Callers should compute
    autofit on the first frame, then thread the returned value back in.

    ``prev_centroid`` / ``centroid_alpha`` FREEZE/SMOOTH the orbit center the
    same way ``focal_length`` freezes the zoom.  The cloud is rotated around
    ``centroid = pts.mean(0)`` recomputed every frame; that raw mean jitters
    with depth-normalization noise, so the whole cloud (and any marbles sharing
    this ``proj``) translates frame-to-frame.  Pass the previous frame's
    ``proj['centroid']`` back in with a small ``centroid_alpha`` (e.g. 0.15) to
    EMA it: ``centroid = alpha*raw + (1-alpha)*prev`` — high-freq swim removed,
    slow object motion still tracked.  Default ``(None, 1.0)`` reproduces the
    raw per-frame mean EXACTLY (bit-identical), so existing callers are unchanged.

    Empty input returns ``(empty, empty, empty, empty, 0.0)`` so the caller
    can splat onto a bg fill without a special-case path.
    """
    H, W = depth_hxw.shape
    out_H, out_W = out_hw
    empty = (np.empty(0, dtype=np.int32),) * 4
    empty_ret = (*empty, 0.0, None)

    # The depth model emits RELATIVE INVERSE depth (disparity-like: near = large
    # value). Convert to the same pseudo-metric Z the tracker unprojects with
    # (``_depth_to_Z``: near = small Z, far = large Z) BEFORE unprojecting the
    # cloud. Feeding the raw inverse depth straight in as Z turns the scene
    # inside-out (near objects pushed far, far objects pulled near) and is the
    # dominant source of the "warped sheet" distortion — converting here makes
    # the point cloud geometry consistent with the camera the tracker sees.
    depth_hxw = _depth_to_Z(np.asarray(depth_hxw, dtype=np.float32), depth_lo, depth_hi)

    ys, xs = np.meshgrid(
        np.arange(0, H, point_subsample),
        np.arange(0, W, point_subsample),
        indexing="ij",
    )
    ys = ys.flatten()
    xs = xs.flatten()
    z = depth_hxw[ys, xs].astype(np.float32)
    valid = (z > 1e-3) & np.isfinite(z)
    if not valid.any():
        return empty_ret
    ys, xs, z = ys[valid], xs[valid], z[valid]
    X = (xs.astype(np.float32) - intr.cx) * z / intr.fx
    Y = (ys.astype(np.float32) - intr.cy) * z / intr.fy
    pts = np.stack([X, Y, z], axis=-1)

    # Rotate around the cloud centroid for a 3/4 view, then re-center the
    # rotated cloud on the optical axis (only the original Z offset is kept).
    # This is equivalent to orbiting a virtual camera around the cloud at a
    # fixed look-at: the centroid always projects to the tile center, so a
    # large yaw won't shift the cloud off the edge of the tile (as restoring
    # the full 3D centroid after rotation would).
    raw_centroid = pts.mean(axis=0)
    if prev_centroid is None:
        centroid = raw_centroid
    else:
        ca = float(centroid_alpha)
        centroid = (ca * raw_centroid
                    + (1.0 - ca) * np.asarray(prev_centroid, dtype=np.float32)).astype(np.float32)
    pts_c = pts - centroid
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    Ry = np.array([[ np.cos(yaw), 0., np.sin(yaw)],
                   [          0., 1.,          0.],
                   [-np.sin(yaw), 0., np.cos(yaw)]], dtype=np.float32)
    Rx = np.array([[1.,            0.,             0.],
                   [0.,  np.cos(pitch), -np.sin(pitch)],
                   [0.,  np.sin(pitch),  np.cos(pitch)]], dtype=np.float32)
    R_pc = (Rx @ Ry).astype(np.float32)   # camera-frame -> view rotation
    pts_rot = pts_c @ R_pc.T
    pts_rot[:, 2] += centroid[2]

    z_view = pts_rot[:, 2]
    in_front = z_view > 1e-3
    if not in_front.any():
        return empty_ret
    pts_rot = pts_rot[in_front]
    z_view = z_view[in_front]
    ys = ys[in_front]
    xs = xs[in_front]

    # Uniform focal length keeps 3D circles round.  Frame to the CAMERA's
    # aspect ratio (a W/H-wide window centered in the tile) rather than the full
    # 960-px tile width: the scene is wider than it is tall, so fitting to the
    # tile width collapses the cloud into a thin horizontal band.  Fitting to a
    # camera-aspect window instead fills the tile height and keeps the view
    # consistent with the source camera's framing (the "calibrated frustum").
    # Auto-fit uses a high percentile of the cloud extent (robust to a few stray
    # far points); callers freeze the value across frames by threading it back in.
    if focal_length is None or focal_length <= 0.0:
        margin = 0.98   # fill the tile (a frozen frame-0 focal stays stable, no pop)
        cam_aspect = float(W) / float(H)                 # source camera aspect
        # camera-aspect window, but never wider than the tile itself (so narrow
        # 1×5 tiles don't overflow / get culled).
        eff_W = min(out_H * cam_aspect, float(out_W))
        # Frame to the BULK of the cloud (90th pct) so it fills the tile; the
        # outer ~10% of points spill into the margin / crop at the edges.
        x_extent = float(np.percentile(np.abs(pts_rot[:, 0]), 90)) + 1e-3
        y_extent = float(np.percentile(np.abs(pts_rot[:, 1]), 90)) + 1e-3
        z_ref = float(np.median(z_view))
        f_x = (margin * 0.5 * eff_W) * z_ref / x_extent
        f_y = (margin * 0.5 * out_H) * z_ref / y_extent
        f = float(min(f_x, f_y))
    else:
        f = float(focal_length)
    out_fx = out_fy = f
    out_cx = out_W * 0.5
    out_cy = out_H * 0.5

    u = (pts_rot[:, 0] * out_fx / z_view + out_cx).astype(np.int32)
    v = (pts_rot[:, 1] * out_fy / z_view + out_cy).astype(np.int32)

    in_bounds = (u >= 0) & (u < out_W) & (v >= 0) & (v < out_H)
    u = u[in_bounds]; v = v[in_bounds]
    z_view = z_view[in_bounds]
    ys = ys[in_bounds]; xs = xs[in_bounds]

    # Projection parameters so callers can render OTHER 3D entities (e.g. the
    # particle covariance ellipsoids in ROW4) into the SAME rotated view —
    # aligned with this point cloud.
    proj = {"centroid": centroid.astype(np.float32), "R": R_pc,
            "focal": float(f), "out_cx": float(out_cx), "out_cy": float(out_cy)}
    order = np.argsort(-z_view, kind="stable")
    return u[order], v[order], ys[order], xs[order], f, proj


# Memoized disk-offset tables for the vectorized splat (one np.array each for
# dy and dx, in the order points get repeated along the K axis).
_DISK_OFFSETS_CACHE: dict = {}


def _disk_offsets(point_size: int) -> Tuple[np.ndarray, np.ndarray]:
    cached = _DISK_OFFSETS_CACHE.get(point_size)
    if cached is not None:
        return cached
    r = int(point_size)
    if r <= 0:
        offs = (np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32))
    else:
        dys, dxs = [], []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    dys.append(dy); dxs.append(dx)
        offs = (np.array(dys, dtype=np.int32), np.array(dxs, dtype=np.int32))
    _DISK_OFFSETS_CACHE[point_size] = offs
    return offs


def _splat_pointcloud(u: np.ndarray, v: np.ndarray, colors: np.ndarray,
                       out_hw: Tuple[int, int], point_size: int,
                       bg_bgr: Tuple[int, int, int]) -> np.ndarray:
    """Painter-order disk splat into a fresh bg-filled canvas.

    Caller must have sorted ``u, v, colors`` far→near so that later writes
    to the same pixel naturally overwrite earlier (nearer) ones.  Vectorizes
    the disk by broadcasting ``(N,) + (K,)`` rather than running K separate
    fancy-index writes — assignments stay painter-correct because the K
    offsets are interleaved per point (so all K writes for the farthest
    point happen before any write for the next point).
    """
    out_H, out_W = out_hw
    img = np.full((out_H, out_W, 3), bg_bgr, dtype=np.uint8)
    if u.size == 0:
        return img
    offs_dy, offs_dx = _disk_offsets(point_size)
    K = offs_dy.shape[0]
    if K == 1 and int(offs_dy[0]) == 0 and int(offs_dx[0]) == 0:
        img[v, u] = colors
        return img
    v_off = (v[:, None] + offs_dy[None, :]).reshape(-1)
    u_off = (u[:, None] + offs_dx[None, :]).reshape(-1)
    c_off = np.broadcast_to(colors[:, None, :], (colors.shape[0], K, 3)).reshape(-1, 3)
    mask = (v_off >= 0) & (v_off < out_H) & (u_off >= 0) & (u_off < out_W)
    img[v_off[mask], u_off[mask]] = c_off[mask]
    return img


def render_pointcloud_tile(depth_hxw: np.ndarray,
                            color_hxw_bgr: np.ndarray,
                            intr: Intrinsics = DEFAULT_INTRINSICS,
                            yaw_deg: float = POINTCLOUD_YAW_DEG,
                            pitch_deg: float = POINTCLOUD_PITCH_DEG,
                            point_subsample: int = 2,
                            point_size: int = 2,
                            bg_bgr: Tuple[int, int, int] = (18, 18, 26),
                            out_hw: Tuple[int, int] = (360, 960),
                            focal_length: Optional[float] = None,
                            ) -> np.ndarray:
    """Render a 3/4-view 3D point-cloud splat of the scene.

    Every (subsampled) pixel of ``depth_hxw`` is unprojected to a camera-frame
    3D point using ``intr``; the cloud is rotated yaw_deg around the
    vertical axis through its centroid and pitched pitch_deg downward, then
    projected back into a virtual image of size ``out_hw``.  Each splat takes
    its color from ``color_hxw_bgr`` (which should be one of the per-pixel
    mask tiles — pixel_by_cluster or pixel_by_particle — so the resulting
    point cloud is colored by the GenMatter blob/hyperblob assignment).

    Painter's algorithm with a back-to-front draw order; no z-buffer.  At
    640x360 input with subsample=2 this is ~14400 points.

    Thin wrapper around ``_build_pointcloud_projection`` + ``_splat_pointcloud``
    so callers that need both cluster- and particle-colored tiles can amortize
    the projection with ``render_pointcloud_tiles_pair``.  When ``focal_length``
    is None the projection auto-fits per frame (jitters); pass a frozen value
    to stabilize framing across frames.
    """
    u, v, ys, xs, _f, _proj = _build_pointcloud_projection(
        depth_hxw, intr, yaw_deg, pitch_deg, point_subsample, out_hw,
        focal_length=focal_length)
    if u.size == 0:
        out_H, out_W = out_hw
        return np.full((out_H, out_W, 3), bg_bgr, dtype=np.uint8)
    colors = color_hxw_bgr[ys, xs].astype(np.uint8)
    return _splat_pointcloud(u, v, colors, out_hw, point_size, bg_bgr)


def render_pointcloud_tiles_pair(
    depth_hxw: np.ndarray,
    color_a_bgr: np.ndarray,
    color_b_bgr: np.ndarray,
    intr: Intrinsics = DEFAULT_INTRINSICS,
    yaw_deg: float = POINTCLOUD_YAW_DEG,
    pitch_deg: float = POINTCLOUD_PITCH_DEG,
    point_subsample: int = 2,
    point_size: int = 2,
    bg_bgr: Tuple[int, int, int] = (18, 18, 26),
    out_hw: Tuple[int, int] = (360, 960),
    focal_length: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Render two pointcloud tiles that share geometry but use different color
    sources.  Builds the projection once and splats twice — saves roughly half
    the per-frame pointcloud cost vs. two ``render_pointcloud_tile`` calls.

    Returns ``(tile_a, tile_b, focal_length, proj)``.  Pass ``focal_length``
    back in on subsequent calls to freeze framing across frames; pass ``proj``
    to ``render_particle_ellipsoid_tile`` to draw aligned 3D ellipsoids (ROW4).
    """
    out_H, out_W = out_hw
    u, v, ys, xs, f_used, proj = _build_pointcloud_projection(
        depth_hxw, intr, yaw_deg, pitch_deg, point_subsample, out_hw,
        focal_length=focal_length)
    if u.size == 0:
        blank = np.full((out_H, out_W, 3), bg_bgr, dtype=np.uint8)
        return blank.copy(), blank.copy(), f_used, proj
    colors_a = color_a_bgr[ys, xs].astype(np.uint8)
    colors_b = color_b_bgr[ys, xs].astype(np.uint8)
    # Resolve the shared painter geometry once (winner point per pixel) and gather
    # both colorings — bit-identical to two _splat_pointcloud calls, ~2x cheaper.
    tile_a, tile_b = _splat_pointcloud_multi(u, v, [colors_a, colors_b],
                                             out_hw, point_size, bg_bgr)
    return tile_a, tile_b, f_used, proj


def _splat_pointcloud_multi(u: np.ndarray, v: np.ndarray,
                            colors_list, out_hw: Tuple[int, int],
                            point_size: int, bg_bgr: Tuple[int, int, int]):
    """Splat SEVERAL color sources that share the SAME projected geometry.

    The disk-offset broadcast (``v_off``/``u_off``) and the in-bounds ``mask`` are
    a pure function of ``(u, v, point_size, out_hw)`` — identical for every tile —
    so we compute them ONCE and only re-scatter the (different) colors per tile.
    Bit-for-bit identical to calling ``_splat_pointcloud`` once per color source
    (same offsets, same mask, same painter-interleaved write order)."""
    out_H, out_W = out_hw
    if u.size == 0:
        return [np.full((out_H, out_W, 3), bg_bgr, dtype=np.uint8) for _ in colors_list]
    offs_dy, offs_dx = _disk_offsets(point_size)
    K = offs_dy.shape[0]
    if K == 1 and int(offs_dy[0]) == 0 and int(offs_dx[0]) == 0:
        imgs = []
        for colors in colors_list:
            img = np.full((out_H, out_W, 3), bg_bgr, dtype=np.uint8)
            img[v, u] = colors
            imgs.append(img)
        return imgs
    v_off = (v[:, None] + offs_dy[None, :]).reshape(-1)
    u_off = (u[:, None] + offs_dx[None, :]).reshape(-1)
    mask = (v_off >= 0) & (v_off < out_H) & (u_off >= 0) & (u_off < out_W)
    vm = v_off[mask]
    um = u_off[mask]
    # Which source point each in-bounds disk pixel belongs to (row-major over the
    # (N, K) point x offset broadcast == np.repeat(arange(N), K)).
    pt_idx_m = np.repeat(np.arange(u.size, dtype=np.intp), K)[mask]
    # Painter's algorithm with last-write-wins means each output pixel ends up
    # showing the NEAREST point that covers it. That winner is a pure function of
    # GEOMETRY (the disk scatter + far->near order), so it is IDENTICAL for every
    # color source. Resolve it ONCE into a (H*W,) winner-point buffer (one big
    # last-wins scatter, exactly the NumPy semantics the per-tile splat relied on),
    # then each tile is a cheap gather over only the covered pixels instead of
    # re-scattering all N*K disk writes. Bit-identical to splatting each tile.
    lin = vm.astype(np.intp) * out_W + um
    winner = np.full(out_H * out_W, -1, dtype=np.intp)
    winner[lin] = pt_idx_m                      # last-wins == nearest == painter result
    covered = winner >= 0
    win_pts = winner[covered]
    imgs = []
    for colors in colors_list:
        flat = np.empty((out_H * out_W, 3), dtype=np.uint8)
        flat[:] = np.asarray(bg_bgr, dtype=np.uint8)
        flat[covered] = colors[win_pts]
        imgs.append(flat.reshape(out_H, out_W, 3))
    return imgs


def render_pointcloud_tiles_multi(
    depth_hxw: np.ndarray,
    color_sources,
    intr: Intrinsics = DEFAULT_INTRINSICS,
    yaw_deg: float = POINTCLOUD_YAW_DEG,
    pitch_deg: float = POINTCLOUD_PITCH_DEG,
    point_subsample: int = 2,
    point_size: int = 2,
    bg_bgr: Tuple[int, int, int] = (18, 18, 26),
    out_hw: Tuple[int, int] = (360, 960),
    focal_length: Optional[float] = None,
    prev_centroid: Optional[np.ndarray] = None,
    centroid_alpha: float = 1.0,
    depth_lo: Optional[float] = None,
    depth_hi: Optional[float] = None,
):
    """Render N point-cloud tiles that share geometry but differ only in color.

    Builds the projection ONCE (unproject + rotate + cull + depth-sort) and the
    splat geometry ONCE, then paints each ``color_sources[i]`` (an ``(H, W, 3)``
    BGR tile sampled at the surviving source pixels). Output is bit-identical to
    a ``render_pointcloud_tiles_pair`` + per-extra ``render_pointcloud_tile`` at
    the SAME frozen ``focal_length``, but pays the projection + disk broadcast +
    bounds mask once instead of once per tile.

    Returns ``(tiles, focal_length, proj)`` where ``tiles`` is a list aligned to
    ``color_sources``."""
    out_H, out_W = out_hw
    u, v, ys, xs, f_used, proj = _build_pointcloud_projection(
        depth_hxw, intr, yaw_deg, pitch_deg, point_subsample, out_hw,
        focal_length=focal_length, prev_centroid=prev_centroid,
        centroid_alpha=centroid_alpha, depth_lo=depth_lo, depth_hi=depth_hi)
    if u.size == 0:
        blank = np.full((out_H, out_W, 3), bg_bgr, dtype=np.uint8)
        return [blank.copy() for _ in color_sources], f_used, proj
    colors_list = [np.asarray(c)[ys, xs].astype(np.uint8) for c in color_sources]
    tiles = _splat_pointcloud_multi(u, v, colors_list, out_hw, point_size, bg_bgr)
    return tiles, f_used, proj


def render_particle_ellipsoid_tile(means_3d: np.ndarray, covs_3d: np.ndarray,
                                   palette: np.ndarray, proj: Optional[dict], *,
                                   out_hw: Tuple[int, int] = (360, 960),
                                   sigma_scale: float = 1.5,
                                   bg_bgr: Tuple[int, int, int] = (18, 18, 26),
                                   alpha: float = 0.6) -> np.ndarray:
    """Render the PROBABILISTIC particles as 3D covariance ellipsoids, in the
    SAME rotated view as the point cloud (``proj`` returned by
    ``render_pointcloud_tiles_pair``) so ROW4 lines up with ROW3.

    Each particle is a 3D Gaussian (``means_3d[i]`` + ``covs_3d[i]``): we rotate
    it into the view frame, perspective-project the mean with the point cloud's
    focal, project the covariance to a 2D ellipse, and draw a filled translucent
    Gaussian — depth-sorted far→near. This shows each particle's position AND
    spread/uncertainty in 3D, rather than hard per-pixel colors.
    """
    out_H, out_W = out_hw
    canvas = np.full((out_H, out_W, 3), bg_bgr, dtype=np.uint8)
    means = None if means_3d is None else np.asarray(means_3d, dtype=np.float32)
    if proj is None or means is None or means.shape[0] == 0:
        return canvas
    R = np.asarray(proj["R"], dtype=np.float32)
    centroid = np.asarray(proj["centroid"], dtype=np.float32)
    f = float(proj["focal"])
    if f <= 0.0:
        return canvas
    pc_intr = Intrinsics(fx=f, fy=f, cx=float(proj["out_cx"]), cy=float(proj["out_cy"]))
    m_rot = (means - centroid) @ R.T            # match the cloud's rotation
    m_rot[:, 2] += centroid[2]
    uv = project_3d_to_2d(m_rot, pc_intr)
    overlay = np.zeros_like(canvas)
    mask = np.zeros((out_H, out_W), dtype=np.uint8)
    n = means.shape[0]
    for i in np.argsort(-m_rot[:, 2]):          # painter: far → near
        u, v = uv[i]
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        ui, vi = int(round(u)), int(round(v))
        if ui < -200 or ui > out_W + 200 or vi < -200 or vi > out_H + 200:
            continue
        cov_rot = R @ np.asarray(covs_3d[i], dtype=np.float64) @ R.T
        axes, ang = covariance_to_ellipse_2d(cov_rot, m_rot[i], pc_intr,
                                             sigma_scale=sigma_scale,
                                             max_half=min(out_W, out_H) * 0.45)
        color = (int(palette[i, 0]), int(palette[i, 1]), int(palette[i, 2]))
        cv2.ellipse(overlay, (ui, vi), (int(axes[0]), int(axes[1])), ang, 0, 360,
                    color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.ellipse(mask, (ui, vi), (int(axes[0]), int(axes[1])), ang, 0, 360,
                    255, thickness=-1, lineType=cv2.LINE_AA)
        cv2.ellipse(canvas, (ui, vi), (int(axes[0]), int(axes[1])), ang, 0, 360,
                    color, thickness=1, lineType=cv2.LINE_AA)
    m = (mask[..., None].astype(np.float32) / 255.0) * alpha
    out = canvas.astype(np.float32) * (1.0 - m) + overlay.astype(np.float32) * m
    return out.clip(0, 255).astype(np.uint8)


# ---------------- Clean 3D particles: shaded "marbles" + multi-view + flicker-smoothing ----------------

_LIT_SPHERE_CACHE: dict = {}


def make_lit_sphere_sprite(size: int = 72,
                           light_dir: Tuple[float, float, float] = (-0.45, -0.55, 1.0),
                           ambient: float = 0.46, diffuse: float = 0.62,
                           specular: float = 0.22, shininess: float = 22.0,
                           edge: float = 0.16) -> np.ndarray:
    """Precompute a unit lit-sphere SPRITE: an ``(size, size, 2)`` float32 image
    whose channel 0 is per-pixel LUMINANCE (ambient + Lambert diffuse + a soft
    specular highlight of a sphere lit from ``light_dir``) and channel 1 is a
    smooth COVERAGE alpha (1 inside the disk, anti-aliased to 0 at the rim).

    Warping this ONE sprite to each projected particle ellipse (``cv2.warpAffine``)
    and compositing far->near yields SOLID shaded 3-D 'marbles' — no 2-D outline,
    near marbles occlude far ones — for a fraction of a per-pixel raymarch cost.
    Luminance peaks ~1.3 at the lit pole (a clipped specular glint) and falls to
    ``ambient`` at the rim, so tinting by a particle's avg-RGB reads as a shaded
    ball of that colour. Memoized by params."""
    keyp = (size, light_dir, ambient, diffuse, specular, shininess, edge)
    cached = _LIT_SPHERE_CACHE.get(keyp)
    if cached is not None:
        return cached
    ax = (np.arange(size, dtype=np.float32) - (size - 1) * 0.5) / ((size - 1) * 0.5)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")            # image coords in [-1, 1]
    rr = np.sqrt(xx * xx + yy * yy)
    inside = rr <= 1.0
    nz = np.sqrt(np.clip(1.0 - rr * rr, 0.0, 1.0))         # sphere height -> normal z
    L = np.asarray(light_dir, dtype=np.float32)
    L = L / (np.linalg.norm(L) + 1e-8)
    ndotl = np.clip(xx * L[0] + yy * L[1] + nz * L[2], 0.0, 1.0)
    spec = specular * np.power(ndotl, shininess)           # cheap glossy highlight
    lum = (ambient + diffuse * ndotl + spec).astype(np.float32)
    lum = np.where(inside, lum, 0.0)
    cov = np.clip((1.0 - rr) / max(edge, 1e-3), 0.0, 1.0).astype(np.float32)
    sprite = np.stack([lum, cov], axis=-1).astype(np.float32)
    _LIT_SPHERE_CACHE[keyp] = sprite
    return sprite


def blob_alpha_from_weights(w_ema: np.ndarray, *, total_points: float = float(N_KEEP),
                            min_points: float = 3.0,
                            full_points: float = 11.0) -> np.ndarray:
    """Map per-blob posterior-mean weights (already TEMPORALLY EMA-smoothed by the
    caller) to a per-blob visibility ALPHA in ``[0, 1]`` via a smoothstep — so weak
    particles FADE rather than pop (the flicker fix, read straight off the
    probabilistic trace).

    ``w_ema`` is the Dirichlet-multinomial posterior mean ``(beta + n_b)/sum``
    (``extract_blob_weights``); ``w_ema * total_points`` is therefore the blob's
    effective datapoint COUNT. A blob owning ``< min_points`` is invisible,
    ``>= full_points`` fully opaque, with a smooth Hermite ramp between. Using a
    STABLE ``total_points`` (not the per-frame inlier count) keeps the thresholds
    from drifting frame-to-frame — a moving threshold would itself flicker."""
    counts = np.asarray(w_ema, dtype=np.float32) * float(total_points)
    denom = max(float(full_points) - float(min_points), 1e-6)
    t = np.clip((counts - float(min_points)) / denom, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)    # smoothstep (C1, no kink)


_GLOW_SPRITE_CACHE: dict = {}


def make_glow_sprite(size: int = 72, falloff: float = 2.0) -> np.ndarray:
    """Precompute a soft radial GLOW sprite: a ``(size, size)`` float32 ALPHA map
    (smooth ``(1-r)**falloff`` falloff to 0 at the rim; no luminance channel).
    Warped behind a marble with its axes scaled up and composited in the particle's
    SEGMENTATION color, it reads as a faint highlighter halo around the particle —
    the semantic channel of a fill+glow multiplex. Memoized by params."""
    keyp = (size, falloff)
    cached = _GLOW_SPRITE_CACHE.get(keyp)
    if cached is not None:
        return cached
    ax = (np.arange(size, dtype=np.float32) - (size - 1) * 0.5) / ((size - 1) * 0.5)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    sprite = (np.clip(1.0 - rr, 0.0, 1.0) ** float(falloff)).astype(np.float32)
    _GLOW_SPRITE_CACHE[keyp] = sprite
    return sprite


def render_particle_marbles_tile(means_3d: np.ndarray, covs_3d: np.ndarray,
                                 colors_bgr: np.ndarray, proj: Optional[dict], *,
                                 out_hw: Tuple[int, int] = (360, 960),
                                 sigma_scale: float = 1.6,
                                 alpha_per: Optional[np.ndarray] = None,
                                 bg_bgr: Tuple[int, int, int] = (18, 18, 26),
                                 base_tile: Optional[np.ndarray] = None,
                                 base_dim: float = 0.40,
                                 sprite: Optional[np.ndarray] = None,
                                 max_half_frac: float = 0.22,
                                 min_axis: float = 1.5,
                                 glow_colors: Optional[np.ndarray] = None,
                                 glow_alpha_per: Optional[np.ndarray] = None,
                                 glow_scale: float = 1.7) -> np.ndarray:
    """Draw the tracker's particles as SOLID shaded 3-D ellipsoidal 'marbles' in
    the SAME rotated view as a point-cloud tile (``proj`` from
    ``render_pointcloud_tiles_multi`` / ``render_pointcloud_views``), so the
    particles line up with the cloud.

    For each particle we rotate its REAL ``means_3d`` / ``covs_3d`` into the view
    frame (NO re-fit — exactly the cloud's R / centroid / focal — so the viz can't
    diverge from tracking), project the mean and linearize the covariance to a 2-D
    ellipse, ``warpAffine`` the precomputed lit-sphere ``sprite`` onto that
    ellipse, tint by ``colors_bgr`` (avg-RGB per blob), and alpha-composite
    far->near (painter occlusion: near marbles cover far — kills the 2-D-outline
    spaghetti). ``alpha_per`` (per-blob, in ``[0, 1]``, from
    ``blob_alpha_from_weights``) gates each marble so weak particles fade out
    smoothly. Optionally drawn over ``base_tile`` (e.g. a dimmed point cloud) for
    spatial context.

    ``glow_colors``/``glow_alpha_per`` (default None = OFF, bit-exact for existing
    callers) add a per-particle SEMANTIC GLOW: a soft radial halo (``make_glow_sprite``,
    axes x ``glow_scale``) in each particle's segmentation color at the given faint
    alpha, composited as an UNDER-PASS (all halos far->near, then all fills) so the
    halos union into a continuous highlight field behind a dense cluster instead of
    being covered by the fills. Fill = what the particle looks like; glow = which
    object it belongs to.

    Returns an ``(out_H, out_W, 3)`` uint8 BGR tile. Purely additive; touches no
    inference state."""
    out_H, out_W = out_hw
    if base_tile is not None:
        canvas = (np.asarray(base_tile, np.float32) * float(base_dim)).clip(0, 255)
        if canvas.shape[:2] != (out_H, out_W):
            canvas = cv2.resize(canvas, (out_W, out_H))
    else:
        canvas = np.full((out_H, out_W, 3), bg_bgr, dtype=np.float32)
    means = None if means_3d is None else np.asarray(means_3d, dtype=np.float32)
    if proj is None or means is None or means.shape[0] == 0:
        return canvas.clip(0, 255).astype(np.uint8)
    f = float(proj["focal"])
    if f <= 0.0:
        return canvas.clip(0, 255).astype(np.uint8)
    R = np.asarray(proj["R"], dtype=np.float32)
    centroid = np.asarray(proj["centroid"], dtype=np.float32)
    pc_intr = Intrinsics(fx=f, fy=f, cx=float(proj["out_cx"]), cy=float(proj["out_cy"]))
    if sprite is None:
        sprite = make_lit_sphere_sprite()
    S = sprite.shape[0]
    rs = (S - 1) * 0.5
    n = means.shape[0]
    if alpha_per is None:
        alpha_per = np.ones(n, dtype=np.float32)
    else:
        alpha_per = np.asarray(alpha_per, dtype=np.float32)
    colors_bgr = np.asarray(colors_bgr, dtype=np.float32)
    use_glow = glow_colors is not None
    if use_glow:
        glow_colors = np.asarray(glow_colors, dtype=np.float32)
        glow_alpha_per = (np.ones(n, dtype=np.float32) if glow_alpha_per is None
                          else np.asarray(glow_alpha_per, dtype=np.float32))
        glow_sprite = make_glow_sprite()
        grs = (glow_sprite.shape[0] - 1) * 0.5
    m_rot = (means - centroid) @ R.T                       # match the cloud's rotation
    m_rot[:, 2] += centroid[2]
    uv = project_3d_to_2d(m_rot, pc_intr)
    max_half = min(out_W, out_H) * float(max_half_frac)
    n_col = len(colors_bgr)
    if use_glow:
        # SEMANTIC GLOW UNDER-PASS: composite EVERY particle's soft cluster-color halo
        # (axes x glow_scale) BEFORE any marble fill, far -> near. A single in-loop glow gets
        # mostly covered — by its own opaque fill and by neighboring marbles in a dense
        # cluster; as a separate under-layer the halos UNION into a continuous faint field
        # behind the cluster (highlighter-style), which is the readable version.
        for i in np.argsort(-m_rot[:, 2]):
            if i >= glow_colors.shape[0] or i >= glow_alpha_per.shape[0] or i >= alpha_per.shape[0]:
                continue
            g_a = float(glow_alpha_per[i])
            if g_a <= 0.01 or float(alpha_per[i]) <= 0.02:  # no halo for invisible particles
                continue
            u, v = uv[i]
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            cov_rot = R @ np.asarray(covs_3d[i], dtype=np.float64) @ R.T
            (ha, hb), ang = covariance_to_ellipse_2d(cov_rot, m_rot[i], pc_intr,
                                                     sigma_scale=sigma_scale, max_half=max_half)
            gha = max(float(ha), min_axis) * float(glow_scale)
            ghb = max(float(hb), min_axis) * float(glow_scale)
            th = np.deg2rad(ang)
            cs_, sn_ = np.cos(th), np.sin(th)
            Ag = np.array([[gha * cs_ / grs, -ghb * sn_ / grs],
                           [gha * sn_ / grs,  ghb * cs_ / grs]], dtype=np.float64)
            radg = int(np.ceil(max(gha, ghb))) + 2
            cu, cv_ = int(round(float(u))), int(round(float(v)))
            gx0 = max(cu - radg, 0); gx1 = min(cu + radg + 1, out_W)
            gy0 = max(cv_ - radg, 0); gy1 = min(cv_ + radg + 1, out_H)
            if gx1 <= gx0 or gy1 <= gy0:
                continue
            tg = np.array([float(u), float(v)], dtype=np.float64) - Ag @ np.array([grs, grs])
            Mg = np.array([[Ag[0, 0], Ag[0, 1], tg[0] - gx0],
                           [Ag[1, 0], Ag[1, 1], tg[1] - gy0]], dtype=np.float64)
            gwarp = cv2.warpAffine(glow_sprite, Mg, (gx1 - gx0, gy1 - gy0),
                                   flags=cv2.INTER_LINEAR, borderValue=0.0)
            ga_eff = (gwarp * g_a)[..., None]
            gcol = glow_colors[i][None, None, :]
            regg = canvas[gy0:gy1, gx0:gx1]
            canvas[gy0:gy1, gx0:gx1] = regg * (1.0 - ga_eff) + gcol * ga_eff
    for i in np.argsort(-m_rot[:, 2]):                     # painter: far -> near
        if i >= n_col:        # color LUT can be 1 short of means (SAM seed yields an
            continue          # extra blob); skip the unmatched particle rather than crash
        a_vis = float(alpha_per[i])
        if a_vis <= 0.02:                                  # culled (smoothly faded out)
            continue
        u, v = uv[i]
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        cov_rot = R @ np.asarray(covs_3d[i], dtype=np.float64) @ R.T
        (ha, hb), ang = covariance_to_ellipse_2d(cov_rot, m_rot[i], pc_intr,
                                                 sigma_scale=sigma_scale, max_half=max_half)
        ha = max(float(ha), min_axis)
        hb = max(float(hb), min_axis)
        th = np.deg2rad(ang)
        cs_, sn_ = np.cos(th), np.sin(th)
        # Forward affine mapping the sprite (centered, radius rs) onto the ellipse:
        # sprite +x radius -> major axis (ha @ ang); sprite +y radius -> minor axis.
        A = np.array([[ha * cs_ / rs, -hb * sn_ / rs],
                      [ha * sn_ / rs,  hb * cs_ / rs]], dtype=np.float64)
        rad = int(np.ceil(max(ha, hb))) + 2
        cu, cv_ = int(round(float(u))), int(round(float(v)))
        x0 = max(cu - rad, 0); x1 = min(cu + rad + 1, out_W)
        y0 = max(cv_ - rad, 0); y1 = min(cv_ + rad + 1, out_H)
        if x1 <= x0 or y1 <= y0:                           # entirely off-tile
            continue
        t = np.array([float(u), float(v)], dtype=np.float64) - A @ np.array([rs, rs])
        M = np.array([[A[0, 0], A[0, 1], t[0] - x0],
                      [A[1, 0], A[1, 1], t[1] - y0]], dtype=np.float64)
        warped = cv2.warpAffine(sprite, M, (x1 - x0, y1 - y0),
                                flags=cv2.INTER_LINEAR, borderValue=0.0)
        lum = warped[..., 0]
        cov = warped[..., 1]
        a_eff = (cov * a_vis)[..., None]
        col = colors_bgr[i][None, None, :] * lum[..., None]
        reg = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = reg * (1.0 - a_eff) + col * a_eff
    return canvas.clip(0, 255).astype(np.uint8)


def render_pointcloud_views(
    depth_hxw: np.ndarray,
    color_sources,
    intr: Intrinsics = DEFAULT_INTRINSICS,
    yaws: Tuple[float, ...] = (-18.0, 0.0, 18.0),
    pitch_deg: float = POINTCLOUD_PITCH_DEG,
    point_subsample: int = 2,
    point_size: int = 2,
    bg_bgr: Tuple[int, int, int] = (18, 18, 26),
    out_hw: Tuple[int, int] = (360, 960),
    focal_lengths=None,
):
    """Render the SAME point cloud from several yaw angles (multi-view); each yaw
    shares one projection + winner-buffer splat across all ``color_sources``.

    A single-camera depth map only captures the visible surface, so rotating the
    cloud reveals empty space behind the foreground (a hole behind the object).
    Flanking a hole-free near-frontal ``0deg`` view with symmetric ``+/-`` yaws
    makes that gap read as an honest rotation artifact rather than a ghost
    duplicate.

    Returns ``(views, focal_lengths, projs)`` where ``views[k]`` is the list of
    per-colour tiles at ``yaws[k]``, ``focal_lengths[k]`` the frozen focal for that
    yaw (thread it back in next frame to stabilize framing), and ``projs[k]`` the
    projection dict for that yaw (pass to ``render_particle_marbles_tile`` to align
    marbles with that exact view). Purely additive."""
    yaws = list(yaws)
    if focal_lengths is None:
        focal_lengths = [None] * len(yaws)
    views, out_focals, projs = [], [], []
    for k, yaw in enumerate(yaws):
        tiles, f_used, proj = render_pointcloud_tiles_multi(
            depth_hxw, color_sources, intr, yaw_deg=float(yaw), pitch_deg=pitch_deg,
            point_subsample=point_subsample, point_size=point_size, bg_bgr=bg_bgr,
            out_hw=out_hw, focal_length=focal_lengths[k])
        views.append(tiles)
        out_focals.append(focal_lengths[k] if focal_lengths[k] is not None else f_used)
        projs.append(proj)
    return views, out_focals, projs



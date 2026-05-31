"""Real-time GenMatter++ adapter for the StreamingVision pipeline.

This wraps the GenMatter++ DINO-features Gibbs tracker (the realtime-demo
branch, vendored under StreamingVision/genmatterpp) so it can be driven by
the latest-value-slot pipeline in render_demo.py — one Gibbs sweep per
incoming RGB frame, with persistent blob/hyperblob state.

The vendored branch ships its own JIT-stable per-step
`tracking.dino.f_tracking_sweep_dino` for offline padded scans.  For
streaming we extract just the active-step body (predict blob_means via
velocity, splice in new observations, run blob_tracking_gibbs_dino) into a
`@jax.jit` standalone function — single compile, reused every frame.

Calibration to 640x360 test.mp4
-------------------------------
  * 1/8 spatial downsample: 80 x 45 = 3600 candidate pixels (stride 8).
  * Random subsample 2925 of them (= 3 * Gibbs internal batch_size 975).
  * num_blobs = 64, num_hyperblobs = 4 (configurable via YAML).
  * Intrinsics + per-blob hyperparameters loaded from
    ``configs/streaming_default.yaml`` (or ``streaming_general.yaml`` once
    the multi-video calibrator has produced it).  ``sigma_F`` defaults to
    2.0 (the realtime-demo branch's setting; lower values produced an
    outlier-dominated visualization).
  * Depth model returns relative inverse depth; we min/max-normalize per
    frame and invert to a pseudo-Z in roughly [0.83, 5.0] m.
  * DINO PCA: fit a 384 -> 32 basis on the first frame's per-pixel features
    and freeze it; per-dim z-score so the model's sigma_F is appropriately
    scaled (matches DAVIS's pre-normalized `gaussian_means/_stds`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
import yaml

# Vendored GenMatter++ realtime-demo branch.  `genmatterpp/__init__.py` adds
# its own directory to sys.path so the inner `genmatter` package is importable
# without modifying any vendored file.
import genmatterpp  # noqa: F401 — side effect: adds genmatterpp/ to sys.path

import jax
import jax.numpy as jnp
from jax.random import key as jkey

from genmatter.datatypes import *  # noqa: F401,F403 — load first for circular import
from genmatter.model_3d import *   # noqa: F401,F403 — see VENDORED.md circular-import note
from genmatter.inference import *  # noqa: F401,F403
from genmatter.utils import (
    make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob,
    make_hierarchical_kmeans_chm_with_SAM_segmentations,
    sample_covariance_matrix_numpy,
)

from genmatter.tracking.dino import (
    GenMatter_Hyperparams_DINO,
    GenMatter_model_dino,
    blob_tracking_gibbs_dino,
    init_gibbs_sweep_dino,
    model_jimportance,  # pre-jitted GenMatter_model_dino.importance
    estimate_feature_sigmas_from_chm,
    FEATURE_SIGMA_FLOOR,
    # Module-level Gibbs building blocks (re-exported into dino's namespace via
    # `from genmatter.inference import *`).  Imported here so the streaming-local
    # sweep override below can replicate dino.blob_tracking_gibbs_dino's schedule
    # without editing the vendored file (VENDORED.md: no edits to genmatterpp/).
    gibbs_blob_assignments_dino,
    gibbs_blob_weights,
    gibbs_blob_means,
    gibbs_blob_features_dino,
    gibbs_blob_vel_means,
    gibbs_blob_vel_covs,
    gibbs_hyperblob_assignments_dino,
    gibbs_hyperblob_means,
    gibbs_hyperblob_covs,
    gibbs_hyperblob_rot,
    gibbs_hyperblob_trans,
    gibbs_hyperblob_features_dino,
)

from genjax import ChoiceMapBuilder as _C


# ---------------- Calibration constants ----------------

@dataclass(frozen=True)
class Intrinsics:
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 180.0


DEFAULT_INTRINSICS = Intrinsics()

STRIDE = 8                 # 1/8 spatial downsample of 640x360 -> 80x45 = 3600
N_KEEP = 2925              # exact multiple of Gibbs internal batch_size 975
NUM_BLOBS = 64
NUM_HYPERBLOBS = 4
FEATURE_DIM = 32           # PCA-reduce DINOv2-S 384-dim to this; fit on frame 0


# ---------------- 64-entry BGR palette ----------------

def _build_palette(n: int = NUM_BLOBS, seed: int = 0) -> np.ndarray:
    """Visually-distinct random BGR palette, matches GenMatter++ DAVIS viz style
    (run_davis_tracking.py:700 — random per-blob color)."""
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
# tracked object (the car / first judoka). The earlier muted-gray override here
# was tied to the GT-propagation cluster hack (since removed), where id 0 meant
# the bg sentinel — under the real frozen-hyperblob view it would paint the
# tracked object gray and bury it in the background, so we give id 0 a vivid,
# saturated color instead (so the seeded object visually pops).
HYPERBLOB_PALETTE[0] = np.array([60, 220, 60], dtype=np.uint8)   # vivid green (BGR)


def _build_vivid_fg_colors(n: int) -> np.ndarray:
    """``n`` maximally-distinct VIVID/saturated BGR colors for the foreground
    (seeded-object) hyperblobs in the CLUSTER tiles.

    Hue is spread on the golden-ratio sequence (so even a handful of instances
    land far apart on the wheel) at full saturation/value, so each tracked
    object reads as a punchy, high-chroma color that pops against the muted
    background. Id 0 is pinned to the established vivid green so the primary
    tracked object keeps its identity across the prior renders.
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


# ---------------- Geometry ----------------

def subsample_indices(h: int = 360, w: int = 640, stride: int = STRIDE,
                       n_keep: int = N_KEEP, seed: int = 0) -> np.ndarray:
    """Return flat indices into the (h // stride) x (w // stride) downsampled
    grid — one entry per kept datapoint.  Sorted so neighbour lookups in
    rendering stay coherent.
    """
    gh, gw = h // stride, w // stride
    total = gh * gw
    if n_keep > total:
        raise ValueError(f"n_keep={n_keep} > grid {gh}x{gw}={total}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(total)[:n_keep]
    idx.sort()
    return idx.astype(np.int32)


def _depth_to_Z(d_raw: np.ndarray) -> np.ndarray:
    """DepthAnythingV2-Small outputs relative inverse depth (disparity-ish).
    Map to a pseudo-Z roughly in [0.83, 5.0] m — closer pixels get small Z,
    farther pixels get larger Z.  Scale is arbitrary; what matters is it's
    consistent across frames so the K-means-derived priors stay valid.
    """
    d_min = float(d_raw.min())
    d_max = float(d_raw.max())
    d_norm = (d_raw - d_min) / (d_max - d_min + 1e-6)   # [0, 1]
    Z = 1.0 / (d_norm * 0.8 + 0.2)                       # [0.83, 5.0]
    return Z.astype(np.float32)


def unproject(depth_hxw: np.ndarray, flow_2xhxw: np.ndarray,
               indices: np.ndarray, intr: Intrinsics = DEFAULT_INTRINSICS,
               stride: int = STRIDE) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (depth, optical flow) into 3D positions and 3D velocities at
    the chosen subsampled pixel locations.

    depth_hxw : (H, W) float — raw depth-model output (relative inverse depth)
    flow_2xhxw: (2, H, W) float — SEA-RAFT forward flow, [fx, fy] in pixels
    indices   : (N,) int — flat indices into the (H/stride) x (W/stride) grid

    Returns
    -------
    positions  : (N, 3) float32 — (X, Y, Z) at current frame
    velocities : (N, 3) float32 — (X, Y, Z) at next frame minus current
    """
    H, W = depth_hxw.shape
    gh, gw = H // stride, W // stride
    iy = (indices // gw).astype(np.int32)   # grid row
    ix = (indices %  gw).astype(np.int32)   # grid col
    py = iy * stride                         # pixel row, 0..H-stride
    px = ix * stride                         # pixel col

    Z_full = _depth_to_Z(depth_hxw)
    Z = Z_full[py, px]
    X = (px.astype(np.float32) - intr.cx) * Z / intr.fx
    Y = (py.astype(np.float32) - intr.cy) * Z / intr.fy
    positions = np.stack([X, Y, Z], axis=1)

    fx_pix = flow_2xhxw[0]
    fy_pix = flow_2xhxw[1]
    du = fx_pix[py, px]
    dv = fy_pix[py, px]
    # Next-frame pixel; clamp to image so we can sample depth there.
    nx = np.clip(px.astype(np.float32) + du, 0, W - 1)
    ny = np.clip(py.astype(np.float32) + dv, 0, H - 1)
    Z_next = Z_full[ny.astype(np.int32), nx.astype(np.int32)]
    Xn = (nx - intr.cx) * Z_next / intr.fx
    Yn = (ny - intr.cy) * Z_next / intr.fy
    velocities = np.stack([Xn - X, Yn - Y, Z_next - Z], axis=1).astype(np.float32)
    return positions.astype(np.float32), velocities


# ---------------- DINO features ----------------

def _upsample_features_to_grid(feat: np.ndarray, target_h: int, target_w: int,
                                grid_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """feat: (P, D) flat DINO patches (P = gh_p * gw_p).  We reshape to
    (gh_p, gw_p, D), then resize each channel to (target_h, target_w) via
    cv2.INTER_LINEAR.  cv2.resize handles multichannel only up to 4; for
    arbitrary D we batch over channels.

    grid_hw : (gh_p, gw_p) patch grid.  Pass the FeatureWorker's grid_hw for a
    non-square (dense) DINO grid; if None, assume a square grid (back-compat).
    """
    n = feat.shape[0]
    if grid_hw is not None:
        gh_p, gw_p = int(grid_hw[0]), int(grid_hw[1])
    else:
        gh_p = gw_p = int(round(np.sqrt(n)))
    assert gh_p * gw_p == n, f"patch grid {gh_p}x{gw_p} != {n} patches"
    D = feat.shape[1]
    g = feat.reshape(gh_p, gw_p, D).astype(np.float32)
    # cv2.resize processes up to 4 channels per call; iterate in blocks of 4.
    out = np.empty((target_h, target_w, D), dtype=np.float32)
    for c0 in range(0, D, 4):
        c1 = min(c0 + 4, D)
        out[:, :, c0:c1] = cv2.resize(g[:, :, c0:c1], (target_w, target_h),
                                       interpolation=cv2.INTER_LINEAR)
    return out


def dino_features_to_datapoints(feat: np.ndarray, indices: np.ndarray,
                                 pca_basis: Optional[np.ndarray] = None,
                                 pca_mean: Optional[np.ndarray] = None,
                                 pca_std: Optional[np.ndarray] = None,
                                 stride: int = STRIDE,
                                 image_hw: Tuple[int, int] = (360, 640),
                                 target_dim: int = FEATURE_DIM,
                                 feat_grid_hw: Optional[Tuple[int, int]] = None,
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Upsample DINO patches to the 80x45 subsampled grid, gather at
    `indices`, then project through a frozen (or freshly-fit) PCA basis and
    z-score-normalize per-dim.  This matches GenMatter++'s DAVIS pipeline
    which pre-normalizes features with `gaussian_means` + `gaussian_stds`
    so the model's hard-coded `sigma_F = 0.2` is appropriately scaled.

    feat       : (P, D_dino) raw DINOv2-S patch features (P=256, D_dino=384).
    indices    : (N,) flat indices into (H/stride) x (W/stride) grid.
    pca_basis  : (D_dino, target_dim) or None; if None, fit on these features.
    pca_mean   : (D_dino,) or None.
    pca_std    : (target_dim,) or None; if None, computed post-projection.

    Returns
    -------
    features      : (N, target_dim) float32 — PCA+z-score features.
    pca_basis     : (D_dino, target_dim) float32.
    pca_mean      : (D_dino,) float32.
    pca_std       : (target_dim,) float32 — frozen post-projection std.
    """
    H, W = image_hw
    gh, gw = H // stride, W // stride
    up = _upsample_features_to_grid(feat, gh, gw, grid_hw=feat_grid_hw)  # (gh, gw, D_dino)
    flat = up.reshape(gh * gw, -1)                        # (gh*gw, D_dino)
    sub = flat[indices]                                   # (N, D_dino)
    if pca_basis is None:
        pca_mean = sub.mean(0).astype(np.float32)
        X = sub - pca_mean
        C = (X.T @ X) / (X.shape[0] - 1)
        _, eigvecs = np.linalg.eigh(C)
        pca_basis = eigvecs[:, -target_dim:].astype(np.float32)
        projected = (X @ pca_basis).astype(np.float32)
        # Freeze the per-dim std at frame 0 so subsequent frames stay in the
        # same coordinate frame as the blob_features prior.
        pca_std = projected.std(0).astype(np.float32) + 1e-3
        normalized = projected / pca_std
    else:
        projected = ((sub - pca_mean) @ pca_basis).astype(np.float32)
        normalized = projected / pca_std
    return normalized.astype(np.float32), pca_basis, pca_mean, pca_std


# ---------------- GenMatter++ init + per-frame step ----------------

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "streaming_default.yaml"


def load_yaml_hypers(config_path: Optional[Path] = None) -> dict:
    """Load the ``tracking.hyperparams`` section of a streaming YAML config
    (clone of the GenMatter++ branch's ``custom_default.yaml`` shape)."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _build_hypers(kmeans_chm, features_np, yaml_cfg, *, num_blobs, num_hyperblobs,
                   num_datapoints, debug_capture: Optional[dict] = None):
    """Derive Psi_* from the K-means cluster covs (empirical), then layer the
    YAML-driven hyperparameters on top.  Mirrors the branch's
    `_build_hypers_from_kmeans` (`genmatter/tracking/dino.py:957`) but reads
    its scalar priors from our streaming YAML so calibrated values can be
    consumed directly.

    When ``tracking.use_calibrated_priors`` is true, reads ``Psi_B/H/V``,
    ``nu_B/H/V``, ``mu_H`` from ``tracking.hyperparams`` (the calibrator's
    output) instead of deriving them from the K-means seed.  Plumbing for the
    self-supervised calibrator (see plan
    `~/.claude/plans/piped-discovering-pebble.md`)."""
    roi_blob_indices = jnp.arange(num_blobs)
    roi_hyperblob_indices = jnp.arange(num_hyperblobs)

    empirical_mu_H = jnp.median(kmeans_chm['datapoints', 'datapoint_positions'], axis=0)
    empirical_Psi_B = jnp.median(kmeans_chm['blobs', 'blob_covs'][roi_blob_indices], axis=0)
    empirical_Psi_H = jnp.median(kmeans_chm['hyperblobs', 'hyperblob_covs'][roi_hyperblob_indices], axis=0)
    empirical_Psi_V = jnp.median(kmeans_chm['blobs', 'blob_vel_covs'][roi_blob_indices], axis=0)
    # max(.,1) guards the SAM-init path, where many small hyperblobs/blobs can
    # drive the per-group mean below 1 (int -> 0 -> degenerate inverse-Wishart).
    # For the default k-means init these are ~16 / ~45, so the floor is inert.
    mean_blobs_per_hyperblob = num_blobs / max(num_hyperblobs, 1)
    empirical_nu_H = f_(max(int(mean_blobs_per_hyperblob), 1))  # noqa: F405 — f_ from datatypes
    mean_points_per_blob = num_datapoints / max(num_blobs, 1)
    empirical_nu_B = empirical_nu_V = f_(max(int(mean_points_per_blob), 1))  # noqa: F405

    gaussian_means = features_np.mean(0).astype(np.float32)
    gaussian_stds = features_np.std(0).astype(np.float32) + 1e-3

    hp = yaml_cfg["tracking"]["hyperparams"]
    use_calibrated = bool(yaml_cfg["tracking"].get("use_calibrated_priors", False))
    if use_calibrated:
        mu_H_value = jnp.asarray(hp["mu_H"], dtype=jnp.float32)
        Psi_B_value = jnp.asarray(hp["Psi_B"], dtype=jnp.float32)
        Psi_H_value = jnp.asarray(hp["Psi_H"], dtype=jnp.float32)
        Psi_V_value = jnp.asarray(hp["Psi_V"], dtype=jnp.float32)
        nu_B_value = f_(float(hp["nu_B"]))  # noqa: F405
        nu_H_value = f_(float(hp["nu_H"]))  # noqa: F405
        nu_V_value = f_(float(hp["nu_V"]))  # noqa: F405
    else:
        mu_H_value = empirical_mu_H
        Psi_B_value = empirical_Psi_B
        Psi_H_value = empirical_Psi_H
        Psi_V_value = empirical_Psi_V
        nu_B_value = empirical_nu_B
        nu_H_value = empirical_nu_H
        nu_V_value = empirical_nu_V
    if debug_capture is not None:
        # Exposes the K-means-derived values to the calibrator's plumbing
        # smoke test so it can emit a streaming_general_smoke.yaml that
        # round-trips byte-for-byte through the use_calibrated_priors path.
        debug_capture["empirical_mu_H"] = np.asarray(empirical_mu_H, dtype=np.float32)
        debug_capture["empirical_Psi_B"] = np.asarray(empirical_Psi_B, dtype=np.float32)
        debug_capture["empirical_Psi_H"] = np.asarray(empirical_Psi_H, dtype=np.float32)
        debug_capture["empirical_Psi_V"] = np.asarray(empirical_Psi_V, dtype=np.float32)
        debug_capture["empirical_nu_B"] = float(empirical_nu_B)
        debug_capture["empirical_nu_H"] = float(empirical_nu_H)
        debug_capture["empirical_nu_V"] = float(empirical_nu_V)
    # Step 2 — empirical-Bayes feature variances: when calibrate_feature_sigmas
    # is set, override the YAML sigma_F / sigma_F_H (the "sigma_F~9 too loose"
    # bug) with the within-partition feature variance estimated directly from the
    # k-means seed.  Same seam that already empirical-Bayes-fits the geometry
    # priors above; native JAX, runs once at init.
    if yaml_cfg["tracking"].get("calibrate_feature_sigmas", False):
        sigma_F_est, sigma_F_H_est = estimate_feature_sigmas_from_chm(kmeans_chm, num_hyperblobs)
        sigma_F_value = f_(float(jnp.maximum(sigma_F_est, FEATURE_SIGMA_FLOOR)))  # noqa: F405
        sigma_F_H_value = f_(float(jnp.maximum(sigma_F_H_est, FEATURE_SIGMA_FLOOR)))  # noqa: F405
        print(f"[genmatter_rt._build_hypers] calibrate_feature_sigmas: "
              f"sigma_F {float(hp['sigma_F']):.4g} -> {float(sigma_F_value):.4g}, "
              f"sigma_F_H {float(hp.get('sigma_F_H', 2.0)):.4g} -> {float(sigma_F_H_value):.4g}",
              flush=True)
    else:
        sigma_F_value = f_(float(hp["sigma_F"]))  # noqa: F405
        sigma_F_H_value = f_(float(hp.get("sigma_F_H", 2.0)))  # noqa: F405
    return GenMatter_Hyperparams_DINO.create(
        mu_F=jnp.array(gaussian_means),
        sigma_F_prior=jnp.array(gaussian_stds),
        sigma_F=sigma_F_value,
        sigma_F_H=sigma_F_H_value,
        outlier_prob=f_(float(hp["outlier_prob"])),  # noqa: F405
        outlier_velocity_gamma_shape=f_(float(hp["outlier_velocity_gamma_shape"])),  # noqa: F405
        outlier_velocity_gamma_rate=f_(float(hp["outlier_velocity_gamma_rate"])),  # noqa: F405
        alpha=f_(float(hp["alpha"])),  # noqa: F405
        beta=f_(float(hp["beta"])),  # noqa: F405
        mu_H=mu_H_value,
        sigma_H=f_(float(hp["sigma_H"])),  # noqa: F405
        nu_H=nu_H_value,
        Psi_H=Psi_H_value,
        nu_B=nu_B_value,
        Psi_B=Psi_B_value,
        sigma_V=f_(float(hp["sigma_V"])),  # noqa: F405
        nu_V=nu_V_value,
        Psi_V=Psi_V_value,
        translation_gaussian_scale=snp(f_(float(hp["translation_gaussian_scale"]))),  # noqa: F405
        translation_max_radius=snp(float(hp["translation_max_radius"])),  # noqa: F405
        translation_num_radii_cells=snp(int(hp["translation_num_radii_cells"])),  # noqa: F405
        translation_theta_step_deg=snp(float(hp["translation_theta_step_deg"])),  # noqa: F405
        rotation_vmf_kappa=snp(f_(float(hp["rotation_vmf_kappa"]))),  # noqa: F405
        rotation_angle_max_deg=snp(float(hp["rotation_angle_max_deg"])),  # noqa: F405
        rotation_angle_step_deg=snp(float(hp["rotation_angle_step_deg"])),  # noqa: F405
        n_hyperblobs=num_hyperblobs,
        n_blobs=num_blobs,
        n_datapoints=num_datapoints,
    )


def instance_mask_to_rgb_grid(mask_native: np.ndarray, grid_h: int,
                              grid_w: int) -> np.ndarray:
    """Convert a native-resolution uint16 SAM2 instance-id map into the
    ``(grid_h, grid_w, 3)`` RGB pseudo-color mask that ``init_state``'s
    SAM-frame-0 branch (and ``make_hierarchical_kmeans_chm_with_SAM_segmentations``)
    expect: instance id 0 -> background white ``[255, 255, 255]``; each non-zero
    id -> a distinct color.

    The downstream SAM helper only cares that (a) background is exactly white and
    (b) every distinct instance id maps to a distinct non-white color (it calls
    ``np.unique`` over the colors and assigns sequential labels).  We therefore
    encode the integer id *bijectively* into the three channels
    (b = id & 255, g = (id >> 8) & 255, r = (id >> 16) & 255), which is
    collision-free across the entire uint16 id range — SAM2 ids reach the
    hundreds, so a fixed-size palette would alias distinct instances onto a single
    hyperblob.  No non-zero uint16 id can encode to pure white (that needs
    id == 0xFFFFFF), so the background sentinel stays unambiguous.

    Resize is INTER_NEAREST so ids are never interpolated into nonexistent
    colors — mirrors the live demo's downsample (render_demo.py:552) but starts
    from an id-map instead of an RGB PNG.
    """
    mask = np.asarray(mask_native)
    if mask.ndim == 3:
        mask = mask[..., 0]
    small = cv2.resize(mask.astype(np.int32), (grid_w, grid_h),
                       interpolation=cv2.INTER_NEAREST).astype(np.int64)
    ids = np.clip(small, 0, None)
    rgb = np.empty((grid_h, grid_w, 3), dtype=np.uint8)
    rgb[..., 0] = (ids & 0xFF).astype(np.uint8)
    rgb[..., 1] = ((ids >> 8) & 0xFF).astype(np.uint8)
    rgb[..., 2] = ((ids >> 16) & 0xFF).astype(np.uint8)
    rgb[ids == 0] = (255, 255, 255)             # background sentinel
    return rgb


def _seed_grid_to_datapoint_instance_ids(sam_segmentation: np.ndarray,
                                          subsample_indices: np.ndarray) -> Tuple[np.ndarray, int]:
    """Map the ``(gh, gw, 3)`` RGB seed grid to a per-datapoint integer instance
    id, byte-identical to ``make_hierarchical_kmeans_chm_with_SAM_segmentations``'s
    own color->label assignment (background white ``[255,255,255]`` -> 0; every
    other distinct color -> sequential 1..K in ``np.unique`` order).  Returns
    ``(instance_ids_N, num_object_instances)``.
    """
    flat = np.asarray(sam_segmentation).reshape(-1, 3)
    unique_colors = np.unique(flat, axis=0)
    color_to_label = {}
    label = 0
    for color in unique_colors:
        if np.array_equal(color, [255, 255, 255]):
            color_to_label[tuple(int(c) for c in color)] = 0
        else:
            label += 1
            color_to_label[tuple(int(c) for c in color)] = label
    integer_mask_full = np.array(
        [color_to_label[tuple(int(c) for c in px)] for px in flat], dtype=np.int64)
    inst_ids = integer_mask_full[np.asarray(subsample_indices)]
    return inst_ids, int(label)


def _purify_object_seed_chm(kmeans_chm, sam_segmentation: np.ndarray,
                            subsample_indices: np.ndarray):
    """Seed-purity fix (``tracking.pure_object_seed``): rebuild the blob ->
    hyperblob assignment so each OBJECT-instance hyperblob contains EXACTLY the
    blobs whose frame-0 datapoints are majority-inside that instance's mask, and
    all other blobs are pushed into BACKGROUND hyperblobs (the background is NOT
    collapsed — its existing multi-hyperblob split is preserved).

    Host-side CHM construction at init (not in the per-frame hot loop), so a plain
    numpy rebuild is fine.  The object-instance hyperblob ids stay the LOW ids
    ``0..num_object_instances-1`` (matching the SAM builder's convention that the
    vivid-fg cluster palette + ``freeze_hyperblob_assignment`` rely on); only the
    blob->hyperblob membership (and the derived per-hyperblob geometry/feature
    arrays) is rewritten.  The frozen membership is therefore the PURIFIED one.

    No-op-equivalent when the seed grid already partitions blobs cleanly (the
    common GT-seeded case, where the CHM is already pure); meaningful when the
    seed instances are noisy (SAM) so a blob's datapoints straddle instance
    boundaries.
    """
    inst_ids, num_obj = _seed_grid_to_datapoint_instance_ids(
        sam_segmentation, subsample_indices)            # (N,), int

    blob_assign = np.asarray(kmeans_chm['datapoints', 'blob_assignments']).astype(np.int64)
    blob_to_hb_old = np.asarray(kmeans_chm['blobs', 'hyperblob_assignments']).astype(np.int64)
    num_blobs = int(blob_to_hb_old.shape[0])
    num_hb_old = int(np.asarray(kmeans_chm['hyperblobs', 'hyperblob_means']).shape[0])

    # Per-blob majority instance id (0 = background) over its frame-0 datapoints.
    blob_majority_inst = np.zeros(num_blobs, dtype=np.int64)
    for b in range(num_blobs):
        m = (blob_assign == b)
        if m.any():
            vals = inst_ids[m]
            counts = np.bincount(vals, minlength=num_obj + 1)
            blob_majority_inst[b] = int(np.argmax(counts))

    # New blob->hyperblob map: object instance j -> hyperblob (j-1) (low ids);
    # every other (background) blob keeps its OLD hyperblob, shifted up by num_obj
    # so background ids never collide with the object ids.  This preserves the
    # background's existing multi-hyperblob split (no collapse).
    blob_to_hb_new = np.empty(num_blobs, dtype=np.int64)
    is_obj_blob = blob_majority_inst > 0
    blob_to_hb_new[is_obj_blob] = blob_majority_inst[is_obj_blob] - 1
    blob_to_hb_new[~is_obj_blob] = num_obj + blob_to_hb_old[~is_obj_blob]

    # Compact the realized hyperblob ids to a contiguous 0..K-1 range, keeping the
    # object ids first (so vivid-fg palette + num_fg accounting stay correct).
    used = np.unique(blob_to_hb_new)
    remap = {int(old): new for new, old in enumerate(used)}
    blob_to_hb_new = np.array([remap[int(h)] for h in blob_to_hb_new], dtype=np.int64)
    num_hb_new = int(len(used))

    # Rebuild per-hyperblob geometry from the member blobs' datapoints (means/covs
    # /weights) and the member blobs' velocities (rot=I, trans=mean vel) — the
    # same statistics the SAM builder computes, now over the purified partition.
    positions = np.asarray(kmeans_chm['datapoints', 'datapoint_positions'])
    vels = np.asarray(kmeans_chm['datapoints', 'datapoint_vels'])
    total_points = positions.shape[0]
    hb_means = np.zeros((num_hb_new, 3), dtype=np.float32)
    hb_covs = np.tile(np.eye(3, dtype=np.float32) * 0.01, (num_hb_new, 1, 1))
    hb_weights = np.zeros((num_hb_new,), dtype=np.float32)
    hb_rot = np.tile(np.eye(3, dtype=np.float32), (num_hb_new, 1, 1))
    hb_trans = np.zeros((num_hb_new, 3), dtype=np.float32)
    dp_hb = blob_to_hb_new[blob_assign]                 # per-datapoint new hyperblob
    global_mean = positions.mean(axis=0)
    for h in range(num_hb_new):
        pts = positions[dp_hb == h]
        if pts.shape[0] > 0:
            hb_means[h] = pts.mean(axis=0)
            hb_weights[h] = pts.shape[0] / total_points
            if pts.shape[0] > 3:
                hb_covs[h] = sample_covariance_matrix_numpy(pts).astype(np.float32)
            hb_trans[h] = vels[dp_hb == h].mean(axis=0)
        else:
            hb_means[h] = global_mean

    chm = kmeans_chm
    chm = chm | _C['blobs', 'hyperblob_assignments'].set(jnp.array(blob_to_hb_new))
    chm = chm | _C['hyperblobs', 'hyperblob_means'].set(jnp.array(hb_means))
    chm = chm | _C['hyperblobs', 'hyperblob_covs'].set(jnp.array(hb_covs))
    chm = chm | _C['hyperblobs', 'hyperblob_weights'].set(jnp.array(hb_weights))
    chm = chm | _C['hyperblobs', 'hyperblob_rot_vels'].set(jnp.array(hb_rot))
    chm = chm | _C['hyperblobs', 'hyperblob_trans_vels'].set(jnp.array(hb_trans))
    return chm, num_hb_new


def init_state(positions: np.ndarray, velocities: np.ndarray, features: np.ndarray,
                key, *, yaml_cfg: Optional[dict] = None,
                num_blobs: int = NUM_BLOBS, num_hyperblobs: int = NUM_HYPERBLOBS,
                num_warmup_sweeps: Optional[int] = None, verbose: bool = False,
                sam_segmentation: Optional[np.ndarray] = None,
                subsample_indices: Optional[np.ndarray] = None,
                debug_capture: Optional[dict] = None):
    """Bootstrap the GenMatter_State_DINO on the first valid (depth, flow,
    features) tuple.  Runs K-means + ``num_warmup_sweeps`` Gibbs iterations
    (the slow startup, dominated by JIT compile of the Gibbs sweep).

    SAM-anchored semantic init (Step 3): when ``tracking.use_sam_frame0`` is set
    and a frame-0 SAM mask is supplied, each SAM instance seeds its own
    hyperblob (and unsegmented regions get k-means hyperblobs) via
    ``make_hierarchical_kmeans_chm_with_SAM_segmentations`` instead of the flat
    position-only k-means — so streaming clusters START semantic.  ``num_blobs``
    / ``num_hyperblobs`` then follow the SAM partition (the returned state's
    actual counts), not the YAML values.

    ``sam_segmentation`` must be the RGB pseudo-color SAM mask already
    downsampled to the stride-8 grid that ``subsample_indices`` indexes into
    (i.e. shape ``(H//stride, W//stride, 3)``), so the per-datapoint instance
    labels line up with ``positions``/``velocities``.
    """
    def _log(msg):
        if verbose:
            print(f"[genmatter_rt.init_state +{_t.monotonic() - _t_start:.1f}s] {msg}", flush=True)
    import time as _t
    _t_start = _t.monotonic()

    if yaml_cfg is None:
        yaml_cfg = load_yaml_hypers()
    tracking_cfg = yaml_cfg["tracking"]
    if num_warmup_sweeps is None:
        num_warmup_sweeps = int(tracking_cfg.get("init_gibbs_sweeps", 15))
    N = positions.shape[0]
    tracked_points = positions[None, ...]                       # (1, N, 3)
    tracked_motion_vectors = velocities[None, ...]              # (1, N, 3)

    use_sam = (bool(tracking_cfg.get("use_sam_frame0", False))
               and sam_segmentation is not None and subsample_indices is not None)
    if use_sam:
        sam_h, sam_w = sam_segmentation.shape[:2]
        _log(f"entering SAM-frame-0 hierarchical init ({sam_h}x{sam_w} mask)")
        kmeans_chm, _roi_b, _roi_h, num_hyperblobs = make_hierarchical_kmeans_chm_with_SAM_segmentations(
            tracked_points,
            num_blobs,
            np.asarray(sam_segmentation),
            (sam_h, sam_w),
            subsampled_indices=np.asarray(subsample_indices),
            motion_vectors=tracked_motion_vectors,
        )
        _log(f"SAM init done ({num_hyperblobs} hyperblobs); building hypers + overlays")
        # Seed-purity fix (default OFF -> bit-exact to the prior path).  Rewrite
        # the blob->hyperblob membership so each object-instance hyperblob holds
        # EXACTLY the blobs majority-inside that instance and all other blobs go
        # to (still-multiple) background hyperblobs.  Host-side CHM rebuild at
        # init (not the per-frame loop); the purified membership is what
        # freeze_hyperblob_assignment then freezes.
        if bool(tracking_cfg.get("pure_object_seed", False)):
            kmeans_chm, num_hyperblobs = _purify_object_seed_chm(
                kmeans_chm, np.asarray(sam_segmentation), np.asarray(subsample_indices))
            _log(f"pure_object_seed: purified CHM -> {num_hyperblobs} hyperblobs")
    else:
        # All datapoints are ROI; pass num_roi_hyperblobs == num_hyperblobs so all
        # hyperblob slots get blob assignments — initialize_model_with_dino doesn't
        # expose num_roi_hyperblobs, so call the lower-level helper directly and
        # then overlay the DINO feature stats (mirrors run_davis_tracking.py:324-336).
        seg_mask = np.ones(N, dtype=bool)
        _log("entering K-means hierarchical init")
        kmeans_chm, _roi_b, _roi_h = make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(
            tracked_points,
            num_blobs,
            num_hyperblobs,
            segmentation_mask=seg_mask,
            motion_vectors=tracked_motion_vectors,
            num_roi_blobs=None,
            subsampled_indices=None,
            num_roi_hyperblobs=num_hyperblobs,
        )
        _log("K-means done; building hypers + ChoiceMap overlays")

    # Overlay DINO features into the choice map (per-blob feature mean +
    # per-datapoint features) — same pattern as initialize_model_with_dino.
    # Size blob_features to the chm's actual blob-slot count (blob_means rows),
    # which the SAM-init partition sets dynamically; unpopulated slots stay zero.
    frame_features = features
    blob_assignments_np = np.array(kmeans_chm['datapoints', 'blob_assignments'])
    feature_dim = frame_features.shape[1]
    num_blob_slots = int(np.asarray(kmeans_chm['blobs', 'blob_means']).shape[0])
    blob_features = np.zeros((num_blob_slots, feature_dim), dtype=np.float32)
    for i in range(num_blob_slots):
        points_in_blob = np.where(blob_assignments_np == i)[0]
        if len(points_in_blob) > 0:
            blob_features[i] = np.mean(frame_features[points_in_blob], axis=0)
    kmeans_chm = kmeans_chm | _C['datapoints', 'datapoint_features'].set(jnp.array(frame_features))
    kmeans_chm = kmeans_chm | _C['blobs', 'blob_features'].set(jnp.array(blob_features))
    # Per-hyperblob feature means: vectorized segment-mean of blob_features by the
    # k-means blob->hyperblob assignments (no python loop), so the DINO model's
    # 'hyperblob_features' address is constrained at init for feature-aware clusters.
    hb_assign_init = np.asarray(kmeans_chm['blobs', 'hyperblob_assignments']).astype(np.int64)
    num_hb_init = int(np.asarray(kmeans_chm['hyperblobs', 'hyperblob_means']).shape[0])
    hb_sums = np.zeros((num_hb_init, feature_dim), dtype=np.float32)
    hb_counts = np.zeros((num_hb_init, 1), dtype=np.float32)
    np.add.at(hb_sums, hb_assign_init, blob_features)
    np.add.at(hb_counts, hb_assign_init, 1.0)
    hyperblob_features = hb_sums / np.maximum(hb_counts, 1.0)
    kmeans_chm = kmeans_chm | _C['hyperblobs', 'hyperblob_features'].set(jnp.array(hyperblob_features))

    num_datapoints = kmeans_chm['datapoints', 'datapoint_positions'].shape[0]
    num_blobs_actual = kmeans_chm['blobs', 'hyperblob_assignments'].shape[0]
    num_hb_actual = kmeans_chm['hyperblobs', 'hyperblob_means'].shape[0]

    hypers = _build_hypers(kmeans_chm, features, yaml_cfg,
                            num_blobs=num_blobs_actual,
                            num_hyperblobs=num_hb_actual,
                            num_datapoints=num_datapoints,
                            debug_capture=debug_capture)

    _log("running model_jimportance (first JIT compile of model)")
    key, k_imp = jax.random.split(key)
    init_tr, _ = model_jimportance(k_imp, kmeans_chm, (hypers,))
    init_tr.get_retval().datapoints_state.blob_assignments.block_until_ready()
    state = init_tr.get_retval()
    _log("model_jimportance done")

    _log(f"running init_gibbs_sweep_dino (num_sweeps={num_warmup_sweeps})")
    key, k_warm = jax.random.split(key)
    gibbs_wtrs = init_gibbs_sweep_dino(k_warm, state, num_sweeps=num_warmup_sweeps)
    state = gibbs_wtrs[-1].retval
    state.datapoints_state.blob_assignments.block_until_ready()
    _log("init_gibbs_sweep_dino done")

    # After init, swap to the tracking-time outlier_prob (typically several
    # orders of magnitude lower than init) so frame-to-frame Gibbs doesn't
    # keep generating phantom outlier assignments.
    tracking_outlier = float(tracking_cfg.get("tracking_outlier_prob", 1e-28))
    state = state.replace({'hypers': {'outlier_prob': f_(tracking_outlier)}})  # noqa: F405
    return state, key


def blob_tracking_gibbs_dino_streaming(key, genmatter_state, *,
                                       feature_aware_final: bool,
                                       final_outlier: bool,
                                       freeze_hyperblob_assignment: bool = False,
                                       blob_means_updates: int = 15,
                                       freeze_blob_features: bool = False,
                                       feature_update_damping=None,
                                       blob_feat_anchor=None,
                                       hb_feat_anchor=None):
    """Streaming-local re-implementation of
    ``genmatter.tracking.dino.blob_tracking_gibbs_dino`` (dino.py:743-832) whose
    FINAL assignment step (the dino.py:826 ``update_blob_assignments_with_outlier``
    step) is configurable, so we can ablate it WITHOUT editing the vendored file.

    The vendored schedule ends its assignment phase with a POSITION-ONLY +
    outlier-injecting assignment, so the labels the renderer reads are
    position-driven (DINO features washed out) with outliers added as the last
    word — the suspected cause of "per-pixel CLUSTER view worse than the raw
    DINO PCA, too many outliers".

    Flags (closed over as Python bools at build time -> NOT jax-traced, so the
    static ``fori_loop`` counts and the choice of final-step body are baked into
    one compiled program; never forces a mid-run recompile):

      feature_aware_final=False  (baseline)  : final step is IDENTICAL to dino.py
          -> gibbs_blob_assignments_dino(position_only=True,
                                         disable_outlier_prob=False) + weights.
          Reproduces current streaming behavior bit-for-bit.

      feature_aware_final=True   (fix)       : final step is a FULL position +
          feature (+ velocity, since neither position_only nor velocity_only nor
          feature_only is set) likelihood assignment
          -> gibbs_blob_assignments_dino(position_only=False, feature_only=False,
                                         disable_outlier_prob=(not final_outlier))
             + weights.
          ``final_outlier`` toggles whether outliers are enabled on that step.

    Everything else (3x hyperblob, 2x feature-only assign, 1x position-only
    assign, 15x means, 3x features, then 15x vel-means, 15x vel-covs, 3x
    features, 3x hyperblob) is byte-identical to dino.py:818-830.

    ``freeze_hyperblob_assignment`` (Python bool, build-time) — when True the
    per-frame ``hyperblob_update_loop`` SKIPS the blob->hyperblob membership
    re-sampling (``gibbs_hyperblob_assignments_dino``, dino.py:804) while KEEPING
    the geometry/appearance updates (means/covs/rot/trans/features_dino).  This
    makes cluster identity sticky: the frame-0 seeded blob->hyperblob mapping is
    preserved across the stream so the ``pixel_by_cluster`` view tracks the same
    object instead of re-partitioning into coarse spatial clusters every frame.
    Default False reproduces the vendored behavior bit-for-bit.

    ``blob_means_updates`` (Python int, build-time) — number of
    ``gibbs_blob_means`` refinement iterations per frame (the dino.py:824
    ``fori_loop(0, 15, update_blob_means, ...)``).  Each iteration re-solves the
    blob-mean posterior, which blends the constant-velocity PREDICTION
    (``next_blob_means``, the prior carried in via the hyperblob mean + velocity
    affine terms) against the DATA term ``blob_cov_inv * N_l * S_l`` — so MORE
    iterations let a frozen object blob's mean be pulled further toward whatever
    datapoints currently fall in it (including background that leaked in, since
    DINO appearance is shared near the object).  FEWER iterations keep the mean
    nearer the velocity prediction (anti-drift; approximates a stronger
    velocity-anchored prior — lever 2 — without editing the vendored
    ``gibbs_blob_means``).  CONVERGE-round anti-bleed lever: default 15
    reproduces dino.py/the vendored schedule bit-for-bit.

    ``feature_update_damping`` / ``blob_feat_anchor`` / ``hb_feat_anchor``
    (anti-drift, DAMPED generalization of ``freeze_blob_features``) — when
    ``blob_feat_anchor`` is supplied (TRACED arrays, NOT static), every per-frame
    Gibbs feature update is followed by a blend back toward the frame-0 anchor:
    ``f = (1 - d)*anchor + d*f_gibbs`` with ``d = feature_update_damping`` (a
    TRACED jnp scalar in [0, 1]).  d=1 reproduces the vendored pure-Gibbs update;
    d=0 holds the frame-0 appearance frozen (== ``freeze_blob_features``); d in
    (0, 1) is a slow-EMA that lets a deforming object's appearance follow while
    resisting runaway drift into leaked background.  Because the anchor and the
    damping are TRACED args, ONE compiled program serves every damping value and
    every video of equal shape (no per-value / per-video recompile).  When
    ``blob_feat_anchor is None`` (the default) NONE of this runs and the existing
    vendored / ``freeze_blob_features`` static paths are byte-identical.
    """
    # Build-time (Python, NOT traced) branch: blend only when an anchor is given.
    _blend = blob_feat_anchor is not None

    def update_blob_assignments_position_only(i, carry):
        key, st = carry
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_assignments_dino(
            gibbs_key, st, position_only=True, disable_outlier_prob=True
        )
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_weights(gibbs_key, st)
        return key, st

    def update_blob_assignments_feature_only(i, carry):
        key, st = carry
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_assignments_dino(
            gibbs_key, st, feature_only=True, disable_outlier_prob=False
        )
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_weights(gibbs_key, st)
        return key, st

    def update_blob_assignments_final_baseline(i, carry):
        # dino.py:755-763 update_blob_assignments_with_outlier: position-only,
        # outliers ENABLED.  The vendored "last assignment word".
        key, st = carry
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_assignments_dino(
            gibbs_key, st, position_only=True, disable_outlier_prob=False
        )
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_weights(gibbs_key, st)
        return key, st

    def update_blob_assignments_final_feature_aware(i, carry):
        # Fix: FULL position+feature+velocity likelihood as the last assignment,
        # so DINO features (and motion) drive the labels the renderer reads.
        key, st = carry
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_assignments_dino(
            gibbs_key, st, position_only=False, feature_only=False,
            disable_outlier_prob=(not final_outlier),
        )
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_weights(gibbs_key, st)
        return key, st

    def update_blob_velocities(i, carry):
        key, st = carry
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_vel_means(gibbs_key, st)
        return key, st

    def update_blob_velocity_covariances(i, carry):
        key, st = carry
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_vel_covs(gibbs_key, st)
        return key, st

    def update_blob_means(i, carry):
        key, st = carry
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_means(gibbs_key, st)
        return key, st

    def update_blob_features_dino(i, carry):
        key, st = carry
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_features_dino(gibbs_key, st)
        if _blend:
            # Damped appearance update: pull the freshly-Gibbs'd blob features
            # back toward the frozen frame-0 anchor by (1 - damping).  d=1 ->
            # pure Gibbs (vendored); d=0 -> frozen anchor.  Traced scalar/arrays,
            # so no per-value / per-video recompile.
            f_g = st.blobs_state.blob_features
            f_b = ((1.0 - feature_update_damping) * blob_feat_anchor
                   + feature_update_damping * f_g)
            st = st.replace({'blobs_state': {'blob_features': f_b}})
        return key, st

    def hyperblob_update_loop(i, carry):
        key, st = carry
        # freeze_hyperblob_assignment is a Python bool closed over at build time
        # (NOT jax-traced), so this branch is resolved once at compile: when True
        # the per-frame blob->hyperblob membership re-sampling (dino.py:804) is
        # SKIPPED, preserving the frame-0 (k-means / SAM / GT) seeded cluster
        # identity, while the geometry/appearance updates below still let each
        # cluster's mean/cov/rot/trans/feature follow its member blobs as they move.
        if not freeze_hyperblob_assignment:
            key, gibbs_key = jax.random.split(key)
            st = gibbs_hyperblob_assignments_dino(gibbs_key, st)
        key, gibbs_key = jax.random.split(key)
        st = gibbs_hyperblob_means(gibbs_key, st)
        key, gibbs_key = jax.random.split(key)
        st = gibbs_hyperblob_covs(gibbs_key, st)
        key, gibbs_key = jax.random.split(key)
        st = gibbs_hyperblob_rot(gibbs_key, st)
        key, gibbs_key = jax.random.split(key)
        st = gibbs_hyperblob_trans(gibbs_key, st)
        # freeze_blob_features (build-time bool): when True, the per-frame appearance
        # (hyperblob feature mean) update is SKIPPED so the frame-0 seeded appearance
        # is held FROZEN across the stream (mirrors the frozen-centroid feature
        # classifier that beats the drifting tracker). Default False = bit-exact.
        # When the DAMPED path is active (_blend) the update RUNS but is blended
        # back toward the frame-0 hyperblob anchor (d=0 == freeze; d=1 == vendored).
        if _blend:
            key, gibbs_key = jax.random.split(key)
            st = gibbs_hyperblob_features_dino(gibbs_key, st)
            hf_g = st.hyperblobs_state.hyperblob_features
            hf_b = ((1.0 - feature_update_damping) * hb_feat_anchor
                    + feature_update_damping * hf_g)
            st = st.replace({'hyperblobs_state': {'hyperblob_features': hf_b}})
        elif not freeze_blob_features:
            key, gibbs_key = jax.random.split(key)
            st = gibbs_hyperblob_features_dino(gibbs_key, st)
        return key, st

    # Python-bool branch at BUILD time (not traced): pick the final-step body.
    final_step = (update_blob_assignments_final_feature_aware
                  if feature_aware_final
                  else update_blob_assignments_final_baseline)

    key, genmatter_state = jax.lax.fori_loop(0, 3, hyperblob_update_loop, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 2, update_blob_assignments_feature_only, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 1, update_blob_assignments_position_only, (key, genmatter_state))
    # CONVERGE anti-bleed lever: blob_means_updates (default 15 = dino.py) sets how
    # many times the blob-mean posterior is re-solved.  Python int closed over at
    # build time -> static fori_loop bound (no recompile mid-run).
    key, genmatter_state = jax.lax.fori_loop(0, blob_means_updates, update_blob_means, (key, genmatter_state))
    # freeze_blob_features (build-time bool): 0 feature-mean updates => the blob
    # appearance stays at its frame-0 init (anti-drift). Default 3 = vendored.
    # The DAMPED path always runs the 3 updates (the blend toward the anchor,
    # inside update_blob_features_dino, applies the damping continuously).
    _feat_updates = 3 if (_blend or not freeze_blob_features) else 0
    key, genmatter_state = jax.lax.fori_loop(0, _feat_updates, update_blob_features_dino, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 1, final_step, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 15, update_blob_velocities, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 15, update_blob_velocity_covariances, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, _feat_updates, update_blob_features_dino, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 3, hyperblob_update_loop, (key, genmatter_state))

    return genmatter_state


# Per-frame tracking-config flags controlling the FINAL Gibbs assignment step
# (see ``blob_tracking_gibbs_dino_streaming``).  Set by ``init_state`` from the
# YAML ``tracking`` block so ``_step_jit`` / the multi-sweep step (built before
# the per-frame loop) close over the right Python bools.  Defaults reproduce the
# vendored behavior exactly (feature_aware_final=False, final_outlier=True), so
# any config that doesn't set these keys is unchanged.
_FEATURE_AWARE_FINAL_DEFAULT = False
_FINAL_OUTLIER_DEFAULT = True
# When True the per-frame Gibbs SKIPS blob->hyperblob membership re-sampling
# (dino.py:804), making cluster identity STICKY across frames (the frame-0
# seeded blob->hyperblob mapping is preserved) while geometry/appearance still
# follow the member blobs.  Default False preserves current behavior exactly.
_FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT = False
# Number of per-frame gibbs_blob_means refinements (the dino.py:824
# fori_loop(0, 15, ...)).  Default 15 reproduces the vendored schedule exactly;
# lowering it is the CONVERGE anti-bleed lever (keeps frozen object blob means
# near the constant-velocity prediction instead of chasing leaked background
# datapoints).  jit-static (closed over at build time).
_BLOB_MEANS_UPDATES_DEFAULT = 15
# When True the per-frame Gibbs SKIPS the blob/hyperblob DINO feature-mean updates
# (gibbs_blob_features_dino / gibbs_hyperblob_features_dino), holding the frame-0
# seeded APPEARANCE frozen across the stream. Motivation (pvc_loop, 2026-05-30): a
# frozen-centroid feature classifier scores ~0.71 region-J while the drifting
# tracker scores ~0.50 from the SAME seed — the per-frame appearance update drifts
# the object representation into background. jit-static; default False = bit-exact.
_FREEZE_BLOB_FEATURES_DEFAULT = False


@partial(jax.jit, static_argnames=("feature_aware_final", "final_outlier",
                                    "freeze_hyperblob_assignment",
                                    "blob_means_updates", "freeze_blob_features"))
def _step_jit(key, state, positions, velocities, features,
              feature_aware_final: bool = _FEATURE_AWARE_FINAL_DEFAULT,
              final_outlier: bool = _FINAL_OUTLIER_DEFAULT,
              freeze_hyperblob_assignment: bool = _FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT,
              blob_means_updates: int = _BLOB_MEANS_UPDATES_DEFAULT,
              freeze_blob_features: bool = _FREEZE_BLOB_FEATURES_DEFAULT,
              blob_feat_anchor=None, hb_feat_anchor=None,
              feature_update_damping=None):
    """Body of `f_tracking_sweep` from run_davis_tracking.py:626-640, but
    standalone so we can call it per-frame instead of via lax.scan."""
    next_blob_means = state.blobs_state.blob_vel_means + state.blobs_state.blob_means
    state = state.replace({'blobs_state': {'blob_means': next_blob_means}})
    state = state.replace({
        'datapoints_state': {
            'datapoint_positions': positions,
            'datapoint_vels': velocities,
            'datapoint_features': features,
        }
    })
    key, gibbs_key = jax.random.split(key)
    state = blob_tracking_gibbs_dino_streaming(
        gibbs_key, state,
        feature_aware_final=feature_aware_final, final_outlier=final_outlier,
        freeze_hyperblob_assignment=freeze_hyperblob_assignment,
        blob_means_updates=blob_means_updates,
        freeze_blob_features=freeze_blob_features,
        feature_update_damping=feature_update_damping,
        blob_feat_anchor=blob_feat_anchor,
        hb_feat_anchor=hb_feat_anchor,
    )
    return state, key


def step(state, positions: np.ndarray, velocities: np.ndarray, features: np.ndarray, key,
         *, feature_aware_final: bool = _FEATURE_AWARE_FINAL_DEFAULT,
         final_outlier: bool = _FINAL_OUTLIER_DEFAULT,
         freeze_hyperblob_assignment: bool = _FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT,
         blob_means_updates: int = _BLOB_MEANS_UPDATES_DEFAULT,
         freeze_blob_features: bool = _FREEZE_BLOB_FEATURES_DEFAULT,
         blob_feat_anchor=None, hb_feat_anchor=None,
         feature_update_damping=None):
    """One Gibbs sweep on the current frame.  Returns (new_state, new_key).

    ``blob_feat_anchor``/``hb_feat_anchor``/``feature_update_damping`` (all
    default None) activate the DAMPED anti-drift feature update (see
    ``blob_tracking_gibbs_dino_streaming``); they are passed to ``_step_jit`` as
    TRACED args (not static) so damping value + per-video anchor never recompile.
    """
    pj = jnp.asarray(positions)
    vj = jnp.asarray(velocities)
    fj = jnp.asarray(features)
    damp = None if feature_update_damping is None else jnp.asarray(
        feature_update_damping, dtype=jnp.float32)
    ba = None if blob_feat_anchor is None else jnp.asarray(blob_feat_anchor)
    ha = None if hb_feat_anchor is None else jnp.asarray(hb_feat_anchor)
    # Static (Python-bool/int) args -> _step_jit specializes one compile per combo.
    state, key = _step_jit(key, state, pj, vj, fj,
                           feature_aware_final=bool(feature_aware_final),
                           final_outlier=bool(final_outlier),
                           freeze_hyperblob_assignment=bool(freeze_hyperblob_assignment),
                           blob_means_updates=int(blob_means_updates),
                           freeze_blob_features=bool(freeze_blob_features),
                           blob_feat_anchor=ba, hb_feat_anchor=ha,
                           feature_update_damping=damp)
    return state, key


# ---- Multi-sweep per-frame step (calibration only; streaming demo uses step) ----
#
# Calibration deepens the per-frame measurement update (Level 1 of the two-level
# inference in ~/.claude/plans/velvet-knitting-nova.md) from 1 Gibbs sweep to N:
# predict + splice ONCE per frame, then ``jax.lax.scan`` blob_tracking_gibbs_dino
# N times inside a single jit boundary.  Re-predicting inside the scan would
# advance blob_means by N velocity steps (wrong dynamics), so the predict is
# hoisted out — mirroring ``f_tracking_sweep_dino`` active_step (dino.py:889) and
# ``_init_gibbs_scan_step`` (dino.py:849, whose scan body is also gibbs-only).
_MULTI_SWEEP_CACHE: dict = {}


def _make_multi_sweep_step(num_sweeps: int,
                           feature_aware_final: bool = _FEATURE_AWARE_FINAL_DEFAULT,
                           final_outlier: bool = _FINAL_OUTLIER_DEFAULT,
                           freeze_hyperblob_assignment: bool = _FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT,
                           blob_means_updates: int = _BLOB_MEANS_UPDATES_DEFAULT,
                           freeze_blob_features: bool = _FREEZE_BLOB_FEATURES_DEFAULT):
    """Build one jit'd N-sweep step; the caller caches it per distinct
    ``(num_sweeps, feature_aware_final, final_outlier,
    freeze_hyperblob_assignment, blob_means_updates)``.  All are closed over as
    Python values (the scan length is static; the flags select the
    final-assignment body, whether hyperblob membership is re-sampled, and the
    per-frame blob-mean refinement count in
    ``blob_tracking_gibbs_dino_streaming`` at build time), so they never force a
    recompile mid-run."""
    @jax.jit
    def _multi_step(key, state, positions, velocities, features,
                    blob_feat_anchor=None, hb_feat_anchor=None,
                    feature_update_damping=None):
        # Predict (constant-velocity dynamics) + splice the new frame ONCE.
        next_blob_means = state.blobs_state.blob_vel_means + state.blobs_state.blob_means
        state = state.replace({'blobs_state': {'blob_means': next_blob_means}})
        state = state.replace({
            'datapoints_state': {
                'datapoint_positions': positions,
                'datapoint_vels': velocities,
                'datapoint_features': features,
            }
        })

        def _sweep(carry, _):
            key, state = carry
            key, gibbs_key = jax.random.split(key)   # key threaded via carry
            # blob_feat_anchor/hb_feat_anchor/feature_update_damping are TRACED
            # runtime args (None when not damping) -> the anti-drift blend is
            # baked once per (blend on/off) structure, not per damping value or
            # per video.
            state = blob_tracking_gibbs_dino_streaming(
                gibbs_key, state,
                feature_aware_final=feature_aware_final, final_outlier=final_outlier,
                freeze_hyperblob_assignment=freeze_hyperblob_assignment,
                blob_means_updates=blob_means_updates,
                freeze_blob_features=freeze_blob_features,
                feature_update_damping=feature_update_damping,
                blob_feat_anchor=blob_feat_anchor,
                hb_feat_anchor=hb_feat_anchor,
            )
            return (key, state), None

        (key, state), _ = jax.lax.scan(_sweep, (key, state), None, length=num_sweeps)
        return state, key
    return _multi_step


def step_multi_sweep(state, positions: np.ndarray, velocities: np.ndarray,
                     features: np.ndarray, key, num_sweeps: int,
                     *, feature_aware_final: bool = _FEATURE_AWARE_FINAL_DEFAULT,
                     final_outlier: bool = _FINAL_OUTLIER_DEFAULT,
                     freeze_hyperblob_assignment: bool = _FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT,
                     blob_means_updates: int = _BLOB_MEANS_UPDATES_DEFAULT,
                     freeze_blob_features: bool = _FREEZE_BLOB_FEATURES_DEFAULT,
                     blob_feat_anchor=None, hb_feat_anchor=None,
                     feature_update_damping=None):
    """N-sweep variant of ``step()``.  Falls through to ``step()`` when
    ``num_sweeps <= 1`` so the streaming code path pays no extra JIT compile or
    scan-of-1 overhead.  Caches one compiled implementation per
    ``(num_sweeps, feature_aware_final, final_outlier,
    freeze_hyperblob_assignment, blob_means_updates, freeze_blob_features,
    blend)``.  The damped-feature anchors / damping are TRACED runtime args (not
    in the cache key beyond the blend on/off bit), so one program serves all
    damping values and all videos of equal shape."""
    if num_sweeps <= 1:
        return step(state, positions, velocities, features, key,
                    feature_aware_final=feature_aware_final,
                    final_outlier=final_outlier,
                    freeze_hyperblob_assignment=freeze_hyperblob_assignment,
                    blob_means_updates=blob_means_updates,
                    freeze_blob_features=freeze_blob_features,
                    blob_feat_anchor=blob_feat_anchor,
                    hb_feat_anchor=hb_feat_anchor,
                    feature_update_damping=feature_update_damping)
    pj = jnp.asarray(positions)
    vj = jnp.asarray(velocities)
    fj = jnp.asarray(features)
    blend = blob_feat_anchor is not None
    ba = None if blob_feat_anchor is None else jnp.asarray(blob_feat_anchor)
    ha = None if hb_feat_anchor is None else jnp.asarray(hb_feat_anchor)
    damp = None if feature_update_damping is None else jnp.asarray(
        feature_update_damping, dtype=jnp.float32)
    cache_key = (int(num_sweeps), bool(feature_aware_final), bool(final_outlier),
                 bool(freeze_hyperblob_assignment), int(blob_means_updates),
                 bool(freeze_blob_features), bool(blend))
    fn = _MULTI_SWEEP_CACHE.get(cache_key)
    if fn is None:
        fn = _make_multi_sweep_step(num_sweeps, bool(feature_aware_final),
                                    bool(final_outlier),
                                    bool(freeze_hyperblob_assignment),
                                    int(blob_means_updates),
                                    bool(freeze_blob_features))
        _MULTI_SWEEP_CACHE[cache_key] = fn
    state, key = fn(key, state, pj, vj, fj, ba, ha, damp)
    return state, key


# Sentinel index marking outlier datapoints in the rendered palette.  The
# Gibbs sampler uses index == num_blobs for outliers; we clip those before
# the palette lookup and color them with this entry instead.
# Outliers (Gibbs couldn't assign a datapoint to any cluster) get a SOLID, light,
# neutral gray — perceptually distinct from the saturated cluster palettes and
# clearly readable as "unassigned", instead of a muddy darkened tint.
OUTLIER_BGR = np.array([205, 205, 210], dtype=np.uint8)


def extract_assignments(state) -> Tuple[np.ndarray, np.ndarray]:
    """Pull blob_assignments and the per-datapoint hyperblob lookup off the
    state, handling the outlier index (=num_blobs) by mapping it to the
    sentinel value -1 in both arrays.

    Returns (blob_assignments_N, hyperblob_per_dp_N), both int32 with -1
    indicating outlier.
    """
    blob_a = np.asarray(state.datapoints_state.blob_assignments)
    hb_lookup = np.asarray(state.blobs_state.hyperblob_assignments)
    num_blobs = hb_lookup.shape[0]
    is_outlier = blob_a >= num_blobs
    # Clip into the valid range for the lookup, then mask back to -1.
    safe = np.minimum(blob_a, num_blobs - 1)
    hyperblob_per_dp = hb_lookup[safe].astype(np.int32)
    blob_out = blob_a.astype(np.int32)
    blob_out[is_outlier] = -1
    hyperblob_per_dp[is_outlier] = -1
    return blob_out, hyperblob_per_dp


@jax.jit
def _blob_weights_mean_jit(state):
    """Closed-form Dirichlet-multinomial posterior MEAN of the per-blob weights.

    Models ``dense_eval_blob_weights`` (dino.py:643) — same ``segment_sum`` over
    the sparse blob_assignments — but returns the posterior mean
    ``w_b = (beta + n_b) / (n_blobs * beta + N_in)`` instead of a Dirichlet
    sample, so the matter-weighted-Jaccard metric is deterministic across runs
    (Rao-Blackwellized, ~10x cheaper, no PRNG).  Outliers (assignment ==
    n_blobs) fall outside ``[0, n_blobs)`` and are dropped by ``segment_sum``.
    """
    num_blobs = state.hypers.n_blobs
    prior_beta = state.hypers.beta
    blob_idxs = state.datapoints_state.blob_assignments
    blob_counts = jax.ops.segment_sum(jnp.ones_like(blob_idxs), blob_idxs,
                                      num_segments=num_blobs)
    new_betas = prior_beta + blob_counts
    return new_betas / jnp.sum(new_betas)


def extract_blob_weights(state, key) -> Tuple[np.ndarray, object]:
    """Posterior-mean per-blob weights (length ``n_blobs``) for the matter-
    weighted Jaccard scorer.  Returns ``(weights_np, new_key)``.  The single
    ``np.asarray`` host transfer happens once per frame in the calibration loop,
    never in an inner kernel; the mean is deterministic, so ``key`` is only
    advanced for API symmetry with the sampling variant.
    """
    w = _blob_weights_mean_jit(state)
    weights_np = np.asarray(w)
    key, _ = jax.random.split(key)
    return weights_np, key


def run_tracker_from_cache(positions, velocities, features, indices, *, yaml_cfg,
                           num_blobs: int = NUM_BLOBS, num_hyperblobs: int = NUM_HYPERBLOBS,
                           num_sweeps: int = 1, capture_blob_weights: bool = False,
                           sam_segmentation: Optional[np.ndarray] = None,
                           key_seed: int = 0,
                           capture_data_loglik: bool = False) -> dict:
    """Run ONLY the GenMatter Gibbs tracker over PRE-COMPUTED perception outputs
    (positions/velocities/features as cached in the pseudo-label .npz).

    The depth+flow+DINO perception is invariant to the calibration hyperparameters
    — only the Gibbs tracker consumes them — so recomputing it on every EM /
    accept-test trial is pure waste (and starves the GPU).  This skips perception
    entirely and is numerically identical to ``run_streaming_live``'s tracker path
    for the same frames: same K-means init on frame 0, same per-frame
    ``step_multi_sweep``, same PRNGKey(key_seed).

    positions/velocities : (T, N, 3); features : (T, N, FEATURE_DIM); indices : (N,).
    Returns the same dict shape the calibrator's tracker runner produces.

    ``sam_segmentation`` : optional ``(H//stride, W//stride, 3)`` RGB pseudo-color
    frame-0 SAM mask (see ``instance_mask_to_rgb_grid``).  When supplied AND
    ``yaml_cfg['tracking']['use_sam_frame0']`` is set, the frame-0 ``init_state``
    seeds one hyperblob per SAM instance instead of flat k-means.  ``indices`` is
    reused as the per-datapoint subsample map into that grid (it already indexes
    the same stride-8 grid), so the SAM mask supplied here must correspond to the
    SAME video frame as ``positions[0]`` (the cache's first frame) for a 0-frame
    init offset.
    """
    import time as _time
    pos = np.asarray(positions, dtype=np.float32)
    vel = np.asarray(velocities, dtype=np.float32)
    feat = np.asarray(features, dtype=np.float32)
    T = int(pos.shape[0])
    key = jax.random.PRNGKey(key_seed)
    state = None
    blob_a_list, hb_a_list, blob_w_list = [], [], []
    n_blobs_val = None
    outlier_fracs, step_walls = [], []
    # Phase-B probabilistic objective: per-frame complete-data log-likelihood
    # term breakdown, accumulated where the state is LIVE (no offline
    # re-materialization).  Self-supervised (uses no GT / SAM labels).
    loglik_terms_list = []
    loglik_blob_anchor = None   # frame-0 blob appearance, captured post-warmup below
    if capture_data_loglik:
        import _data_loglik as _dll
        _loglik_terms_jit = jax.jit(_dll.frame_data_loglik_terms)
    # Final-assignment ablation flags (default = vendored behavior); see
    # ``blob_tracking_gibbs_dino_streaming``.  Python bools closed over by the
    # jit'd step at build time, so they never trigger a mid-run recompile.
    _trk = yaml_cfg.get("tracking", {})
    feature_aware_final = bool(_trk.get("feature_aware_final_assignment",
                                        _FEATURE_AWARE_FINAL_DEFAULT))
    final_outlier = bool(_trk.get("final_assignment_outlier",
                                  _FINAL_OUTLIER_DEFAULT))
    # When set, the per-frame Gibbs keeps the frame-0 seeded blob->hyperblob
    # mapping (cluster identity is sticky).  init_state's warmup
    # (init_gibbs_sweep_dino -> _init_gibbs_scan_step) is BLOB-ONLY and never
    # re-samples hyperblob membership, so the frame-0 semantic seed survives
    # warmup; this freeze applies from the very first per-frame sweep below.
    freeze_hyperblob_assignment = bool(_trk.get("freeze_hyperblob_assignment",
                                                _FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT))
    # CONVERGE anti-bleed lever: per-frame blob-mean refinement count (default 15
    # = vendored).  jit-static; read from YAML so the cached ablation/render share
    # one knob.
    blob_means_updates = int(_trk.get("blob_means_updates_per_frame",
                                      _BLOB_MEANS_UPDATES_DEFAULT))
    # Anti-drift: freeze the frame-0 seeded blob/hyperblob appearance (no per-frame
    # DINO feature-mean update). jit-static; read from YAML; default False = bit-exact.
    freeze_blob_features = bool(_trk.get("freeze_blob_features",
                                         _FREEZE_BLOB_FEATURES_DEFAULT))
    # DAMPED anti-drift (generalizes freeze): feature_update_damping in [0, 1] blends
    # each per-frame Gibbs feature update back toward the frame-0 anchor
    # (d=0 == freeze, d=1 == vendored). EXPLICIT damping takes precedence; the blend
    # path activates only when an explicit damping < 1.0 is set, so configs that set
    # only freeze_blob_features keep their exact (validated) static-freeze numerics,
    # and damping-unset / damping=1.0 stays bit-exact vendored. The frame-0 anchor is
    # captured post-warmup below and passed as a TRACED arg (no recompile per video).
    _damp_raw = _trk.get("feature_update_damping", None)
    feature_update_damping = (float(_damp_raw) if _damp_raw is not None else None)
    use_damp = (feature_update_damping is not None) and (feature_update_damping < 1.0)
    blob_feat_anchor = None
    hb_feat_anchor = None
    for t in range(T):
        if state is None:
            # init_state runs K-means + warmup Gibbs on frame 0, then one step on
            # the same frame (mirrors run_streaming_live's first tracked frame).
            state, key = init_state(pos[t], vel[t], feat[t], key, yaml_cfg=yaml_cfg,
                                    num_blobs=num_blobs, num_hyperblobs=num_hyperblobs,
                                    sam_segmentation=sam_segmentation,
                                    subsample_indices=indices,
                                    verbose=False)
            # Capture the post-warmup frame-0 appearance as the damping anchor
            # (before the first per-frame step can drift it). Traced from here on.
            if use_damp:
                blob_feat_anchor = jnp.asarray(state.blobs_state.blob_features)
                hb_feat_anchor = jnp.asarray(state.hyperblobs_state.hyperblob_features)
            # The data_loglik objective scores features against this SAME frozen
            # frame-0 anchor (captured regardless of damping), so the objective
            # measures appearance consistency vs the seed instead of rewarding the
            # drift that fitting the live blob_features would.
            if capture_data_loglik:
                loglik_blob_anchor = jnp.asarray(state.blobs_state.blob_features)
            state, key = step_multi_sweep(state, pos[t], vel[t], feat[t], key,
                                          num_sweeps=num_sweeps,
                                          feature_aware_final=feature_aware_final,
                                          final_outlier=final_outlier,
                                          freeze_hyperblob_assignment=freeze_hyperblob_assignment,
                                          blob_means_updates=blob_means_updates,
                                          freeze_blob_features=freeze_blob_features,
                                          blob_feat_anchor=blob_feat_anchor,
                                          hb_feat_anchor=hb_feat_anchor,
                                          feature_update_damping=feature_update_damping)
            state.datapoints_state.blob_assignments.block_until_ready()
        else:
            t0 = _time.monotonic()
            state, key = step_multi_sweep(state, pos[t], vel[t], feat[t], key,
                                          num_sweeps=num_sweeps,
                                          feature_aware_final=feature_aware_final,
                                          final_outlier=final_outlier,
                                          freeze_hyperblob_assignment=freeze_hyperblob_assignment,
                                          blob_means_updates=blob_means_updates,
                                          freeze_blob_features=freeze_blob_features,
                                          blob_feat_anchor=blob_feat_anchor,
                                          hb_feat_anchor=hb_feat_anchor,
                                          feature_update_damping=feature_update_damping)
            state.datapoints_state.blob_assignments.block_until_ready()
            step_walls.append(_time.monotonic() - t0)
        blob_a, hyperblob_a = extract_assignments(state)
        blob_a_list.append(blob_a)
        hb_a_list.append(hyperblob_a)
        outlier_fracs.append(float(np.mean(blob_a == -1)))
        if capture_data_loglik:
            terms = _loglik_terms_jit(state, loglik_blob_anchor)
            loglik_terms_list.append({k: float(v) for k, v in terms.items()})
        if capture_blob_weights:
            w, key = extract_blob_weights(state, key)
            blob_w_list.append(w)
            if n_blobs_val is None:
                n_blobs_val = int(w.shape[0])
    out = {
        "blob_a": np.stack(blob_a_list, axis=0).astype(np.int32),
        "hyperblob_a": np.stack(hb_a_list, axis=0).astype(np.int32),
        "matter_fps": float(1.0 / np.median(step_walls)) if step_walls else float("nan"),
        "outlier_frac_p95": float(np.percentile(outlier_fracs, 95)) if outlier_fracs else float("nan"),
    }
    if capture_blob_weights:
        out["blob_w"] = (np.stack(blob_w_list, axis=0).astype(np.float32)
                         if blob_w_list else None)
        out["n_blobs"] = n_blobs_val
        out["indices"] = np.asarray(indices)
    if capture_data_loglik and loglik_terms_list:
        # Video-level objective = mean over frames of each per-datapoint-mean term
        # (comparable across videos). "data_loglik" is the headline scalar (the
        # full complete-data density); the rest are exposed for the Phase-B proto
        # to pick the best GT-correlating formulation.
        keys = loglik_terms_list[0].keys()
        agg = {k: float(np.mean([d[k] for d in loglik_terms_list])) for k in keys}
        out["data_loglik_terms"] = agg
        # Headline = the ANCHOR-referenced objective (the naive live-feature "full"
        # rewards drift). _pick_data_loglik selects the configured variant from the
        # terms dict; this is only the fallback scalar.
        out["data_loglik"] = agg.get("anchor_pos_feat", agg.get("feat_anchor", float("nan")))
    return out


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


def extract_blob_means_and_covs(state) -> Tuple[np.ndarray, np.ndarray]:
    """Return (blob_means_Nx3, blob_covs_Nx3x3) as host numpy arrays."""
    return (np.asarray(state.blobs_state.blob_means),
            np.asarray(state.blobs_state.blob_covs))


def extract_hyperblob_means_and_covs(state) -> Tuple[np.ndarray, np.ndarray]:
    """Return (hyperblob_means_Kx3, hyperblob_covs_Kx3x3)."""
    return (np.asarray(state.hyperblobs_state.hyperblob_means),
            np.asarray(state.hyperblobs_state.hyperblob_covs))


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
                           stride: int = STRIDE) -> np.ndarray:
    """Per-blob mean BGR at this frame's datapoints — the LOCKED 'colour of each
    particle'. Compute it ONCE (frame 1) and thread it back into
    ``render_matter_tile(blob_color_lut=...)`` so each particle keeps a fixed
    colour across frames (no per-frame colour drift / flicker), even as it moves.

    Returns a ``(BLOB_PALETTE.shape[0], 3)`` uint8 LUT indexed by blob id. Blobs
    with no datapoints this frame fall back to the global mean colour. The grid
    cell each datapoint occupies (``indices``) is coloured from ``rgb_guide``
    downsampled to the stride grid."""
    gh, gw = h // stride, w // stride
    guide = rgb_guide if rgb_guide.shape[:2] == (h, w) else \
        cv2.resize(rgb_guide, (w, h), interpolation=cv2.INTER_AREA)
    cell_rgb = cv2.resize(guide, (gw, gh), interpolation=cv2.INTER_AREA
                          ).reshape(-1, 3).astype(np.float32)
    lab = np.asarray(blob_assignments).reshape(-1)
    idx = np.asarray(indices).reshape(-1)
    n = BLOB_PALETTE.shape[0]
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


def render_matter_tile(blob_assignments: np.ndarray,
                        hyperblob_per_dp: np.ndarray,
                        indices: np.ndarray,
                        h: int = 360, w: int = 640,
                        stride: int = STRIDE,
                        rgb_guide: Optional[np.ndarray] = None,
                        num_fg_hyperblobs: Optional[int] = None,
                        particle_avg_rgb: bool = False,
                        blob_color_lut: Optional[np.ndarray] = None,
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
    n_blobs = blob_palette.shape[0]
    n_hypers = hyper_palette.shape[0]

    # Datapoints with assignment < 0 are real Gibbs outliers.  We keep them
    # out of the NN-fill source set so unfilled grid cells can't inherit
    # outlier-gray transitively (which used to amplify a few real outliers
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
        blob_bgr = blob_palette[blob_full].astype(np.uint8)
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
    depth_hxw = _depth_to_Z(np.asarray(depth_hxw, dtype=np.float32))

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
    # yaw of 25° won't shift the cloud off the left edge of the tile the way
    # it did when we restored the full 3D centroid after rotation.
    centroid = pts.mean(axis=0)
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
    # tile width collapsed the cloud into a thin horizontal band.  Fitting to a
    # camera-aspect window instead fills the tile height and keeps the view
    # consistent with the source camera's framing (the "calibrated frustum").
    # Auto-fit uses the 96 th percentile (robust to a few stray far points);
    # callers freeze the value across frames by threading the return back in.
    if focal_length is None or focal_length <= 0.0:
        margin = 0.98   # fill the tile (FROZEN frame-0 focal stays stable, no pop)
        cam_aspect = float(W) / float(H)                 # source camera aspect
        # camera-aspect window, but never wider than the tile itself (so narrow
        # 1×5 tiles don't overflow / get culled).
        eff_W = min(out_H * cam_aspect, float(out_W))
        # 90th pct (was 96th): frame to the BULK of the cloud so it fills the tile;
        # the outer ~10% of points spill into the margin / crop at the edges.
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
    tile_a = _splat_pointcloud(u, v, colors_a, out_hw, point_size, bg_bgr)
    tile_b = _splat_pointcloud(u, v, colors_b, out_hw, point_size, bg_bgr)
    return tile_a, tile_b, f_used, proj


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


# ---------------- Self-test ----------------

if __name__ == "__main__":
    import time

    print(f"jax {jax.__version__}; running synthetic init+step smoke test")
    rng = np.random.default_rng(7)

    # Synthetic 640x360 streams.
    depth = rng.uniform(0.3, 1.0, (360, 640)).astype(np.float32)
    flow = rng.normal(0, 0.5, (2, 360, 640)).astype(np.float32)
    feat = rng.normal(0, 1, (256, 384)).astype(np.float32)  # 16x16 DINO patches

    indices = subsample_indices(360, 640, STRIDE, N_KEEP, seed=0)
    print(f"  subsample_indices: shape={indices.shape}, range=[{indices.min()}, {indices.max()}]")

    positions, velocities = unproject(depth, flow, indices, DEFAULT_INTRINSICS, STRIDE)
    print(f"  unproject: positions={positions.shape} velocities={velocities.shape} "
          f"X range [{positions[:,0].min():.2f}, {positions[:,0].max():.2f}] "
          f"Z range [{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")

    features, pca_basis, pca_mean, pca_std = dino_features_to_datapoints(feat, indices)
    print(f"  dino_features: features={features.shape} pca_basis={pca_basis.shape} "
          f"feat std={features.std():.3f} (expect ~1)")

    t0 = time.monotonic()
    state, key = init_state(positions, velocities, features, jkey(0))
    t1 = time.monotonic()
    print(f"  init_state: {t1 - t0:.2f}s (includes K-means + 15 warm-up Gibbs + JIT compile)")

    # 5 step calls.
    latencies = []
    for i in range(5):
        # Vary features slightly so something updates.
        feat_pert = feat + 0.01 * rng.normal(0, 1, feat.shape).astype(np.float32)
        feats_p, _, _, _ = dino_features_to_datapoints(feat_pert, indices, pca_basis, pca_mean, pca_std)
        pos_p, vel_p = unproject(depth + 0.01 * rng.normal(0, 1, depth.shape).astype(np.float32),
                                  flow, indices, DEFAULT_INTRINSICS, STRIDE)
        ts = time.monotonic()
        state, key = step(state, pos_p, vel_p, feats_p, key)
        # block_until_ready on a downstream leaf so the wall time is real.
        state.datapoints_state.blob_assignments.block_until_ready()
        latencies.append(time.monotonic() - ts)
        print(f"  step {i}: {latencies[-1]*1000:.1f} ms")
    print(f"  steady-state step latency (calls 2-5): "
          f"{np.mean(latencies[1:])*1000:.1f} ms mean")

    # Render the four streaming tiles.
    blob_a, hyperblob_a = extract_assignments(state)
    print(f"  blob_assignments: shape={blob_a.shape} unique={len(np.unique(blob_a))}")
    pixel_by_particle, pixel_by_cluster = render_matter_tile(
        blob_a, hyperblob_a, indices, 360, 640, STRIDE)
    print(f"  pixel_by_particle={pixel_by_particle.shape} "
          f"pixel_by_cluster={pixel_by_cluster.shape}")

    # Cluster (hyperblob) + particle (blob) ellipse tiles need a base RGB; use
    # a placeholder mid-gray for the synthetic test.
    base = np.full((360, 640, 3), 80, dtype=np.uint8)
    blob_means, blob_covs = extract_blob_means_and_covs(state)
    hb_means, hb_covs = extract_hyperblob_means_and_covs(state)
    particles_tile = render_centroid_tile(blob_means, blob_covs, BLOB_PALETTE[:blob_means.shape[0]],
                                           base, DEFAULT_INTRINSICS, sigma_scale=1.5, alpha=0.55)
    clusters_tile = render_centroid_tile(hb_means, hb_covs,
                                          HYPERBLOB_PALETTE[:hb_means.shape[0]],
                                          base, DEFAULT_INTRINSICS, sigma_scale=1.5, alpha=0.55)
    print(f"  particles_tile={particles_tile.shape} clusters_tile={clusters_tile.shape}")

    pc_cluster = render_pointcloud_tile(depth, pixel_by_cluster, DEFAULT_INTRINSICS)
    pc_particle = render_pointcloud_tile(depth, pixel_by_particle, DEFAULT_INTRINSICS)
    print(f"  pointcloud_cluster={pc_cluster.shape} pointcloud_particle={pc_particle.shape}")
    print("smoke test ok")

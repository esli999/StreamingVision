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
    ``configs/streaming_default.yaml`` (or ``streaming_tuned.yaml`` after
    bayesopt).  ``sigma_F`` defaults to 2.0 (the realtime-demo branch's
    setting; lower values produced an outlier-dominated visualization).
  * Depth model returns relative inverse depth; we min/max-normalize per
    frame and invert to a pseudo-Z in roughly [0.83, 5.0] m.
  * DINO PCA: fit a 384 -> 32 basis on the first frame's per-pixel features
    and freeze it; per-dim z-score so the model's sigma_F is appropriately
    scaled (matches DAVIS's pre-normalized `gaussian_means/_stds`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
)

from genmatter.tracking.dino import (
    GenMatter_Hyperparams_DINO,
    GenMatter_model_dino,
    blob_tracking_gibbs_dino,
    init_gibbs_sweep_dino,
    model_jimportance,  # pre-jitted GenMatter_model_dino.importance
    estimate_feature_sigmas_from_chm,
    FEATURE_SIGMA_FLOOR,
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


BLOB_PALETTE = _build_palette(NUM_BLOBS, seed=0)
# Sized generously (not just NUM_HYPERBLOBS=4) so the SAM-frame-0 semantic-init
# path — which yields one hyperblob per SAM instance plus k-means hyperblobs for
# unsegmented regions, typically 15-40 total — still renders distinct colors.
HYPERBLOB_PALETTE = _build_palette(max(NUM_HYPERBLOBS, 64), seed=42)


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
                   num_datapoints):
    """Derive Psi_* from the K-means cluster covs (empirical), then layer the
    YAML-driven hyperparameters on top.  Mirrors the branch's
    `_build_hypers_from_kmeans` (`genmatter/tracking/dino.py:957`) but reads
    its scalar priors from our streaming YAML so bayesopt-tuned values can be
    consumed directly."""
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
        mu_H=empirical_mu_H,
        sigma_H=f_(float(hp["sigma_H"])),  # noqa: F405
        nu_H=empirical_nu_H,
        Psi_H=empirical_Psi_H,
        nu_B=empirical_nu_B,
        Psi_B=empirical_Psi_B,
        sigma_V=f_(float(hp["sigma_V"])),  # noqa: F405
        nu_V=empirical_nu_V,
        Psi_V=empirical_Psi_V,
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


def init_state(positions: np.ndarray, velocities: np.ndarray, features: np.ndarray,
                key, *, yaml_cfg: Optional[dict] = None,
                num_blobs: int = NUM_BLOBS, num_hyperblobs: int = NUM_HYPERBLOBS,
                num_warmup_sweeps: Optional[int] = None, verbose: bool = False,
                sam_segmentation: Optional[np.ndarray] = None,
                subsample_indices: Optional[np.ndarray] = None):
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
                            num_datapoints=num_datapoints)

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


@jax.jit
def _step_jit(key, state, positions, velocities, features):
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
    state = blob_tracking_gibbs_dino(gibbs_key, state)
    return state, key


def step(state, positions: np.ndarray, velocities: np.ndarray, features: np.ndarray, key):
    """One Gibbs sweep on the current frame.  Returns (new_state, new_key)."""
    pj = jnp.asarray(positions)
    vj = jnp.asarray(velocities)
    fj = jnp.asarray(features)
    state, key = _step_jit(key, state, pj, vj, fj)
    return state, key


# Sentinel index marking outlier datapoints in the rendered palette.  The
# Gibbs sampler uses index == num_blobs for outliers; we clip those before
# the palette lookup and color them with this entry instead.
OUTLIER_BGR = np.array([60, 60, 60], dtype=np.uint8)  # dark gray


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


def render_matter_tile(blob_assignments: np.ndarray,
                        hyperblob_per_dp: np.ndarray,
                        indices: np.ndarray,
                        h: int = 360, w: int = 640,
                        stride: int = STRIDE) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns (blob_bgr, hyperblob_bgr).  Both are uint8 (h, w, 3) BGR.
    """
    gh, gw = h // stride, w // stride

    blob_palette = BLOB_PALETTE
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

    blob_full = cv2.resize(blob_grid.astype(np.int32), (w, h),
                            interpolation=cv2.INTER_NEAREST)
    hyper_full = cv2.resize(hyper_grid.astype(np.int32), (w, h),
                             interpolation=cv2.INTER_NEAREST)
    outlier_full = cv2.resize(outlier_grid.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)

    blob_bgr = blob_palette[blob_full].astype(np.uint8)
    hyper_bgr = hyper_palette[hyper_full].astype(np.uint8)
    if outlier_full.any():
        m = outlier_full[..., None]
        tint = OUTLIER_BGR[None, None, :].astype(np.uint16)
        blob_bgr = np.where(m,
                             ((blob_bgr.astype(np.uint16) + tint) // 2).astype(np.uint8),
                             blob_bgr)
        hyper_bgr = np.where(m,
                              ((hyper_bgr.astype(np.uint16) + tint) // 2).astype(np.uint8),
                              hyper_bgr)
    return blob_bgr, hyper_bgr


# Default 3D-view rotation for the ROW3 point-cloud tiles. ~25° yaw + slight
# downward pitch gives a 3/4 view that shows depth structure without losing
# orientation cues. Use the same rotation for both tiles so the cluster and
# particle views are spatially comparable.
POINTCLOUD_YAW_DEG = 25.0
POINTCLOUD_PITCH_DEG = -10.0


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
    empty_ret = (*empty, 0.0)

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
    pts_rot = pts_c @ (Rx @ Ry).T
    pts_rot[:, 2] += centroid[2]

    z_view = pts_rot[:, 2]
    in_front = z_view > 1e-3
    if not in_front.any():
        return empty_ret
    pts_rot = pts_rot[in_front]
    z_view = z_view[in_front]
    ys = ys[in_front]
    xs = xs[in_front]

    # Uniform focal length keeps 3D circles round.  If the caller didn't
    # supply one, auto-fit so 98 % of the cloud lands inside ~90 % of the
    # tile, using the 98 th percentile rather than abs-max so a handful of
    # stray far-away points don't zoom the whole view out.  IMPORTANT: the
    # autofit jitters frame-to-frame because the percentile XY extents
    # fluctuate with depth-normalization noise even when the 3D scene is
    # stable, so callers should compute autofit ONCE and pass the returned
    # value back in to freeze the framing.
    if focal_length is None or focal_length <= 0.0:
        margin = 0.9
        x_extent = float(np.percentile(np.abs(pts_rot[:, 0]), 98)) + 1e-3
        y_extent = float(np.percentile(np.abs(pts_rot[:, 1]), 98)) + 1e-3
        z_ref = float(np.median(z_view))
        f_x = (margin * 0.5 * out_W) * z_ref / x_extent
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

    order = np.argsort(-z_view, kind="stable")
    return u[order], v[order], ys[order], xs[order], f


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
                            point_size: int = 1,
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
    u, v, ys, xs, _f = _build_pointcloud_projection(
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
    point_size: int = 1,
    bg_bgr: Tuple[int, int, int] = (18, 18, 26),
    out_hw: Tuple[int, int] = (360, 960),
    focal_length: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Render two pointcloud tiles that share geometry but use different color
    sources.  Builds the projection once and splats twice — saves roughly half
    the per-frame pointcloud cost vs. two ``render_pointcloud_tile`` calls.

    Returns ``(tile_a, tile_b, focal_length)``.  Pass ``focal_length`` back in
    on subsequent calls to freeze framing across frames.
    """
    out_H, out_W = out_hw
    u, v, ys, xs, f_used = _build_pointcloud_projection(
        depth_hxw, intr, yaw_deg, pitch_deg, point_subsample, out_hw,
        focal_length=focal_length)
    if u.size == 0:
        blank = np.full((out_H, out_W, 3), bg_bgr, dtype=np.uint8)
        return blank.copy(), blank.copy(), f_used
    colors_a = color_a_bgr[ys, xs].astype(np.uint8)
    colors_b = color_b_bgr[ys, xs].astype(np.uint8)
    tile_a = _splat_pointcloud(u, v, colors_a, out_hw, point_size, bg_bgr)
    tile_b = _splat_pointcloud(u, v, colors_b, out_hw, point_size, bg_bgr)
    return tile_a, tile_b, f_used


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

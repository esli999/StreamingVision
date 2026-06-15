"""Real-time GenMatter++ adapter for the StreamingVision pipeline.

This wraps the GenMatter++ DINO-features Gibbs tracker (vendored under
StreamingVision/genmatterpp) so it can be driven by the latest-value-slot
pipeline in render_demo.py — one Gibbs sweep per incoming RGB frame, with
persistent blob/hyperblob state.

The vendored package ships its own JIT-stable per-step
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
    2.0; lower values produced an outlier-dominated visualization.
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

# Vendored GenMatter++ package.  `genmatterpp/__init__.py` adds its own
# directory to sys.path so the inner `genmatter` package is importable
# without modifying any vendored file.
import genmatterpp  # noqa: F401 — side effect: adds genmatterpp/ to sys.path

import jax
import jax.numpy as jnp
from jax.random import key as jkey
from jax.scipy.stats import multivariate_normal as _mvn_stats
from jax.scipy.stats import norm as _norm_stats

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


def _depth_to_Z(d_raw: np.ndarray, lo: Optional[float] = None,
                hi: Optional[float] = None) -> np.ndarray:
    """DepthAnythingV2-Small outputs relative inverse depth (disparity-ish).
    Map to a pseudo-Z roughly in [0.83, 5.0] m — closer pixels get small Z,
    farther pixels get larger Z.  Scale is arbitrary; what matters is it's
    CONSISTENT across frames so the K-means-derived priors stay valid.

    ``lo``/``hi`` are the normalization bounds. When BOTH are None (the default)
    they fall back to this frame's own min/max — the original behaviour, which is
    BIT-EXACT but NOT temporally consistent (the bounds move every frame, so a
    static pixel's Z drifts; see ``stabilize_depth``). Pass FROZEN bounds (e.g.
    frame-0 robust percentiles from ``stabilize_depth``) to get the consistency the
    docstring promises: a static background pixel then maps to a constant Z.
    """
    if lo is None or hi is None:
        lo = float(d_raw.min())
        hi = float(d_raw.max())
    d_norm = np.clip((d_raw - lo) / (hi - lo + 1e-6), 0.0, 1.0)   # [0, 1]
    Z = 1.0 / (d_norm * 0.8 + 0.2)                                # [0.83, 5.0]
    return Z.astype(np.float32)


def stabilize_depth(d_raw: np.ndarray, state: Optional[dict], *,
                    ema_alpha: float = 0.5, fit_subsample: int = 4,
                    huber_iters: int = 2):
    """Temporally stabilize Depth-Anything's scale-shift-invariant disparity so a
    STATIC scene keeps a CONSTANT pseudo-depth across frames — the principled fix
    for the per-frame-normalization drift (replaces re-min/max-ing every frame).

    Depth-Anything V2 is trained scale-shift-invariant: its raw output is relative
    inverse depth defined only up to a per-frame affine ``disp_t ≈ a_t·disp_true +
    b_t``. That per-frame affine drifts (and per-frame min/max estimates it from the
    two noisiest pixels), so the static background's unprojected position swims. The
    fix: align each frame's raw depth to a FROZEN frame-0 reference by a ROBUST
    affine (scale ``s`` + shift ``o``, median/MAD init + IRLS-Huber so the moving
    foreground is down-weighted as an outlier to the static model), EMA the aligned
    map, and normalize later with FROZEN bounds.

    ``d_raw``: raw model output, (H, W) float. ``state``: dict carried across frames
    (pass None on frame 0). Returns ``(d_stab, lo, hi, state)`` — feed
    ``_depth_to_Z(d_stab, lo, hi)``. Global affine preserves real object motion and
    real camera parallax; only the per-frame scale/shift drift is removed.
    """
    d = np.asarray(d_raw, dtype=np.float32)
    if state is None:
        state = {}
    if state.get("ref_s") is None:
        # Frame 0: freeze the reference subsample + robust normalization bounds.
        lo, hi = np.percentile(d, [1.0, 99.0])
        state["ref_s"] = d.reshape(-1)[::fit_subsample].copy()
        state["lo"], state["hi"] = float(lo), float(hi)
        state["ema"] = d.copy()
        return d, state["lo"], state["hi"], state

    rs = state["ref_s"]
    ds = d.reshape(-1)[::fit_subsample]
    # robust affine  s*ds + o ~= rs : median/MAD init, then IRLS-Huber refinement.
    med_d, med_r = np.median(ds), np.median(rs)
    mad_d = np.median(np.abs(ds - med_d)) + 1e-6
    mad_r = np.median(np.abs(rs - med_r)) + 1e-6
    s = float(mad_r / mad_d)
    o = float(med_r - s * med_d)
    for _ in range(huber_iters):
        resid = s * ds + o - rs
        sigma = 1.4826 * (np.median(np.abs(resid - np.median(resid))) + 1e-6)
        k = 1.345 * sigma
        ar = np.abs(resid)
        w = np.where(ar <= k, 1.0, k / (ar + 1e-9)).astype(np.float32)
        Sww = float(w.sum()); Sx = float((w * ds).sum()); Sy = float((w * rs).sum())
        Sxx = float((w * ds * ds).sum()); Sxy = float((w * ds * rs).sum())
        det = Sxx * Sww - Sx * Sx
        if abs(det) < 1e-12:
            break
        s = (Sxy * Sww - Sx * Sy) / det
        o = (Sxx * Sy - Sx * Sxy) / det
    d_align = (s * d + o).astype(np.float32)
    ema = state.get("ema")
    state["ema"] = d_align if ema is None else (
        ema_alpha * d_align + (1.0 - ema_alpha) * ema).astype(np.float32)
    return state["ema"], state["lo"], state["hi"], state


def unproject(depth_hxw: np.ndarray, flow_2xhxw: np.ndarray,
               indices: np.ndarray, intr: Intrinsics = DEFAULT_INTRINSICS,
               stride: int = STRIDE, depth_lo: Optional[float] = None,
               depth_hi: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (depth, optical flow) into 3D positions and 3D velocities at
    the chosen subsampled pixel locations.

    depth_hxw : (H, W) float — raw depth-model output (relative inverse depth)
    flow_2xhxw: (2, H, W) float — SEA-RAFT forward flow, [fx, fy] in pixels
    indices   : (N,) int — flat indices into the (H/stride) x (W/stride) grid
    depth_lo/depth_hi : FROZEN depth-normalization bounds (from ``stabilize_depth``);
        None (default) = per-frame min/max = BIT-EXACT original behaviour.

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

    Z_full = _depth_to_Z(depth_hxw, depth_lo, depth_hi)
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
    YAML-driven hyperparameters on top.  Mirrors the vendored
    `_build_hypers_from_kmeans` (`genmatter/tracking/dino.py`) but reads
    its scalar priors from our streaming YAML so calibrated values can be
    consumed directly.

    When ``tracking.use_calibrated_priors`` is true, reads ``Psi_B/H/V``,
    ``nu_B/H/V``, ``mu_H`` from ``tracking.hyperparams`` (the offline
    calibrator's output) instead of deriving them from the K-means seed."""
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
        # Expose the K-means-derived values so the offline calibrator can emit a
        # YAML that round-trips byte-for-byte through the use_calibrated_priors
        # path.
        debug_capture["empirical_mu_H"] = np.asarray(empirical_mu_H, dtype=np.float32)
        debug_capture["empirical_Psi_B"] = np.asarray(empirical_Psi_B, dtype=np.float32)
        debug_capture["empirical_Psi_H"] = np.asarray(empirical_Psi_H, dtype=np.float32)
        debug_capture["empirical_Psi_V"] = np.asarray(empirical_Psi_V, dtype=np.float32)
        debug_capture["empirical_nu_B"] = float(empirical_nu_B)
        debug_capture["empirical_nu_H"] = float(empirical_nu_H)
        debug_capture["empirical_nu_V"] = float(empirical_nu_V)
    # Empirical-Bayes feature variances: when calibrate_feature_sigmas is set,
    # override the YAML sigma_F / sigma_F_H with the within-partition feature
    # variance estimated directly from the k-means seed.  Same seam that already
    # empirical-Bayes-fits the geometry priors above; native JAX, runs once at
    # init.
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
    colors — mirrors the live demo's downsample (render_demo.py) but starts
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

    SAM-anchored semantic init: when ``tracking.use_sam_frame0`` is set
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
        # Seed-purity fix (default OFF -> bit-exact to the standard path).  Rewrite
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
        # then overlay the DINO feature stats (mirrors run_davis_tracking.py).
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


def _feature_temp_final_assignment(key, st, *, tau_feat, final_outlier: bool,
                                   feat_anchor=None, use_anchor: bool = False,
                                   stickiness=None, prev_assignments=None):
    """Re-implemented FINAL blob assignment with a DINO-feature TEMPERATURE.

    Reproduces the vendored feature-aware-final logits (``gibbs_blob_assignments_dino``
    with position_only=False/feature_only=False -> pos+vel+feat summed RAW, no cov
    inflation) but scales the feature term by ``tau_feat`` (a TRACED scalar)::

        logits[n, i] = pos_ll[n,i] + vel_ll[n,i] + tau_feat * feat_ll[n,i] + log_mix[i]

    ``tau_feat == 1`` reproduces the vendored summed assignment to ~1e-6 (the only
    difference is jax.scipy vs genjax logpdf; the categorical sampler is bit-identical).
    ``tau_feat > 1`` up-weights the DINO features at the step that writes the labels
    the renderer/scorer read — pushing the model tracker toward a frozen-centroid DINO
    classifier.  ``tau_feat -> inf`` approximates ``feature_only``.

    ``use_anchor`` (with ``feat_anchor``, the FROZEN frame-0 blob appearance): score
    the feature term against the anchor instead of the live (possibly drifted)
    ``blob_features``.  A no-op when the live features equal the anchor (damping d=0).
    All math is jax.scipy (mvn/normal == genjax to ~1e-6)."""
    posterior_key, _ = jax.random.split(key)
    batch_size = 975                          # match the vendored Gibbs batch size exactly
    h = st.hypers
    N = int(h.n_datapoints)
    ds = st.datapoints_state
    bs = st.blobs_state
    pos = ds.datapoint_positions              # (N, 3)
    vel = ds.datapoint_vels                   # (N, 3)
    feat = ds.datapoint_features              # (N, D)
    bm, bc = bs.blob_means, bs.blob_covs              # (L, 3), (L, 3, 3)
    bvm, bvc = bs.blob_vel_means, bs.blob_vel_covs    # (L, 3), (L, 3, 3)
    bf = bs.blob_features                     # (L, D)
    feat_ref = feat_anchor if (use_anchor and feat_anchor is not None) else bf
    sqrtF = jnp.sqrt(h.sigma_F)

    # Precompute each blob's PRECISION + log-det ONCE (L small 3x3 matrices) so the
    # per-datapoint Gaussian log-density is a cheap einsum quadratic form — NOT an
    # N*L triangular solve (the vmap'd jax.scipy mvn.logpdf re-Choleskys per pair and
    # OOMs the solve workspace when both the vendored + re-impl programs are resident).
    _LOG2PI = float(np.log(2.0 * np.pi))

    def _prep(covs):                          # covs (L,3,3) -> (prec (L,3,3), logdet (L,))
        prec = jnp.linalg.inv(covs)
        _, logdet = jnp.linalg.slogdet(covs)
        return prec, logdet
    prec_b, logdet_b = _prep(bc)
    prec_v, logdet_v = _prep(bvc)
    _cst3 = -1.5 * _LOG2PI                     # -0.5 * k * log(2pi), k=3

    def _gauss_ll(x_B3, mu_L3, prec_L33, logdet_L):
        # (B,3),(L,3),(L,3,3),(L,) -> (B,L) Gaussian log-density (== mvn.logpdf).
        d = x_B3[:, None, :] - mu_L3[None, :, :]               # (B,L,3)
        maha = jnp.einsum('bli,lij,blj->bl', d, prec_L33, d)   # (B,L)
        return _cst3 - 0.5 * (logdet_L[None, :] + maha)        # (B,L)

    blob_weights = bs.blob_weights
    outlier_prob = h.outlier_prob if final_outlier else 0.0
    ext = jnp.concatenate(
        [blob_weights, jnp.asarray([outlier_prob], dtype=blob_weights.dtype)])
    log_mix = jnp.log(ext / jnp.sum(ext))     # (L+1,)
    a = h.outlier_velocity_gamma_shape
    b = h.outlier_velocity_gamma_rate

    def batch(carry, batch_idx):
        idxs = jnp.arange(batch_size) + batch_idx * batch_size
        p = pos[idxs]; v = vel[idxs]; f = feat[idxs]           # (B,3),(B,3),(B,D)
        pos_ll = _gauss_ll(p, bm, prec_b, logdet_b)            # (B,L)
        vel_ll = _gauss_ll(v, bvm, prec_v, logdet_v)           # (B,L)
        # Feature term: sum_d N(f_d | feat_ref_d, sqrt(sigma_F)) — no solve.
        feat_ll = jnp.sum(_norm_stats.logpdf(
            f[:, None, :], feat_ref[None, :, :], sqrtF), axis=-1)   # (B,L)
        log_liks = pos_ll + vel_ll + tau_feat * feat_ll        # (B,L)
        if final_outlier:
            speed = jnp.linalg.norm(v, axis=-1)                # (B,)
            log_gamma_vel = ((a - 1) * jnp.log(speed + 1e-8) - b * speed
                             - a * jnp.log(1.0 / b) - jax.lax.lgamma(a))   # (B,)
        else:
            log_gamma_vel = jnp.zeros(p.shape[0])              # outlier mix is -inf
        raw = jnp.concatenate([log_liks, log_gamma_vel[:, None]], axis=1)  # (B,L+1)
        return carry, raw + log_mix[None, :]

    num_full = N // batch_size
    _, bl = jax.lax.scan(batch, None, jnp.arange(num_full))
    all_logprobs = bl.reshape(num_full * batch_size, -1)            # (N, L+1)
    if (stickiness is not None) and (prev_assignments is not None):
        # TEMPORAL HYSTERESIS: add `stickiness` to each datapoint's PREVIOUS-frame
        # blob logit so a static scene point keeps its blob instead of being
        # re-sampled every frame (the categorical assignment churn that amplifies
        # background-particle jitter beyond the input cloud).  A genuinely moving
        # point still switches when the data evidence beats `stickiness`.
        # Outlier (prev < 0) maps to the last (outlier) column.
        Lp1 = all_logprobs.shape[1]
        prev_col = jnp.where(prev_assignments >= 0, prev_assignments, Lp1 - 1)
        all_logprobs = all_logprobs + stickiness * jax.nn.one_hot(
            prev_col, Lp1, dtype=all_logprobs.dtype)
    updated = jax.random.categorical(posterior_key, all_logprobs)
    return st.replace({'datapoints_state': {'blob_assignments': updated}})


def _robust_blob_means(key, st, *, robust_delta):
    """ROBUST (occlusion/outlier-aware) re-implementation of the vendored
    ``gibbs_blob_means`` (genmatterpp/genmatter/inference.py): the IDENTICAL
    conjugate Gaussian posterior (hyperblob prior + datapoint likelihood + velocity
    affine), EXCEPT each datapoint's contribution to its blob's DATA term is
    down-weighted by a Huber weight on its Mahalanobis distance to the blob's CURRENT
    mean (under blob_cov).  An OCCLUDED datapoint (its grid pixel briefly shows the
    occluder, so its position jumps UN-predictably to the occluder's depth) is a far
    outlier -> w ~ 0 -> it cannot pull the blob mean, which falls back to its prior
    (hyperblob + constant-velocity prediction) and HOLDS.  A genuinely MOVING blob is
    NOT penalized: the per-frame predict has already advanced its mean along its
    velocity, so its datapoints sit near the predicted mean -> w = 1; only UNPREDICTED
    jumps (occlusion / depth spikes) are down-weighted.  The per-frame fori_loop over
    this step makes it iteratively-reweighted (IRLS).  ``robust_delta`` = Huber knee in
    sigma (Mahalanobis) units (a STATISTICAL constant, ~3-sigma; not a per-video fit).
    Mirrors the vendored Gaussian sample so semantics match; used ONLY when
    ``use_robust_mean`` (default off routes to the vendored equal-weight step, bit-exact)."""
    posterior_key, _ = jax.random.split(key)
    ds, bs, hs = st.datapoints_state, st.blobs_state, st.hyperblobs_state
    x = ds.datapoint_positions                                   # [N, 3]
    a = ds.blob_assignments                                      # [N]
    blob_means, blob_covs = bs.blob_means, bs.blob_covs          # [L,3],[L,3,3]
    L = st.hypers.n_blobs
    d = blob_means.shape[-1]
    sigmaV = st.hypers.sigma_V
    blob_cov_inv = jnp.linalg.inv(blob_covs)                     # [L,3,3]

    # Huber weighting is disabled here (shrinking N inflates the sample noise); this
    # function instead supplies the in-tracker temporal MEAN PRIOR, with robust_delta
    # used as the prior strength.  w = 1 reproduces the vendored equal-weight sums.
    w = jnp.ones(x.shape[0], dtype=x.dtype)
    one_hot = jax.nn.one_hot(a, L, dtype=x.dtype)              # [N,L] (outliers a<0 -> all-zero row)
    N_l = jnp.einsum('n,nl->l', w, one_hot)                    # [L]
    S_l = jnp.einsum('ni,nl->li', x * w[:, None], one_hot)     # [L,3]

    # --- vendored conjugate posterior (mirrors inference.py gibbs_blob_means) ---
    hb_assign = bs.hyperblob_assignments
    muH_l = hs.hyperblob_means[hb_assign]                        # [L,3]
    hyperblob_cov_inv = jnp.linalg.inv(hs.hyperblob_covs[hb_assign])   # [L,3,3]
    A_l = hs.hyperblob_rot_vels[hb_assign] - jnp.eye(d)          # [L,3,3]
    b_l = hs.hyperblob_trans_vels[hb_assign] - jnp.einsum('lij,lj->li', A_l, muH_l)
    residuals = bs.blob_vel_means - b_l                          # [L,3]
    lhs_affine = jnp.einsum('lij,lik->ljk', A_l, A_l)           # [L,3,3]
    rhs_affine = jnp.einsum('lij,li->lj', A_l, residuals)       # [L,3]
    P_post = (hyperblob_cov_inv + blob_cov_inv * N_l[:, None, None]
              + lhs_affine / sigmaV)                            # [L,3,3]
    weighted_mean = (jnp.einsum('lij,lj->li', hyperblob_cov_inv, muH_l)
                     + jnp.einsum('lij,lj->li', blob_cov_inv, S_l)
                     + rhs_affine / sigmaV)                     # [L,3]
    # In-tracker temporal MEAN PRIOR (Kalman process-noise): pull each blob mean toward
    # its velocity-PREDICTED position (= blob_means at this update's entry; _step_jit set
    # it to blob_vel_means+prev before the sweeps).  Prior precision = strength x the DATA
    # precision (scale-free): strength=1 weights the prediction equal to all this frame's
    # datapoints.  A STATIC blob HOLDS vs depth jitter; a MOVING blob is anchored to its
    # advanced prediction, so it still tracks.  (strength = robust_delta.)
    data_prec = blob_cov_inv * N_l[:, None, None]              # [L,3,3]
    # MOTION-ADAPTIVE gate: only a STATIC blob (small blob_vel_means) gets the hold-prior;
    # a MOVING blob (a moving object, or a moving-CAMERA background) has large velocity ->
    # gate ~0 -> no prior -> it tracks freely. This lets a static background region stiffen
    # WITHOUT damping any foreground motion or a moving-camera background.
    vmag = jnp.linalg.norm(bs.blob_vel_means, axis=-1)         # [L]
    gate = 1.0 / (1.0 + (vmag / 0.03) ** 2)                    # [L] ~1 slow, ~0 fast
    sv = (robust_delta * gate)[:, None]                        # [L,1]
    P_post = P_post + sv[:, :, None] * data_prec               # [L,3,3]
    weighted_mean = weighted_mean + sv * jnp.einsum('lij,lj->li', data_prec, blob_means)
    cov_post = jnp.linalg.inv(P_post)                           # [L,3,3]
    mean_post = jnp.einsum('lij,lj->li', cov_post, weighted_mean)
    new_blob_means = jax.random.multivariate_normal(posterior_key, mean_post, cov_post)
    return st.replace({'blobs_state': {'blob_means': new_blob_means}})


def blob_tracking_gibbs_dino_streaming(key, genmatter_state, *,
                                       feature_aware_final: bool,
                                       final_outlier: bool,
                                       freeze_hyperblob_assignment: bool = False,
                                       blob_means_updates: int = 15,
                                       freeze_blob_features: bool = False,
                                       feature_update_damping=None,
                                       blob_feat_anchor=None,
                                       hb_feat_anchor=None,
                                       use_feature_temp_final: bool = False,
                                       final_feature_temp=None,
                                       final_assignment_anchor: bool = False,
                                       use_final_stickiness: bool = False,
                                       final_assignment_stickiness=None,
                                       use_robust_mean: bool = False,
                                       mean_update_robust_delta=None):
    """Streaming-local re-implementation of
    ``genmatter.tracking.dino.blob_tracking_gibbs_dino`` whose FINAL assignment
    step (the ``update_blob_assignments_with_outlier`` step) is configurable, so
    we can ablate it WITHOUT editing the vendored file.

    The vendored schedule ends its assignment phase with a POSITION-ONLY +
    outlier-injecting assignment, so the labels the renderer reads are
    position-driven (DINO features washed out) with outliers added as the last
    word.

    Flags (closed over as Python bools at build time -> NOT jax-traced, so the
    static ``fori_loop`` counts and the choice of final-step body are baked into
    one compiled program; never forces a mid-run recompile):

      feature_aware_final=False  (baseline)  : final step matches the vendored
          schedule -> gibbs_blob_assignments_dino(position_only=True,
                                         disable_outlier_prob=False) + weights.
          Reproduces the vendored streaming behavior bit-for-bit.

      feature_aware_final=True               : final step is a FULL position +
          feature (+ velocity, since neither position_only nor velocity_only nor
          feature_only is set) likelihood assignment
          -> gibbs_blob_assignments_dino(position_only=False, feature_only=False,
                                         disable_outlier_prob=(not final_outlier))
             + weights.
          ``final_outlier`` toggles whether outliers are enabled on that step.

    Everything else (3x hyperblob, 2x feature-only assign, 1x position-only
    assign, 15x means, 3x features, then 15x vel-means, 15x vel-covs, 3x
    features, 3x hyperblob) is byte-identical to the vendored schedule.

    ``freeze_hyperblob_assignment`` (Python bool, build-time) — when True the
    per-frame ``hyperblob_update_loop`` SKIPS the blob->hyperblob membership
    re-sampling (``gibbs_hyperblob_assignments_dino``) while KEEPING the
    geometry/appearance updates (means/covs/rot/trans/features_dino).  This
    makes cluster identity sticky: the frame-0 seeded blob->hyperblob mapping is
    preserved across the stream so the ``pixel_by_cluster`` view tracks the same
    object instead of re-partitioning into coarse spatial clusters every frame.
    Default False reproduces the vendored behavior bit-for-bit.

    ``blob_means_updates`` (Python int, build-time) — number of
    ``gibbs_blob_means`` refinement iterations per frame (the vendored
    ``fori_loop(0, 15, update_blob_means, ...)``).  Each iteration re-solves the
    blob-mean posterior, which blends the constant-velocity PREDICTION
    (``next_blob_means``, the prior carried in via the hyperblob mean + velocity
    affine terms) against the DATA term ``blob_cov_inv * N_l * S_l`` — so MORE
    iterations let a frozen object blob's mean be pulled further toward whatever
    datapoints currently fall in it (including background that leaked in, since
    DINO appearance is shared near the object).  FEWER iterations keep the mean
    nearer the velocity prediction (anti-drift; approximates a stronger
    velocity-anchored prior without editing the vendored ``gibbs_blob_means``).
    Default 15 reproduces the vendored schedule bit-for-bit.

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
    # Build-time (Python, NOT traced) branch: run the DAMPED feature blend only when
    # BOTH an anchor AND an explicit damping are given.  Decoupled from the anchor
    # alone so the feature-temperature final assignment (below) can reference the same
    # frame-0 anchor (``final_assignment_anchor``) WITHOUT activating the damping blend
    # (e.g. tau-only configs at damping unset).
    _blend = (blob_feat_anchor is not None) and (feature_update_damping is not None)

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
        # Vendored update_blob_assignments_with_outlier: position-only, outliers
        # ENABLED.  The vendored "last assignment word".
        key, st = carry
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_assignments_dino(
            gibbs_key, st, position_only=True, disable_outlier_prob=False
        )
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_weights(gibbs_key, st)
        return key, st

    def update_blob_assignments_final_feature_aware(i, carry):
        # FULL position+feature+velocity likelihood as the last assignment, so
        # DINO features (and motion) drive the labels the renderer reads.
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
        if use_robust_mean:
            # Occlusion/outlier-robust IRLS mean update (down-weights datapoints
            # inconsistent with their blob's predicted mean). Build-time branch ->
            # default-off keeps the vendored equal-weight step bit-exact.
            st = _robust_blob_means(gibbs_key, st, robust_delta=mean_update_robust_delta)
        else:
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
        # the per-frame blob->hyperblob membership re-sampling is SKIPPED,
        # preserving the frame-0 (k-means / SAM / GT) seeded cluster identity,
        # while the geometry/appearance updates below still let each cluster's
        # mean/cov/rot/trans/feature follow its member blobs as they move.
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
        # is held FROZEN across the stream (anti-drift). Default False = bit-exact.
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

    def update_blob_assignments_final_feature_temp(i, carry):
        # Re-implemented feature-aware final assignment with a TRACED feature
        # TEMPERATURE (and optional anchor-referenced feature term).  Replaces the
        # vendored summed assignment so the DINO features can DRIVE the final labels.
        key, st = carry
        key, gibbs_key = jax.random.split(key)
        st = _feature_temp_final_assignment(
            gibbs_key, st, tau_feat=final_feature_temp, final_outlier=final_outlier,
            feat_anchor=blob_feat_anchor, use_anchor=final_assignment_anchor)
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_weights(gibbs_key, st)
        return key, st

    # Previous-frame assignment carried INTO this step (captured before any of the
    # per-frame Gibbs sweeps mutate it) -> the temporal anchor for the sticky final.
    _prev_assign = genmatter_state.datapoints_state.blob_assignments

    def update_blob_assignments_final_sticky(i, carry):
        # Feature-aware final assignment (tau_feat=1, == feature_aware_final) PLUS a
        # temporal stickiness bonus toward the previous-frame blob, so static scene
        # points stop hopping between blobs (the churn behind background jitter)
        # while moving points still switch when the evidence wins.
        key, st = carry
        key, gibbs_key = jax.random.split(key)
        st = _feature_temp_final_assignment(
            gibbs_key, st, tau_feat=1.0, final_outlier=final_outlier,
            feat_anchor=blob_feat_anchor, use_anchor=final_assignment_anchor,
            stickiness=final_assignment_stickiness, prev_assignments=_prev_assign)
        key, gibbs_key = jax.random.split(key)
        st = gibbs_blob_weights(gibbs_key, st)
        return key, st

    # Python-bool branch at BUILD time (not traced): pick the final-step body.
    # Stickiness takes precedence, then the feature-temperature re-impl (opt-in via
    # use_feature_temp_final); default-off keeps the vendored feature-aware /
    # baseline paths byte-identical.
    if use_final_stickiness:
        final_step = update_blob_assignments_final_sticky
    elif use_feature_temp_final:
        final_step = update_blob_assignments_final_feature_temp
    elif feature_aware_final:
        final_step = update_blob_assignments_final_feature_aware
    else:
        final_step = update_blob_assignments_final_baseline

    key, genmatter_state = jax.lax.fori_loop(0, 3, hyperblob_update_loop, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 2, update_blob_assignments_feature_only, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 1, update_blob_assignments_position_only, (key, genmatter_state))
    # STICKINESS (when on): run the temporally-anchored assignment BEFORE the mean
    # update so the rendered blob_means are built from coherent memberships — the
    # position_only churn (background points reassigned every frame) is what jitters
    # the means; a sticky-only-at-the-end fix stabilizes the output labels but not
    # the means. Build-time gated (0 iters when off) => default byte-identical.
    _pre_mean_sticky = 1 if use_final_stickiness else 0
    key, genmatter_state = jax.lax.fori_loop(
        0, _pre_mean_sticky, update_blob_assignments_final_sticky, (key, genmatter_state))
    # Anti-bleed lever: blob_means_updates (default 15 = vendored) sets how many
    # times the blob-mean posterior is re-solved.  Python int closed over at
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
# When True the per-frame Gibbs SKIPS blob->hyperblob membership re-sampling,
# making cluster identity STICKY across frames (the frame-0 seeded
# blob->hyperblob mapping is preserved) while geometry/appearance still follow
# the member blobs.  Default False preserves the vendored behavior exactly.
_FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT = False
# Number of per-frame gibbs_blob_means refinements (the vendored
# fori_loop(0, 15, ...)).  Default 15 reproduces the vendored schedule exactly;
# lowering it is the anti-bleed lever (keeps frozen object blob means near the
# constant-velocity prediction instead of chasing leaked background datapoints).
# jit-static (closed over at build time).
_BLOB_MEANS_UPDATES_DEFAULT = 15
# When True the per-frame Gibbs SKIPS the blob/hyperblob DINO feature-mean updates
# (gibbs_blob_features_dino / gibbs_hyperblob_features_dino), holding the frame-0
# seeded APPEARANCE frozen across the stream. Motivation: the per-frame appearance
# update can drift the object representation into background; freezing it keeps the
# frame-0 appearance as a stable anchor. jit-static; default False = bit-exact.
_FREEZE_BLOB_FEATURES_DEFAULT = False
# Weight on the instance-purity term in the combined data_loglik objective.  The
# anchor term is ~ -40 (per-datapoint mean loglik); purity is ~ [-log K, 0].  This
# scales purity so its variation across configs is comparable to the anchor term's.
_PURITY_WEIGHT = 20.0
# TEMPORAL ASSIGNMENT STICKINESS (hysteresis) for the FINAL blob assignment: a
# log-prob bonus added to each datapoint's PREVIOUS-frame blob, so a static scene
# point keeps its blob instead of being re-sampled by the categorical step every
# frame (a large fraction of background datapoints are otherwise reassigned per
# frame, the dominant amplifier of background-particle jitter).  A moving point
# still switches once the data evidence beats the bonus, so it does NOT over-stick.
# None/0 = OFF = bit-exact vendored final assignment.  When ON the final step routes
# through the feature-aware re-impl (tau_feat=1) + the bonus.  TRACED scalar (one
# compile across kappa values).
_FINAL_ASSIGNMENT_STICKINESS_DEFAULT = None


@partial(jax.jit, static_argnames=("feature_aware_final", "final_outlier",
                                    "freeze_hyperblob_assignment",
                                    "blob_means_updates", "freeze_blob_features",
                                    "use_feature_temp_final", "final_assignment_anchor",
                                    "use_final_stickiness", "use_robust_mean"))
def _step_jit(key, state, positions, velocities, features,
              feature_aware_final: bool = _FEATURE_AWARE_FINAL_DEFAULT,
              final_outlier: bool = _FINAL_OUTLIER_DEFAULT,
              freeze_hyperblob_assignment: bool = _FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT,
              blob_means_updates: int = _BLOB_MEANS_UPDATES_DEFAULT,
              freeze_blob_features: bool = _FREEZE_BLOB_FEATURES_DEFAULT,
              blob_feat_anchor=None, hb_feat_anchor=None,
              feature_update_damping=None,
              use_feature_temp_final: bool = False,
              final_feature_temp=None,
              final_assignment_anchor: bool = False,
              use_final_stickiness: bool = False,
              final_assignment_stickiness=None,
              use_robust_mean: bool = False,
              mean_update_robust_delta=None):
    """Body of the vendored `f_tracking_sweep` active step, but standalone so we
    can call it per-frame instead of via lax.scan."""
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
        use_feature_temp_final=use_feature_temp_final,
        final_feature_temp=final_feature_temp,
        final_assignment_anchor=final_assignment_anchor,
        use_final_stickiness=use_final_stickiness,
        final_assignment_stickiness=final_assignment_stickiness,
        use_robust_mean=use_robust_mean,
        mean_update_robust_delta=mean_update_robust_delta,
    )
    return state, key


def step(state, positions: np.ndarray, velocities: np.ndarray, features: np.ndarray, key,
         *, feature_aware_final: bool = _FEATURE_AWARE_FINAL_DEFAULT,
         final_outlier: bool = _FINAL_OUTLIER_DEFAULT,
         freeze_hyperblob_assignment: bool = _FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT,
         blob_means_updates: int = _BLOB_MEANS_UPDATES_DEFAULT,
         freeze_blob_features: bool = _FREEZE_BLOB_FEATURES_DEFAULT,
         blob_feat_anchor=None, hb_feat_anchor=None,
         feature_update_damping=None,
         use_feature_temp_final: bool = False,
         final_feature_temp=None,
         final_assignment_anchor: bool = False,
         use_final_stickiness: bool = False,
         final_assignment_stickiness=None,
         use_robust_mean: bool = False,
         mean_update_robust_delta=None):
    """One Gibbs sweep on the current frame.  Returns (new_state, new_key).

    ``blob_feat_anchor``/``hb_feat_anchor``/``feature_update_damping`` (all
    default None) activate the DAMPED anti-drift feature update (see
    ``blob_tracking_gibbs_dino_streaming``); they are passed to ``_step_jit`` as
    TRACED args (not static) so damping value + per-video anchor never recompile.

    ``use_feature_temp_final`` / ``final_feature_temp`` / ``final_assignment_anchor``
    select the re-implemented feature-temperature final assignment; ``final_feature_temp``
    is a TRACED scalar (one compile across tau values), the two bools are static.
    """
    pj = jnp.asarray(positions)
    vj = jnp.asarray(velocities)
    fj = jnp.asarray(features)
    damp = None if feature_update_damping is None else jnp.asarray(
        feature_update_damping, dtype=jnp.float32)
    ba = None if blob_feat_anchor is None else jnp.asarray(blob_feat_anchor)
    ha = None if hb_feat_anchor is None else jnp.asarray(hb_feat_anchor)
    tau = (jnp.asarray(1.0 if final_feature_temp is None else final_feature_temp,
                       dtype=jnp.float32) if use_feature_temp_final else None)
    stick = (jnp.asarray(final_assignment_stickiness, dtype=jnp.float32)
             if use_final_stickiness else None)
    rdelta = (jnp.asarray(mean_update_robust_delta, dtype=jnp.float32)
              if use_robust_mean else None)
    # Static (Python-bool/int) args -> _step_jit specializes one compile per combo.
    state, key = _step_jit(key, state, pj, vj, fj,
                           feature_aware_final=bool(feature_aware_final),
                           final_outlier=bool(final_outlier),
                           freeze_hyperblob_assignment=bool(freeze_hyperblob_assignment),
                           blob_means_updates=int(blob_means_updates),
                           freeze_blob_features=bool(freeze_blob_features),
                           blob_feat_anchor=ba, hb_feat_anchor=ha,
                           feature_update_damping=damp,
                           use_feature_temp_final=bool(use_feature_temp_final),
                           final_feature_temp=tau,
                           final_assignment_anchor=bool(final_assignment_anchor),
                           use_final_stickiness=bool(use_final_stickiness),
                           final_assignment_stickiness=stick,
                           use_robust_mean=bool(use_robust_mean),
                           mean_update_robust_delta=rdelta)
    return state, key


# ---- Multi-sweep per-frame step (calibration only; streaming demo uses step) ----
#
# Calibration deepens the per-frame measurement update from 1 Gibbs sweep to N:
# predict + splice ONCE per frame, then ``jax.lax.scan`` blob_tracking_gibbs_dino
# N times inside a single jit boundary.  Re-predicting inside the scan would
# advance blob_means by N velocity steps (wrong dynamics), so the predict is
# hoisted out — mirroring the vendored ``f_tracking_sweep_dino`` active_step and
# ``_init_gibbs_scan_step`` (whose scan body is also gibbs-only).
_MULTI_SWEEP_CACHE: dict = {}


def _make_multi_sweep_step(num_sweeps: int,
                           feature_aware_final: bool = _FEATURE_AWARE_FINAL_DEFAULT,
                           final_outlier: bool = _FINAL_OUTLIER_DEFAULT,
                           freeze_hyperblob_assignment: bool = _FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT,
                           blob_means_updates: int = _BLOB_MEANS_UPDATES_DEFAULT,
                           freeze_blob_features: bool = _FREEZE_BLOB_FEATURES_DEFAULT,
                           use_feature_temp_final: bool = False,
                           final_assignment_anchor: bool = False,
                           use_final_stickiness: bool = False,
                           use_robust_mean: bool = False):
    """Build one jit'd N-sweep step; the caller caches it per distinct
    ``(num_sweeps, feature_aware_final, final_outlier,
    freeze_hyperblob_assignment, blob_means_updates, freeze_blob_features,
    blend, use_feature_temp_final, final_assignment_anchor)``.  All are closed over
    as Python values (the scan length is static; the flags select the
    final-assignment body, whether hyperblob membership is re-sampled, and the
    per-frame blob-mean refinement count in
    ``blob_tracking_gibbs_dino_streaming`` at build time), so they never force a
    recompile mid-run.  ``final_feature_temp`` is a TRACED runtime arg."""
    @jax.jit
    def _multi_step(key, state, positions, velocities, features,
                    blob_feat_anchor=None, hb_feat_anchor=None,
                    feature_update_damping=None, final_feature_temp=None,
                    final_assignment_stickiness=None, mean_update_robust_delta=None):
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
            # blob_feat_anchor/hb_feat_anchor/feature_update_damping/final_feature_temp
            # are TRACED runtime args (None when inactive) -> the anti-drift blend and
            # the feature-temperature are baked once per (structure) combo, not per
            # damping/tau value or per video.
            state = blob_tracking_gibbs_dino_streaming(
                gibbs_key, state,
                feature_aware_final=feature_aware_final, final_outlier=final_outlier,
                freeze_hyperblob_assignment=freeze_hyperblob_assignment,
                blob_means_updates=blob_means_updates,
                freeze_blob_features=freeze_blob_features,
                feature_update_damping=feature_update_damping,
                blob_feat_anchor=blob_feat_anchor,
                hb_feat_anchor=hb_feat_anchor,
                use_feature_temp_final=use_feature_temp_final,
                final_feature_temp=final_feature_temp,
                final_assignment_anchor=final_assignment_anchor,
                use_final_stickiness=use_final_stickiness,
                final_assignment_stickiness=final_assignment_stickiness,
                use_robust_mean=use_robust_mean,
                mean_update_robust_delta=mean_update_robust_delta,
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
                     feature_update_damping=None,
                     use_feature_temp_final: bool = False,
                     final_feature_temp=None,
                     final_assignment_anchor: bool = False,
                     use_final_stickiness: bool = False,
                     final_assignment_stickiness=None,
                     use_robust_mean: bool = False,
                     mean_update_robust_delta=None):
    """N-sweep variant of ``step()``.  Falls through to ``step()`` when
    ``num_sweeps <= 1`` so the streaming code path pays no extra JIT compile or
    scan-of-1 overhead.  Caches one compiled implementation per
    ``(num_sweeps, feature_aware_final, final_outlier,
    freeze_hyperblob_assignment, blob_means_updates, freeze_blob_features,
    blend, use_feature_temp_final, final_assignment_anchor)``.  The damped-feature
    anchors / damping / feature-temperature are TRACED runtime args (not in the cache
    key beyond their on/off bits), so one program serves all damping/tau values and
    all videos of equal shape."""
    if num_sweeps <= 1:
        return step(state, positions, velocities, features, key,
                    feature_aware_final=feature_aware_final,
                    final_outlier=final_outlier,
                    freeze_hyperblob_assignment=freeze_hyperblob_assignment,
                    blob_means_updates=blob_means_updates,
                    freeze_blob_features=freeze_blob_features,
                    blob_feat_anchor=blob_feat_anchor,
                    hb_feat_anchor=hb_feat_anchor,
                    feature_update_damping=feature_update_damping,
                    use_feature_temp_final=use_feature_temp_final,
                    final_feature_temp=final_feature_temp,
                    final_assignment_anchor=final_assignment_anchor,
                    use_final_stickiness=use_final_stickiness,
                    final_assignment_stickiness=final_assignment_stickiness,
                    use_robust_mean=use_robust_mean,
                    mean_update_robust_delta=mean_update_robust_delta)
    pj = jnp.asarray(positions)
    vj = jnp.asarray(velocities)
    fj = jnp.asarray(features)
    # The damping blend activates only when an explicit damping is ALSO set (mirrors
    # ``_blend`` in blob_tracking_gibbs_dino_streaming) so an anchor passed purely for
    # the anchor-referenced final assignment does not trigger the blend.
    blend = (blob_feat_anchor is not None) and (feature_update_damping is not None)
    ba = None if blob_feat_anchor is None else jnp.asarray(blob_feat_anchor)
    ha = None if hb_feat_anchor is None else jnp.asarray(hb_feat_anchor)
    damp = None if feature_update_damping is None else jnp.asarray(
        feature_update_damping, dtype=jnp.float32)
    tau = (jnp.asarray(1.0 if final_feature_temp is None else final_feature_temp,
                       dtype=jnp.float32) if use_feature_temp_final else None)
    stick = (jnp.asarray(final_assignment_stickiness, dtype=jnp.float32)
             if use_final_stickiness else None)
    rdelta = (jnp.asarray(mean_update_robust_delta, dtype=jnp.float32)
              if use_robust_mean else None)
    cache_key = (int(num_sweeps), bool(feature_aware_final), bool(final_outlier),
                 bool(freeze_hyperblob_assignment), int(blob_means_updates),
                 bool(freeze_blob_features), bool(blend),
                 bool(use_feature_temp_final), bool(final_assignment_anchor),
                 bool(use_final_stickiness), bool(use_robust_mean))
    fn = _MULTI_SWEEP_CACHE.get(cache_key)
    if fn is None:
        fn = _make_multi_sweep_step(num_sweeps, bool(feature_aware_final),
                                    bool(final_outlier),
                                    bool(freeze_hyperblob_assignment),
                                    int(blob_means_updates),
                                    bool(freeze_blob_features),
                                    bool(use_feature_temp_final),
                                    bool(final_assignment_anchor),
                                    bool(use_final_stickiness),
                                    bool(use_robust_mean))
        _MULTI_SWEEP_CACHE[cache_key] = fn
    state, key = fn(key, state, pj, vj, fj, ba, ha, damp, tau, stick, rdelta)
    return state, key


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

    Models the vendored ``dense_eval_blob_weights`` — same ``segment_sum`` over
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


def extract_blob_means_and_covs(state) -> Tuple[np.ndarray, np.ndarray]:
    """Return (blob_means_Nx3, blob_covs_Nx3x3) as host numpy arrays."""
    return (np.asarray(state.blobs_state.blob_means),
            np.asarray(state.blobs_state.blob_covs))


def extract_hyperblob_means_and_covs(state) -> Tuple[np.ndarray, np.ndarray]:
    """Return (hyperblob_means_Kx3, hyperblob_covs_Kx3x3)."""
    return (np.asarray(state.hyperblobs_state.hyperblob_means),
            np.asarray(state.hyperblobs_state.hyperblob_covs))


# ---------------- Self-test ----------------

if __name__ == "__main__":
    import time
    import genmatter_viz as _gv  # render smoke-test only (no module-level dep)

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
    pixel_by_particle, pixel_by_cluster = _gv.render_matter_tile(
        blob_a, hyperblob_a, indices, 360, 640, STRIDE)
    print(f"  pixel_by_particle={pixel_by_particle.shape} "
          f"pixel_by_cluster={pixel_by_cluster.shape}")

    # Cluster (hyperblob) + particle (blob) ellipse tiles need a base RGB; use
    # a placeholder mid-gray for the synthetic test.
    base = np.full((360, 640, 3), 80, dtype=np.uint8)
    blob_means, blob_covs = extract_blob_means_and_covs(state)
    hb_means, hb_covs = extract_hyperblob_means_and_covs(state)
    particles_tile = _gv.render_centroid_tile(blob_means, blob_covs, _gv.BLOB_PALETTE[:blob_means.shape[0]],
                                           base, DEFAULT_INTRINSICS, sigma_scale=1.5, alpha=0.55)
    clusters_tile = _gv.render_centroid_tile(hb_means, hb_covs,
                                          _gv.HYPERBLOB_PALETTE[:hb_means.shape[0]],
                                          base, DEFAULT_INTRINSICS, sigma_scale=1.5, alpha=0.55)
    print(f"  particles_tile={particles_tile.shape} clusters_tile={clusters_tile.shape}")

    pc_cluster = _gv.render_pointcloud_tile(depth, pixel_by_cluster, DEFAULT_INTRINSICS)
    pc_particle = _gv.render_pointcloud_tile(depth, pixel_by_particle, DEFAULT_INTRINSICS)
    print(f"  pointcloud_cluster={pc_cluster.shape} pointcloud_particle={pc_particle.shape}")
    print("smoke test ok")

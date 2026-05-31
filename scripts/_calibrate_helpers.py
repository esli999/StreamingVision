"""Helpers for ``scripts/calibrate_consistency.py``.

JAX-native per-frame sufficient-statistics primitives and JAX ARI for
consistency scoring. Conjugate posterior MAP estimators (closed-form scalar /
matrix math) stay in numpy — they run once per EM iter and don't benefit from
GPU.

The hot paths (M-step pooling, ARI) use ``jax.ops.segment_sum`` over a
fixed-K_MAX label space and ``jax.vmap`` over T frames, so each primitive
compiles once per cached (T, N, D) shape and is reused across all 38 videos.

Pseudo-label generation (sklearn KMeans inside SAM instances, motion KMeans)
runs once at the start of the calibration session and is cached as .npz on
disk, so it's not on the iterative hot path. Kept in numpy/sklearn for
clarity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import jax
import jax.numpy as jnp


# Upper bound on any cluster id across all partitions, set once. This bounds the
# segment_sum label space for the M-step sufficient stats and the (K_MAX × K_MAX)
# ARI contingency table; ids >= K_MAX are silently dropped by segment_sum and the
# OUTLIER_FOLD_ID = K_MAX-1 bucket is reserved for outliers/background.
#
# Sized 256 (was 128) so the stronger inference strategy's tracker blob budget
# (num_blobs=128, which SAM-frame-0 init can nudge slightly higher) is fully
# covered — every tracker blob id 0..~150 lands well below the outlier-fold
# sentinel (255), so no real blob is dropped from the conjugate M-step or aliased
# onto the outlier bucket. It also roughly halves the residual truncation of the
# high-cumulative-id pseudo SAM instances (Z_sam can reach the low hundreds with
# detect_new_objects on long videos). The ARI table is 256*256 = 65536 cells —
# still cheap on GPU, and ARI is now a local-video diagnostic only.
K_MAX = 256

# Outlier-fold sentinel id used by callers before feeding into ARI / segment_sum.
OUTLIER_FOLD_ID = K_MAX - 1


# ----------------------------------------------------------------------
# Pseudo-label loading + KMeans refinement (one-time, cached → numpy)
# ----------------------------------------------------------------------

def load_z_sam_for_frame(vid: str, frame_idx: int, *, kind: str,
                          repo_root: Path) -> Optional[np.ndarray]:
    """Load the uint16 instance-id PNG SAM mask for a (video, frame)."""
    if kind == "local":
        path = (repo_root / "assets" / "custom_videos" / vid /
                "pseudo_gt_sam" / "segmasks" / vid / f"{frame_idx:05d}.png")
    elif kind == "davis":
        path = (repo_root / "assets" / "tapvid_davis_30_videos_processed" /
                "sam2_propagated" / vid / f"{frame_idx:05d}.png")
    else:
        raise ValueError(f"unknown kind {kind!r}")
    if not path.is_file():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def gather_z_sam_at_grid(mask_native: np.ndarray, indices: np.ndarray,
                         work_hw: Tuple[int, int], stride: int) -> np.ndarray:
    """Downsample a native-resolution SAM mask to the stride-8 grid and gather."""
    h, w = work_hw
    gh = h // stride
    gw = w // stride
    small = cv2.resize(mask_native.astype(np.int32), (gw, gh),
                       interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return small.reshape(-1)[indices].astype(np.int32)


def within_instance_dino_kmeans(features: np.ndarray, z_sam: np.ndarray, *,
                                  min_inst_size: int = 50,
                                  max_k_per_inst: int = 8,
                                  per_subcluster_size: int = 100,
                                  seed: int = 0) -> np.ndarray:
    """Refine SAM instances with DINO-feature KMeans (sklearn, cached output)."""
    from sklearn.cluster import KMeans

    n = z_sam.shape[0]
    out = np.full(n, -1, dtype=np.int32)
    next_id = 0
    bg_mask = (z_sam == 0)
    if bg_mask.any():
        out[bg_mask] = next_id
        next_id += 1
    for inst_id in np.unique(z_sam):
        if inst_id == 0:
            continue
        mask = (z_sam == inst_id)
        size = int(mask.sum())
        if size < min_inst_size:
            out[mask] = next_id
            next_id += 1
            continue
        k = min(max(2, size // per_subcluster_size), max_k_per_inst)
        feat_inst = features[mask]
        try:
            sub = KMeans(n_clusters=k, random_state=seed, n_init=4).fit_predict(feat_inst)
        except Exception:
            sub = np.zeros(size, dtype=np.int32)
        out[mask] = next_id + sub.astype(np.int32)
        next_id += int(sub.max()) + 1
    return out


def motion_kmeans(velocities: np.ndarray, *, k: int = 8, seed: int = 0) -> np.ndarray:
    """KMeans on per-datapoint 3D velocities (sklearn, cached output)."""
    from sklearn.cluster import KMeans
    try:
        return KMeans(n_clusters=k, random_state=seed, n_init=4).fit_predict(velocities).astype(np.int32)
    except Exception:
        return np.zeros(velocities.shape[0], dtype=np.int32)


# ----------------------------------------------------------------------
# Conjugate posterior MAP estimators (scalar / 3×3 — numpy is fine)
# ----------------------------------------------------------------------

def map_variance(resid_sq_sum: float, n: int, default_var: float,
                 n_0: float = 100.0) -> float:
    """Normal-Inverse-Gamma posterior mean of variance.

    Prior: Inv-Gamma centered at default_var with effective "n_0 datapoints".
    Posterior mean: (n_0 · default_var + Σ x²) / (n_0 + n).
    """
    return float((n_0 * default_var + resid_sq_sum) / (n_0 + n))


def map_wishart(pooled_scatter: np.ndarray, pooled_n: float,
                default_psi: np.ndarray, nu_0: float = 4.0) -> np.ndarray:
    """Inv-Wishart posterior mean of scale matrix.

    pooled_scatter = Σ_c (n_c − 1) S_c; pooled_n = Σ_c (n_c − 1).
    Posterior mean = (nu_0 · default_psi + pooled_scatter) / (nu_0 + pooled_n).
    """
    return ((nu_0 * default_psi + pooled_scatter) /
            (nu_0 + pooled_n)).astype(np.float32)


def map_crp_alpha(K_observed: float, N: int) -> Optional[float]:
    """Empirical-Bayes solve for CRP concentration given observed K."""
    from scipy.optimize import brentq
    if not np.isfinite(K_observed) or K_observed <= 1.0 or K_observed >= N:
        return None
    def f(a: float) -> float:
        return a * np.log((N + a) / a) - K_observed
    try:
        return float(brentq(f, 0.01, 1000.0))
    except Exception:
        return None


# ----------------------------------------------------------------------
# JAX-native per-frame primitives (segment_sum on K_MAX label space)
# ----------------------------------------------------------------------

def _frame_resid_sq_sum_impl(obs_ND: jnp.ndarray, z_N: jnp.ndarray
                              ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Σ of squared residual from per-cluster mean. Returns (resid_sq_sum, n_valid).

    Datapoints with z_N == OUTLIER_FOLD_ID are excluded from both the sum and n.
    """
    N, D = obs_ND.shape
    sums = jax.ops.segment_sum(obs_ND, z_N, num_segments=K_MAX)        # (K_MAX, D)
    counts = jax.ops.segment_sum(jnp.ones(N), z_N, num_segments=K_MAX)  # (K_MAX,)
    means = sums / jnp.maximum(counts[:, None], 1.0)
    resid = obs_ND - means[z_N]
    valid = ((z_N >= 0) & (z_N != OUTLIER_FOLD_ID) & (z_N < K_MAX)
             ).astype(jnp.float32)
    sum_sq = jnp.sum((resid ** 2) * valid[:, None])
    n_valid = jnp.sum(valid) * float(D)
    return sum_sq, n_valid


def _frame_per_cluster_scatter_impl(obs_N3: jnp.ndarray, z_N: jnp.ndarray
                                     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-cluster sum-of-squared-deviations (K_MAX, 3, 3) + cluster counts (K_MAX,).

    Host masks out the OUTLIER_FOLD_ID bucket after the JIT call.
    """
    N = obs_N3.shape[0]
    sums = jax.ops.segment_sum(obs_N3, z_N, num_segments=K_MAX)        # (K_MAX, 3)
    counts = jax.ops.segment_sum(jnp.ones(N), z_N, num_segments=K_MAX)  # (K_MAX,)
    means = sums / jnp.maximum(counts[:, None], 1.0)
    diff = obs_N3 - means[z_N]                                          # (N, 3)
    outers = diff[:, :, None] * diff[:, None, :]                         # (N, 3, 3)
    scatter = jax.ops.segment_sum(outers, z_N, num_segments=K_MAX)      # (K_MAX, 3, 3)
    return scatter, counts


def _frame_hyperblob_impl(feat_ND: jnp.ndarray, pos_N3: jnp.ndarray,
                           z_instance_N: jnp.ndarray, z_fine_N: jnp.ndarray
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                       jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-frame two-level (instance, sub-cluster) sufficient stats.

    Caller convention: invalid datapoints (SAM background, tracker outliers)
    must have z_instance_N == OUTLIER_FOLD_ID before calling. This unifies the
    pseudo (background) and tracker (outlier) exclusion paths.

    Returns tuple of jnp arrays:
        sigma_F_H_resid_sq_sum (scalar)
        sigma_F_H_n_valid (scalar): count of valid blobs
        Psi_H_scatter_per_inst (K_MAX, 3, 3): per-instance scatter of blob pos means
        inst_blob_counts (K_MAX,): valid blobs per instance (filter + nu_H median)
        inst_centroid_positions (K_MAX, 3): per-instance centroid (for sigma_H + mu_H)
        inst_centroid_counts (K_MAX,): datapoints per instance (mask for centroids)
    """
    N, D = feat_ND.shape
    feat_sums = jax.ops.segment_sum(feat_ND, z_fine_N, num_segments=K_MAX)
    pos_sums = jax.ops.segment_sum(pos_N3, z_fine_N, num_segments=K_MAX)
    blob_counts = jax.ops.segment_sum(jnp.ones(N), z_fine_N, num_segments=K_MAX)
    blob_count_safe = jnp.maximum(blob_counts, 1.0)
    blob_feat_means = feat_sums / blob_count_safe[:, None]
    blob_pos_means = pos_sums / blob_count_safe[:, None]

    # Each z_fine segment shares a z_instance value (by construction of both
    # within_instance_dino_kmeans and the tracker's hb_lookup[blob_a]).
    # segment_max returns INT_MIN for empty segments; mask via valid_blob.
    blob_inst = jax.ops.segment_max(z_instance_N, z_fine_N, num_segments=K_MAX)
    valid_blob = ((blob_counts >= 8.0) & (blob_inst >= 0) &
                   (blob_inst != OUTLIER_FOLD_ID))
    valid_f = valid_blob.astype(jnp.float32)

    # Clamp invalid blob_inst to 0 so downstream segment_sum gets a legal
    # segment id; the valid_f=0 mask ensures the bad blob contributes nothing.
    blob_inst_safe = jnp.where(valid_blob, blob_inst, 0)

    inst_feat_sums = jax.ops.segment_sum(blob_feat_means * valid_f[:, None],
                                          blob_inst_safe, num_segments=K_MAX)
    inst_pos_sums = jax.ops.segment_sum(blob_pos_means * valid_f[:, None],
                                         blob_inst_safe, num_segments=K_MAX)
    inst_blob_counts = jax.ops.segment_sum(valid_f, blob_inst_safe, num_segments=K_MAX)
    inst_count_safe = jnp.maximum(inst_blob_counts, 1.0)
    inst_feat_means = inst_feat_sums / inst_count_safe[:, None]
    inst_pos_means = inst_pos_sums / inst_count_safe[:, None]

    blob_feat_resid = (blob_feat_means - inst_feat_means[blob_inst_safe]) * valid_f[:, None]
    sigma_F_H_resid_sq_sum = jnp.sum(blob_feat_resid ** 2)
    sigma_F_H_n_valid = jnp.sum(valid_f)

    blob_pos_resid = (blob_pos_means - inst_pos_means[blob_inst_safe]) * valid_f[:, None]
    outers = blob_pos_resid[:, :, None] * blob_pos_resid[:, None, :]
    Psi_H_scatter_per_inst = jax.ops.segment_sum(outers, blob_inst_safe, num_segments=K_MAX)

    # Per-instance centroid positions on datapoint level. The
    # OUTLIER_FOLD_ID bucket aggregates excluded datapoints — host masks it
    # before computing sigma_H / mu_H.
    inst_pos_pt_sums = jax.ops.segment_sum(pos_N3, z_instance_N, num_segments=K_MAX)
    inst_counts_pt = jax.ops.segment_sum(jnp.ones(N), z_instance_N, num_segments=K_MAX)
    inst_centroid_positions = inst_pos_pt_sums / jnp.maximum(inst_counts_pt[:, None], 1.0)

    return (sigma_F_H_resid_sq_sum, sigma_F_H_n_valid, Psi_H_scatter_per_inst,
            inst_blob_counts, inst_centroid_positions, inst_counts_pt)


def _frame_ari_impl(a_N: jnp.ndarray, b_N: jnp.ndarray) -> jnp.ndarray:
    """Adjusted Rand Index for one frame. Inputs must be in [0, K_MAX)."""
    N = a_N.shape[0]
    flat = a_N * K_MAX + b_N
    cont_flat = jax.ops.segment_sum(jnp.ones(N), flat, num_segments=K_MAX * K_MAX)
    cont = cont_flat.reshape(K_MAX, K_MAX)

    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_c2_cont = jnp.sum(comb2(cont))
    a_sums = cont.sum(axis=1)
    b_sums = cont.sum(axis=0)
    sum_c2_a = jnp.sum(comb2(a_sums))
    sum_c2_b = jnp.sum(comb2(b_sums))
    total = jnp.sum(cont)
    total_c2 = comb2(total)
    expected = sum_c2_a * sum_c2_b / jnp.maximum(total_c2, 1.0)
    max_idx = (sum_c2_a + sum_c2_b) / 2.0
    denom = max_idx - expected
    return jnp.where(denom > 0.0, (sum_c2_cont - expected) / denom, 0.0)


# Vmap over T frames + JIT — single compile per (T, N, D) shape, reused across
# the 5-iter EM loop and the 6-vid crossval accept-tests.
_frame_resid_sq_sum_vT = jax.jit(jax.vmap(_frame_resid_sq_sum_impl))
_frame_per_cluster_scatter_vT = jax.jit(jax.vmap(_frame_per_cluster_scatter_impl))
_frame_hyperblob_vT = jax.jit(jax.vmap(_frame_hyperblob_impl))
_frame_ari_vT = jax.jit(jax.vmap(_frame_ari_impl))


# ----------------------------------------------------------------------
# Per-video sufficient-statistics aggregator (host-side accumulation)
# ----------------------------------------------------------------------

def _agg_features(feat_TND: jnp.ndarray, z_TN: jnp.ndarray) -> dict:
    rs_per_T, n_per_T = _frame_resid_sq_sum_vT(feat_TND, z_TN)
    return {"resid_sq_sum": float(jnp.sum(rs_per_T)),
            "n": int(jnp.sum(n_per_T))}


def _agg_scatter_layer(obs_TN3: jnp.ndarray, z_TN: jnp.ndarray,
                        min_count: int = 4) -> dict:
    """Σ scatter and per-cluster sizes across (T × K_MAX) with count >= min_count.
    Outlier bucket (OUTLIER_FOLD_ID) masked out so tracker outliers don't form
    a phantom coherent cluster.
    """
    scatter_TK33, counts_TK = _frame_per_cluster_scatter_vT(obs_TN3, z_TN)
    counts = np.asarray(counts_TK)                  # (T, K_MAX)
    scatter = np.asarray(scatter_TK33)              # (T, K_MAX, 3, 3)
    valid = counts >= float(min_count)
    valid[:, OUTLIER_FOLD_ID] = False
    if not valid.any():
        return {"pooled_scatter": None, "pooled_n": 0.0,
                "sizes": [], "K_per_frame": []}
    valid_scatter = scatter[valid]                   # (M, 3, 3)
    valid_counts = counts[valid]                     # (M,)
    pooled_scatter = valid_scatter.sum(axis=0)
    pooled_n = float((valid_counts - 1.0).sum())
    sizes = valid_counts.astype(np.int64).tolist()
    # Per-frame K = non-empty segments excluding the outlier bucket
    nonempty = counts > 0
    nonempty[:, OUTLIER_FOLD_ID] = False
    K_per_frame = nonempty.sum(axis=1).astype(np.int64).tolist()
    return {"pooled_scatter": pooled_scatter, "pooled_n": pooled_n,
            "sizes": sizes, "K_per_frame": K_per_frame}


def _agg_hyperblob(feat_TND: jnp.ndarray, pos_TN3: jnp.ndarray,
                    z_inst_TN: jnp.ndarray, z_fine_TN: jnp.ndarray,
                    min_inst_blob_count: int = 2) -> dict:
    """Caller convention: z_inst_TN must already have background / outliers
    mapped to OUTLIER_FOLD_ID."""
    sfH_rs_T, sfH_n_T, Psi_H_TK33, inst_blob_counts_TK, inst_centroids_TK3, \
        inst_counts_pt_TK = _frame_hyperblob_vT(feat_TND, pos_TN3, z_inst_TN, z_fine_TN)
    D = feat_TND.shape[2]
    sigma_F_H_resid_sq_sum = float(jnp.sum(sfH_rs_T))
    sigma_F_H_n = int(jnp.sum(sfH_n_T) * D)

    Psi_H = np.asarray(Psi_H_TK33)                  # (T, K_MAX, 3, 3)
    blob_counts = np.asarray(inst_blob_counts_TK)   # (T, K_MAX)
    valid_inst = blob_counts >= float(min_inst_blob_count)
    Psi_H_scatters = Psi_H[valid_inst].tolist() if valid_inst.any() else []
    Psi_H_sizes = blob_counts[valid_inst].astype(np.int64).tolist() if valid_inst.any() else []
    if Psi_H_scatters:
        pooled_Psi_H = np.sum(np.stack(Psi_H_scatters, axis=0), axis=0)
        pooled_n_H = float(sum(s - 1 for s in Psi_H_sizes))
    else:
        pooled_Psi_H = None
        pooled_n_H = 0.0

    # Per-frame instance count = non-empty instances excluding OUTLIER_FOLD_ID
    inst_pt_counts = np.asarray(inst_counts_pt_TK)
    inst_active = inst_pt_counts > 0
    inst_active[:, OUTLIER_FOLD_ID] = False
    K_inst_per_frame = inst_active.sum(axis=1).astype(np.int64).tolist()

    inst_centroids = np.asarray(inst_centroids_TK3)
    centroids_valid = inst_centroids[inst_active]
    if centroids_valid.shape[0] >= 2:
        sigma_H_emp = float(np.std(centroids_valid, axis=0).mean())
        mu_H_emp = np.median(centroids_valid, axis=0).astype(np.float32)
    else:
        sigma_H_emp = float("nan")
        mu_H_emp = None

    return {
        "sigma_F_H_resid_sq_sum": sigma_F_H_resid_sq_sum,
        "sigma_F_H_n": sigma_F_H_n,
        "pooled_Psi_H_scatter": pooled_Psi_H,
        "pooled_Psi_H_n": pooled_n_H,
        "Psi_H_sizes": Psi_H_sizes,
        "K_inst_per_frame": K_inst_per_frame,
        "sigma_H_emp": sigma_H_emp,
        "mu_H_emp": mu_H_emp,
    }


def _agg_outlier(vel_TN3: np.ndarray, blob_a_TN: np.ndarray, *,
                  z_sam_TN: Optional[np.ndarray] = None) -> dict:
    """Outlier source: blob_a == -1 (tracker outliers; plan §5 fix).

    Numpy is fine here — input is already on host (tracker output) and we just
    need a mask + norm. z_sam_TN is logged-only.
    """
    T, N, _ = vel_TN3.shape
    outlier_mask = (blob_a_TN == -1)
    outlier_fracs = outlier_mask.mean(axis=1).astype(np.float64).tolist()
    out_speeds: List[float] = []
    if outlier_mask.any():
        for t in range(T):
            sel = outlier_mask[t]
            if sel.any():
                sp = np.linalg.norm(vel_TN3[t][sel], axis=1)
                out_speeds.extend(sp.tolist())
    sam_bg_fracs: List[float] = []
    if z_sam_TN is not None:
        sam_bg_fracs = (z_sam_TN == 0).mean(axis=1).astype(np.float64).tolist()
    return {
        "outlier_fracs_per_frame": outlier_fracs,
        "outlier_speeds": out_speeds,
        "sam_bg_fracs_per_frame": sam_bg_fracs,
    }


def _agg_translation(vel_TN3: jnp.ndarray) -> dict:
    """Per-datapoint velocity magnitudes for translation_* posteriors."""
    speeds = np.linalg.norm(np.asarray(vel_TN3).reshape(-1, 3), axis=1)
    speeds = speeds[np.isfinite(speeds)]
    return {"speeds": speeds.astype(np.float32)}


def per_video_sufficient_stats(labels_TN: dict, blob_a_TN: np.ndarray,
                                 hb_a_TN: np.ndarray) -> Tuple[dict, dict]:
    """Drive the JAX per-frame primitives across this video's T frames under
    BOTH the pseudo-label partition (Z_dino/Z_motion/Z_sam) and the tracker
    partition (blob_a/hb_a).
    """
    pos = jnp.asarray(labels_TN["positions"])
    vel = jnp.asarray(labels_TN["velocities"])
    feat = jnp.asarray(labels_TN["features"])
    z_sam = jnp.asarray(labels_TN["Z_sam"])
    z_dino = jnp.asarray(labels_TN["Z_dino"])
    z_motion = jnp.asarray(labels_TN["Z_motion"])

    T = min(int(blob_a_TN.shape[0]), int(pos.shape[0]))
    pos, vel, feat = pos[:T], vel[:T], feat[:T]
    z_sam, z_dino, z_motion = z_sam[:T], z_dino[:T], z_motion[:T]

    # Tracker partitions: outliers (-1) folded to OUTLIER_FOLD_ID = K_MAX - 1.
    # This collapses all tracker outliers into a single cluster on the K_MAX
    # label space; the JIT excludes that bucket from variance / scatter sums.
    blob_a_np = blob_a_TN[:T].astype(np.int32)
    hb_a_np = hb_a_TN[:T].astype(np.int32)
    blob_a_safe = np.where(blob_a_np >= 0, blob_a_np, OUTLIER_FOLD_ID).astype(np.int32)
    hb_a_safe = np.where(hb_a_np >= 0, hb_a_np, OUTLIER_FOLD_ID).astype(np.int32)
    blob_a_j = jnp.asarray(blob_a_safe)
    hb_a_j = jnp.asarray(hb_a_safe)
    vel_np_full = np.asarray(vel)

    # _agg_hyperblob expects the instance-level partition to mark its excluded
    # bucket with OUTLIER_FOLD_ID. For pseudo (z_sam): background (0) → OUTLIER_FOLD_ID.
    # For tracker (hb_a_safe): outliers already mapped above.
    z_sam_np = np.asarray(z_sam, dtype=np.int32)
    z_sam_for_hb = jnp.asarray(np.where(z_sam_np == 0, OUTLIER_FOLD_ID, z_sam_np).astype(np.int32))

    pseudo = {
        "features": _agg_features(feat, z_dino),
        "position": _agg_scatter_layer(pos, z_dino),
        "velocity": _agg_scatter_layer(vel, z_motion),
        "hyperblob": _agg_hyperblob(feat, pos, z_sam_for_hb, z_dino),
        "translation": _agg_translation(vel),
    }
    tracker = {
        "features": _agg_features(feat, blob_a_j),
        "position": _agg_scatter_layer(pos, blob_a_j),
        "velocity": _agg_scatter_layer(vel, blob_a_j),
        "hyperblob": _agg_hyperblob(feat, pos, hb_a_j, blob_a_j),
        "outlier": _agg_outlier(vel_np_full, blob_a_np,
                                  z_sam_TN=z_sam_np),
    }
    return pseudo, tracker


# ----------------------------------------------------------------------
# Cross-video aggregator: pool sufficient stats + apply conjugate posteriors
# ----------------------------------------------------------------------

def _avg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    vals = [v for v in (a, b) if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _avg_arr(a: Optional[np.ndarray], b: Optional[np.ndarray]
              ) -> Optional[np.ndarray]:
    arrs = [v for v in (a, b) if v is not None]
    if not arrs:
        return None
    return np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)


def aggregate_posteriors(
    per_vid_pseudo: Dict[str, dict],
    per_vid_tracker: Dict[str, dict],
    defaults: dict,
    *,
    N_per_frame: int = 2925,
) -> dict:
    """Pool sufficient stats across all videos (separately for each partition)
    and apply MAP. Per the plan §2, MAPs from pseudo + tracker are averaged
    per-layer to mix their biases (K-means-tight pseudo + current-hypers tracker).
    """
    out: dict = {}

    # σ_F
    def _pool_feat(per_vid: dict):
        rs = sum(d["features"]["resid_sq_sum"] for d in per_vid.values() if "features" in d)
        n = sum(d["features"]["n"] for d in per_vid.values() if "features" in d)
        return rs, n
    rs_p, n_p = _pool_feat(per_vid_pseudo)
    rs_t, n_t = _pool_feat(per_vid_tracker)
    sigma_F_p = map_variance(rs_p, n_p, defaults["sigma_F"]) if n_p > 0 else None
    sigma_F_t = map_variance(rs_t, n_t, defaults["sigma_F"]) if n_t > 0 else None
    out["sigma_F"] = _avg(sigma_F_p, sigma_F_t) or float("nan")

    # Psi_B + nu_B
    def _pool_scatter(per_vid: dict, layer: str):
        scatters: List[np.ndarray] = []
        ns: List[float] = []
        sizes: List[int] = []
        for d in per_vid.values():
            sec = d.get(layer, {})
            if sec.get("pooled_scatter") is not None:
                scatters.append(sec["pooled_scatter"])
                ns.append(sec["pooled_n"])
            sizes.extend(sec.get("sizes", []))
        if not scatters:
            return None, 0.0, None
        pooled_scatter = np.sum(np.stack(scatters, axis=0), axis=0)
        pooled_n = float(sum(ns))
        med_size = float(np.median(sizes)) if sizes else None
        return pooled_scatter, pooled_n, med_size

    B_p, nB_p, nu_B_p = _pool_scatter(per_vid_pseudo, "position")
    B_t, nB_t, nu_B_t = _pool_scatter(per_vid_tracker, "position")
    Psi_B_p = map_wishart(B_p, nB_p, defaults["Psi_B"]) if B_p is not None else None
    Psi_B_t = map_wishart(B_t, nB_t, defaults["Psi_B"]) if B_t is not None else None
    out["Psi_B"] = _avg_arr(Psi_B_p, Psi_B_t)
    out["nu_B"] = _avg(nu_B_p, nu_B_t) or float("nan")

    # alpha (CRP)
    K_dino_all: List[int] = []
    for d in per_vid_pseudo.values():
        K_dino_all.extend(d.get("position", {}).get("K_per_frame", []))
    K_dino_median = float(np.median(K_dino_all)) if K_dino_all else float("nan")
    alpha_eb = map_crp_alpha(K_dino_median, N_per_frame)
    out["alpha"] = alpha_eb if alpha_eb is not None else float("nan")

    # σ_F_H
    def _pool_sfH(per_vid: dict):
        rs = sum(d.get("hyperblob", {}).get("sigma_F_H_resid_sq_sum", 0.0)
                 for d in per_vid.values())
        n = sum(d.get("hyperblob", {}).get("sigma_F_H_n", 0)
                for d in per_vid.values())
        return rs, n
    sfH_rs_p, sfH_n_p = _pool_sfH(per_vid_pseudo)
    sfH_rs_t, sfH_n_t = _pool_sfH(per_vid_tracker)
    sfH_p = map_variance(sfH_rs_p, sfH_n_p, defaults["sigma_F_H"]) if sfH_n_p > 0 else None
    sfH_t = map_variance(sfH_rs_t, sfH_n_t, defaults["sigma_F_H"]) if sfH_n_t > 0 else None
    out["sigma_F_H"] = _avg(sfH_p, sfH_t) or float("nan")

    # Psi_H + nu_H
    def _pool_hyper_scatter(per_vid: dict):
        scatters: List[np.ndarray] = []
        ns: List[float] = []
        sizes: List[int] = []
        for d in per_vid.values():
            sec = d.get("hyperblob", {})
            if sec.get("pooled_Psi_H_scatter") is not None:
                scatters.append(sec["pooled_Psi_H_scatter"])
                ns.append(sec["pooled_Psi_H_n"])
            sizes.extend(sec.get("Psi_H_sizes", []))
        if not scatters:
            return None, 0.0, None
        pooled = np.sum(np.stack(scatters, axis=0), axis=0)
        pooled_n = float(sum(ns))
        med_size = float(np.median(sizes)) if sizes else None
        return pooled, pooled_n, med_size

    H_p, nH_p, nu_H_p = _pool_hyper_scatter(per_vid_pseudo)
    H_t, nH_t, nu_H_t = _pool_hyper_scatter(per_vid_tracker)
    Psi_H_p = map_wishart(H_p, nH_p, defaults["Psi_H"]) if H_p is not None else None
    Psi_H_t = map_wishart(H_t, nH_t, defaults["Psi_H"]) if H_t is not None else None
    out["Psi_H"] = _avg_arr(Psi_H_p, Psi_H_t)
    out["nu_H"] = _avg(nu_H_p, nu_H_t) or float("nan")

    # beta (CRP on Z_sam)
    K_inst_all: List[int] = []
    for d in per_vid_pseudo.values():
        K_inst_all.extend(d.get("hyperblob", {}).get("K_inst_per_frame", []))
    K_inst_median = float(np.median(K_inst_all)) if K_inst_all else float("nan")
    beta_eb = map_crp_alpha(K_inst_median, N_per_frame)
    out["beta"] = beta_eb if beta_eb is not None else float("nan")

    # sigma_H + mu_H (per-instance centroid scatter from pseudo Z_sam only —
    # the tracker hyperblob_a uses a much coarser 4-way partition that doesn't
    # carry SAM's semantic instance structure)
    sigma_Hs: List[float] = []
    mu_Hs: List[np.ndarray] = []
    for d in per_vid_pseudo.values():
        sec = d.get("hyperblob", {})
        sh = sec.get("sigma_H_emp")
        if sh is not None and np.isfinite(sh):
            sigma_Hs.append(sh)
        mh = sec.get("mu_H_emp")
        if mh is not None:
            mu_Hs.append(mh)
    out["sigma_H"] = float(np.median(sigma_Hs)) if sigma_Hs else float("nan")
    out["mu_H"] = np.median(np.stack(mu_Hs, axis=0), axis=0).astype(np.float32) \
        if mu_Hs else None

    # Psi_V + nu_V
    V_p, nV_p, nu_V_p = _pool_scatter(per_vid_pseudo, "velocity")
    V_t, nV_t, nu_V_t = _pool_scatter(per_vid_tracker, "velocity")
    Psi_V_p = map_wishart(V_p, nV_p, defaults["Psi_V"]) if V_p is not None else None
    Psi_V_t = map_wishart(V_t, nV_t, defaults["Psi_V"]) if V_t is not None else None
    out["Psi_V"] = _avg_arr(Psi_V_p, Psi_V_t)
    out["nu_V"] = _avg(nu_V_p, nu_V_t) or float("nan")

    # sigma_V (isotropic velocity-prior variance; the #1 motion lever). Closed-form
    # Normal-Inv-Gamma proxy: reuse the SAME pooled velocity scatter as Psi_V and
    # take the isotropic mean variance = trace(pooled_scatter) / 3 over the pooled
    # datapoints. _pool_scatter's pooled_n is Σ(n_c − 1); the per-dim residual count
    # is 3·pooled_n. This is an approximation (the within-motion-cluster velocity
    # variance stands in for the constant-velocity process noise), but it is bounded
    # by SANITY_FLOORS and gated by the motion-group accept-test, so a bad estimate
    # never ships. Averaged across the pseudo + tracker partitions like Psi_V/nu_V.
    sv_default = float(defaults.get("sigma_V", 1.0e-1))
    sigma_V_p = (map_variance(float(np.trace(V_p)), 3.0 * nV_p, sv_default)
                 if V_p is not None and nV_p > 0 else None)
    sigma_V_t = (map_variance(float(np.trace(V_t)), 3.0 * nV_t, sv_default)
                 if V_t is not None and nV_t > 0 else None)
    out["sigma_V"] = _avg(sigma_V_p, sigma_V_t) or float("nan")

    # Outlier: tracker partition only (blob_a == -1)
    outlier_fracs: List[float] = []
    outlier_speeds: List[float] = []
    sam_bg_fracs: List[float] = []
    for d in per_vid_tracker.values():
        sec = d.get("outlier", {})
        outlier_fracs.extend(sec.get("outlier_fracs_per_frame", []))
        outlier_speeds.extend(sec.get("outlier_speeds", []))
        sam_bg_fracs.extend(sec.get("sam_bg_fracs_per_frame", []))
    out["outlier_prob"] = float(np.median(outlier_fracs)) if outlier_fracs else float("nan")
    out["sam_bg_outlier_prob_log"] = float(np.median(sam_bg_fracs)) \
        if sam_bg_fracs else float("nan")

    speeds = np.asarray(outlier_speeds, dtype=np.float64)
    speeds = speeds[np.isfinite(speeds) & (speeds >= 0)]
    out["outlier_n"] = int(speeds.size)
    if speeds.size < 50:
        out["gamma_shape"] = float("nan")
        out["gamma_rate"] = float("nan")
    else:
        m = float(speeds.mean())
        v = float(speeds.var(ddof=1))
        if v <= 0.0 or m <= 0.0:
            out["gamma_shape"] = float("nan")
            out["gamma_rate"] = float("nan")
        else:
            out["gamma_rate"] = m / v
            out["gamma_shape"] = m * (m / v)

    # translation_max_radius + translation_gaussian_scale
    speeds_all: List[np.ndarray] = []
    for d in per_vid_pseudo.values():
        sp = d.get("translation", {}).get("speeds")
        if sp is not None and sp.size:
            speeds_all.append(sp)
    if speeds_all:
        speeds_cat = np.concatenate(speeds_all)
        out["translation_max_radius"] = float(np.percentile(speeds_cat, 99))
        out["translation_gaussian_scale"] = float(np.median(speeds_cat))
    else:
        out["translation_max_radius"] = float("nan")
        out["translation_gaussian_scale"] = float("nan")

    return out


# ----------------------------------------------------------------------
# Consistency score (JAX ARI vmap'd over T)
# ----------------------------------------------------------------------

def consistency_score(blob_a_TN: np.ndarray, hyperblob_a_TN: np.ndarray,
                       z_dino_TN: np.ndarray, z_motion_TN: np.ndarray,
                       z_sam_TN: np.ndarray) -> dict:
    """Mean ARI over frames for tracker partition × pseudo-label pairs.

    Outliers (blob == -1) folded into a single extra cluster id (OUTLIER_FOLD_ID =
    K_MAX − 1). ARI is permutation- and label-count-invariant so this is robust.
    The 3 ARIs (fine/coarse/semantic) are JIT'd via the same vmap'd primitive
    and dispatched in 3 vmap calls per video.
    """
    T = blob_a_TN.shape[0]

    def _fold(x: np.ndarray) -> jnp.ndarray:
        clipped = np.where(x >= 0, x, OUTLIER_FOLD_ID).astype(np.int32)
        return jnp.asarray(np.clip(clipped, 0, K_MAX - 1))

    blob_a = _fold(blob_a_TN)
    hb_a = _fold(hyperblob_a_TN)
    z_dino = jnp.asarray(np.clip(z_dino_TN.astype(np.int32), 0, K_MAX - 1))
    z_motion = jnp.asarray(np.clip(z_motion_TN.astype(np.int32), 0, K_MAX - 1))
    z_sam = jnp.asarray(np.clip(z_sam_TN.astype(np.int32), 0, K_MAX - 1))

    s_fine_T = _frame_ari_vT(blob_a, z_dino)
    s_coarse_T = _frame_ari_vT(blob_a, z_motion)
    s_semantic_T = _frame_ari_vT(hb_a, z_sam)

    s_fine_m = float(jnp.mean(s_fine_T))
    s_coarse_m = float(jnp.mean(s_coarse_T))
    s_semantic_m = float(jnp.mean(s_semantic_T))
    valid = [v for v in (s_fine_m, s_coarse_m, s_semantic_m) if np.isfinite(v)]
    s_total = float(np.mean(valid)) if valid else float("nan")
    return {"S_fine": s_fine_m, "S_coarse": s_coarse_m,
            "S_semantic": s_semantic_m, "S": s_total}

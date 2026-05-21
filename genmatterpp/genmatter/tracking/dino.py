"""DINO + Gibbs tracking core (shared by custom CLI and DAVIS experiments)."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import config as _genmatter_config

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import key as jkey

import genjax
from genjax import Const, gen, Pytree

from genmatter.datatypes import *
from genmatter.dataloader import extract_3d_points_and_motion_vectors_data
from genmatter.inference import *
from genmatter.model_3d import *
from genmatter.utils import *

_JAX_CACHE_CONFIGURED = False


@dataclass
class DinoCompiledProgram:
    """Compile-once handle: fixed max frame count for padded temporal scans."""

    max_frames: int
    warmed: bool = False

    @property
    def max_tracking_steps(self) -> int:
        return self.max_frames - 1


_DEFAULT_COMPILED_PROGRAM: DinoCompiledProgram | None = None


def configure_jax_cache(cache_dir: str | None = None) -> Path:
    """Enable persistent JAX compilation cache (idempotent)."""
    global _JAX_CACHE_CONFIGURED
    root = Path(cache_dir or os.path.join(os.getcwd(), ".jax_cache"))
    root.mkdir(parents=True, exist_ok=True)
    if not _JAX_CACHE_CONFIGURED:
        jax.config.update("jax_compilation_cache_dir", str(root))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.experimental.compilation_cache.compilation_cache.set_cache_dir(str(root))
        _JAX_CACHE_CONFIGURED = True
    return root


@dataclass
class DinoTrackingHyperparams:
    sigma_F: float = 0.2
    outlier_prob: float = 5.0
    outlier_velocity_gamma_shape: float = 5.0
    outlier_velocity_gamma_rate: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    sigma_H: float = 25.0
    sigma_V: float = 1e14
    translation_gaussian_scale: float = 0.2
    translation_max_radius: float = 0.35
    translation_num_radii_cells: int = 15
    translation_theta_step_deg: float = 15.0
    rotation_vmf_kappa: float = 100.0
    rotation_angle_max_deg: float = 25.0
    rotation_angle_step_deg: float = 0.375


@dataclass
class DinoTrackingParams:
    num_blobs: int = 500
    num_hyperblobs: int = 9
    datapoint_retain_pct: float = 78.125
    random_seed: int = 42
    focal_length: float = 520.0
    use_sam_frame0: bool = True
    init_gibbs_sweeps: int = 15
    tracking_outlier_prob: float = 1e-28
    dense_disable_outlier_prob: bool = False
    measure_fps: bool = True
    hyperparams: DinoTrackingHyperparams = field(default_factory=DinoTrackingHyperparams)


@dataclass
class DinoTrackingInputs:
    video_id: str
    motion_npz: Path
    dino_npz: Path
    sam_frame0_png: Path | None = None
    segmentation_mask_frame0: np.ndarray | None = None
    max_frames: int | None = None


@dataclass
class DinoTrackingTimings:
    load_seconds: float = 0.0
    kmeans_init_seconds: float = 0.0
    importance_seconds: float = 0.0
    jit_init_gibbs_seconds: float = 0.0
    init_gibbs_seconds: float = 0.0
    init_gibbs_fps: float = 0.0
    jit_tracking_seconds: float = 0.0
    tracking_seconds: float = 0.0
    tracking_fps: float = 0.0
    jit_dense_seconds: float = 0.0
    dense_seconds: float = 0.0
    dense_fps: float = 0.0
    num_frames: int = 0
    num_tracking_steps: int = 0


@dataclass
class DinoTrackingResult:
    video_id: str
    tracking_data: list[dict[str, Any]]
    img_dims: tuple[int, int]
    subsampled_indices: np.ndarray
    focal_length: float
    timings: DinoTrackingTimings
    gaussian_means: np.ndarray
    gaussian_stds: np.ndarray


def dino_params_from_config(tracking_cfg) -> DinoTrackingParams:
    """Build ``DinoTrackingParams`` from ``CustomConfig.tracking``."""
    hp = tracking_cfg.hyperparams
    return DinoTrackingParams(
        num_blobs=tracking_cfg.num_blobs,
        num_hyperblobs=tracking_cfg.num_hyperblobs,
        datapoint_retain_pct=tracking_cfg.datapoint_retain_pct,
        random_seed=tracking_cfg.random_seed,
        focal_length=tracking_cfg.focal_length,
        use_sam_frame0=tracking_cfg.use_sam_frame0,
        init_gibbs_sweeps=tracking_cfg.init_gibbs_sweeps,
        tracking_outlier_prob=tracking_cfg.tracking_outlier_prob,
        dense_disable_outlier_prob=getattr(
            tracking_cfg, "dense_disable_outlier_prob", False
        ),
        measure_fps=getattr(tracking_cfg, "measure_fps", True),
        hyperparams=DinoTrackingHyperparams(
            sigma_F=hp.sigma_F,
            outlier_prob=hp.outlier_prob,
            outlier_velocity_gamma_shape=hp.outlier_velocity_gamma_shape,
            outlier_velocity_gamma_rate=hp.outlier_velocity_gamma_rate,
            alpha=hp.alpha,
            beta=hp.beta,
            sigma_H=hp.sigma_H,
            sigma_V=hp.sigma_V,
            translation_gaussian_scale=hp.translation_gaussian_scale,
            translation_max_radius=hp.translation_max_radius,
            translation_num_radii_cells=hp.translation_num_radii_cells,
            translation_theta_step_deg=hp.translation_theta_step_deg,
            rotation_vmf_kappa=hp.rotation_vmf_kappa,
            rotation_angle_max_deg=hp.rotation_angle_max_deg,
            rotation_angle_step_deg=hp.rotation_angle_step_deg,
        ),
    )


# ============================================================================
# ============================================================================

@Pytree.dataclass
class GenMatter_Hyperparams_DINO(Super_Pytree):
    outlier_prob: jnp.float32 = Super_Pytree.field()
    outlier_velocity_gamma_shape: jnp.float32 = Super_Pytree.field()
    outlier_velocity_gamma_rate: jnp.float32 = Super_Pytree.field()
    alpha: jnp.float32 = Super_Pytree.field()
    beta: jnp.float32 = Super_Pytree.field()
    mu_H: jnp.ndarray = Super_Pytree.field()
    sigma_H: jnp.float32 = Super_Pytree.field()
    nu_H: jnp.float32 = Super_Pytree.field()
    Psi_H: jnp.ndarray = Super_Pytree.field()
    nu_B: jnp.float32 = Super_Pytree.field()
    Psi_B: jnp.ndarray = Super_Pytree.field()
    sigma_V: jnp.float32 = Super_Pytree.field()
    nu_V: jnp.float32 = Super_Pytree.field()
    Psi_V: jnp.ndarray = Super_Pytree.field()
    mu_F: jnp.ndarray = Super_Pytree.field()
    sigma_F_prior: jnp.ndarray = Super_Pytree.field()
    sigma_F: jnp.float32 = Super_Pytree.field()
    translation_max_radius: StaticJnp = Super_Pytree.static()
    translation_num_radii_cells: StaticJnp = Super_Pytree.static()
    translation_theta_step_deg: StaticJnp = Super_Pytree.static()
    translation_gaussian_scale: StaticJnp = Super_Pytree.static()
    rotation_vmf_kappa: StaticJnp = Super_Pytree.static()
    rotation_angle_max_deg: StaticJnp = Super_Pytree.static()
    rotation_angle_step_deg: StaticJnp = Super_Pytree.static()
    n_hyperblobs: jnp.int32 = Super_Pytree.static()
    n_blobs: jnp.int32 = Super_Pytree.static()
    n_datapoints: jnp.int32 = Super_Pytree.static()
    discrete_translation: Precomputed_DiscreteDistribution = Super_Pytree.static()
    discrete_rotation: Precomputed_DiscreteDistribution = Super_Pytree.static()

    def __eq__(self, other):
        if jax.tree_util.tree_structure(self) != jax.tree_util.tree_structure(other):
            return False
        leaves1 = jax.tree_util.tree_leaves(self)
        leaves2 = jax.tree_util.tree_leaves(other)
        bools = [jnp.all(l1 == l2) for l1, l2 in zip(leaves1, leaves2)]
        return jnp.all(jnp.array(bools))

    @classmethod
    def create(cls, **kwargs):
        return GenMatter_Hyperparams.create.__func__(cls, **kwargs)

@Pytree.dataclass
class GenMatter_Blobs_State_DINO(Super_Pytree):
    hyperblob_assignments: jnp.ndarray
    blob_weights: jnp.ndarray
    blob_means: jnp.ndarray
    blob_covs: jnp.ndarray
    blob_vel_means: jnp.ndarray
    blob_vel_covs: jnp.ndarray
    blob_features: jnp.ndarray

@Pytree.dataclass
class GenMatter_Datapoints_State_DINO(Super_Pytree):
    blob_assignments: jnp.ndarray
    datapoint_positions: jnp.ndarray
    datapoint_vels: jnp.ndarray
    datapoint_features: jnp.ndarray

@Pytree.dataclass
class GenMatter_State_DINO(Super_Pytree):
    hypers: GenMatter_Hyperparams_DINO
    hyperblobs_state: GenMatter_Hyperblobs_State
    blobs_state: GenMatter_Blobs_State_DINO
    datapoints_state: GenMatter_Datapoints_State_DINO

@gen
def GenMatter_model_dino(hypers: GenMatter_Hyperparams_DINO):
    hyperblobs_state = GenMatter_hyperblobs_model(hypers) @ 'hyperblobs'
    blobs_state = GenMatter_blobs_model_dino(hypers, hyperblobs_state) @ 'blobs'
    datapoints_state = GenMatter_datapoints_model_dino(hypers, blobs_state) @ 'datapoints'
    return GenMatter_State_DINO(
        hypers=hypers,
        hyperblobs_state=hyperblobs_state,
        blobs_state=blobs_state,
        datapoints_state=datapoints_state
    )

@gen
def GenMatter_blobs_model_dino(hypers: GenMatter_Hyperparams_DINO, hyperblobs_state: GenMatter_Hyperblobs_State):
    sample_shape = Const((hypers.n_blobs,))
    hyperblob_assignments = genjax.categorical(
        probs=hyperblobs_state.hyperblob_weights,
        sample_shape=sample_shape
    ) @ 'hyperblob_assignments'
    blob_weights = genjax.dirichlet(jnp.repeat(hypers.beta, hypers.n_blobs)) @ 'blob_weights'
    assigned_hyperblob_per_blob = hyperblobs_state[hyperblob_assignments]
    blob_covs = inverse_wishart(hypers.nu_B, hypers.Psi_B, sample_shape=sample_shape) @ 'blob_covs'
    blob_means = genjax.mv_normal(
        assigned_hyperblob_per_blob.hyperblob_means,
        assigned_hyperblob_per_blob.hyperblob_covs
    ) @ 'blob_means'
    blob_vel_means_ = assigned_hyperblob_per_blob.hyperblob_trans_vels + jnp.einsum(
        'nij,nj->ni',
        assigned_hyperblob_per_blob.hyperblob_rot_vels - jnp.repeat(jnp.eye(3)[None, ...], hypers.n_blobs, axis=0),
        blob_means - assigned_hyperblob_per_blob.hyperblob_means
    )
    blob_vel_means = genjax.normal(blob_vel_means_, jnp.sqrt(hypers.sigma_V)) @ 'blob_vel_means'
    blob_vel_covs = inverse_wishart(hypers.nu_V, hypers.Psi_V, sample_shape=sample_shape) @ 'blob_vel_covs'
    blob_features = genjax.normal(hypers.mu_F, jnp.sqrt(hypers.sigma_F_prior**2)) @ 'blob_features'
    return GenMatter_Blobs_State_DINO(
        hyperblob_assignments=hyperblob_assignments,
        blob_weights=blob_weights,
        blob_means=blob_means,
        blob_covs=blob_covs,
        blob_vel_means=blob_vel_means,
        blob_vel_covs=blob_vel_covs,
        blob_features=blob_features,
    )

@gen
def GenMatter_datapoints_model_dino(hypers: GenMatter_Hyperparams_DINO, blobs_state: GenMatter_Blobs_State_DINO):
    blob_assignments = genjax.categorical(
        probs=blobs_state.blob_weights,
        sample_shape=Const((hypers.n_datapoints,))
    ) @ 'blob_assignments'
    assigned_blob_per_datapoint = blobs_state[blob_assignments]
    datapoint_positions = genjax.mv_normal(
        assigned_blob_per_datapoint.blob_means,
        assigned_blob_per_datapoint.blob_covs
    ) @ 'datapoint_positions'
    datapoint_vels = genjax.mv_normal(
        assigned_blob_per_datapoint.blob_vel_means,
        assigned_blob_per_datapoint.blob_vel_covs
    ) @ 'datapoint_vels'
    datapoint_features = genjax.normal(
        assigned_blob_per_datapoint.blob_features,
        jnp.sqrt(hypers.sigma_F)
    ) @ 'datapoint_features'
    return GenMatter_Datapoints_State_DINO(
        blob_assignments=blob_assignments,
        datapoint_positions=datapoint_positions,
        datapoint_vels=datapoint_vels,
        datapoint_features=datapoint_features,
    )

@gen
def blob_datapoint_likelihood_model_dino(blob_state: GenMatter_Blobs_State_DINO, sigma_F: jnp.float32):
    datapoint_position = genjax.mv_normal(blob_state.blob_means, blob_state.blob_covs) @ 'datapoint_position'
    datapoint_vel = genjax.mv_normal(blob_state.blob_vel_means, blob_state.blob_vel_covs) @ 'datapoint_vel'
    datapoint_feature = genjax.normal(blob_state.blob_features, jnp.sqrt(sigma_F)) @ 'datapoint_feature'
    return None

# JIT compile
model_jsimulate = jax.jit(GenMatter_model_dino.simulate)
model_jimportance = jax.jit(GenMatter_model_dino.importance)

# ============================================================================
# ============================================================================

def extract_dino_features(stimulus, pca_features, img_dims, num_timesteps):
    """Extract DINO features reshaped to match tracking data."""
    T, h, w, num_features = pca_features.shape
    target_h, target_w = img_dims

    if (h, w) != (target_h, target_w):
        tracked_features = np.zeros((num_timesteps, target_h, target_w, num_features), dtype=pca_features.dtype)
        for t in range(num_timesteps):
            for c in range(num_features):
                tracked_features[t, :, :, c] = cv2.resize(
                    pca_features[t, :, :, c],
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR
                )
    else:
        tracked_features = pca_features[:num_timesteps]

    tracked_features = tracked_features.reshape(num_timesteps, -1, num_features)
    return tracked_features


def sample_datapoints_percentage(
    tracked_points, tracked_motion_vectors, percentage, seed=None, same_indices_all_timesteps=True
):
    """Randomly keep *percentage* of datapoints (same indices at all timesteps if requested)."""
    if seed is not None:
        np.random.seed(seed)

    T, num_points = tracked_points.shape[:2]
    num_points_to_keep = max(1, int(num_points * percentage / 100.0))

    if same_indices_all_timesteps:
        sampled_indices = np.random.choice(num_points, num_points_to_keep, replace=False)
        sampled_indices = np.sort(sampled_indices)
        sampled_tracked_points = tracked_points[:, sampled_indices, ...]
        sampled_tracked_motion_vectors = tracked_motion_vectors[:, sampled_indices, ...]
        return sampled_tracked_points, sampled_tracked_motion_vectors, sampled_indices
    sampled_indices_batched = np.zeros((T, num_points_to_keep), dtype=int)
    sampled_tracked_points = np.zeros((T, num_points_to_keep) + tracked_points.shape[2:], dtype=tracked_points.dtype)
    sampled_tracked_motion_vectors = np.zeros(
        (T, num_points_to_keep) + tracked_motion_vectors.shape[2:], dtype=tracked_motion_vectors.dtype
    )
    for t in range(T):
        idx = np.random.choice(num_points, num_points_to_keep, replace=False)
        idx = np.sort(idx)
        sampled_indices_batched[t] = idx
        sampled_tracked_points[t] = tracked_points[t, idx, ...]
        sampled_tracked_motion_vectors[t] = tracked_motion_vectors[t, idx, ...]
    return sampled_tracked_points, sampled_tracked_motion_vectors, sampled_indices_batched


def initialize_model_with_dino(
    tracked_points,
    num_blobs,
    num_hyperblobs,
    segmentation_mask,
    motion_vectors,
    tracked_features,
    img_dims,
    *,
    use_sam_frame0: bool,
    sam_frame0_path: Path | None,
    subsampled_indices=None,
):
    """Initialize model with DINO features."""
    from genjax import ChoiceMapBuilder as C

    if use_sam_frame0:
        if sam_frame0_path is None or not sam_frame0_path.is_file():
            raise FileNotFoundError(f"SAM frame-0 mask not found: {sam_frame0_path}")
        segmentation_mask = cv2.imread(str(sam_frame0_path), cv2.IMREAD_COLOR)
        segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2RGB)
        kmeans_chm, roi_blob_indices, roi_hyperblob_indices, num_hyperblobs_actual = make_hierarchical_kmeans_chm_with_SAM_segmentations(
            tracked_points,
            num_blobs,
            segmentation_mask,
            img_dims,
            motion_vectors=motion_vectors,
            subsampled_indices=subsampled_indices,
        )
    else:
        kmeans_chm, roi_blob_indices, roi_hyperblob_indices = make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(
            tracked_points,
            num_blobs,
            num_hyperblobs,
            segmentation_mask=segmentation_mask,
            motion_vectors=motion_vectors,
            num_roi_blobs=None,
            subsampled_indices=subsampled_indices,
        )
        num_hyperblobs_actual = num_hyperblobs

    frame_features = tracked_features[0]
    blob_assignments = np.array(kmeans_chm['datapoints', 'blob_assignments'])
    num_blobs_actual = len(np.unique(blob_assignments))
    num_features = frame_features.shape[1]
    blob_features = np.zeros((num_blobs_actual, num_features), dtype=np.float32)

    for i in range(num_blobs_actual):
        points_in_blob = np.where(blob_assignments == i)[0]
        if len(points_in_blob) > 0:
            blob_features[i] = np.mean(frame_features[points_in_blob], axis=0)

    kmeans_chm = kmeans_chm | C['datapoints', 'datapoint_features'].set(a_(frame_features))
    kmeans_chm = kmeans_chm | C['blobs', 'blob_features'].set(a_(blob_features))

    return kmeans_chm, roi_blob_indices, roi_hyperblob_indices, num_hyperblobs_actual

# ============================================================================
# ============================================================================

def gibbs_blob_features_dino(key, genmatter_state):
    """Gibbs update for blob DINO features."""
    posterior_key, _ = jax.random.split(key)
    datapoint_features = genmatter_state.datapoints_state.datapoint_features
    blob_assignments = genmatter_state.datapoints_state.blob_assignments
    mu_F = genmatter_state.hypers.mu_F
    sigma_F_prior = genmatter_state.hypers.sigma_F_prior
    sigma_F = genmatter_state.hypers.sigma_F
    L = genmatter_state.hypers.n_blobs

    N_l = jax.ops.segment_sum(jnp.ones(datapoint_features.shape[0]), blob_assignments, num_segments=L)
    S_l = stable_segment_sum(datapoint_features, blob_assignments, L)
    prior_precision = 1.0 / (sigma_F_prior ** 2)
    likelihood_precision = 1.0 / sigma_F
    posterior_precision = prior_precision[None, :] + N_l[:, None] * likelihood_precision
    posterior_mean_numerator = (mu_F[None, :] * prior_precision[None, :] + S_l * likelihood_precision)
    posterior_mean = posterior_mean_numerator / posterior_precision
    posterior_std = 1.0 / jnp.sqrt(posterior_precision)
    has_points = N_l > 0
    posterior_mean = jnp.where(has_points[:, None], posterior_mean, mu_F[None, :])
    posterior_std = jnp.where(has_points[:, None], posterior_std, sigma_F_prior[None, :])
    new_blob_features = genjax.normal.sample(posterior_key, posterior_mean, posterior_std)
    return genmatter_state.replace({'blobs_state': {'blob_features': new_blob_features}})


@jax.jit
def dense_eval_blob_assignments(key, genmatter_state, dense_positions, dense_vels, dense_features,
                                disable_outlier_prob=False):
    """Dense evaluation of blob assignments on the full pixel grid (matches run_davis_subsampling)."""
    from genjax import ChoiceMapBuilder as C

    posterior_key, _ = jax.random.split(key)
    batch_size = 975

    hypers = genmatter_state.hypers
    num_blobs = hypers.n_blobs
    num_datapoints = dense_positions.shape[0]
    datapoint_positions = dense_positions
    datapoint_vels = dense_vels
    datapoint_features = dense_features
    blobs_state = genmatter_state.blobs_state

    blob_weights = blobs_state.blob_weights
    outlier_prob = jnp.where(disable_outlier_prob, 0.0, hypers.outlier_prob)
    extended_weights = jnp.concatenate([blob_weights, jnp.array([outlier_prob])])
    normalized_weights = extended_weights / jnp.sum(extended_weights)
    log_mixture_weights = jnp.log(normalized_weights)
    sigma_F = hypers.sigma_F

    def compute_local_density(point_idx):
        chm = (
            C["datapoint_position"].set(datapoint_positions[point_idx]) |
            C["datapoint_vel"].set(datapoint_vels[point_idx]) |
            C["datapoint_feature"].set(datapoint_features[point_idx])
        )
        log_liks = jax.vmap(
            lambda i: blob_datapoint_likelihood_model_dino.assess(chm, (blobs_state[i], sigma_F))[0]
        )(jnp.arange(num_blobs))
        v = datapoint_vels[point_idx]
        speed = jnp.linalg.norm(v)
        alpha = hypers.outlier_velocity_gamma_shape
        beta = hypers.outlier_velocity_gamma_rate
        log_gamma_vel = (
            (alpha - 1) * jnp.log(speed + 1e-8)
            - beta * speed
            - alpha * jnp.log(1. / beta)
            - jax.lax.lgamma(alpha)
        )
        log_gamma_vel = jnp.where(disable_outlier_prob, 0.0, log_gamma_vel)
        raw_log_liks = jnp.concatenate([log_liks, jnp.array([log_gamma_vel])])
        return raw_log_liks + log_mixture_weights

    def batch_compute_logprobs(carry, batch_idx):
        batch_start = batch_idx * batch_size
        batch_indices = jnp.arange(batch_size) + batch_start
        batch_logprobs = jax.vmap(compute_local_density)(batch_indices)
        return carry, batch_logprobs

    num_full_batches = num_datapoints // batch_size
    _, batched_logprobs = jax.lax.scan(
        batch_compute_logprobs,
        None,
        jnp.arange(num_full_batches)
    )

    all_logprobs = batched_logprobs.reshape(num_full_batches * batch_size, -1)
    dense_eval_assignments = genjax.categorical.sample(posterior_key, logits=all_logprobs)

    return dense_eval_assignments


@jax.jit
def dense_eval_blob_weights(key, genmatter_state: GenMatter_State, dense_assignments: jnp.ndarray):
    posterior_key, _ = jax.random.split(key)

    num_blobs = genmatter_state.hypers.n_blobs
    prior_beta = genmatter_state.hypers.beta
    blob_idxs = dense_assignments

    blob_counts = jax.ops.segment_sum(
        jnp.ones_like(blob_idxs),
        blob_idxs,
        num_segments=num_blobs
    )

    new_betas = prior_beta + blob_counts
    new_blob_weights = genjax.dirichlet.sample(posterior_key, new_betas)

    return new_blob_weights


def gibbs_blob_assignments_dino(key, genmatter_state, position_only=False,
                                velocity_only=False, disable_outlier_prob=False, feature_only=False):
    """Gibbs update for blob assignments with DINO feature likelihood."""
    from genjax import ChoiceMapBuilder as C

    posterior_key, _ = jax.random.split(key)
    batch_size = 975

    hypers = genmatter_state.hypers
    num_blobs = hypers.n_blobs
    num_datapoints = hypers.n_datapoints
    datapoint_positions = genmatter_state.datapoints_state.datapoint_positions
    datapoint_vels = genmatter_state.datapoints_state.datapoint_vels
    datapoint_features = genmatter_state.datapoints_state.datapoint_features
    blobs_state = genmatter_state.blobs_state

    gibbs_blob_vel_covs = jnp.where(
        jnp.logical_or(position_only, feature_only),
        1e14 * blobs_state.blob_vel_covs,
        blobs_state.blob_vel_covs
    )
    gibbs_blob_covs = jnp.where(
        jnp.logical_or(velocity_only, feature_only),
        1e14 * blobs_state.blob_covs,
        blobs_state.blob_covs
    )
    blobs_state = blobs_state.replace({
        'blob_vel_covs': gibbs_blob_vel_covs,
        'blob_covs': gibbs_blob_covs
    })

    blob_weights = blobs_state.blob_weights
    outlier_prob = jnp.where(disable_outlier_prob, 0.0, hypers.outlier_prob)
    extended_weights = jnp.concatenate([blob_weights, jnp.array([outlier_prob])])
    normalized_weights = extended_weights / jnp.sum(extended_weights)
    log_mixture_weights = jnp.log(normalized_weights)
    sigma_F = hypers.sigma_F

    def compute_local_density(point_idx):
        chm = (
            C["datapoint_position"].set(datapoint_positions[point_idx]) |
            C["datapoint_vel"].set(datapoint_vels[point_idx]) |
            C["datapoint_feature"].set(datapoint_features[point_idx])
        )
        log_liks = jax.vmap(
            lambda i: blob_datapoint_likelihood_model_dino.assess(chm, (blobs_state[i], sigma_F))[0]
        )(jnp.arange(num_blobs))
        v = datapoint_vels[point_idx]
        speed = jnp.linalg.norm(v)
        alpha = hypers.outlier_velocity_gamma_shape
        beta = hypers.outlier_velocity_gamma_rate
        log_gamma_vel = (
            (alpha - 1) * jnp.log(speed + 1e-8)
            - beta * speed
            - alpha * jnp.log(1. / beta)
            - jax.lax.lgamma(alpha)
        )
        log_gamma_vel = jnp.where(disable_outlier_prob, 0.0, log_gamma_vel)
        raw_log_liks = jnp.concatenate([log_liks, jnp.array([log_gamma_vel])])
        return raw_log_liks + log_mixture_weights

    def batch_compute_logprobs(carry, batch_idx):
        batch_start = batch_idx * batch_size
        batch_indices = jnp.arange(batch_size) + batch_start
        batch_logprobs = jax.vmap(compute_local_density)(batch_indices)
        return carry, batch_logprobs

    num_full_batches = num_datapoints // batch_size
    _, batched_logprobs = jax.lax.scan(
        batch_compute_logprobs,
        None,
        jnp.arange(num_full_batches)
    )

    all_logprobs = batched_logprobs.reshape(num_full_batches * batch_size, -1)
    updated_assignments = genjax.categorical.sample(posterior_key, logits=all_logprobs)

    return genmatter_state.replace({
        'datapoints_state': {'blob_assignments': updated_assignments}
    })

def blob_tracking_gibbs_dino(key, genmatter_state):
    """Single-frame Gibbs updates."""
    def update_blob_assignments_position_only(i, carry):
        key, genmatter_state = carry
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_assignments_dino(
            gibbs_key, genmatter_state, position_only=True, disable_outlier_prob=True
        )
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_weights(gibbs_key, genmatter_state)
        return key, genmatter_state

    def update_blob_assignments_with_outlier(i, carry):
        key, genmatter_state = carry
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_assignments_dino(
            gibbs_key, genmatter_state, position_only=True, disable_outlier_prob=False
        )
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_weights(gibbs_key, genmatter_state)
        return key, genmatter_state

    def update_blob_assignments_feature_only(i, carry):
        key, genmatter_state = carry
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_assignments_dino(
            gibbs_key, genmatter_state, feature_only=True, disable_outlier_prob=False
        )
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_weights(gibbs_key, genmatter_state)
        return key, genmatter_state

    def update_blob_velocities(i, carry):
        key, genmatter_state = carry
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_vel_means(gibbs_key, genmatter_state)
        return key, genmatter_state

    def update_blob_velocity_covariances(i, carry):
        key, genmatter_state = carry
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_vel_covs(gibbs_key, genmatter_state)
        return key, genmatter_state

    def update_blob_means(i, carry):
        key, genmatter_state = carry
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_means(gibbs_key, genmatter_state)
        return key, genmatter_state

    def update_blob_features_dino(i, carry):
        key, genmatter_state = carry
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_features_dino(gibbs_key, genmatter_state)
        return key, genmatter_state

    def hyperblob_update_loop(i, carry):
        key, genmatter_state = carry
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_hyperblob_means(gibbs_key, genmatter_state)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_hyperblob_covs(gibbs_key, genmatter_state)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_hyperblob_rot(gibbs_key, genmatter_state)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_hyperblob_trans(gibbs_key, genmatter_state)
        return key, genmatter_state

    key, genmatter_state = jax.lax.fori_loop(0, 3, hyperblob_update_loop, (key, genmatter_state))
    # # added this to see if it helps below
    # key, genmatter_state = jax.lax.fori_loop(0, 3, update_blob_assignments_feature_only, (key, genmatter_state))
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Keep this schedule identical to run_davis_subsampling so 1/128 results match exactly.
    key, genmatter_state = jax.lax.fori_loop(0, 1, update_blob_assignments_position_only, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 15, update_blob_means, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 3, update_blob_features_dino, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 1, update_blob_assignments_with_outlier, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 15, update_blob_velocities, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 15, update_blob_velocity_covariances, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 3, update_blob_features_dino, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 3, hyperblob_update_loop, (key, genmatter_state))

    return genmatter_state


def _pad_temporal_stack(arr: np.ndarray, num_real: int, max_frames: int) -> np.ndarray:
    """Pad time axis (axis 0) by repeating the last real frame."""
    if num_real >= max_frames:
        return arr[:max_frames]
    pad = max_frames - num_real
    last = arr[num_real - 1]
    padding = np.repeat(last[np.newaxis, ...], pad, axis=0)
    return np.concatenate([arr[:num_real], padding], axis=0)


def _to_jax_temporal_stack(arr: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(arr)


def _init_gibbs_scan_step(carry, _):
    key, genmatter_state = carry
    key, gibbs_key = jax.random.split(key)
    genmatter_state = gibbs_blob_assignments_dino(
        gibbs_key, genmatter_state, position_only=True, disable_outlier_prob=True
    )
    key, gibbs_key = jax.random.split(key)
    genmatter_state = gibbs_blob_weights(gibbs_key, genmatter_state)
    key, gibbs_key = jax.random.split(key)
    genmatter_state = gibbs_blob_means(gibbs_key, genmatter_state)
    key, gibbs_key = jax.random.split(key)
    genmatter_state = gibbs_blob_covs(gibbs_key, genmatter_state)
    key, gibbs_key = jax.random.split(key)
    genmatter_state = gibbs_blob_vel_means(gibbs_key, genmatter_state)
    key, gibbs_key = jax.random.split(key)
    genmatter_state = gibbs_blob_vel_covs(gibbs_key, genmatter_state)
    key, gibbs_key = jax.random.split(key)
    genmatter_state = gibbs_blob_features_dino(gibbs_key, genmatter_state)
    return (key, genmatter_state), genmatter_TraceWrapper(force_retval=genmatter_state)


def init_gibbs_sweep_dino(key, genmatter_state, num_sweeps=30):
    """Custom Gibbs sweeps for DINO initialization (eager scan; matches pre-refactor numerics)."""
    (_, _), traces = jax.lax.scan(
        _init_gibbs_scan_step,
        (key, genmatter_state),
        jnp.arange(num_sweeps),
    )
    return GenMatter_Gibbs_TraceWrapper(
        genmatter_TraceWrapper(force_retval=genmatter_state),
        traces,
    )


@jax.jit
def f_tracking_sweep_dino(carry, timestep_idx):
    """One temporal tracking step; inactive steps are no-ops (no RNG use)."""
    key, genmatter_state, tracked_points, tracked_motion_vectors, tracked_features, num_real = carry
    is_active = timestep_idx < num_real

    def active_step():
        next_blob_means = (
            genmatter_state.blobs_state.blob_vel_means + genmatter_state.blobs_state.blob_means
        )
        state = genmatter_state.replace({"blobs_state": {"blob_means": next_blob_means}})
        state = state.replace(
            {
                "datapoints_state": {
                    "datapoint_positions": tracked_points[timestep_idx],
                    "datapoint_vels": tracked_motion_vectors[timestep_idx],
                    "datapoint_features": tracked_features[timestep_idx],
                }
            }
        )
        key_new, gibbs_key = jax.random.split(key)
        state = blob_tracking_gibbs_dino(gibbs_key, state)
        new_carry = (
            key_new,
            state,
            tracked_points,
            tracked_motion_vectors,
            tracked_features,
            num_real,
        )
        return new_carry, genmatter_TraceWrapper(force_retval=state)

    def inactive_step():
        new_carry = (
            key,
            genmatter_state,
            tracked_points,
            tracked_motion_vectors,
            tracked_features,
            num_real,
        )
        return new_carry, genmatter_TraceWrapper(force_retval=genmatter_state)

    return jax.lax.cond(is_active, active_step, inactive_step)


def genmatter_tracking_gibbs_dino(
    key,
    init_genmatter_state,
    tracked_points,
    tracked_motion_vectors,
    tracked_features,
    outlier_prob,
    *,
    num_real_frames: int,
    max_frames: int,
):
    """Track over time with DINO features; padded scan length ``max_frames - 1``."""
    init_genmatter_state = init_genmatter_state.replace(
        {"hypers": {"outlier_prob": f_(outlier_prob)}}
    )
    num_real = jnp.int32(num_real_frames)
    timestep_indices = jnp.arange(1, max_frames)
    _, stacked_genmatter_wtrs = jax.lax.scan(
        f_tracking_sweep_dino,
        (
            key,
            init_genmatter_state,
            tracked_points,
            tracked_motion_vectors,
            tracked_features,
            num_real,
        ),
        timestep_indices,
        unroll=1,
    )
    return GenMatter_Gibbs_TraceWrapper(
        genmatter_TraceWrapper(force_retval=init_genmatter_state),
        stacked_genmatter_wtrs,
    )


def genmatter_tracking_gibbs_dino_legacy(
    key,
    init_genmatter_state,
    tracked_points,
    tracked_motion_vectors,
    tracked_features,
    outlier_prob,
):
    """Unpadded tracking (reference for parity tests only)."""
    init_genmatter_state = init_genmatter_state.replace(
        {"hypers": {"outlier_prob": f_(outlier_prob)}}
    )
    timestep_indices = jnp.arange(1, len(tracked_points))
    _, stacked_genmatter_wtrs = jax.lax.scan(
        f_tracking_sweep_dino,
        (
            key,
            init_genmatter_state,
            tracked_points,
            tracked_motion_vectors,
            tracked_features,
            jnp.int32(len(tracked_points)),
        ),
        timestep_indices,
        unroll=1,
    )
    return GenMatter_Gibbs_TraceWrapper(
        genmatter_TraceWrapper(force_retval=init_genmatter_state),
        stacked_genmatter_wtrs,
    )


def compile_dino_tracking_program(
    max_frames: int | None = None,
    *,
    cache_dir: str | None = None,
) -> DinoCompiledProgram:
    """Build compile-once program handle (fixed ``max_frames`` for padded scans)."""
    configure_jax_cache(cache_dir)
    if max_frames is not None:
        mf = max_frames
    else:
        # StreamingVision patch: fall back to env var if DAVIS NPZs aren't present,
        # so custom-video bayesopt runs don't require a DAVIS install.
        try:
            mf = _genmatter_config.tapvid_davis_max_frames()
        except FileNotFoundError:
            import os
            env_mf = os.environ.get("GENMATTER_DEFAULT_MAX_FRAMES")
            if env_mf is None:
                raise
            mf = int(env_mf)
    return DinoCompiledProgram(max_frames=mf)


def get_default_dino_compiled_program() -> DinoCompiledProgram:
    global _DEFAULT_COMPILED_PROGRAM
    if _DEFAULT_COMPILED_PROGRAM is None:
        _DEFAULT_COMPILED_PROGRAM = compile_dino_tracking_program()
    return _DEFAULT_COMPILED_PROGRAM


def _prepare_padded_tracking_arrays(
    tracked_points: np.ndarray,
    tracked_motion_vectors: np.ndarray,
    tracked_features: np.ndarray,
    *,
    num_real_frames: int,
    max_frames: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    pts = _to_jax_temporal_stack(
        _pad_temporal_stack(tracked_points, num_real_frames, max_frames)
    )
    mvs = _to_jax_temporal_stack(
        _pad_temporal_stack(tracked_motion_vectors, num_real_frames, max_frames)
    )
    feats = _to_jax_temporal_stack(
        _pad_temporal_stack(tracked_features, num_real_frames, max_frames)
    )
    return pts, mvs, feats


def _frame_data_from_trace(
    frame,
    tracked_points_full,
    tracked_motion_vectors_full,
    tracked_features_full,
    frame_idx: int,
    num_datapoints_full: int,
) -> dict[str, Any]:
    return {
        "n_blobs": int(frame.retval.hypers.n_blobs),
        "n_hyperblobs": int(frame.retval.hypers.n_hyperblobs),
        "n_datapoints": num_datapoints_full,
        "blob_assignments": None,  # filled by dense eval caller
        "datapoint_positions": np.array(tracked_points_full[frame_idx]),
        "datapoint_vels": np.array(tracked_motion_vectors_full[frame_idx]),
        "datapoint_features": np.array(tracked_features_full[frame_idx]),
        "blob_weights": None,
        "blob_means": np.array(frame.retval.blobs_state.blob_means),
        "blob_covs": np.array(frame.retval.blobs_state.blob_covs),
        "blob_vel_means": np.array(frame.retval.blobs_state.blob_vel_means),
        "blob_vel_covs": np.array(frame.retval.blobs_state.blob_vel_covs),
        "blob_features": np.array(frame.retval.blobs_state.blob_features),
        "hyperblob_assignments": np.array(frame.retval.blobs_state.hyperblob_assignments),
        "hyperblob_weights": np.array(frame.retval.hyperblobs_state.hyperblob_weights),
        "hyperblob_means": np.array(frame.retval.hyperblobs_state.hyperblob_means),
        "hyperblob_trans_vels": np.array(frame.retval.hyperblobs_state.hyperblob_trans_vels),
        "hyperblob_rot_vels": np.array(frame.retval.hyperblobs_state.hyperblob_rot_vels),
    }


def _build_hypers_from_kmeans(kmeans_chm, roi_blob_indices, roi_hyperblob_indices, num_hyperblobs, num_blobs, num_datapoints, gaussian_means, gaussian_stds, hp: DinoTrackingHyperparams):
    empirical_mu_H = jnp.median(kmeans_chm["datapoints", "datapoint_positions"], axis=0)
    empirical_sigma_H = hp.sigma_H
    empirical_Psi_B = jnp.median(kmeans_chm["blobs", "blob_covs"][roi_blob_indices], axis=0)
    empirical_Psi_H = jnp.median(kmeans_chm["hyperblobs", "hyperblob_covs"][roi_hyperblob_indices], axis=0)
    empirical_Psi_V = jnp.median(kmeans_chm["blobs", "blob_vel_covs"][roi_blob_indices], axis=0)
    mean_blobs_per_roi_hyperblob = jnp.sum(
        jnp.isin(kmeans_chm["blobs", "hyperblob_assignments"], roi_hyperblob_indices)
    ) / len(roi_hyperblob_indices)
    empirical_nu_H = f_(int(mean_blobs_per_roi_hyperblob))
    mean_points_per_roi_blob = jnp.sum(
        jnp.isin(kmeans_chm["datapoints", "blob_assignments"], roi_blob_indices)
    ) / len(roi_blob_indices)
    empirical_nu_B = empirical_nu_V = f_(int(mean_points_per_roi_blob))

    return GenMatter_Hyperparams_DINO.create(
        mu_F=jnp.array(gaussian_means),
        sigma_F_prior=jnp.array(gaussian_stds),
        sigma_F=f_(hp.sigma_F),
        outlier_prob=f_(hp.outlier_prob),
        outlier_velocity_gamma_shape=f_(hp.outlier_velocity_gamma_shape),
        outlier_velocity_gamma_rate=f_(hp.outlier_velocity_gamma_rate),
        alpha=f_(hp.alpha),
        beta=f_(hp.beta),
        mu_H=empirical_mu_H,
        sigma_H=f_(empirical_sigma_H),
        nu_H=empirical_nu_H,
        Psi_H=empirical_Psi_H,
        nu_B=empirical_nu_B,
        Psi_B=empirical_Psi_B,
        sigma_V=f_(hp.sigma_V),
        nu_V=empirical_nu_V,
        Psi_V=empirical_Psi_V,
        translation_gaussian_scale=snp(f_(hp.translation_gaussian_scale)),
        translation_max_radius=snp(hp.translation_max_radius),
        translation_num_radii_cells=snp(hp.translation_num_radii_cells),
        translation_theta_step_deg=snp(hp.translation_theta_step_deg),
        rotation_vmf_kappa=snp(f_(hp.rotation_vmf_kappa)),
        rotation_angle_max_deg=snp(hp.rotation_angle_max_deg),
        rotation_angle_step_deg=snp(hp.rotation_angle_step_deg),
        n_hyperblobs=num_hyperblobs,
        n_blobs=num_blobs,
        n_datapoints=num_datapoints,
    )


def run_dino_tracking(
    inputs: DinoTrackingInputs,
    params: DinoTrackingParams,
    *,
    compiled: DinoCompiledProgram | None = None,
    on_step_start: Callable[[str], None] | None = None,
    on_dense_progress: Callable[[int, int], None] | None = None,
) -> DinoTrackingResult:
    """Run full DINO tracking pipeline; JIT compile time excluded from FPS."""
    program = compiled or get_default_dino_compiled_program()
    configure_jax_cache()
    timings = DinoTrackingTimings()
    needs_jit_warmup = not program.warmed
    hp = params.hyperparams

    def _step(name: str) -> None:
        if on_step_start is not None:
            on_step_start(name)

    t0 = time.perf_counter()
    _step("load")
    pca_data = np.load(inputs.dino_npz)
    pca_features_unnormalized = pca_data["pca_features_unnormalized"]
    gaussian_means = pca_data["gaussian_means"]
    gaussian_stds = pca_data["gaussian_stds"]
    pca_data.close()

    motion_dir = str(inputs.motion_npz.parent)
    tracked_points_full, tracked_motion_vectors_full, num_data_tsteps, img_dims = (
        extract_3d_points_and_motion_vectors_data(motion_dir, inputs.video_id)
    )
    if inputs.max_frames is not None:
        num_data_tsteps = min(num_data_tsteps, inputs.max_frames)
        tracked_points_full = tracked_points_full[:num_data_tsteps]
        tracked_motion_vectors_full = tracked_motion_vectors_full[:num_data_tsteps]

    tracked_features_full = extract_dino_features(
        inputs.video_id, pca_features_unnormalized, img_dims, num_data_tsteps
    )
    timings.load_seconds = time.perf_counter() - t0
    timings.num_frames = int(num_data_tsteps)

    seg_mask = inputs.segmentation_mask_frame0
    if seg_mask is None and not params.use_sam_frame0:
        raise ValueError("segmentation_mask_frame0 required when use_sam_frame0 is false")

    tracked_points, tracked_motion_vectors, subsampled_indices = sample_datapoints_percentage(
        tracked_points_full,
        tracked_motion_vectors_full,
        params.datapoint_retain_pct,
        seed=params.random_seed,
        same_indices_all_timesteps=True,
    )
    tracked_features = tracked_features_full[:, subsampled_indices, :]

    num_real_frames = int(tracked_points.shape[0])
    max_frames = program.max_frames
    if num_real_frames > max_frames:
        raise ValueError(
            f"Video {inputs.video_id} has {num_real_frames} frames > compiled max_frames={max_frames}"
        )
    tracked_points_jax, tracked_motion_vectors_jax, tracked_features_jax = (
        _prepare_padded_tracking_arrays(
            tracked_points,
            tracked_motion_vectors,
            tracked_features,
            num_real_frames=num_real_frames,
            max_frames=max_frames,
        )
    )

    _step("init")
    t0 = time.perf_counter()
    kmeans_chm, roi_blob_indices, roi_hyperblob_indices, num_hyperblobs = initialize_model_with_dino(
        tracked_points,
        params.num_blobs,
        params.num_hyperblobs,
        seg_mask,
        tracked_motion_vectors,
        tracked_features,
        img_dims,
        use_sam_frame0=params.use_sam_frame0,
        sam_frame0_path=inputs.sam_frame0_png,
        subsampled_indices=subsampled_indices,
    )
    timings.kmeans_init_seconds = time.perf_counter() - t0

    num_datapoints = kmeans_chm["datapoints", "datapoint_positions"].shape[0]
    num_blobs = kmeans_chm["blobs", "hyperblob_assignments"].shape[0]

    hypers = _build_hypers_from_kmeans(
        kmeans_chm,
        roi_blob_indices,
        roi_hyperblob_indices,
        num_hyperblobs,
        num_blobs,
        num_datapoints,
        gaussian_means,
        gaussian_stds,
        hp,
    )

    key = jkey(params.random_seed)
    key, key_importance = jax.random.split(key)
    t0 = time.perf_counter()
    init_tr, _ = model_jimportance(key_importance, kmeans_chm, (hypers,))
    init_genmatter_state = init_tr.get_retval()
    timings.importance_seconds = time.perf_counter() - t0

    key, init_gibbs_key = jax.random.split(key)
    _step("init_gibbs")
    if needs_jit_warmup and params.measure_fps:
        t_jit = time.perf_counter()
    t0 = time.perf_counter()
    gibbs_wtrs = init_gibbs_sweep_dino(
        init_gibbs_key, init_genmatter_state, num_sweeps=params.init_gibbs_sweeps
    )
    if needs_jit_warmup and params.measure_fps:
        timings.jit_init_gibbs_seconds = time.perf_counter() - t_jit
    timings.init_gibbs_seconds = time.perf_counter() - t0
    if params.init_gibbs_sweeps > 0 and timings.init_gibbs_seconds > 0:
        timings.init_gibbs_fps = params.init_gibbs_sweeps / timings.init_gibbs_seconds
    init_genmatter_state = gibbs_wtrs[-1].retval

    key, tracking_key = jax.random.split(key)
    outlier = params.tracking_outlier_prob
    num_tracking_steps = num_real_frames - 1
    timings.num_tracking_steps = num_tracking_steps

    tracking_kwargs = dict(
        num_real_frames=num_real_frames,
        max_frames=max_frames,
    )

    _step("tracking")
    if needs_jit_warmup and params.measure_fps:
        t_jit = time.perf_counter()
    t0 = time.perf_counter()
    tracking_wtrs = genmatter_tracking_gibbs_dino(
        tracking_key,
        init_genmatter_state,
        tracked_points_jax,
        tracked_motion_vectors_jax,
        tracked_features_jax,
        outlier,
        **tracking_kwargs,
    )
    if needs_jit_warmup and params.measure_fps:
        timings.jit_tracking_seconds = time.perf_counter() - t_jit
    timings.tracking_seconds = time.perf_counter() - t0
    if params.measure_fps and num_tracking_steps > 0 and timings.tracking_seconds > 0:
        timings.tracking_fps = num_tracking_steps / timings.tracking_seconds

    program.warmed = True

    num_datapoints_full = tracked_points_full.shape[1]
    tracking_data: list[dict[str, Any]] = []

    _step("dense")
    t_dense = time.perf_counter()
    num_frames = num_real_frames
    for frame_idx in range(num_frames):
        if on_dense_progress is not None:
            on_dense_progress(frame_idx + 1, num_frames)
        frame = tracking_wtrs[frame_idx]
        key, k_assign = jax.random.split(key)
        dense_assignments = dense_eval_blob_assignments(
            key=k_assign,
            genmatter_state=frame.retval,
            dense_positions=tracked_points_full[frame_idx],
            dense_vels=tracked_motion_vectors_full[frame_idx],
            dense_features=tracked_features_full[frame_idx],
            disable_outlier_prob=params.dense_disable_outlier_prob,
        )
        key, k_w = jax.random.split(key)
        dense_weights = dense_eval_blob_weights(
            key=k_w,
            genmatter_state=frame.retval,
            dense_assignments=dense_assignments,
        )
        fd = _frame_data_from_trace(
            frame,
            tracked_points_full,
            tracked_motion_vectors_full,
            tracked_features_full,
            frame_idx,
            num_datapoints_full,
        )
        fd["blob_assignments"] = np.array(dense_assignments)
        fd["blob_weights"] = np.array(dense_weights)
        tracking_data.append(fd)

    timings.dense_seconds = time.perf_counter() - t_dense
    if needs_jit_warmup and params.measure_fps:
        timings.jit_dense_seconds = timings.dense_seconds
    if num_frames > 0 and timings.dense_seconds > 0:
        timings.dense_fps = num_frames / timings.dense_seconds

    return DinoTrackingResult(
        video_id=inputs.video_id,
        tracking_data=tracking_data,
        img_dims=img_dims,
        subsampled_indices=subsampled_indices,
        focal_length=params.focal_length,
        timings=timings,
        gaussian_means=np.array(gaussian_means),
        gaussian_stds=np.array(gaussian_stds),
    )


def save_dense_tracking_npz(
    path: Path,
    result: DinoTrackingResult,
    *,
    rgb_frames_dir: Path | None = None,
    motion_npz: Path | None = None,
    dino_npz: Path | None = None,
    sam_frame0_png: Path | None = None,
) -> int:
    """Stack per-frame tracking dicts into one NPZ for future Rerun visualization."""
    td = result.tracking_data
    T = len(td)
    if T == 0:
        raise ValueError("tracking_data is empty")

    def stack_key(key: str, dtype=None):
        arr = np.stack([f[key] for f in td], axis=0)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    payload = {
        "blob_assignments": stack_key("blob_assignments", np.int32),
        "blob_weights": stack_key("blob_weights", np.float32),
        "blob_means": stack_key("blob_means", np.float32),
        "blob_covs": stack_key("blob_covs", np.float32),
        "blob_vel_means": stack_key("blob_vel_means", np.float32),
        "blob_vel_covs": stack_key("blob_vel_covs", np.float32),
        "blob_features": stack_key("blob_features", np.float32),
        "hyperblob_assignments": stack_key("hyperblob_assignments", np.int32),
        "hyperblob_weights": stack_key("hyperblob_weights", np.float32),
        "hyperblob_means": stack_key("hyperblob_means", np.float32),
        "hyperblob_trans_vels": stack_key("hyperblob_trans_vels", np.float32),
        "hyperblob_rot_vels": stack_key("hyperblob_rot_vels", np.float32),
        "datapoint_positions": stack_key("datapoint_positions", np.float32),
        "datapoint_vels": stack_key("datapoint_vels", np.float32),
        "datapoint_features": stack_key("datapoint_features", np.float32),
        "n_blobs": np.array([f["n_blobs"] for f in td], dtype=np.int32),
        "n_hyperblobs": np.array([f["n_hyperblobs"] for f in td], dtype=np.int32),
        "subsampled_indices": result.subsampled_indices.astype(np.int32),
        "img_dims": np.array(result.img_dims, dtype=np.int32),
        "focal_length": np.float32(result.focal_length),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    import json

    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "video_id": result.video_id,
                "num_frames": T,
                "img_dims": list(result.img_dims),
                "focal_length": result.focal_length,
                "rgb_frames_dir": str(rgb_frames_dir) if rgb_frames_dir else "",
                "motion_npz": str(motion_npz) if motion_npz else "",
                "dino_npz": str(dino_npz) if dino_npz else "",
                "sam_frame0_png": str(sam_frame0_png) if sam_frame0_png else "",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path.stat().st_size


# GenMatter Tracking with DINO Features - Batch Processing

import os
import json
import time
import gc
import jax

# JAX compilation cache setup
cache_dir = os.path.join(os.getcwd(), ".jax_cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.experimental.compilation_cache.compilation_cache.set_cache_dir(cache_dir)

import jax.numpy as jnp
from jax.random import key as jkey

import numpy as np
import pickle
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import config

from genmatter.datatypes import *
from genmatter.model_3d import *
from genmatter.inference import *
from genmatter.dataloader import *
from genmatter.utils import *
from genmatter.evaluation import *
from genmatter.bootstrap_stats import (
    BOOTSTRAP_N_SAMPLES,
    BOOTSTRAP_RANDOM_SEED,
    bootstrap_mean_ci_95,
)

import genjax
from genjax import Const, gen, Pytree

# ============================================================================
# Configuration
# ============================================================================

# FPS measurement flag
MEASURE_FPS = True

VIDEO_NAMES = list(config.TAPVID_DAVIS_VIDEO_NAMES)

NUM_INIT_PARTICLES_ON_MASK = None
USE_SAM_FRAME0 = True
SAVE_3WIDE_VIDEO = False  # overridden by davis_run_cli when run as __main__
SKIP_COMPLETED = False  # overwritten by davis_run_cli.configure_experiment_module

# Paths
DAVIS_3D_MOTION_PATH = str(config.DAVIS_3D_MOTION_PATH)
DAVIS_SEGMASKS_PATH = str(config.DAVIS_SEGMASKS_PATH)
DAVIS_RGB_PATH = str(config.DAVIS_RGB_PATH)
DINO_PATH_TEMPLATE = str(config.DAVIS_DINO_PATH / '{}_dino_pca_per_pixel.npz')
SAM_FRAME0_PATH_TEMPLATE = str(config.DAVIS_SAM_FRAME0_PATH / '{}_SAM_frame0.png')

# Output directory (overridden by davis_run_cli from --gt-init)
EXPERIMENT_SAVE_DIR = str(config.DAVIS_ABLATION_OUTPUT_DIR_SAM)

# Model hyperparameters
NUM_BLOBS = 500
NUM_HYPERBLOBS_ORIGINAL = 9
# NUM_HYPERBLOBS_ORIGINAL = 1
FOCAL_LENGTH = 520.0
BLOB_COUNTING_THRESHOLD = 0
RANDOM_SEED = 42

# ============================================================================
# Model Definition with DINO Features
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
# Helper Functions
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

def sample_datapoints_percentage(tracked_points, tracked_motion_vectors, percentage, seed=None, same_indices_all_timesteps=True):
    """
    Randomly sample a percentage of datapoints from tracked_points and tracked_motion_vectors
    at each timestep.

    Args:
        tracked_points: Array of shape (T, num_points, ...)
        tracked_motion_vectors: Array of shape (T, num_points, ...)
        percentage: Percentage of points to keep (0-100)
        seed: Random seed for reproducibility
        same_indices_all_timesteps: If True, sample once and reuse for all timesteps

    Returns:
        tuple: (sampled_tracked_points, sampled_tracked_motion_vectors, sampled_indices_info)
            - If same_indices_all_timesteps: sampled_indices_info is shape (num_points_to_keep,)
            - Else: sampled_indices_info is shape (T, num_points_to_keep)
    """
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
    else:
        sampled_indices_batched = np.zeros((T, num_points_to_keep), dtype=int)
        sampled_tracked_points = np.zeros((T, num_points_to_keep) + tracked_points.shape[2:], dtype=tracked_points.dtype)
        sampled_tracked_motion_vectors = np.zeros((T, num_points_to_keep) + tracked_motion_vectors.shape[2:], dtype=tracked_motion_vectors.dtype)
        for t in range(T):
            idx = np.random.choice(num_points, num_points_to_keep, replace=False)
            idx = np.sort(idx)
            sampled_indices_batched[t] = idx
            sampled_tracked_points[t] = tracked_points[t, idx, ...]
            sampled_tracked_motion_vectors[t] = tracked_motion_vectors[t, idx, ...]
        return sampled_tracked_points, sampled_tracked_motion_vectors, sampled_indices_batched

def initialize_model_with_dino(tracked_points, num_blobs, num_hyperblobs,
                               segmentation_mask, motion_vectors, tracked_features, img_dims, video_name,
                               subsampled_indices=None):
    """Initialize model with DINO features."""
    from genjax import ChoiceMapBuilder as C

    if USE_SAM_FRAME0:
        segmentation_mask = cv2.imread(SAM_FRAME0_PATH_TEMPLATE.format(video_name), cv2.IMREAD_COLOR)
        segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2RGB)
        kmeans_chm, roi_blob_indices, roi_hyperblob_indices, num_hyperblobs_actual = make_hierarchical_kmeans_chm_with_SAM_segmentations(
            tracked_points, num_blobs, segmentation_mask, img_dims,
            motion_vectors=motion_vectors,
            subsampled_indices=subsampled_indices,
        )
    else:
        kmeans_chm, roi_blob_indices, roi_hyperblob_indices = make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(
            tracked_points, num_blobs, num_hyperblobs,
            segmentation_mask=segmentation_mask,
            motion_vectors=motion_vectors,
            num_roi_blobs = NUM_INIT_PARTICLES_ON_MASK,
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
# Inference Functions
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
    """Dense evaluation of blob assignments."""
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
def dense_eval_blob_weights(key, genmatter_state : GenMatter_State, dense_assignments : jnp.ndarray): 
    posterior_key, _ = jax.random.split(key)

    # Get the data and parameters from the trace
    num_blobs = genmatter_state.hypers.n_blobs
    prior_beta = genmatter_state.hypers.beta
    blob_idxs = dense_assignments  # [N]

    # Compute counts via segment_sum
    blob_counts = jax.ops.segment_sum(
        jnp.ones_like(blob_idxs),  # [N]
        blob_idxs,                 # [N] (each datapoint's assigned blob)
        num_segments=num_blobs     # total number of blobs
    )  # [L]

    # Posterior Dirichlet parameters
    new_betas = prior_beta + blob_counts  # [L]

    # Sample new blob weights
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
    key, genmatter_state = jax.lax.fori_loop(0, 1, update_blob_assignments_position_only, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 15, update_blob_means, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 3, update_blob_features_dino, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 1, update_blob_assignments_with_outlier, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 15, update_blob_velocities, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 15, update_blob_velocity_covariances, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 3, update_blob_features_dino, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 3, hyperblob_update_loop, (key, genmatter_state))

    return genmatter_state

def genmatter_tracking_gibbs_dino(key, init_genmatter_state, tracked_points,
                              tracked_motion_vectors, tracked_features, outlier_prob):
    """Track over time with DINO features."""
    init_genmatter_state = init_genmatter_state.replace({'hypers': {'outlier_prob': f_(outlier_prob)}})

    @jax.jit
    def f_tracking_sweep(carry, timestep_idx):
        key, genmatter_state, tracked_points, tracked_motion_vectors, tracked_features = carry
        next_blob_means = genmatter_state.blobs_state.blob_vel_means + genmatter_state.blobs_state.blob_means
        genmatter_state = genmatter_state.replace({'blobs_state': {'blob_means': next_blob_means}})
        genmatter_state = genmatter_state.replace({
            'datapoints_state': {
                'datapoint_positions': tracked_points[timestep_idx],
                'datapoint_vels': tracked_motion_vectors[timestep_idx],
                'datapoint_features': tracked_features[timestep_idx]
            }
        })
        key, gibbs_key = jax.random.split(key)
        genmatter_state = blob_tracking_gibbs_dino(gibbs_key, genmatter_state)
        return (key, genmatter_state, tracked_points, tracked_motion_vectors, tracked_features), \
               genmatter_TraceWrapper(force_retval=genmatter_state)

    timestep_indices = jnp.arange(1, len(tracked_points))
    _, stacked_genmatter_wtrs = jax.lax.scan(
        f_tracking_sweep,
        (key, init_genmatter_state, tracked_points, tracked_motion_vectors, tracked_features),
        timestep_indices,
        unroll=1
    )

    return GenMatter_Gibbs_TraceWrapper(
        genmatter_TraceWrapper(force_retval=init_genmatter_state),
        stacked_genmatter_wtrs
    )

def init_gibbs_sweep_dino(key, genmatter_state, num_sweeps=30):
    """Custom Gibbs sweeps for DINO initialization."""
    def gibbs_iteration(carry, i):
        key, genmatter_state = carry
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_assignments_dino(gibbs_key, genmatter_state, position_only=True, disable_outlier_prob=True)
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

    (key, final_state), traces = jax.lax.scan(gibbs_iteration, (key, genmatter_state), jnp.arange(num_sweeps))
    return GenMatter_Gibbs_TraceWrapper(genmatter_TraceWrapper(force_retval=genmatter_state), traces)

# ============================================================================
# Evaluation Functions
# ============================================================================
#
# Primary comparison metric: matter-weighted recall, precision, and Jaccard with
# frame-0 blob weights (evaluate_single_davis_video).
# ============================================================================


def create_3wide_video(tracking_data, video_name, rgb_path, img_dims, segmentation_masks,
                       num_blobs, save_path, subsampled_indices=None, is_subsampled=False,
                       ):
    """Create 3-wide synchronized video."""
    from glob import glob

    rgb_dir = os.path.join(rgb_path, video_name)
    rgb_files = sorted(glob(os.path.join(rgb_dir, "*.jpg")))

    if len(rgb_files) == 0:
        print(f"Warning: No RGB frames found for {video_name}")
        return

    img_height, img_width = img_dims
    # Seed numpy RNG for consistent visualization colors
    np.random.seed(RANDOM_SEED)
    blob_colors = np.random.randint(0, 255, size=(num_blobs, 3), dtype=np.uint8)
    combined_frames = []

    num_frames = min(len(tracking_data), len(rgb_files))

    # Determine object particles from frame 0 using segmentation mask overlap
    frame0 = tracking_data[0]
    blob_assignments_frame0 = frame0['blob_assignments']
    n_blobs_frame0 = NUM_BLOBS
    gt_mask_frame0 = segmentation_masks[0]

    print(f"number of blobs in frame 0: {n_blobs_frame0}")
    
    # Count pixels per blob in frame 0
    blob_pixel_counts_frame0 = np.bincount(
        blob_assignments_frame0[blob_assignments_frame0 < n_blobs_frame0],
        minlength=n_blobs_frame0
    )
    
    # Determine which blobs are object vs background based on mask overlap
    # Pixels assigned to each blob vs mask overlap
    object_blobs_frame0 = set()
    background_blobs_frame0 = set()
    
    for blob_idx in range(n_blobs_frame0):
        # Find pixels assigned to this blob
        blob_pixel_mask = (blob_assignments_frame0 == blob_idx)
        
        # Check if any of these pixels are on the segmentation mask
        pixels_on_mask = np.sum(gt_mask_frame0[blob_pixel_mask])
        
        if pixels_on_mask > 0:
            object_blobs_frame0.add(blob_idx)
        else:
            background_blobs_frame0.add(blob_idx)

    for frame_idx in tqdm(range(num_frames), desc="Rendering 3-wide video frames"):
        rgb_frame = cv2.imread(rgb_files[frame_idx])
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        frame = tracking_data[frame_idx]
        blob_assignments = frame['blob_assignments']
        n_blobs = NUM_BLOBS

        valid_mask = blob_assignments < n_blobs
        
        # Create binary mask based on object particles identified from frame 0
        # Convert object_blobs_frame0 set to array for vectorized operations
        object_blobs_array = np.array(list(object_blobs_frame0))
        
        # Vectorized check: for each blob_assignment, check if it's in object_blobs_frame0
        # Only consider valid blob assignments (< n_blobs) to avoid outlier blobs
        object_blob_mask = valid_mask & np.isin(blob_assignments, object_blobs_array)
        binary_mask = np.where(object_blob_mask, 255, 0).astype(np.uint8)
        
        background_color = np.array([128, 128, 128], dtype=np.uint8)

        blob_assignment_frame = np.zeros((len(blob_assignments), 3), dtype=np.uint8)
        blob_assignment_frame[~object_blob_mask] = background_color
        
        # Only assign colors to valid object blobs (ignore outlier blobs >= n_blobs)
        valid_object_mask = object_blob_mask & valid_mask
        if np.any(valid_object_mask):
            valid_object_indices = blob_assignments[valid_object_mask]
            blob_assignment_frame[valid_object_mask] = blob_colors[valid_object_indices]
        
        # Get RGB dimensions first
        rgb_h, rgb_w = rgb_frame.shape[:2]
        
        # Handle reshaping based on subsampling
        if is_subsampled:
            # For subsampled data, reconstruct full image with correct spatial positions
            # Create full-size arrays with background values
            full_binary = np.zeros(img_height * img_width, dtype=np.uint8)
            full_blob_frame = np.full((img_height * img_width, 3), background_color, dtype=np.uint8)
            
            # Place subsampled data at correct positions
            full_binary[subsampled_indices] = binary_mask
            full_blob_frame[subsampled_indices] = blob_assignment_frame
            
            # Reshape to image dimensions
            binary_image = full_binary.reshape(img_height, img_width)
            blob_assignment_frame = full_blob_frame.reshape(img_height, img_width, 3)
            
            # Resize to match RGB if dimensions differ
            if (rgb_h, rgb_w) != (img_height, img_width):
                binary_image = cv2.resize(binary_image, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
                blob_assignment_frame = cv2.resize(blob_assignment_frame, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        else:
            # No subsampling: reshape to image dimensions
            binary_image = binary_mask.reshape(img_height, img_width)
            blob_assignment_frame = blob_assignment_frame.reshape(img_height, img_width, 3)
            
            # Resize to match RGB if dimensions differ
            if (rgb_h, rgb_w) != (img_height, img_width):
                binary_image = cv2.resize(binary_image, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
                blob_assignment_frame = cv2.resize(blob_assignment_frame, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        
        # Create segmentation frame from binary image
        segmentation_frame = np.stack([binary_image, binary_image, binary_image], axis=-1)

        SCALE_FACTOR = 0.5
        target_w = int(rgb_w * SCALE_FACTOR)
        target_h = int(rgb_h * SCALE_FACTOR)
        rgb_frame = cv2.resize(rgb_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        segmentation_frame = cv2.resize(segmentation_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        blob_assignment_frame = cv2.resize(blob_assignment_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        combined_frame = np.concatenate([rgb_frame, segmentation_frame, blob_assignment_frame], axis=1)
        combined_frames.append(combined_frame)

    # Write frames to temporary location, then use ffmpeg subprocess to encode with H.264
    # This avoids JAX fork issues since we call ffmpeg AFTER all JAX computation is done
    if len(combined_frames) == 0:
        print(f"Warning: No frames to write for {save_path}")
        return
    else:
        print(f"Writing {len(combined_frames)} frames to temporary location")

    import tempfile
    import subprocess
    import shutil

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    try:
        # Write frames as PNGs
        for i, frame in tqdm(enumerate(combined_frames), desc="Writing frames to temporary location"):
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)

        # Use ffmpeg to create H.264 video
        # This is safe to call after JAX initialization since all compute is done
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',
            '-i', os.path.join(temp_dir, 'frame_%05d.png'),
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Force even dimensions
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            save_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Saved 3-wide video: {save_path}")
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)

# ============================================================================
# Cleanup Function
# ============================================================================

def cleanup_between_runs():
    """Clear JAX state, environment variables, and memory between experimental runs."""
    # This function helps prevent memory leaks that can accumulate across runs
    
    # Clear matplotlib figure cache to prevent memory leaks
    plt.close('all')
    
    # Clear OpenCV cache if any
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    # Clear JAX device memory explicitly
    # Note: PJRT backend doesn't support explicit memory release operations
    # (device.clear_cache() and defragment() both cause fatal errors).
    # JAX's automatic memory management + Python GC will handle memory cleanup.
    # Explicit deletion of variables + multiple GC cycles is sufficient.
    
    # Clear JAX compilation cache if it exists
    try:
        # Clear in-memory compilation cache
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
    except:
        pass
    
    # Try to clear JAX backend cache if available
    try:
        import jax._src.lib.xla_bridge as xla_bridge
        if hasattr(xla_bridge, 'get_backend'):
            backend = xla_bridge.get_backend()
            if hasattr(backend, 'defragment'):
                # This might fail on PJRT, but try it
                try:
                    backend.defragment()
                except:
                    pass
    except:
        pass
    
    # Force garbage collection multiple times to ensure cleanup
    # Run multiple times because some objects may have circular references
    for _ in range(10):  # Increased to 10 for more aggressive cleanup
        gc.collect()

# ============================================================================
# Main Processing Function
# ============================================================================

def process_video(video_name, subsampling_percentage=100.0, subsampled_indices=None, run_save_dir=None):
    """Process a single video and return results."""
    print(f"\n{'='*80}")
    print(f"Processing: {video_name} | Subsampling: {subsampling_percentage}%")
    print(f"{'='*80}")

    try:
        # Load data
        dino_path = DINO_PATH_TEMPLATE.format(video_name)
        if not os.path.exists(dino_path):
            print(f"DINO features not found: {dino_path}")
            return None

        pca_data = np.load(dino_path)
        pca_features_unnormalized = pca_data['pca_features_unnormalized']
        gaussian_means = pca_data['gaussian_means']
        gaussian_stds = pca_data['gaussian_stds']
        # Close the npz file to free memory
        pca_data.close()

        tracked_points_full, tracked_motion_vectors_full, num_data_tsteps, img_dims = \
            extract_3d_points_and_motion_vectors_data(DAVIS_3D_MOTION_PATH, video_name)

        num_datapoints_full = tracked_points_full.shape[1]

        # Limit to first 90 frames for jello-trim
        if video_name == "jello_trim":
            num_data_tsteps = min(num_data_tsteps, 90)
            tracked_points = tracked_points_full[:num_data_tsteps]
            tracked_motion_vectors = tracked_motion_vectors_full[:num_data_tsteps]
            print(f"Limited to first {num_data_tsteps} frames for jello_trim")

        tracked_features_full = extract_dino_features(video_name, pca_features_unnormalized, img_dims, num_data_tsteps)

        first_frame_seg = get_segmentation_mask(
            video_name, 0, DAVIS_SEGMASKS_PATH, img_dims=img_dims, flatten=True
        )

        # Subsample datapoints consistently across all timesteps
        if subsampled_indices is None and subsampling_percentage < 100.0:
            tracked_points, tracked_motion_vectors, subsampled_indices = sample_datapoints_percentage(
                tracked_points_full, tracked_motion_vectors_full, subsampling_percentage, seed=RANDOM_SEED, same_indices_all_timesteps=True
            )
            # Subsample DINO features with same indices
            tracked_features = tracked_features_full[:, subsampled_indices, :]
            # Subsample the GT mask for frame 0 to align to datapoints where needed
            first_frame_seg_sub = first_frame_seg[subsampled_indices]
        else:
            first_frame_seg_sub = first_frame_seg
            tracked_features = tracked_features_full
            tracked_points = tracked_points_full
            tracked_motion_vectors = tracked_motion_vectors_full

        # Initialize
        kmeans_chm, roi_blob_indices, roi_hyperblob_indices, num_hyperblobs = initialize_model_with_dino(
            tracked_points, NUM_BLOBS, NUM_HYPERBLOBS_ORIGINAL,
            first_frame_seg, tracked_motion_vectors, tracked_features, img_dims, video_name,
            subsampled_indices=subsampled_indices
        )

        # Inflate k-means hyperblob covariances (weak cluster-level coupling in the generative init)
        from genjax import ChoiceMapBuilder as C
        num_hyperblobs_actual = kmeans_chm['hyperblobs', 'hyperblob_covs'].shape[0]
        huge_hyperblob_covs = jnp.tile(1e6 * jnp.eye(3)[None, :, :], (num_hyperblobs_actual, 1, 1))
        kmeans_chm = kmeans_chm | C['hyperblobs', 'hyperblob_covs'].set(huge_hyperblob_covs)

        # Create hyperparameters (empirical medians from k-means; same scale as non-ablation subsampling)
        num_datapoints = kmeans_chm['datapoints', 'datapoint_positions'].shape[0]
        num_blobs = kmeans_chm['blobs', 'hyperblob_assignments'].shape[0]
        num_hyperblobs = kmeans_chm['hyperblobs', 'hyperblob_means'].shape[0]

        empirical_mu_H = jnp.median(kmeans_chm['datapoints', 'datapoint_positions'], axis=0)
        empirical_sigma_H = (10 * 0.5) ** 2
        empirical_Psi_B = jnp.median(kmeans_chm['blobs', 'blob_covs'][roi_blob_indices], axis=0)
        empirical_Psi_H = 1e6 * jnp.eye(3)
        empirical_Psi_V = jnp.median(kmeans_chm['blobs', 'blob_vel_covs'][roi_blob_indices], axis=0)
        mean_blobs_per_roi_hyperblob = jnp.sum(
            jnp.isin(kmeans_chm['blobs', 'hyperblob_assignments'], roi_hyperblob_indices)
        ) / len(roi_hyperblob_indices)
        empirical_nu_H = f_(int(mean_blobs_per_roi_hyperblob))
        mean_points_per_roi_blob = jnp.sum(
            jnp.isin(kmeans_chm['datapoints', 'blob_assignments'], roi_blob_indices)
        ) / len(roi_blob_indices)
        empirical_nu_B = empirical_nu_V = f_(int(mean_points_per_roi_blob))

        hypers = GenMatter_Hyperparams_DINO.create(
            mu_F=jnp.array(gaussian_means),
            sigma_F_prior=jnp.array(gaussian_stds),
            # sigma_F=f_(20.0),
            sigma_F=f_(0.2),
            outlier_prob=f_(5e-0), ## this triggers the outlier for the first frame 0
            # outlier_prob=f_(5e-3), ## this triggers the outlier for the first frame 0
            outlier_velocity_gamma_shape=f_(5.0),
            outlier_velocity_gamma_rate=f_(1.0),
            alpha=f_(1.0),
            beta=f_(1.0),
            mu_H=empirical_mu_H,
            sigma_H=empirical_sigma_H,
            nu_H=empirical_nu_H,
            Psi_H=empirical_Psi_H,
            nu_B=empirical_nu_B,
            Psi_B=empirical_Psi_B,
            sigma_V=f_(10e14),
            nu_V=empirical_nu_V,
            Psi_V=empirical_Psi_V,
            translation_gaussian_scale=snp(f_(0.2)),
            translation_max_radius=snp(0.35),
            translation_num_radii_cells=snp(15),
            translation_theta_step_deg=snp(15),
            rotation_vmf_kappa=snp(f_(100)),
            rotation_angle_max_deg=snp(25),
            rotation_angle_step_deg=snp(0.375),
            n_hyperblobs=num_hyperblobs,
            n_blobs=num_blobs,
            n_datapoints=num_datapoints,
        )

        # Initialize model with fixed random seed for reproducibility
        key = jkey(RANDOM_SEED)
        key, key_importance = jax.random.split(key)
        init_tr, _ = model_jimportance(key_importance, kmeans_chm, (hypers,))
        init_genmatter_state = init_tr.get_retval()

        # Initial Gibbs sweeps
        key, init_gibbs_key = jax.random.split(key)
        print("Running initial Gibbs sweeps...")
        gibbs_wtrs = init_gibbs_sweep_dino(init_gibbs_key, init_genmatter_state, num_sweeps=15)
        init_genmatter_state = gibbs_wtrs[-1].retval

        # # Post-Gibbs filtering (gets ignored if USE_SAM_FRAME0 is True)
        # fx = fy = FOCAL_LENGTH
        # cx = img_dims[1] / 2.0
        # cy = img_dims[0] / 2.0

        # blob_means_after_gibbs = np.array(init_genmatter_state.blobs_state.blob_means)
        # hyperblob_assignments_after_gibbs = np.array(init_genmatter_state.blobs_state.hyperblob_assignments)

        # x_2d = (blob_means_after_gibbs[:, 0] / (blob_means_after_gibbs[:, 2] + 1e-8)) * fx + cx
        # y_2d = (blob_means_after_gibbs[:, 1] / (blob_means_after_gibbs[:, 2] + 1e-8)) * fy + cy
        # x_2d = np.clip(x_2d.astype(int), 0, img_dims[1] - 1)
        # y_2d = np.clip(y_2d.astype(int), 0, img_dims[0] - 1)
        # pixel_indices = y_2d * img_dims[1] + x_2d

        # blob_assignments_frame0 = np.array(init_genmatter_state.datapoints_state.blob_assignments)
        # n_blobs_after_gibbs = len(blob_means_after_gibbs)
        # valid_mask = blob_assignments_frame0 < n_blobs_after_gibbs
        # datapoint_to_hyperblob = np.full(len(blob_assignments_frame0), -1, dtype=int)
        # datapoint_to_hyperblob[valid_mask] = hyperblob_assignments_after_gibbs[blob_assignments_frame0[valid_mask]]

        # hyperblob_overlaps = {}
        # for hb_idx in range(num_hyperblobs):
        #     hb_mask = datapoint_to_hyperblob == hb_idx
        #     # Use subsampled segmask if subsampling is active
        #     overlap = np.sum(hb_mask & ((first_frame_seg_sub if subsampled_indices is not None else first_frame_seg) == 1))
        #     hyperblob_overlaps[hb_idx] = overlap

        # object_hyperblob_idx = max(hyperblob_overlaps, key=hyperblob_overlaps.get)

        # if not USE_SAM_FRAME0:
        #     # get ignored if USE_SAM_FRAME0 is True

        #     blobs_in_object = np.where(hyperblob_assignments_after_gibbs == object_hyperblob_idx)[0]
        #     blobs_to_reassign = []
        #     for blob_idx in blobs_in_object:
        #         pixel_idx = pixel_indices[blob_idx]
        #         if pixel_idx >= len(first_frame_seg) or not first_frame_seg[pixel_idx]:
        #             blobs_to_reassign.append(blob_idx)

        #     if len(blobs_to_reassign) > 0:
        #         background_hyperblobs = [hb for hb in range(num_hyperblobs) if hb != object_hyperblob_idx]
        #         background_blob_counts = {hb: np.sum(hyperblob_assignments_after_gibbs == hb) for hb in background_hyperblobs}
        #         target_background_hb = max(background_blob_counts, key=background_blob_counts.get)
        #         updated_hyperblob_assignments = jnp.array(hyperblob_assignments_after_gibbs)
        #         for blob_idx in blobs_to_reassign:
        #             updated_hyperblob_assignments = updated_hyperblob_assignments.at[blob_idx].set(target_background_hb)
        #         init_genmatter_state = init_genmatter_state.replace({
        #             'blobs_state': {'hyperblob_assignments': updated_hyperblob_assignments}
        #         })

        # Track over time with FPS measurement
        key, tracking_key = jax.random.split(key)
        print("Running tracking...")
        
        # Measure FPS if enabled
        fps = None
        if MEASURE_FPS:
            # First run to trigger JIT compilation (not timed)
            print("JIT compiling tracking function...")
            _ = genmatter_tracking_gibbs_dino(
                tracking_key, init_genmatter_state,
                tracked_points, tracked_motion_vectors, tracked_features,
                outlier_prob=1e-28
            )
            
            # Second run for timing (after JIT compilation)
            print("Measuring FPS after JIT compilation...")
            start_time = time.time()
            tracking_wtrs = genmatter_tracking_gibbs_dino(
                tracking_key, init_genmatter_state,
                tracked_points, tracked_motion_vectors, tracked_features,
                outlier_prob=1e-28
            )
            end_time = time.time()
            
            # Calculate FPS
            total_time = end_time - start_time
            num_frames = len(tracked_points) - 1  # Subtract 1 because tracking starts from frame 1
            fps = num_frames / total_time if total_time > 0 else 0.0
            print(f"Tracking FPS: {fps:.2f} frames/second (total time: {total_time:.2f}s for {num_frames} frames)")
        else:
            tracking_wtrs = genmatter_tracking_gibbs_dino(
                tracking_key, init_genmatter_state,
                tracked_points, tracked_motion_vectors, tracked_features,
                outlier_prob=1e-28
            )

        

        # Extract results
        tracking_data = []
        # NOTE that tracking wtrs also contains the first frame, which is good because we need to evaluate the first frame too
        for frame_idx in tqdm(range(len(tracking_wtrs)), desc="Dense Evaluating Blob Assignments"):
            frame = tracking_wtrs[frame_idx]
            #NOTE: This is the dense evaluation of the blob assignments using the full DINO features and motion vectors and positions
            key, dense_eval_assignments_key = jax.random.split(key)
            dense_eval_assignments = dense_eval_blob_assignments(
                key=dense_eval_assignments_key, 
                genmatter_state=frame.retval, 
                dense_positions=tracked_points_full[frame_idx], 
                dense_vels=tracked_motion_vectors_full[frame_idx], 
                dense_features=tracked_features_full[frame_idx], 
                disable_outlier_prob=False
            )

            # Then we get the blob weights from the dense eval assignments
            key, dense_eval_weights_key = jax.random.split(key)
            dense_eval_weights = dense_eval_blob_weights(
                key=dense_eval_weights_key, 
                genmatter_state=frame.retval, 
                dense_assignments=dense_eval_assignments
            )
            frame_data = {
                'n_blobs': frame.retval.hypers.n_blobs,
                'n_hyperblobs': frame.retval.hypers.n_hyperblobs,
                'n_datapoints': num_datapoints_full,
                'blob_assignments': np.array(dense_eval_assignments),
                'datapoint_positions': np.array(tracked_points_full[frame_idx]),
                'datapoint_vels': np.array(tracked_motion_vectors_full[frame_idx]),
                'datapoint_features': np.array(tracked_features_full[frame_idx]),
                'blob_weights': np.array(dense_eval_weights),
                'blob_means': np.array(frame.retval.blobs_state.blob_means),
                'blob_covs': np.array(frame.retval.blobs_state.blob_covs),
                'blob_vel_means': np.array(frame.retval.blobs_state.blob_vel_means),
                'blob_vel_covs': np.array(frame.retval.blobs_state.blob_vel_covs),
                'blob_features': np.array(frame.retval.blobs_state.blob_features),
                'hyperblob_assignments': np.array(frame.retval.blobs_state.hyperblob_assignments),
                'hyperblob_weights': np.array(frame.retval.hyperblobs_state.hyperblob_weights),
                'hyperblob_means': np.array(frame.retval.hyperblobs_state.hyperblob_means),
                'hyperblob_trans_vels': np.array(frame.retval.hyperblobs_state.hyperblob_trans_vels),
                'hyperblob_rot_vels': np.array(frame.retval.hyperblobs_state.hyperblob_rot_vels),
            }
            tracking_data.append(frame_data)

        # Segmentation masks for 3-wide video / viz
        segmentation_masks = []
        for frame_idx in range(len(tracking_data)):
            seg_mask = get_segmentation_mask(
                video_name, frame_idx, DAVIS_SEGMASKS_PATH, img_dims=img_dims, flatten=True
            )
            segmentation_masks.append(seg_mask)

        # Matter-weighted DAVIS metrics
        all_results = {video_name: [tracking_data]}
        experiment_metrics, best_visualization_data = evaluate_single_davis_video(
            davis_name=video_name,
            multiple_genmatter_list=all_results[video_name],
            annotations_path=DAVIS_SEGMASKS_PATH,
            counting_threshold=BLOB_COUNTING_THRESHOLD,
            img_dims=img_dims,
            fps_list=None,
            render_results_video=False,
            experiment_save_dir=None,
            force_below_count_thresh_as_outlier=True,
            subsampled_indices=None
        )

        result = {
            'video_name': video_name,
            'pixel_metrics': experiment_metrics,
            'tracking_data': tracking_data,
            'segmentation_masks': segmentation_masks,
            'img_dims': img_dims,
            'subsampled_indices': None,
            'subsampling_percentage': subsampling_percentage,
            'fps': fps  # Add FPS to result
        }

        print(f"Completed: {video_name}")

        if MEASURE_FPS and fps is not None:
            print(f"  FPS: {fps:.2f} frames/second")

        print(f"\n  MATTER-WEIGHTED (frame-0 blob weights) — primary DAVIS metric:")
        pm = experiment_metrics
        print(f"    Recall:     {pm['avg_matter_weighted_recall_fixed']:.3f}")
        print(f"    Precision:  {pm['avg_matter_weighted_precision_fixed']:.3f}")
        print(f"    Jaccard:    {pm['avg_matter_weighted_jaccard_fixed']:.3f}")
        print(f"    Accuracy:   {pm['avg_matter_weighted_accuracy_fixed']:.3f}")

        # Clean up large intermediate variables before returning
        # These are no longer needed after creating the result
        del tracking_wtrs
        del gibbs_wtrs
        del init_genmatter_state
        del all_results
        del segmentation_masks
        
        # Clean up large input data arrays
        # These are no longer needed after tracking is complete
        del tracked_points
        del tracked_motion_vectors
        del tracked_features
        del pca_features_unnormalized
        
        # Force garbage collection to free memory immediately
        gc.collect()

        return result

    except Exception as e:
        print(f"Error processing {video_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean up memory on error
        cleanup_between_runs()
        return None

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    _davis_dir = _Path(__file__).resolve().parent
    if str(_davis_dir) not in sys.path:
        sys.path.insert(0, str(_davis_dir))
    import davis_run_cli

    _parser = argparse.ArgumentParser(description="DAVIS DINO ablation")
    davis_run_cli.add_frame0_init_args(_parser)
    davis_run_cli.add_save_3wide_video_args(_parser)
    davis_run_cli.add_skip_completed_args(_parser)
    _args = _parser.parse_args()
    davis_run_cli.configure_experiment_module(sys.modules[__name__], _args, "ablation")

    os.makedirs(EXPERIMENT_SAVE_DIR, exist_ok=True)

    all_results = []
    all_accuracies = {}

    print(f"{'='*80}")
    print(f"DINO ablation - Processing {len(VIDEO_NAMES)} videos")
    print(f"  SAM frame-0 init: {USE_SAM_FRAME0}")
    print(f"  Save 3-wide videos: {SAVE_3WIDE_VIDEO}")
    print(f"  Output directory: {EXPERIMENT_SAVE_DIR}")
    if MEASURE_FPS:
        print(f"FPS measurement: ENABLED")
    else:
        print(f"FPS measurement: DISABLED")
    if SKIP_COMPLETED:
        print(f"  Skip completed: YES (per subsample_*/json_results/*_results.json)")
    print(f"{'='*80}\n")

    subsampling_percentages = [0.78125]
    # subsampling_percentages = [100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125]
    # subsampling_percentages = [3.125, 1.5625, 0.78125]

    for video_name in tqdm(VIDEO_NAMES, desc="Overall Progress", position=0):
        for percentage in subsampling_percentages:
            run_dir_name = f"subsample_{str(percentage).replace('.', '_')}"
            run_save_dir = os.path.join(EXPERIMENT_SAVE_DIR, run_dir_name)
            os.makedirs(run_save_dir, exist_ok=True)

            json_results_dir = os.path.join(run_save_dir, "json_results")
            per_run_json_path = os.path.join(json_results_dir, f"{video_name}_results.json")
            if SKIP_COMPLETED and os.path.isfile(per_run_json_path):
                with open(per_run_json_path, "r") as f:
                    vid_metrics = json.load(f)
                all_accuracies[video_name] = vid_metrics
                json_path = os.path.join(run_save_dir, "all_videos_experiment_results.json")
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        existing_results = json.load(f)
                    existing_results.update({video_name: vid_metrics})
                    all_accuracies_to_save = existing_results
                else:
                    all_accuracies_to_save = {video_name: vid_metrics}
                with open(json_path, "w") as f:
                    json.dump(all_accuracies_to_save, f, indent=2)
                print(f"[skip-completed] {video_name} ({percentage}% grid) -> {per_run_json_path}")
                cleanup_between_runs()
                continue

            result = process_video(video_name, subsampling_percentage=percentage, subsampled_indices=None, run_save_dir=run_save_dir)

            # Clean up memory after each subsampling percentage run
            cleanup_between_runs()

            if result is not None:
                all_results.append(result)

                base_dir = run_save_dir if ('subsampling_percentage' in result) else EXPERIMENT_SAVE_DIR

                if SAVE_3WIDE_VIDEO:
                    videos_dir = os.path.join(base_dir, "3wide_videos")
                    os.makedirs(videos_dir, exist_ok=True)
                    video_path = os.path.join(videos_dir, f"{video_name}_3wide_synchronized.mp4")
                    create_3wide_video(
                        result['tracking_data'], video_name, DAVIS_RGB_PATH,
                        result['img_dims'], result['segmentation_masks'],
                        NUM_BLOBS, video_path, subsampled_indices=None, is_subsampled=False
                    )
                    print(f"Saved 3-wide video: {video_path}")

                pm = result['pixel_metrics']
                tracking_data = result['tracking_data']

                all_accuracies[video_name] = {
                    'pixel_metrics': pm,

                    'particle_count_matter_fixed_recall': pm['avg_matter_weighted_recall_fixed'],
                    'particle_count_matter_fixed_precision': pm['avg_matter_weighted_precision_fixed'],
                    'particle_count_matter_fixed_jaccard': pm['avg_matter_weighted_jaccard_fixed'],
                    'particle_count_matter_fixed_accuracy': pm['avg_matter_weighted_accuracy_fixed'],

                    'fps': result.get('fps'),
                }

                # Save per-run JSON
                json_results_dir = os.path.join(base_dir, "json_results")
                os.makedirs(json_results_dir, exist_ok=True)
                per_run_json_path = os.path.join(json_results_dir, f"{video_name}_results.json")
                with open(per_run_json_path, 'w') as f:
                    json.dump(all_accuracies[video_name], f, indent=2)
                print(f"Saved per-run JSON: {per_run_json_path}")

                # Save aggregated JSON after each run completes (incremental updates)
                json_path = os.path.join(base_dir, "all_videos_experiment_results.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        existing_results = json.load(f)
                    existing_results.update(all_accuracies)
                    all_accuracies_to_save = existing_results
                else:
                    all_accuracies_to_save = all_accuracies

                with open(json_path, 'w') as f:
                    json.dump(all_accuracies_to_save, f, indent=2)
                print(f"Saved experiment results (updated): {json_path}")

                # Remove large tracking_data from result to free memory
                if 'tracking_data' in result:
                    del result['tracking_data']
                try:
                    del tracking_data
                except NameError:
                    pass

                # Force garbage collection after processing result
                gc.collect()

        # Clean up environment variables and memory between runs
        # This prevents memory leaks from accumulating across experimental runs
        cleanup_between_runs()

    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Total videos processed: {len(all_results)}/{len(VIDEO_NAMES)}")
    print(f"Failed videos: {len(VIDEO_NAMES) - len(all_results)}")
    print(f"Results saved to: {EXPERIMENT_SAVE_DIR}")

    # Load all results from JSON file to compute aggregate statistics
    json_path = os.path.join(EXPERIMENT_SAVE_DIR, "all_videos_experiment_results.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            all_accuracies_from_json = json.load(f)
    else:
        all_accuracies_from_json = all_accuracies

    # Filter out incomplete results (videos that failed to process)
    valid_results = {
        k: v
        for k, v in all_accuracies_from_json.items()
        if 'particle_count_matter_fixed_jaccard' in v
    }

    print(f"\n{'='*80}")
    print(f"SUMMARY - {len(valid_results)} Videos")
    print(f"{'='*80}")
    print(f"\nMetric guide: matter-weighted recall / precision / Jaccard use frame-0 blob weights (primary DAVIS metric).")
    print(f"\n")

    # Aggregate statistics
    
    # if len(valid_results) == 0:
    #     print("⚠️  No valid results found in JSON file. Cannot compute aggregate statistics.")
    #     return
    
    particle_count_matter_fixed_recall = [m['particle_count_matter_fixed_recall'] for m in valid_results.values()]
    particle_count_matter_fixed_precision = [m['particle_count_matter_fixed_precision'] for m in valid_results.values()]
    particle_count_matter_fixed_jaccard = [m['particle_count_matter_fixed_jaccard'] for m in valid_results.values()]
    particle_count_matter_fixed_accuracy = [m['particle_count_matter_fixed_accuracy'] for m in valid_results.values()]

    # Collect FPS values (handle None values)
    fps_values = [m.get('fps') for m in valid_results.values() if m.get('fps') is not None]

    print(f"AGGREGATE STATISTICS ACROSS ALL VIDEOS:")
    print(
        f"  (95% CIs: percentile bootstrap on video means, B={BOOTSTRAP_N_SAMPLES}, seed={BOOTSTRAP_RANDOM_SEED})"
    )

    print(f"\n  MATTER-WEIGHTED FIXED (primary DAVIS metric):")
    for label, arr in [
        ("Recall", particle_count_matter_fixed_recall),
        ("Precision", particle_count_matter_fixed_precision),
        ("Jaccard", particle_count_matter_fixed_jaccard),
        ("Accuracy", particle_count_matter_fixed_accuracy),
    ]:
        m, lo, hi = bootstrap_mean_ci_95(arr)
        print(f"    {label:18s} {m:.3f} [{lo:.3f}, {hi:.3f}]")

    # Print FPS statistics if available
    if fps_values:
        print(f"\n  5. PERFORMANCE METRICS:")
        fm, flo, fhi = bootstrap_mean_ci_95(fps_values)
        print(f"    FPS:                     {fm:.2f} [{flo:.2f}, {fhi:.2f}] frames/second")
        print(f"    Videos with FPS data:    {len(fps_values)}/{len(valid_results)}")
    else:
        print(f"\n  5. PERFORMANCE METRICS:")
        print(f"    FPS:                     No FPS data available")

    print(f"\n\nPER-VIDEO RESULTS:")
    print(f"{'='*80}\n")

    for video_name, metrics in valid_results.items():
        print(f"  {video_name}:")

        print(f"    Matter-weighted fixed:")
        print(f"      Recall:                  {metrics['particle_count_matter_fixed_recall']:.3f}")
        print(f"      Precision:               {metrics['particle_count_matter_fixed_precision']:.3f}")
        print(f"      Jaccard:                 {metrics['particle_count_matter_fixed_jaccard']:.3f}")
        print(f"      Accuracy:                {metrics['particle_count_matter_fixed_accuracy']:.3f}")

        # Print FPS for this video if available
        video_fps = metrics.get('fps')
        if video_fps is not None:
            print(f"\n    5. PERFORMANCE METRICS:")
            print(f"      FPS:                     {video_fps:.2f} frames/second")
        else:
            print(f"\n    5. PERFORMANCE METRICS:")
            print(f"      FPS:                     Not measured")
        
        print(f"")

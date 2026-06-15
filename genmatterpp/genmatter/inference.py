import jax
import jax.numpy as jnp
import genjax
from genjax import gen
from genjax import ChoiceMapBuilder as C
from .datatypes import GenMatter_State, GenMatter_Gibbs_TraceWrapper, GenMatter_Hyperblobs_State, GenMatter_Blobs_State, GenMatter_Datapoints_State, GenMatter_Hyperparams
from .core_types import f_, truncate_eigenval_ratio, inverse_wishart
from .trace_wrappers import genmatter_TraceWrapper, __genmatter_TraceWrapper__

def stable_segment_sum(data, segment_ids, num_segments):
    # Create one-hot encoding of segment_ids
    one_hot = jax.nn.one_hot(segment_ids, num_segments)
    # Use einsum for stable summation
    return jnp.einsum('nd,ns->sd', data, one_hot)

# def stable_segment_sum(data, segment_ids, num_segments):
#     """Memory-efficient segment sum that avoids large one-hot matrices.
    
#     Args:
#         data: Array of shape [n, d] containing data to sum
#         segment_ids: Array of shape [n] containing segment indices
#         num_segments: Number of segments
        
#     Returns:
#         Array of shape [num_segments, d] with summed data per segment
#     """
#     # Initialize output array with zeros
#     result = jnp.zeros((num_segments, data.shape[1]), dtype=data.dtype)
    
#     # Use a scan to accumulate sums for each segment
#     def update_fn(result, x):
#         idx, values = x
#         # Use index_add to avoid materializing a one-hot matrix
#         return result.at[idx].add(values), None
    
#     # Prepare data as (segment_id, values) pairs
#     pairs = (segment_ids, data)
    
#     # Scan through the data, updating the result
#     final_result, _ = jax.lax.scan(update_fn, result, pairs)
    
#     return final_result

def normal_inverse_wishart_posterior_by_cluster(X, cluster_ids, mu_k, nu0, Psi0, pseudocounts):
    """Compute Normal-Inverse-Wishart posterior parameters for clustered data using segment_sum.
    
    Args:
        X: [T, d] data points
        cluster_ids: [T] ints in 0..K-1 indicating cluster assignments
        mu_k: [K, d] known means per cluster
        nu0: scalar or [K] prior degrees of freedom (can be batched by K or not)
        Psi0: [d, d] or [K, d, d] prior scale matrix (can be batched by K or not)
        pseudocounts: [T] pseudocounts for all points
    Returns:
        nu_posterior: [K] updated degrees of freedom
        Psi_posterior: [K, d, d] updated scale matrices
    
    Note:
        Prior parameters (nu0, Psi0) can be provided as either:
        - Unbatched (scalar for nu0, [d, d] for Psi0) and will be broadcast to all clusters
        - Batched ([K] for nu0, [K, d, d] for Psi0) for cluster-specific priors
    """
    T, d = X.shape
    K = mu_k.shape[0]

    residuals = X - mu_k[cluster_ids]  # [T, d]
    scatter = jnp.einsum('ti,tj->tij', residuals, residuals)  # [T, d, d]

    # Flatten for segment_sum: [T, d*d]
    scatter_flat = scatter.reshape(T, d * d)

    # Segment sum by cluster
    # scatter_sum_flat = jax.ops.segment_sum(scatter_flat * pseudocounts[:, None], cluster_ids, num_segments=K)  # [K, d*d]
    scatter_sum_flat = stable_segment_sum(scatter_flat * pseudocounts[:, None], cluster_ids, K)  # [K, d*d]
    S_k = scatter_sum_flat.reshape(K, d, d)

    # Count points per cluster
    N_k = jax.ops.segment_sum(jnp.ones(T, dtype=X.dtype) * pseudocounts, cluster_ids, num_segments=K)  # [K]

    # Posterior parameters
    nu_posteriors = nu0 + N_k  # [K]

    Psi_posteriors = Psi0 + S_k

    # Apply eigenvalue truncation to ensure numerical stability
    Psi_posteriors = truncate_eigenval_ratio(Psi_posteriors, threshold=1e6)

    return nu_posteriors, Psi_posteriors

def normal_normal_posterior_full_cov_batched_flexible_prior(x, cluster_ids, mu0_k, Sigma0, Sigma_k):
    """
    Args:
        x: [T, d] observed data
        cluster_ids: [T] ints in 0..K-1
        mu0_k: [K, d] prior means
        Sigma0: either scalar or [K, d, d] prior covariance(s)
        Sigma_k: [K, d, d] likelihood covariances

    Returns:
        mu_n_k: [K, d] posterior means
        Sigma_n_k: [K, d, d] posterior covariances
    """
    T, d = x.shape
    K = mu0_k.shape[0]

    # Count points per cluster: [K]
    N_k = jax.ops.segment_sum(jnp.ones(T), cluster_ids, num_segments=K)

    # Sum of x per cluster: [K, d]
    sum_x_k = stable_segment_sum(x, cluster_ids, K)

    # Expand scalar Sigma0 if necessary
    if Sigma0.ndim == 0:
        Sigma0_k = jnp.broadcast_to(Sigma0 * jnp.eye(d), (K, d, d))  # shared scalar prior
    elif Sigma0.ndim == 2:
        Sigma0_k = jnp.broadcast_to(Sigma0, (K, d, d))               # shared full matrix
    else:
        Sigma0_k = Sigma0                                            # already batched

    # Invert Sigma0_k and Sigma_k: [K, d, d]
    Sigma0_inv = jnp.linalg.inv(Sigma0_k)
    Sigma_inv = jnp.linalg.inv(Sigma_k)

    # Posterior covariance: [K, d, d]
    Sigma_n_inv = Sigma0_inv + N_k[:, None, None] * Sigma_inv
    Sigma_n_k = jnp.linalg.inv(Sigma_n_inv)

    # Posterior mean: [K, d]
    term_prior = jnp.einsum('kij,kj->ki', Sigma0_inv, mu0_k)
    term_data  = jnp.einsum('kij,kj->ki', Sigma_inv, sum_x_k)
    mu_n_k = jnp.einsum('kij,kj->ki', Sigma_n_k, term_prior + term_data)

    return mu_n_k, Sigma_n_k



def empty_gibbs(key, genmatter_state : GenMatter_State):
    return genmatter_state

def empty_gibbs_weighted(key, genmatter_state : GenMatter_State, use_weighted_blobs = False):
    return genmatter_state

# Gibbs #1: Update blob weights 
def gibbs_blob_weights(key, genmatter_state : GenMatter_State): 
    posterior_key, _ = jax.random.split(key)

    # Get the data and parameters from the trace
    num_blobs = genmatter_state.hypers.n_blobs
    prior_beta = genmatter_state.hypers.beta
    blob_idxs = genmatter_state.datapoints_state.blob_assignments  # [N]

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

    return genmatter_state.replace({'blobs_state': {'blob_weights': new_blob_weights}})


# Gibbs #2: Update hyperblob weights 
def gibbs_hyperblob_weights(key, genmatter_state : GenMatter_State, use_weighted_blobs = False):
    posterior_key, _ = jax.random.split(key)

    # Get the data and parameters from the trace
    num_hyperblobs = genmatter_state.hypers.n_hyperblobs
    prior_alpha = genmatter_state.hypers.alpha
    hyperblob_assignments = genmatter_state.blobs_state.hyperblob_assignments  # [L]

    # for weighted sums (i.e. weighted by number of datapoints assigned to each blob)
    blob_weights = genmatter_state.blobs_state.blob_weights
    num_blobs = genmatter_state.hypers.n_blobs

    blob_pseudocounts = jnp.where(use_weighted_blobs, blob_weights * num_blobs, jnp.ones_like(hyperblob_assignments, dtype=jnp.float32))

    # Compute counts via segment_sum via blob_pseudocounts (weighted by blob mixture weights)
    hyperblob_counts = jax.ops.segment_sum(
        blob_pseudocounts,  # [L]
        hyperblob_assignments,                 # [L] (each blob's assigned hyperblob)
        num_segments=num_hyperblobs     # total number of hyperblobs
    )  # [K]

    # Posterior Dirichlet parameters
    new_alphas = prior_alpha + hyperblob_counts  # [K]

    # Sample new hyperblob weights
    new_hyperblob_weights = genjax.dirichlet.sample(posterior_key, new_alphas)

    return genmatter_state.replace({'hyperblobs_state': {'hyperblob_weights': new_hyperblob_weights}})


# Likelihood model for blob datapoints used in Gibbs #3 
@gen
def blob_datapoint_likelihood_model_no_assignment(blob_state : GenMatter_Blobs_State):
    datapoint_position = genjax.mv_normal(blob_state.blob_means, blob_state.blob_covs) @ 'datapoint_position'
    datapoint_vel = genjax.mv_normal(blob_state.blob_vel_means, blob_state.blob_vel_covs) @ 'datapoint_vel'
    return None

def gibbs_blob_assignments(key, genmatter_state, position_only=False, velocity_only=False, disable_outlier_prob=False):
    posterior_key, _ = jax.random.split(key)

    hypers = genmatter_state.hypers
    num_blobs = hypers.n_blobs
    num_datapoints = hypers.n_datapoints
    datapoint_positions = genmatter_state.datapoints_state.datapoint_positions
    datapoint_vels = genmatter_state.datapoints_state.datapoint_vels
    blobs_state = genmatter_state.blobs_state

    # disable velocity updates if position_only is True
    gibbs_blob_vel_covs = jnp.where(position_only, 1e14 * blobs_state.blob_vel_covs, blobs_state.blob_vel_covs)
    # disable position updates if velocity_only is True
    gibbs_blob_covs = jnp.where(velocity_only, 1e14 * blobs_state.blob_covs, blobs_state.blob_covs)
    blobs_state = blobs_state.replace({'blob_vel_covs': gibbs_blob_vel_covs})
    blobs_state = blobs_state.replace({'blob_covs': gibbs_blob_covs})

    blob_weights = blobs_state.blob_weights  # [L]
    outlier_prob = hypers.outlier_prob  # scalar

    # if potions only or velocity only is true, set outlier probability to 0 using jnp where
    outlier_prob = jnp.where(disable_outlier_prob, 0.0, outlier_prob)
    # Normalize weights
    extended_weights = jnp.concatenate([blob_weights, jnp.array([outlier_prob])])  # [L+1]
    normalized_weights = extended_weights / jnp.sum(extended_weights)  # [L+1]
    log_mixture_weights = jnp.log(normalized_weights)

    def compute_local_density(point_idx):
        chm = (C["datapoint_position"].set(datapoint_positions[point_idx]) |
               C["datapoint_vel"].set(datapoint_vels[point_idx]))

        # Real blobs
        log_liks = jax.vmap(
            lambda i: blob_datapoint_likelihood_model_no_assignment.assess(
                chm, (blobs_state[i],)
            )[0]
        )(jnp.arange(num_blobs))  # [L]

        # --- Outlier: Gamma log-pdf over ||v||
        v = datapoint_vels[point_idx]
        # speed = jnp.abs(v[2]) #jnp.linalg.norm(v)
        speed = jnp.linalg.norm(v)

        alpha = hypers.outlier_velocity_gamma_shape
        beta = hypers.outlier_velocity_gamma_rate

        log_gamma_vel = (
            (alpha - 1) * jnp.log(speed + 1e-8)
            - beta * speed
            - alpha * jnp.log(1. / beta)
            - jax.lax.lgamma(alpha)
        )

        # jnp where outlier to be 0 if disable_outlier_prob is true
        log_gamma_vel = jnp.where(disable_outlier_prob, 0.0, log_gamma_vel)

        # Combine
        raw_log_liks = jnp.concatenate([log_liks, jnp.array([log_gamma_vel])])  # [L+1]
        return raw_log_liks + log_mixture_weights

    all_logprobs = jax.vmap(compute_local_density)(jnp.arange(num_datapoints))  # [N, L+1]
    updated_assignments = genjax.categorical.sample(posterior_key, logits=all_logprobs)

    return genmatter_state.replace({
        'datapoints_state': {
            'blob_assignments': updated_assignments
        }
    })

def gibbs_blob_assignments_batched(key, genmatter_state, position_only=False, velocity_only=False, disable_outlier_prob=False):
    posterior_key, _ = jax.random.split(key)

    batch_size = 1024  # Changed to divide 4096 evenly

    hypers = genmatter_state.hypers
    num_blobs = hypers.n_blobs
    num_datapoints = hypers.n_datapoints
    datapoint_positions = genmatter_state.datapoints_state.datapoint_positions
    datapoint_vels = genmatter_state.datapoints_state.datapoint_vels
    blobs_state = genmatter_state.blobs_state

    # disable velocity updates if position_only is True
    gibbs_blob_vel_covs = jnp.where(position_only, 1e14 * blobs_state.blob_vel_covs, blobs_state.blob_vel_covs)
    # disable position updates if velocity_only is True
    gibbs_blob_covs = jnp.where(velocity_only, 1e14 * blobs_state.blob_covs, blobs_state.blob_covs)
    blobs_state = blobs_state.replace({'blob_vel_covs': gibbs_blob_vel_covs})
    blobs_state = blobs_state.replace({'blob_covs': gibbs_blob_covs})

    blob_weights = blobs_state.blob_weights  # [L]
    outlier_prob = hypers.outlier_prob  # scalar

    # if potions only or velocity only is true, set outlier probability to 0 using jnp where
    outlier_prob = jnp.where(disable_outlier_prob, 0.0, outlier_prob)
    # Normalize weights
    extended_weights = jnp.concatenate([blob_weights, jnp.array([outlier_prob])])  # [L+1]
    normalized_weights = extended_weights / jnp.sum(extended_weights)  # [L+1]
    log_mixture_weights = jnp.log(normalized_weights)

    def compute_local_density(point_idx):
        chm = (C["datapoint_position"].set(datapoint_positions[point_idx]) |
               C["datapoint_vel"].set(datapoint_vels[point_idx]))

        # Real blobs
        log_liks = jax.vmap(
            lambda i: blob_datapoint_likelihood_model_no_assignment.assess(
                chm, (blobs_state[i],)
            )[0]
        )(jnp.arange(num_blobs))  # [L]

        # --- Outlier: Gamma log-pdf over ||v||
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

        # jnp where outlier to be 0 if disable_outlier_prob is true
        log_gamma_vel = jnp.where(disable_outlier_prob, 0.0, log_gamma_vel)

        # Combine
        raw_log_liks = jnp.concatenate([log_liks, jnp.array([log_gamma_vel])])  # [L+1]
        return raw_log_liks + log_mixture_weights

    # Compute logprobs in batches using scan
    def batch_compute_logprobs(carry, batch_idx):
        batch_start = batch_idx * batch_size
        batch_indices = jnp.arange(batch_size) + batch_start
        batch_logprobs = jax.vmap(compute_local_density)(batch_indices)
        return carry, batch_logprobs
    
    # Calculate number of full batches
    num_full_batches = num_datapoints // batch_size
    
    # Run scan over full batches
    _, batched_logprobs = jax.lax.scan(
        batch_compute_logprobs,
        None,  # carry is unused
        jnp.arange(num_full_batches)
    )
    
    # Reshape to get the full array of logprobs
    all_logprobs = batched_logprobs.reshape(num_full_batches * batch_size, -1)
    
    # Sample from the computed logprobs
    updated_assignments = genjax.categorical.sample(posterior_key, logits=all_logprobs)

    return genmatter_state.replace({
        'datapoints_state': {
            'blob_assignments': updated_assignments
        }
    })

# Likelihood model for hyperblob blobs used in Gibbs #4
@gen
def hyperblob_blob_likelihood_model(hyperblobs_state : GenMatter_Hyperblobs_State, hypers : GenMatter_Hyperparams):
    hyperblob_idx = genjax.categorical(probs = hyperblobs_state.hyperblob_weights) @ 'hyperblob_assignment'
    hyperblob_state = hyperblobs_state[hyperblob_idx]
    blob_mean = genjax.mv_normal(hyperblob_state.hyperblob_means, hyperblob_state.hyperblob_covs) @ 'blob_mean'
    blob_vel_mean_ = hyperblob_state.hyperblob_trans_vels + jnp.matmul(hyperblob_state.hyperblob_rot_vels - jnp.eye(3), blob_mean - hyperblob_state.hyperblob_means)
    blob_vel_mean = genjax.normal(blob_vel_mean_, jnp.sqrt(hypers.sigma_V)) @ 'blob_vel'
    return None

# Gibbs #4: Update hyperblob assignments 
def gibbs_hyperblob_assignments(key, genmatter_state : GenMatter_State):
    posterior_key, _ = jax.random.split(key)

    # get the data and parameters from the trace
    num_hyperblobs = genmatter_state.hypers.n_hyperblobs
    num_blobs = genmatter_state.hypers.n_blobs
    blob_means = genmatter_state.blobs_state.blob_means
    blob_vel_means = genmatter_state.blobs_state.blob_vel_means
    hyperblobs_state = genmatter_state.hyperblobs_state
    hypers = genmatter_state.hypers

    def compute_local_density(blob_idx):
        chm = (C["blob_mean"].set(blob_means[blob_idx]) 
              | C["blob_vel"].set(blob_vel_means[blob_idx]))

        return jax.vmap(
            lambda i: hyperblob_blob_likelihood_model.assess(
                chm.at["hyperblob_assignment"].set(i), (hyperblobs_state, hypers)
            )[0]
        )(jnp.arange(num_hyperblobs))

    local_densities = jax.vmap(compute_local_density)(jnp.arange(num_blobs))

    # Sample new assignments
    updated_hyperblob_assignments = genjax.categorical.sample(posterior_key, logits = local_densities)

    return genmatter_state.replace({'blobs_state': {'hyperblob_assignments': updated_hyperblob_assignments}})

# Gibbs #5: Update hyperblob covariances 
def gibbs_hyperblob_covs(key, genmatter_state, use_weighted_blobs = False):
    posterior_key, _ = jax.random.split(key)

    blob_means = genmatter_state.blobs_state.blob_means
    hyperblob_assignments = genmatter_state.blobs_state.hyperblob_assignments
    assumed_known_means = genmatter_state.hyperblobs_state.hyperblob_means
    prior_nu_H = genmatter_state.hypers.nu_H
    prior_Psi_H = genmatter_state.hypers.Psi_H

    n_hyperblobs = genmatter_state.hypers.n_hyperblobs

    # for weighted sums (i.e. weighted by number of datapoints assigned to each blob)
    blob_weights = genmatter_state.blobs_state.blob_weights
    num_blobs = genmatter_state.hypers.n_blobs

    blob_pseudocounts = jnp.where(use_weighted_blobs, blob_weights * num_blobs, jnp.ones_like(hyperblob_assignments, dtype=jnp.float32))

    # --- 1. Get number of blobs assigned to each hyperblob ---
    N_k = jax.ops.segment_sum(
        blob_pseudocounts, 
        hyperblob_assignments,
        num_segments=n_hyperblobs
    )  # [K]

    # --- 2. Compute normal NIW posterior ---
    nu_posteriors, Psi_posteriors = normal_inverse_wishart_posterior_by_cluster(
        blob_means, hyperblob_assignments, assumed_known_means, prior_nu_H, prior_Psi_H, blob_pseudocounts
    )

    # --- 3. Sample new covariances from posterior ---
    updated_hyperblob_covs = inverse_wishart.sample(posterior_key, nu_posteriors, Psi_posteriors)

    # --- 4. Mask: retain previous covariances if N_k <= 3 (3 min points per hyperblob for scatter matrix to be valid in NIW --> full rank) ---
    current_hyperblob_covs = genmatter_state.hyperblobs_state.hyperblob_covs  # [K, 3, 3]
    mask = (N_k >= 3)[:, None, None]  # [K,1,1] to broadcast over 3x3 matrices

    final_hyperblob_covs = jnp.where(mask, updated_hyperblob_covs, current_hyperblob_covs)

    return genmatter_state.replace({'hyperblobs_state': {'hyperblob_covs': final_hyperblob_covs}})


# Gibbs #6: Update blob covariances 
def gibbs_blob_covs(key, genmatter_state):
    posterior_key, _ = jax.random.split(key)

    datapoint_positions = genmatter_state.datapoints_state.datapoint_positions
    blob_assignments = genmatter_state.datapoints_state.blob_assignments
    assumed_known_means = genmatter_state.blobs_state.blob_means
    prior_nu_B = genmatter_state.hypers.nu_B
    prior_Psi_B = genmatter_state.hypers.Psi_B

    n_blobs = genmatter_state.hypers.n_blobs

    datapoint_pseudocounts = jnp.ones(datapoint_positions.shape[0], dtype=datapoint_positions.dtype)

    # --- 1. Get number of datapoints assigned to each blob ---
    N_l = jax.ops.segment_sum(
        datapoint_pseudocounts, 
        blob_assignments,
        num_segments=n_blobs
    )  # [L]


    # --- 2. Compute normal NIW posterior ---
    nu_posteriors, Psi_posteriors = normal_inverse_wishart_posterior_by_cluster(
        datapoint_positions, blob_assignments, assumed_known_means, prior_nu_B, prior_Psi_B, datapoint_pseudocounts
    )

    # --- 3. Sample new covariances from posterior ---
    updated_blob_covs = inverse_wishart.sample(posterior_key, nu_posteriors, Psi_posteriors)

    # --- 4. Mask: retain previous covariances if N_l <= 3 (3 min points per blob for scatter matrix to be valid in NIW --> full rank)---
    current_blob_covs = genmatter_state.blobs_state.blob_covs  # [L, 3, 3]
    mask = (N_l >= 3)[:, None, None]  # [L,1,1] to broadcast over 3x3 matrices

    final_blob_covs = jnp.where(mask, updated_blob_covs, current_blob_covs)

    return genmatter_state.replace({'blobs_state': {'blob_covs': final_blob_covs}})


# Gibbs #7: Update blob velocity covariances 
def gibbs_blob_vel_covs(key, genmatter_state):
    posterior_key, _ = jax.random.split(key)

    datapoint_vels = genmatter_state.datapoints_state.datapoint_vels
    blob_assignments = genmatter_state.datapoints_state.blob_assignments
    assumed_known_vel_means = genmatter_state.blobs_state.blob_vel_means
    prior_nu_V = genmatter_state.hypers.nu_V
    prior_Psi_V = genmatter_state.hypers.Psi_V

    n_blobs = genmatter_state.hypers.n_blobs

    datapoint_pseudocounts = jnp.ones(datapoint_vels.shape[0], dtype=datapoint_vels.dtype)

    # --- 1. Get number of datapoints assigned to each blob ---
    N_l = jax.ops.segment_sum(
        datapoint_pseudocounts, 
        blob_assignments,
        num_segments=n_blobs
    )  # [L]

    # --- 2. Compute normal NIW posterior ---
    nu_posteriors, Psi_posteriors = normal_inverse_wishart_posterior_by_cluster(
        datapoint_vels, blob_assignments, assumed_known_vel_means, prior_nu_V, prior_Psi_V, datapoint_pseudocounts
    )

    # --- 3. Sample new covariances from posterior ---
    updated_blob_vel_covs = inverse_wishart.sample(posterior_key, nu_posteriors, Psi_posteriors)

    # --- 4. Mask: retain previous covariances if N_l <= 3 (3 min points per blob for scatter matrix to be valid in NIW --> full rank)---
    current_blob_vel_covs = genmatter_state.blobs_state.blob_vel_covs  # [L, 3, 3]
    mask = (N_l >= 3)[:, None, None]  # [L,1,1] to broadcast over 3x3 matrices

    final_blob_vel_covs = jnp.where(mask, updated_blob_vel_covs, current_blob_vel_covs)

    return genmatter_state.replace({'blobs_state': {'blob_vel_covs': final_blob_vel_covs}})

# Gibbs #8: Update blob velocity means 
def gibbs_blob_vel_means(key, genmatter_state):
    posterior_key, _ = jax.random.split(key)

    datapoint_vels = genmatter_state.datapoints_state.datapoint_vels
    blob_assignments = genmatter_state.datapoints_state.blob_assignments
    hyperblob_assignments = genmatter_state.blobs_state.hyperblob_assignments
    blob_means = genmatter_state.blobs_state.blob_means
    assigned_hyperblob_per_blob = genmatter_state.hyperblobs_state[hyperblob_assignments]
    likelihood_blob_vel_covs = genmatter_state.blobs_state.blob_vel_covs
    current_blob_vel_means = genmatter_state.blobs_state.blob_vel_means

    prior_blob_vel_means_ = assigned_hyperblob_per_blob.hyperblob_trans_vels + jnp.einsum('nij,nj->ni', assigned_hyperblob_per_blob.hyperblob_rot_vels - jnp.repeat(jnp.eye(3)[None, ...], genmatter_state.hypers.n_blobs, axis=0), blob_means - assigned_hyperblob_per_blob.hyperblob_means)

    prior_variance = genmatter_state.hypers.sigma_V

    # Count datapoints per blob
    n_blobs = genmatter_state.hypers.n_blobs
    N_l = jax.ops.segment_sum(
        jnp.ones(datapoint_vels.shape[0], dtype=datapoint_vels.dtype), 
        blob_assignments,
        num_segments=n_blobs
    )  # [L]
    
    # Get posterior parameters
    posterior_mus, posterior_covs = normal_normal_posterior_full_cov_batched_flexible_prior(datapoint_vels, blob_assignments, prior_blob_vel_means_, prior_variance, likelihood_blob_vel_covs)
    
    # For blobs with no datapoints, set velocity to zero
    has_points = N_l > 0
    
    # Sample new blob velocity means for blobs with datapoints
    sampled_vel_means = genjax.mv_normal.sample(posterior_key, posterior_mus, posterior_covs)
    
    # Set velocity to zero for empty blobs
    zero_velocities = jnp.zeros_like(posterior_mus)
    posterior_blob_vel_means = jnp.where(has_points[:, None], sampled_vel_means, zero_velocities)
    # posterior_blob_vel_means = jnp.where(has_points[:, None], sampled_vel_means, current_blob_vel_means)
    return genmatter_state.replace({'blobs_state': {'blob_vel_means': posterior_blob_vel_means}})

# Gibbs #9: Update hyperblob means 
def gibbs_hyperblob_means(key, genmatter_state, use_weighted_blobs=False):
    posterior_key, _ = jax.random.split(key)
    
    # Extract data
    blob_means = genmatter_state.blobs_state.blob_means              # [L, 3]
    blob_vel_means = genmatter_state.blobs_state.blob_vel_means      # [L, 3]
    hyperblob_assignments = genmatter_state.blobs_state.hyperblob_assignments  # [L]
    blob_weights = genmatter_state.blobs_state.blob_weights          # [L]

    hyperblob_covs = genmatter_state.hyperblobs_state.hyperblob_covs          # [K, 3, 3]
    rot_vels = genmatter_state.hyperblobs_state.hyperblob_rot_vels            # [K, 3, 3]
    trans_vels = genmatter_state.hyperblobs_state.hyperblob_trans_vels        # [K, 3]

    mu0 = genmatter_state.hypers.mu_H
    sigmaH = genmatter_state.hypers.sigma_H
    sigmaV = genmatter_state.hypers.sigma_V
    K = genmatter_state.hypers.n_hyperblobs
    num_blobs = genmatter_state.hypers.n_blobs

    L = blob_means.shape[0]
    d = blob_means.shape[1]

    # Create pseudocounts based on blob weights if use_weighted_blobs is True
    blob_pseudocounts = jnp.where(use_weighted_blobs, 
                                 blob_weights * num_blobs, 
                                 jnp.ones_like(hyperblob_assignments, dtype=jnp.float32))

    # Segment sums with pseudocounts
    N_k = jax.ops.segment_sum(blob_pseudocounts, hyperblob_assignments, num_segments=K) # [K] weighted num blobs in each hyperblob
    S_k = stable_segment_sum(blob_means * blob_pseudocounts[:, None], hyperblob_assignments, K) # [K, 3] weighted sum of blob means

    hyperblob_covs_inv = jnp.linalg.inv(hyperblob_covs) # [K, 3, 3] inverse of hyperblob covariances

    # Compute transformed velocity means
    A_k = jnp.eye(3) - rot_vels  # [K, 3, 3] affine rotation matrix for each hyperblob
    b_l = trans_vels[hyperblob_assignments] - jnp.einsum("lij,lj->li", A_k[hyperblob_assignments], blob_means) # [L, 3] affine translation vector for each blob
    v_l = blob_vel_means
    residuals = v_l - b_l  # [L, 3]

    # Segment sum residuals with pseudocounts
    residual_sum = stable_segment_sum(residuals * blob_pseudocounts[:, None], hyperblob_assignments, K)  # [K, 3]
    # Then apply A_k^T
    rhs_affine = jnp.einsum("kji,kj->ki", A_k, residual_sum)  # [K, 3]
    lhs_affine = jnp.einsum('kpi,kpj->kij', A_k, A_k)  # [K, 3, 3]

    # Invert prior variance
    prior_precision = jnp.eye(d) / sigmaH  # [3,3] prior precision matrix

    # Posterior precision and mean
    posterior_precision = (
        prior_precision[None, ...] + # [1, 3, 3]
        N_k[:, None, None] * (hyperblob_covs_inv + lhs_affine / sigmaV) # [K, 3, 3]
    )  # [K, 3, 3]

    # Compute weighted mean with pseudocounts
    weighted_mean = (
        (mu0/sigmaH)[None, :] + # [1, 3]
        jnp.einsum('kij,kj->ki', hyperblob_covs_inv, S_k) + # [K, 3]
        rhs_affine / sigmaV # [K, 3]
    )

    posterior_covs = jnp.linalg.inv(posterior_precision)
    posterior_means = jnp.einsum("kij,kj->ki", posterior_covs, weighted_mean)

    # Sample new means
    new_means = genjax.mv_normal.sample(posterior_key, posterior_means, posterior_covs)

    return genmatter_state.replace({'hyperblobs_state': {'hyperblob_means': new_means}})

# Gibbs #10: Update blob means 
def gibbs_blob_means(key, genmatter_state):
    posterior_key, _ = jax.random.split(key)

    # Data
    datapoint_positions = genmatter_state.datapoints_state.datapoint_positions  # [N, 3]
    blob_assignments = genmatter_state.datapoints_state.blob_assignments        # [N]
    blob_vel_means = genmatter_state.blobs_state.blob_vel_means                 # [L, 3]
    hyperblob_assignments = genmatter_state.blobs_state.hyperblob_assignments   # [L]
    blob_covs = genmatter_state.blobs_state.blob_covs                           # [L, 3, 3]

    # Hyperblob info
    mu_H = genmatter_state.hyperblobs_state.hyperblob_means                     # [K, 3]
    trans_vels = genmatter_state.hyperblobs_state.hyperblob_trans_vels         # [K, 3]
    rot_vels = genmatter_state.hyperblobs_state.hyperblob_rot_vels             # [K, 3, 3]

    # Hyperparams
    sigmaV = genmatter_state.hypers.sigma_V
    L = genmatter_state.hypers.n_blobs
    d = genmatter_state.blobs_state.blob_means.shape[-1] # [L,3] --> [3]

    # Assign hyperblob means to blobs
    muH_l = mu_H[hyperblob_assignments]  # [L, 3]

    # Prior precision
    hyperblob_cov_inv = jnp.linalg.inv(genmatter_state.hyperblobs_state.hyperblob_covs[hyperblob_assignments])  # [L, 3, 3]

    # Data terms (from datapoints)
    N_l = jax.ops.segment_sum(jnp.ones(datapoint_positions.shape[0]), blob_assignments, num_segments=L) # [L]
    S_l = stable_segment_sum(datapoint_positions, blob_assignments, L) # [L, 3]

    # Likelihood precision (from blob covariances)
    blob_cov_inv = jnp.linalg.inv(blob_covs)  # [L, 3, 3]

    # Velocity affine likelihood
    A_l = rot_vels[hyperblob_assignments] - jnp.eye(d)  # [L, 3, 3]
    b_l = trans_vels[hyperblob_assignments] - jnp.einsum("lij,lj->li", A_l, muH_l)  # [L, 3]
    v_l = blob_vel_means # [L, 3]
    residuals = v_l - b_l  # [L, 3]

    lhs_affine = jnp.einsum("lij,lik->ljk", A_l, A_l)    # [L, 3, 3]
    rhs_affine = jnp.einsum("lij,li->lj", A_l, residuals)  # [L, 3]

    # Combine into posterior precision and weighted mean
    P_post = (
        hyperblob_cov_inv +  # prior from hyperblob covariance [L, 3, 3]
        blob_cov_inv * N_l[:, None, None] +  # likelihood from datapoints [L, 3, 3]
        lhs_affine / sigmaV  # velocity model [L, 3, 3]
    )  # [L, 3, 3]

    weighted_mean = (
        jnp.einsum("lij,lj->li", hyperblob_cov_inv, muH_l) +  # prior with hyperblob covariance
        jnp.einsum("lij,lj->li", blob_cov_inv, S_l) +         # datapoint positions
        rhs_affine / sigmaV                                   # velocity affine
    )  # [L, 3]

    # Solve for posterior mean
    cov_post = jnp.linalg.inv(P_post)  # [L, 3, 3]
    mean_post = jnp.einsum("lij,lj->li", cov_post, weighted_mean)  # [L, 3]

    # Sample new blob means
    new_blob_means = genjax.mv_normal.sample(posterior_key, mean_post, cov_post)

    return genmatter_state.replace({'blobs_state': {'blob_means': new_blob_means}})


def gibbs_blob_means_ablation(key, genmatter_state):
    """Blob means with zero-mean Gaussian prior (sigma_H); no hyperblob coupling."""
    posterior_key, prior_key = jax.random.split(key)

    datapoint_positions = genmatter_state.datapoints_state.datapoint_positions  # [N, 3]
    blob_assignments = genmatter_state.datapoints_state.blob_assignments        # [N]
    blob_covs = genmatter_state.blobs_state.blob_covs                           # [L, 3, 3]

    n_blobs = genmatter_state.hypers.n_blobs
    prior_blob_means = jnp.zeros((n_blobs, 3))
    prior_variance = genmatter_state.hypers.sigma_H

    N_l = jax.ops.segment_sum(
        jnp.ones(datapoint_positions.shape[0], dtype=datapoint_positions.dtype),
        blob_assignments,
        num_segments=n_blobs
    )

    posterior_mus, posterior_covs = normal_normal_posterior_full_cov_batched_flexible_prior(
        datapoint_positions, blob_assignments, prior_blob_means, prior_variance, blob_covs
    )

    has_points = N_l > 0
    sampled_means = genjax.mv_normal.sample(posterior_key, posterior_mus, posterior_covs)

    prior_cov = jnp.eye(3) * prior_variance
    prior_samples = genjax.mv_normal.sample(
        prior_key,
        prior_blob_means,
        jnp.tile(prior_cov[None, :, :], (n_blobs, 1, 1))
    )

    posterior_blob_means = jnp.where(has_points[:, None], sampled_means, prior_samples)

    return genmatter_state.replace({'blobs_state': {'blob_means': posterior_blob_means}})


def gibbs_blob_vel_means_ablation(key, genmatter_state):
    """Blob velocity means with zero-mean Gaussian prior (sigma_V); no hyperblob coupling."""
    posterior_key, _ = jax.random.split(key)

    datapoint_vels = genmatter_state.datapoints_state.datapoint_vels
    blob_assignments = genmatter_state.datapoints_state.blob_assignments
    likelihood_blob_vel_covs = genmatter_state.blobs_state.blob_vel_covs

    n_blobs = genmatter_state.hypers.n_blobs
    prior_blob_vel_means = jnp.zeros((n_blobs, 3))
    prior_variance = genmatter_state.hypers.sigma_V

    N_l = jax.ops.segment_sum(
        jnp.ones(datapoint_vels.shape[0], dtype=datapoint_vels.dtype),
        blob_assignments,
        num_segments=n_blobs
    )

    posterior_mus, posterior_covs = normal_normal_posterior_full_cov_batched_flexible_prior(
        datapoint_vels, blob_assignments, prior_blob_vel_means, prior_variance, likelihood_blob_vel_covs
    )

    has_points = N_l > 0
    sampled_vel_means = genjax.mv_normal.sample(posterior_key, posterior_mus, posterior_covs)
    zero_velocities = jnp.zeros_like(posterior_mus)
    posterior_blob_vel_means = jnp.where(has_points[:, None], sampled_vel_means, zero_velocities)

    return genmatter_state.replace({'blobs_state': {'blob_vel_means': posterior_blob_vel_means}})


# Gibbs #11: Update hyperblob rotation velocities
def gibbs_hyperblob_rot(key, genmatter_state: GenMatter_State, use_weighted_blobs=False):
    posterior_key, _ = jax.random.split(key)

    hypers = genmatter_state.hypers
    hyperblobs_state = genmatter_state.hyperblobs_state
    blob_means = genmatter_state.blobs_state.blob_means
    blob_vel_means = genmatter_state.blobs_state.blob_vel_means
    blob_weights = genmatter_state.blobs_state.blob_weights
    hyperblob_assignments = genmatter_state.blobs_state.hyperblob_assignments
    hyperblob_trans_vels = hyperblobs_state.hyperblob_trans_vels

    n_hyperblobs = hypers.n_hyperblobs
    num_blobs = hypers.n_blobs
    rot_support = hypers.discrete_rotation.support  # [M, 3, 3]
    rot_logprobs = hypers.discrete_rotation.logprobs  # [M] - prior logprobs
    sigma_V = hypers.sigma_V
    
    # Create pseudocounts based on blob weights if use_weighted_blobs is True
    blob_pseudocounts = jnp.where(use_weighted_blobs, 
                                 blob_weights * num_blobs, 
                                 jnp.ones_like(hyperblob_assignments, dtype=jnp.float32))
    
    # Create a mask for each hyperblob indicating which blobs belong to it
    hyperblob_mask = (hyperblob_assignments[None, :] == jnp.arange(n_hyperblobs)[:, None])  # [K, n_blobs]

    # First compute the (R_k - I) term for each rotation candidate
    # Shape: [M, 3, 3]
    rot_minus_eye = rot_support - jnp.eye(3)
    
    # Pre-compute position differences (μ_ℓ - μ_k) for all blobs
    hyperblob_means_per_blob = hyperblobs_state.hyperblob_means[hyperblob_assignments]  # [n_blobs, 3]
    position_difference = blob_means - hyperblob_means_per_blob  # [n_blobs, 3]
    
    # Pre-compute translation component for each blob
    trans_per_blob = hyperblob_trans_vels[hyperblob_assignments]  # [n_blobs, 3]
    
    # Function to compute log probabilities for all rotation candidates for a single hyperblob
    def compute_rotation_logprobs_for_hyperblob(hk, mask):
        # Reference all blobs (we'll use the mask later)
        all_blob_velocities = blob_vel_means  # [n_blobs, 3]
        all_position_differences = position_difference  # [n_blobs, 3]
        all_trans_components = trans_per_blob  # [n_blobs, 3]
        
        # For each rotation candidate and each blob, compute the expected velocity
        # expected velocity = t_k + (R_k - I)(μ_ℓ - μ_k)
    
        
        # Now compute the rotation effect for each candidate and each blob
        # Shape: [M, n_blobs, 3]
        rotation_effect = jnp.einsum('mij,nj->mni', rot_minus_eye, all_position_differences)
        
        # Compute full expected velocity for each candidate and each blob
        # Shape: [M, n_blobs, 3]
        expected_velocities = all_trans_components[None, :, :] + rotation_effect
        
        # Compute log probability: log N(v_ℓ | expected_velocity, σ²I)
        # Shape: [M, n_blobs]
        squared_diff = jnp.sum((all_blob_velocities[None, :, :] - expected_velocities) ** 2, axis=-1)
        log_liks = (-0.5 * squared_diff / sigma_V) - (1.5 * jnp.log(2 * jnp.pi * sigma_V))
        
        # Apply the mask to only include blobs belonging to this hyperblob
        # and weight by pseudocounts (adding log of pseudocounts since we're in log space)
        # Shape: [M, n_blobs]
        masked_log_liks = jnp.where(mask[None, :], log_liks + jnp.log(blob_pseudocounts[None, :]), 0.0)
        
        # Sum log likelihoods across all blobs for each rotation candidate
        # Shape: [M]
        sum_log_liks = jnp.sum(masked_log_liks, axis=1)
        
        # Add the prior log probability for each rotation candidate
        # Shape: [M]
        return sum_log_liks + rot_logprobs
    
    # Compute log probabilities for all hyperblobs and all rotation candidates
    logprobs_per_hk = jax.vmap(compute_rotation_logprobs_for_hyperblob)(
        jnp.arange(n_hyperblobs),
        hyperblob_mask
    )  # [K, M]
    
    # Sample from the computed distributions
    sampled_indices = genjax.categorical.sample(posterior_key, logits=logprobs_per_hk)  # [K]
    updated_rotations = rot_support[sampled_indices]  # [K, 3, 3]

    return genmatter_state.replace({'hyperblobs_state': {'hyperblob_rot_vels': updated_rotations}})

# Gibbs #12: Update hyperblob translation velocities 
def gibbs_hyperblob_trans(key, genmatter_state: GenMatter_State, use_weighted_blobs=False):
    posterior_key, _ = jax.random.split(key)

    hypers = genmatter_state.hypers
    hyperblobs_state = genmatter_state.hyperblobs_state
    blob_means = genmatter_state.blobs_state.blob_means
    blob_vel_means = genmatter_state.blobs_state.blob_vel_means
    blob_weights = genmatter_state.blobs_state.blob_weights
    hyperblob_assignments = genmatter_state.blobs_state.hyperblob_assignments
    hyperblob_rot_vels = hyperblobs_state.hyperblob_rot_vels

    n_hyperblobs = hypers.n_hyperblobs
    n_blobs = hypers.n_blobs
    trans_support = hypers.discrete_translation.support  # [M, 3]
    trans_logprobs = hypers.discrete_translation.logprobs  # [M] - prior logprobs
    sigma_V = hypers.sigma_V

    # Create pseudocounts based on blob weights if use_weighted_blobs is True
    blob_pseudocounts = jnp.where(use_weighted_blobs, 
                                 blob_weights * n_blobs, 
                                 jnp.ones_like(hyperblob_assignments, dtype=jnp.float32))
    
    # Create a mask for each hyperblob indicating which blobs belong to it
    hyperblob_mask = (hyperblob_assignments[None, :] == jnp.arange(n_hyperblobs)[:, None])  # [K, n_blobs]
    
    # Pre-compute the rotation effect term (R_k - I)(μ_ℓ - μ_k) for all blobs
    eye_matrices = jnp.repeat(jnp.eye(3)[None, ...], n_blobs, axis=0)
    
    # Use hyperblob assignments to get the right hyperblob parameters for each blob
    rot_matrices_per_blob = hyperblob_rot_vels[hyperblob_assignments]  # [n_blobs, 3, 3]
    hyperblob_means_per_blob = hyperblobs_state.hyperblob_means[hyperblob_assignments]  # [n_blobs, 3]
    
    # (R_k - I) term for each blob
    rot_difference = rot_matrices_per_blob - eye_matrices  # [n_blobs, 3, 3]
    
    # (μ_ℓ - μ_k) term for each blob
    position_difference = blob_means - hyperblob_means_per_blob  # [n_blobs, 3]
    
    # Pre-compute (R_k - I)(μ_ℓ - μ_k) for each blob
    rotation_effect = jnp.einsum('nij,nj->ni', rot_difference, position_difference)  # [n_blobs, 3]
    
    # Function to compute log probabilities for all translation candidates for a single hyperblob
    def compute_translation_logprobs_for_hyperblob(hk, mask):
        # Get all blobs (we'll use the mask later)
        all_blob_velocities = blob_vel_means  # [n_blobs, 3]
        all_rotation_effects = rotation_effect  # [n_blobs, 3]
        
        # For each candidate, compute expected velocity for each blob
        # expected velocity = t_k + (R_k - I)(μ_ℓ - μ_k)
        # Shape: [M, n_blobs, 3]
        expected_velocities = trans_support[:, None, :] + all_rotation_effects[None, :, :]
        
        # Compute log probability: log N(v_ℓ | expected_velocity, σ²I)
        # Shape: [M, n_blobs]
        squared_diff = jnp.sum((all_blob_velocities[None, :, :] - expected_velocities) ** 2, axis=-1)
        log_liks = (-0.5 * squared_diff / sigma_V) - (1.5 * jnp.log(2 * jnp.pi * sigma_V))
        
        # Apply the mask to only include blobs belonging to this hyperblob
        # Shape: [M, n_blobs]
        masked_log_liks = jnp.where(mask[None, :], log_liks + jnp.log(blob_pseudocounts[None, :]), 0.0)
        
        # Sum log likelihoods across all blobs for each translation candidate
        # Shape: [M]
        sum_log_liks = jnp.sum(masked_log_liks, axis=1)
        
        # Add the prior log probability for each translation candidate
        # Shape: [M]
        return sum_log_liks + trans_logprobs
    
    
    # Compute log probabilities for all hyperblobs and all translation candidates
    logprobs_per_hk = jax.vmap(compute_translation_logprobs_for_hyperblob)(
        jnp.arange(n_hyperblobs),
        hyperblob_mask
    )  # [K, M]
    
    # Sample from the computed distributions
    sampled_indices = genjax.categorical.sample(posterior_key, logits=logprobs_per_hk)  # [K]
    updated_translations = trans_support[sampled_indices]  # [K, 3]

    return genmatter_state.replace({'hyperblobs_state': {'hyperblob_trans_vels': updated_translations}})


def gibbs_transitioned_blob_means(key, genmatter_state, prior_means, prior_covs):
    posterior_key, _ = jax.random.split(key)

    # Data
    datapoint_positions = genmatter_state.datapoints_state.datapoint_positions  # [N, 3]
    blob_assignments = genmatter_state.datapoints_state.blob_assignments        # [N]
    blob_vel_means = genmatter_state.blobs_state.blob_vel_means                 # [L, 3]
    hyperblob_assignments = genmatter_state.blobs_state.hyperblob_assignments   # [L]
    blob_covs = genmatter_state.blobs_state.blob_covs                           # [L, 3, 3]

    # Hyperblob info
    mu_H = genmatter_state.hyperblobs_state.hyperblob_means                     # [K, 3]
    trans_vels = genmatter_state.hyperblobs_state.hyperblob_trans_vels         # [K, 3]
    rot_vels = genmatter_state.hyperblobs_state.hyperblob_rot_vels             # [K, 3, 3]

    # Hyperparams
    sigmaV = genmatter_state.hypers.sigma_V
    L = genmatter_state.hypers.n_blobs
    d = genmatter_state.blobs_state.blob_means.shape[-1] # [L,3] --> [3]

    # Assign hyperblob means to blobs
    muH_l = prior_means  # [L, 3]

    # Prior precision
    hyperblob_cov_inv = jnp.linalg.inv(prior_covs)  # [L, 3, 3]

    # Data terms (from datapoints)
    N_l = jax.ops.segment_sum(jnp.ones(datapoint_positions.shape[0]), blob_assignments, num_segments=L) # [L]
    S_l = stable_segment_sum(datapoint_positions, blob_assignments, L) # [L, 3]

    # Likelihood precision (from blob covariances)
    blob_cov_inv = jnp.linalg.inv(blob_covs)  # [L, 3, 3]

    # Velocity affine likelihood
    A_l = rot_vels[hyperblob_assignments] - jnp.eye(d)  # [L, 3, 3]
    b_l = trans_vels[hyperblob_assignments] - jnp.einsum("lij,lj->li", A_l, muH_l)  # [L, 3]
    v_l = blob_vel_means # [L, 3]
    residuals = v_l - b_l  # [L, 3]

    lhs_affine = jnp.einsum("lij,lik->ljk", A_l, A_l)    # [L, 3, 3]
    rhs_affine = jnp.einsum("lij,li->lj", A_l, residuals)  # [L, 3]

    # Handle empty blobs by creating a mask
    has_points = N_l > 0
    # Use the mask to zero out the contribution from empty blobs
    N_l_safe = jnp.where(has_points, N_l, 0.0)

    # Combine into posterior precision and weighted mean
    P_post = (
        hyperblob_cov_inv +  # prior from hyperblob covariance [L, 3, 3]
        blob_cov_inv * N_l_safe[:, None, None] +  # likelihood from datapoints [L, 3, 3]
        lhs_affine / sigmaV  # velocity model [L, 3, 3]
    )  # [L, 3, 3]

    weighted_mean = (
        jnp.einsum("lij,lj->li", hyperblob_cov_inv, muH_l) +  # prior with hyperblob covariance
        jnp.einsum("lij,lj->li", blob_cov_inv, S_l) +         # datapoint positions
        rhs_affine / sigmaV                                   # velocity affine
    )  # [L, 3]

    # Solve for posterior mean
    cov_post = jnp.linalg.inv(P_post)  # [L, 3, 3]
    mean_post = jnp.einsum("lij,lj->li", cov_post, weighted_mean)  # [L, 3]

    # For empty blobs, fall back to the prior mean and covariance
    mean_post = jnp.where(has_points[:, None], mean_post, muH_l)
    # For empty blobs, use the prior covariance instead of posterior covariance
    cov_post = jnp.where(has_points[:, None, None], cov_post, prior_covs)
    
    # Sample new blob means
    new_blob_means = genjax.mv_normal.sample(posterior_key, mean_post, cov_post)

    return genmatter_state.replace({'blobs_state': {'blob_means': new_blob_means}})


@jax.jit
def f_gibbs_sweep(carry, gibbs_sweep_idx):
    key, genmatter_state, gibbs_dials, num_gibbs_inner_loops, use_weighted_blobs = carry
    # jprint("MCMC Gibbs sweep {}/{}", gibbs_sweep_idx+1, NUM_GIBBS_SWEEPS)
    
    # Datapoint-related updates 
    def datapoint_update_loop(i, state_key_tuple):
        genmatter_state, key = state_key_tuple
        # Gibbs #3: Update blob assignments
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["blob_assignments"], gibbs_blob_assignments_batched, empty_gibbs, gibbs_key, genmatter_state)
        return (genmatter_state, key)
    
    
    # Blob-related updates 
    def blob_update_loop(i, state_key_tuple):
        genmatter_state, key = state_key_tuple
        # Gibbs #1: Update blob weights
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["blob_weights"], gibbs_blob_weights, empty_gibbs, gibbs_key, genmatter_state)
        
        # Gibbs #4: Update hyperblob assignments
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["hyperblob_assignments"], gibbs_hyperblob_assignments, empty_gibbs, gibbs_key, genmatter_state)
        
        # Gibbs #6: Update blob covariances
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["blob_covs"], gibbs_blob_covs, empty_gibbs, gibbs_key, genmatter_state)
        
        # Gibbs #7: Update blob velocity covariances
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["blob_vel_covs"], gibbs_blob_vel_covs, empty_gibbs, gibbs_key, genmatter_state)
        
        # Gibbs #8: Update blob velocity means
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["blob_vel_means"], gibbs_blob_vel_means, empty_gibbs, gibbs_key, genmatter_state)
        
        # Gibbs #10: Update blob means
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["blob_means"], gibbs_blob_means, empty_gibbs, gibbs_key, genmatter_state)
        
        return (genmatter_state, key)
    
    # Hyperblob-related updates 
    def hyperblob_update_loop(i, state_key_tuple):
        genmatter_state, key, use_weighted_blobs = state_key_tuple
        # Gibbs #2: Update hyperblob weights
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["hyperblob_weights"], gibbs_hyperblob_weights, empty_gibbs_weighted, gibbs_key, genmatter_state, use_weighted_blobs)
        
        # Gibbs #5: Update hyperblob covariances
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["hyperblob_covs"], gibbs_hyperblob_covs, empty_gibbs_weighted, gibbs_key, genmatter_state, use_weighted_blobs)
        
        # Gibbs #9: Update hyperblob means
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["hyperblob_means"], gibbs_hyperblob_means, empty_gibbs_weighted, gibbs_key, genmatter_state, use_weighted_blobs)

        # Gibbs #11: Update hyperblob rotation velocities
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["hyperblob_rot_vels"], gibbs_hyperblob_rot, empty_gibbs_weighted, gibbs_key, genmatter_state, use_weighted_blobs)

        # Gibbs #12: Update hyperblob translation velocities
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["hyperblob_trans_vels"], gibbs_hyperblob_trans, empty_gibbs_weighted, gibbs_key, genmatter_state, use_weighted_blobs)
        
        return (genmatter_state, key, use_weighted_blobs)
    
    genmatter_state, key = jax.lax.fori_loop(0, num_gibbs_inner_loops, datapoint_update_loop, (genmatter_state, key))
    genmatter_state, key = jax.lax.fori_loop(0, num_gibbs_inner_loops, blob_update_loop, (genmatter_state, key))
    genmatter_state, key, _ = jax.lax.fori_loop(0, num_gibbs_inner_loops, hyperblob_update_loop, (genmatter_state, key, use_weighted_blobs))
    
    return (key, genmatter_state, gibbs_dials, num_gibbs_inner_loops, use_weighted_blobs), genmatter_TraceWrapper(force_retval=genmatter_state)

def genmatter_full_gibbs(key, init_genmatter_state, num_gibbs_sweeps, gibbs_dials, use_weighted_blobs=False, num_gibbs_inner_loops=1):
    _, stacked_genmatter_wtrs = jax.lax.scan(f_gibbs_sweep, (key, init_genmatter_state, gibbs_dials, num_gibbs_inner_loops, use_weighted_blobs), jnp.arange(num_gibbs_sweeps))
    gibbs_wtrs = GenMatter_Gibbs_TraceWrapper(genmatter_TraceWrapper(force_retval=init_genmatter_state), stacked_genmatter_wtrs)
    return gibbs_wtrs


def blob_tracking_gibbs(key, genmatter_state : GenMatter_State):

    def update_blob_assignments_position_only(i, carry):
        key, genmatter_state = carry
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_assignments_batched(gibbs_key, genmatter_state, position_only = True, disable_outlier_prob = True)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_weights(gibbs_key, genmatter_state)
        return key, genmatter_state

    def update_blob_assignments_position_w_outlier(i, carry):
        key, genmatter_state = carry
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_assignments_batched(gibbs_key, genmatter_state, position_only = True, disable_outlier_prob = False)
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

    def update_blob_means_with_prior(i, carry):
        key, genmatter_state = carry
        key, gibbs_key = jax.random.split(key)
        genmatter_state = gibbs_blob_means(gibbs_key, genmatter_state)
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
    
    # Increased iterations for better convergence per tracking timestep
    key, genmatter_state = jax.lax.fori_loop(0, 3, hyperblob_update_loop, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 3, update_blob_assignments_position_only, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 10, update_blob_means_with_prior, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 3, update_blob_assignments_position_w_outlier, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 10, update_blob_velocities, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 10, update_blob_velocity_covariances, (key, genmatter_state))
    key, genmatter_state = jax.lax.fori_loop(0, 3, hyperblob_update_loop, (key, genmatter_state))
    
    return genmatter_state

@jax.jit
def f_tracking_sweep(carry, timestep_idx):
    key, genmatter_state, tracked_points, tracked_motion_vectors = carry
    
    # Update blob means with velocity
    next_blob_means = genmatter_state.blobs_state.blob_vel_means + genmatter_state.blobs_state.blob_means
    genmatter_state = genmatter_state.replace({'blobs_state': {'blob_means': next_blob_means}})
    
    # Update datapoint positions for current timestep
    genmatter_state = genmatter_state.replace({
        'datapoints_state': {
            'datapoint_positions': tracked_points[timestep_idx],
            'datapoint_vels': tracked_motion_vectors[timestep_idx]
        }
    })
    
    # Run Gibbs sampling
    key, gibbs_key = jax.random.split(key)
    genmatter_state = blob_tracking_gibbs(gibbs_key, genmatter_state)
    
    return (key, genmatter_state, tracked_points, tracked_motion_vectors), genmatter_TraceWrapper(force_retval=genmatter_state)

def genmatter_tracking_gibbs(key, init_genmatter_state, tracked_points, tracked_motion_vectors, outlier_prob = 1e-24):
    # Set outlier probability to very low value
    init_genmatter_state = init_genmatter_state.replace({'hypers': {'outlier_prob': f_(outlier_prob)}})
    
    # Create timestep indices (skip the first frame since we already have it)
    timestep_indices = jnp.arange(1, len(tracked_points)-1)
    
    # Run the scan over all timesteps with progress bar
    _, stacked_genmatter_wtrs = jax.lax.scan(
        f_tracking_sweep, 
        (key, init_genmatter_state, tracked_points, tracked_motion_vectors), 
        timestep_indices,
        unroll=1
    )
    
    # Create full trace wrapper including initial state
    tracking_wtrs = GenMatter_Gibbs_TraceWrapper(
        genmatter_TraceWrapper(force_retval=init_genmatter_state), 
        stacked_genmatter_wtrs
    )
    
    return tracking_wtrs
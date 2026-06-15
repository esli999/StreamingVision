import genjax
import jax
import numpy as np
import jax.numpy as jnp
from genjax import Const, gen, Pytree
from tensorflow_probability.substrates.jax import distributions as tfd
from .utils import generate_rotation_grid
from .core_types import (
    Precomputed_DiscreteDistribution, discrete_categorical, inverse_wishart, StaticJnp, f_, i_, a_
)
from .datatypes import Super_Pytree

from .trace_wrappers import __genmatter_TraceWrapper__, genmatter_TraceWrapper

@Pytree.dataclass
class GenMatter_Hyperparams(Super_Pytree):
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
        max_radius = kwargs["translation_max_radius"].v
        num_radii = kwargs["translation_num_radii_cells"].v
        theta_step_deg = kwargs["translation_theta_step_deg"].v
        scale = kwargs["translation_gaussian_scale"].v

        radii_positive = jnp.linspace(0.0, max_radius, num_radii)[1:]

        theta_step = jnp.deg2rad(theta_step_deg)
        theta = jnp.arange(0, jnp.pi + theta_step, theta_step)
        directions = []
        for th in theta:
            if jnp.isclose(jnp.sin(th), 0.0):
                x = 0.0
                y = 0.0
                z = jnp.cos(th)
                directions.append(jnp.stack([x, y, z]))
            else:
                n_phi = jnp.maximum(1, jnp.ceil(2 * jnp.pi * jnp.sin(th) / theta_step).astype(int))
                phis = jnp.linspace(0, 2*jnp.pi, n_phi, endpoint=False)
                for phi in phis:
                    x = jnp.sin(th) * jnp.cos(phi)
                    y = jnp.sin(th) * jnp.sin(phi)
                    z = jnp.cos(th)
                    directions.append(jnp.stack([x, y, z]))
        directions = jnp.stack(directions)

        translations_zero = jnp.zeros((1, 3))

        translations_positive = jnp.concatenate(
            [r * directions for r in radii_positive],
            axis=0
        )

        translation_support = jnp.concatenate([translations_zero, translations_positive], axis=0)
        translation_logprobs = jnp.sum(jax.scipy.stats.norm.logpdf(translation_support, 0.0, scale), axis=-1)

        theta_max_deg = kwargs["rotation_angle_max_deg"].v
        theta_step_deg = kwargs["rotation_angle_step_deg"].v
        kappa = kwargs["rotation_vmf_kappa"].v

        directions_rot, rotation_matrices = generate_rotation_grid(theta_max_deg=theta_max_deg, theta_step_deg=theta_step_deg)
        nonzero_mask = jnp.any(directions_rot != 0.0, axis=-1)
        directions_rot = directions_rot[nonzero_mask]
        rotation_matrices = rotation_matrices[nonzero_mask]

        vmf_sampler = tfd.VonMisesFisher(mean_direction=jnp.array([0.0, 0.0, 1.0]), concentration=kappa)
        rotation_logprobs = vmf_sampler.log_prob(a_(directions_rot))

        return cls(
            **kwargs,
            discrete_translation=Precomputed_DiscreteDistribution(support=translation_support, logprobs=translation_logprobs),
            discrete_rotation=Precomputed_DiscreteDistribution(support=a_(rotation_matrices), logprobs=rotation_logprobs)
        )


@Pytree.dataclass
class GenMatter_Hyperblobs_State(Super_Pytree):
    hyperblob_weights: jnp.ndarray
    hyperblob_means: jnp.ndarray
    hyperblob_covs: jnp.ndarray
    hyperblob_trans_vels: jnp.ndarray
    hyperblob_rot_vels: jnp.ndarray

@Pytree.dataclass
class GenMatter_Blobs_State(Super_Pytree):
    hyperblob_assignments: jnp.ndarray
    blob_weights: jnp.ndarray
    blob_means: jnp.ndarray
    blob_covs: jnp.ndarray
    blob_vel_means: jnp.ndarray
    blob_vel_covs: jnp.ndarray

@Pytree.dataclass
class GenMatter_Datapoints_State(Super_Pytree):
    blob_assignments: jnp.ndarray
    datapoint_positions: jnp.ndarray
    datapoint_vels: jnp.ndarray

@Pytree.dataclass
class GenMatter_State(Super_Pytree):
    hypers: GenMatter_Hyperparams
    hyperblobs_state: GenMatter_Hyperblobs_State
    blobs_state: GenMatter_Blobs_State
    datapoints_state: GenMatter_Datapoints_State

@gen
def GenMatter_model_3d(hypers : GenMatter_Hyperparams):
    hyperblobs_state = GenMatter_hyperblobs_model(hypers) @ 'hyperblobs'
    blobs_state = GenMatter_blobs_model(hypers, hyperblobs_state) @ 'blobs'
    datapoints_state = GenMatter_datapoints_model(hypers, blobs_state) @ 'datapoints'

    return GenMatter_State(
        hypers = hypers,
        hyperblobs_state=hyperblobs_state, 
        blobs_state=blobs_state, 
        datapoints_state=datapoints_state
    )

@gen
def GenMatter_hyperblobs_model(hypers : GenMatter_Hyperparams):
    sample_shape = Const((hypers.n_hyperblobs,))
    hyperblob_weights = genjax.dirichlet(jnp.repeat(hypers.alpha, hypers.n_hyperblobs)) @ 'hyperblob_weights'
    hyperblob_covs = inverse_wishart(hypers.nu_H, hypers.Psi_H, sample_shape=sample_shape) @ 'hyperblob_covs'
    hyperblob_means = genjax.normal(hypers.mu_H, jnp.sqrt(hypers.sigma_H), sample_shape=sample_shape) @ 'hyperblob_means' 
    hyperblob_trans_vels = discrete_categorical(hypers.discrete_translation, sample_shape=sample_shape) @ 'hyperblob_trans_vels'
    hyperblob_rot_vels = discrete_categorical(hypers.discrete_rotation, sample_shape=sample_shape) @ 'hyperblob_rot_vels'

    return GenMatter_Hyperblobs_State(
        hyperblob_weights = hyperblob_weights,
        hyperblob_means = hyperblob_means,
        hyperblob_covs = hyperblob_covs,
        hyperblob_trans_vels = hyperblob_trans_vels,
        hyperblob_rot_vels = hyperblob_rot_vels,
    )

@gen
def GenMatter_blobs_model(hypers : GenMatter_Hyperparams, hyperblobs_state : GenMatter_Hyperblobs_State):
    sample_shape = Const((hypers.n_blobs,))
    hyperblob_assignments = genjax.categorical(probs = hyperblobs_state.hyperblob_weights, sample_shape=sample_shape) @ 'hyperblob_assignments'
    blob_weights = genjax.dirichlet(jnp.repeat(hypers.beta, hypers.n_blobs)) @ 'blob_weights'
    assigned_hyperblob_per_blob = hyperblobs_state[hyperblob_assignments]
    blob_covs = inverse_wishart(hypers.nu_B, hypers.Psi_B, sample_shape=sample_shape) @ 'blob_covs'
    blob_means = genjax.mv_normal(assigned_hyperblob_per_blob.hyperblob_means, assigned_hyperblob_per_blob.hyperblob_covs) @ 'blob_means'
    blob_vel_means_ = assigned_hyperblob_per_blob.hyperblob_trans_vels + jnp.einsum('nij,nj->ni', assigned_hyperblob_per_blob.hyperblob_rot_vels - jnp.repeat(jnp.eye(3)[None, ...], hypers.n_blobs, axis=0), blob_means - assigned_hyperblob_per_blob.hyperblob_means)
    blob_vel_means = genjax.normal(blob_vel_means_, jnp.sqrt(hypers.sigma_V)) @ 'blob_vel_means'
    blob_vel_covs = inverse_wishart(hypers.nu_V, hypers.Psi_V, sample_shape=sample_shape) @ 'blob_vel_covs'
    
    return GenMatter_Blobs_State(
        hyperblob_assignments = hyperblob_assignments,
        blob_weights = blob_weights,
        blob_means = blob_means,
        blob_covs = blob_covs,
        blob_vel_means = blob_vel_means,
        blob_vel_covs = blob_vel_covs,
    )

@gen
def GenMatter_datapoints_model(hypers : GenMatter_Hyperparams, blobs_state : GenMatter_Blobs_State):
    blob_assignments = genjax.categorical(probs = blobs_state.blob_weights, sample_shape=Const((hypers.n_datapoints,))) @ 'blob_assignments'
    assigned_blob_per_datapoint = blobs_state[blob_assignments]
    datapoint_positions = genjax.mv_normal(assigned_blob_per_datapoint.blob_means, assigned_blob_per_datapoint.blob_covs) @ 'datapoint_positions'
    datapoint_vels = genjax.mv_normal(assigned_blob_per_datapoint.blob_vel_means, assigned_blob_per_datapoint.blob_vel_covs) @ 'datapoint_vels'

    return GenMatter_Datapoints_State(
        blob_assignments = blob_assignments,
        datapoint_positions = datapoint_positions,
        datapoint_vels = datapoint_vels,
    )
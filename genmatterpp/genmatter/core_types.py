import functools
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from dataclasses import dataclass
from typing import TypeVar, Sequence, Tuple, Union
import genjax
from genjax import exact_density, Const, Pytree
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from .trace_wrappers import Super_Pytree

@dataclass(frozen=True)
class StaticJnp:
    v: jnp.ndarray
    @functools.cached_property
    def np_v(self):
        return np.array(self.v)
    def __eq__(self, other):
        if not isinstance(other, StaticJnp):
            return False
        return bool(np.all(self.np_v == other.np_v))
    @functools.cache
    def __hash__(self):
        return hash(self.np_v.tobytes())

snp = lambda x: StaticJnp(jnp.array(x))
i_ = lambda x: jnp.int32(x)
i8_ = lambda x: jnp.int8(x)
f_ = lambda x: jnp.float32(x)
b_ = lambda x: jnp.bool_(x)
a_ = lambda x: jnp.array(x)

T = TypeVar("T")

@Pytree.dataclass
class Precomputed_DiscreteDistribution(Super_Pytree):
    support: jnp.ndarray
    logprobs: jnp.ndarray

    support_np: np.ndarray = None
    logprobs_np: np.ndarray = None

    def __post_init__(self):
        object.__setattr__(self, "support_np", np.array(self.support))
        object.__setattr__(self, "logprobs_np", np.array(self.logprobs))

    def __eq__(self, other):
        if not isinstance(other, Precomputed_DiscreteDistribution):
            return False
        return bool(np.all(self.support_np == other.support_np) and np.all(self.logprobs_np == other.logprobs_np))
    
    def __hash__(self):
        return hash((
            self.support_np.tobytes(),
            self.logprobs_np.tobytes()
        ))

def discrete_categorical_sample(key, discrete_distribution : Precomputed_DiscreteDistribution, *args, **kwargs):
    sample_shape = Const.unwrap(kwargs.pop("sample_shape", ()))
    return discrete_distribution.support[tfd.Categorical(logits=discrete_distribution.logprobs).sample(seed=key, sample_shape=sample_shape)]

def discrete_categorical_logpdf(v, discrete_distribution : Precomputed_DiscreteDistribution, *args, **kwargs):
    sample_shape = Const.unwrap(kwargs.pop("sample_shape", ()))
    sampled_support_match_bool_extended = (discrete_distribution.support[(slice(None),) + (None,) * len(sample_shape) + (...,)] == v[None, ...])
    sampled_support_match_bool = sampled_support_match_bool_extended.all(tuple(range(1 + len(sample_shape), sampled_support_match_bool_extended.ndim)))
    sampled_support_match_count = sampled_support_match_bool.sum(tuple(range(1, sampled_support_match_bool.ndim)))
    score = jnp.sum(sampled_support_match_count * discrete_distribution.logprobs)
    return score

discrete_categorical = exact_density(discrete_categorical_sample, discrete_categorical_logpdf, "discrete_categorical")

def truncate_eigenval_ratio(matrix, threshold=1e3):
    matrix_shape = matrix.shape
    n = matrix_shape[-1]
    batch_dims = matrix_shape[:-2]
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    eigvals = jnp.where(eigvals < 1e-6, 1e-6, eigvals)
    max_eval = eigvals[..., -1:]
    eigvals_over_max = eigvals / max_eval
    eigvals_over_max = jnp.maximum(eigvals_over_max, 1 / threshold)
    eigvals = eigvals_over_max * max_eval
    diag_eigvals = jnp.zeros((*batch_dims, n, n), dtype=eigvals.dtype)
    diag_indices = jnp.diag_indices(n)
    diag_eigvals = diag_eigvals.at[..., diag_indices[0], diag_indices[1]].set(eigvals)
    new_matrix = jnp.matmul(eigvecs, jnp.matmul(diag_eigvals, jnp.swapaxes(eigvecs, -2, -1)))
    return new_matrix

def wishart_sample(key, nu, Lambda, *args, **kwargs):
    sample_shape = Const.unwrap(kwargs.pop("sample_shape", ()))
    scale_tril = jnp.linalg.cholesky(Lambda)
    dist = tfd.WishartTriL(nu, scale_tril)
    return dist.sample(seed=key, sample_shape=sample_shape)

def wishart_logpdf(x, nu, Lambda, *args, **kwargs):
    scale_tril = jnp.linalg.cholesky(Lambda)
    dist = tfd.WishartTriL(nu, scale_tril)
    return dist.log_prob(x)

wishart = exact_density(wishart_sample, wishart_logpdf, name="wishart")

def inverse_wishart_sample(key, nu, Psi, *args, **kwargs):
    sample_shape = Const.unwrap(kwargs.pop("sample_shape", ()))
    W = wishart.sample(key, nu, jnp.linalg.inv(Psi), sample_shape=Const(sample_shape), *args, **kwargs)
    mtx = truncate_eigenval_ratio(jnp.linalg.inv(W))
    return mtx

def inverse_wishart_logpdf(X, nu, Psi, *args, **kwargs):
    d = X.shape[-1]
    W = jnp.linalg.inv(X)
    wishart_logpdf = wishart.logpdf(W, nu, jnp.linalg.inv(Psi), *args, **kwargs)
    return wishart_logpdf - (d + 1) * jnp.linalg.slogdet(X)[1]

inverse_wishart = exact_density(inverse_wishart_sample, inverse_wishart_logpdf, name="inverse_wishart") 
import functools
import numpy as np
from dataclasses import dataclass, replace
from typing import TypeVar, Sequence, Tuple, Union
from abc import abstractmethod

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import genjax
from genjax import PythonicPytree, exact_density, Pytree, Const

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from .trace_wrappers import __genmatter_TraceWrapper__, Super_Pytree
from .core_types import (
    Precomputed_DiscreteDistribution, discrete_categorical, discrete_categorical_sample, discrete_categorical_logpdf,
    inverse_wishart, inverse_wishart_sample, inverse_wishart_logpdf, wishart, wishart_sample, wishart_logpdf,
    truncate_eigenval_ratio
)
from genjax import Pytree

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

def pytree_concat(pytree_list: Sequence[T], axis: int = 0) -> T:
    """
    Concatenates corresponding JAX arrays in a list of PyTrees along the specified axis.
    Enforces runtime type checking.
    """
    return jtu.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=axis) if isinstance(xs[0], jnp.ndarray) else jnp.array(xs),
        *pytree_list
    )

def pytree_stack(pytree_list: Sequence[T], axis: int = 0) -> T:

    """
    Stacks corresponding JAX arrays in a list of PyTrees along the specified axis.
    """
    return jtu.tree_map(
        lambda *ys: jnp.stack(ys, axis = axis),
    *pytree_list)

def pytree_slice(pytree: T, idx: Union[int, slice, jnp.ndarray, Tuple]) -> T:
    return jtu.tree_map(lambda x: x[idx], pytree)


@Pytree.dataclass
class GenMatter_Gibbs_TraceWrapper(Super_Pytree):
    wtrs: list[__genmatter_TraceWrapper__]
    def __init__(self, init_genmatter_wtr, stacked_genmatter_wtrs):
        self.wtrs = pytree_concat([init_genmatter_wtr.expand_dims(0), stacked_genmatter_wtrs])
        # self.wtrs = [init_genmatter_wtr, *stacked_genmatter_wtrs.unstack()]

    def __getitem__(self, i):
        return self.wtrs[i]
    
    def __len__(self):
        return len(self.wtrs.log_likelihood_)
    
    def __iter__(self):
        return iter(self.wtrs)
    
    def __next__(self):
        return next(self.wtrs)

from .model_3d import GenMatter_State, GenMatter_Hyperblobs_State, GenMatter_Blobs_State, GenMatter_Datapoints_State, GenMatter_Hyperparams
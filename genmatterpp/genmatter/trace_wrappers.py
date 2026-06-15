import jax
import jax.numpy as jnp
from genjax import Pytree
from .utils import *

from typing import TypeVar, Sequence, Tuple, Union
import jax.tree_util as jtu
import numpy as np
from dataclasses import replace

T = TypeVar("T")

class Super_Pytree(Pytree):
    def __getitem__(self: T, idx: Union[int, slice, jnp.ndarray, Tuple]) -> T:
        def safe_index(v, idx):
            if not isinstance(v, jnp.ndarray):
                return v
            if isinstance(idx, tuple):
                if len(idx) > v.ndim:
                    idx = idx[:v.ndim]
            else:
                idx = (idx,)
            try:
                return v[idx]
            except IndexError as e:
                raise IndexError(f"Invalid indexing {idx} for shape {v.shape}") from e
        return jtu.tree_map(lambda v: safe_index(v, idx), self)
    def flatten(self: T) -> T:
        return jtu.tree_map(lambda x: x.flatten() if isinstance(x, jnp.ndarray) else x, self)
    def reshape(self: T, shape: Sequence[int]) -> T:
        def safe_reshape(x, shape):
            if isinstance(x, jnp.ndarray) and jnp.prod(x.shape) == jnp.prod(shape):
                return x.reshape(shape)
            raise ValueError(f"Cannot reshape array of shape {x.shape} to {shape}")
        return jtu.tree_map(lambda x: safe_reshape(x, shape) if isinstance(x, jnp.ndarray) else x, self)
    def expand_dims(self: T, axis: int) -> T:
        return jtu.tree_map(lambda x: jnp.expand_dims(x, axis) if isinstance(x, jnp.ndarray) else x, self)
    def reshape_first_dim(self: T, shape: Sequence[int]) -> T:
        def safe_reshape(x, shape):
            if isinstance(x, jnp.ndarray) and jnp.prod(x.shape) % jnp.prod(shape) == 0:
                return x.reshape((*shape, *x.shape[1:]))
            raise ValueError(f"Cannot reshape first dimension of {x.shape} to {shape}")
        return jtu.tree_map(lambda x: safe_reshape(x, shape) if isinstance(x, jnp.ndarray) else x, self)
    def split(self: T, num_splits: int, axis: int = 0) -> Sequence[T]:
        return [jtu.tree_map(lambda x: jnp.array_split(x, num_splits, axis=axis)[i], self) for i in range(num_splits)]
    def unstack(self: T, axis: int = 0) -> Sequence[T]:
        return [jtu.tree_map(lambda x: x[i], self) for i in range(self.shape[axis])]
    def swapaxes(self: T, axis1: int, axis2: int):
        def swapaxes_(x):
            if hasattr(x, 'shape') and len(x.shape) > max(axis1, axis2):
                return jnp.swapaxes(x, axis1, axis2)
            return x 
        return jax.tree_util.tree_map(swapaxes_, self)
    def replace(self, *args, do_replace_none=False, **kwargs):
        if kwargs:
            if args:
                raise ValueError("Cannot use both positional and keyword arguments")
            return self.replace(kwargs)
        if len(args) != 1 or not isinstance(args[0], dict):
            raise ValueError("Expected a single dictionary argument")
        update_dict = args[0]
        processed_updates = {}
        for field_name, new_value in update_dict.items():
            if new_value is None and not do_replace_none:
                continue
            current_value = getattr(self, field_name)
            if isinstance(new_value, dict) and isinstance(current_value, Super_Pytree):
                processed_updates[field_name] = current_value.replace(new_value)
            else:
                processed_updates[field_name] = new_value
        return replace(self, **processed_updates)

def pytree_concat(pytree_list: Sequence[T], axis: int = 0) -> T:
    return jtu.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=axis) if isinstance(xs[0], jnp.ndarray) else jnp.array(xs),
        *pytree_list
    )
def pytree_stack(pytree_list: Sequence[T], axis: int = 0) -> T:
    return jtu.tree_map(
        lambda *ys: jnp.stack(ys, axis = axis),
    *pytree_list)
def pytree_slice(pytree: T, idx: Union[int, slice, jnp.ndarray, Tuple]) -> T:
    return jtu.tree_map(lambda x: x[idx], pytree)

@Pytree.dataclass
class __genmatter_TraceWrapper__(Super_Pytree):
    retval: any
    log_likelihood_: any
    
    def __init__(self, trace = None, force_retval=None, force_log_likelihood=None):
        self.retval = trace.get_retval() if force_retval is None else force_retval
        self.log_likelihood_ = None if force_log_likelihood is None else force_log_likelihood
    
    def extract_log_likelihood(self, trace):
        raise NotImplementedError("extract_log_likelihood() must be implemented in a subclass.")
    
    def hyperblob_means(self, hyperblob_id: int):
        raise NotImplementedError("hyperblob_means() must be implemented in a subclass.")
    
    def hyperblob_covariance(self, hyperblob_id: int):
        raise NotImplementedError("hyperblob_covariance() must be implemented in a subclass.")
    
    def hyperblob_mixture_weight(self, hyperblob_id: int):
        raise NotImplementedError("hyperblob_mixture_weight() must be implemented in a subclass.")
    
    def hyperblob_trans_vel(self, hyperblob_id: int):
        raise NotImplementedError("hyperblob_trans_vel() must be implemented in a subclass.")
    
    def hyperblob_rot_vel(self, hyperblob_id: int):
        raise NotImplementedError("hyperblob_rot_vel() must be implemented in a subclass.")
    
    def hyperblob_blobs(self, hyperblob_id: int):
        raise NotImplementedError("hyperblob_blobs() must be implemented in a subclass.")
    
    def all_hyperblob_means(self):
        raise NotImplementedError("all_hyperblob_means() must be implemented in a subclass.")
    
    def all_hyperblob_covariances(self):
        raise NotImplementedError("all_hyperblob_covariances() must be implemented in a subclass.")
    
    def all_hyperblob_mixture_weights(self):
        raise NotImplementedError("all_hyperblob_mixture_weights() must be implemented in a subclass.")
    
    def all_hyperblob_trans_vels(self):
        raise NotImplementedError("all_hyperblob_trans_vels() must be implemented in a subclass.")
    
    def all_hyperblob_rot_vels(self):
        raise NotImplementedError("all_hyperblob_rot_vels() must be implemented in a subclass.")
    
    def blob_mean(self, blob_id: int):
        raise NotImplementedError("blob_mean() must be implemented in a subclass.")
    
    def blob_covariance(self, blob_id: int):
        raise NotImplementedError("blob_covariance() must be implemented in a subclass.")
    
    def blob_velocity_mean(self, blob_id: int):
        raise NotImplementedError("blob_velocity_mean() must be implemented in a subclass.")
    
    def blob_velocity_covariance(self, blob_id: int):
        raise NotImplementedError("blob_velocity_covariance() must be implemented in a subclass.")
    
    def blob_mixture_weight(self, blob_id: int):
        raise NotImplementedError("blob_mixture_weight() must be implemented in a subclass.")
    
    def blob_hyperblob(self, blob_id: int):
        raise NotImplementedError("blob_hyperblob() must be implemented in a subclass.")
    
    def blob_datapoints(self, blob_id: int):
        raise NotImplementedError("blob_datapoints() must be implemented in a subclass.")
    
    def blob_datapoints_indices(self, blob_id: int):
        raise NotImplementedError("blob_datapoints_indices() must be implemented in a subclass.")
    
    def all_blob_means(self):
        raise NotImplementedError("all_blob_means() must be implemented in a subclass.")
    
    def all_blob_covariances(self):
        raise NotImplementedError("all_blob_covariances() must be implemented in a subclass.")
    
    def all_blob_velocity_means(self):
        raise NotImplementedError("all_blob_velocity_means() must be implemented in a subclass.")
    
    def all_blob_velocity_covariances(self):
        raise NotImplementedError("all_blob_velocity_covariances() must be implemented in a subclass.")
    
    def all_blob_mixture_weights(self):
        raise NotImplementedError("all_blob_mixture_weights() must be implemented in a subclass.")
    
    def all_blob_hyperblob_assignments(self):
        raise NotImplementedError("all_blob_hyperblob_assignments() must be implemented in a subclass.")
    
    def datapoint_position(self, datapoint_id: int):
        raise NotImplementedError("datapoint_position() must be implemented in a subclass.")
    
    def datapoint_velocity(self, datapoint_id: int):
        raise NotImplementedError("datapoint_velocity() must be implemented in a subclass.")
    
    def datapoint_blob(self, datapoint_id: int):
        raise NotImplementedError("datapoint_blob() must be implemented in a subclass.")
    
    def datapoint_hyperblob(self, datapoint_id: int):
        raise NotImplementedError("datapoint_hyperblob() must be implemented in a subclass.")
    
    def all_datapoint_positions(self):
        raise NotImplementedError("all_datapoint_positions() must be implemented in a subclass.")
    
    def all_datapoint_velocities(self):
        raise NotImplementedError("all_datapoint_velocities() must be implemented in a subclass.")
    
    def all_datapoint_blob_assignments(self):
        raise NotImplementedError("all_datapoint_blob_assignments() must be implemented in a subclass.")
    
    def all_datapoint_hyperblob_assignments(self):
        raise NotImplementedError("all_datapoint_hyperblob_assignments() must be implemented in a subclass.")
    
    @property
    def num_hyperblobs(self):
        raise NotImplementedError("num_hyperblobs property must be implemented in a subclass.")
    
    @property
    def num_blobs(self):
        raise NotImplementedError("num_blobs property must be implemented in a subclass.")
    
    @property
    def num_datapoints(self):
        raise NotImplementedError("num_datapoints property must be implemented in a subclass.")
    
    @property
    def log_likelihood(self):
        raise NotImplementedError("log_likelihood property must be implemented in a subclass.")

from genjax import Pytree

@Pytree.dataclass(init=True, has_implicitly_inherited_fields=True)
class genmatter_TraceWrapper(__genmatter_TraceWrapper__):
    def __init__(self, trace = None, force_retval=None, force_log_likelihood=None):
        super().__init__(trace, force_retval, force_log_likelihood)
        self.log_likelihood_ = self.extract_log_likelihood(trace) if force_log_likelihood is None else force_log_likelihood
    
    def extract_log_likelihood(self, trace):
        return jnp.array(0)
    
    def hyperblob_means(self, hyperblob_id: int):
        if hyperblob_id == self.num_hyperblobs:
            raise ValueError("Cannot get means for outlier hyperblob")
        return np.array(self.retval.hyperblobs_state.hyperblob_means)[hyperblob_id]
    
    def hyperblob_covariance(self, hyperblob_id: int):
        if hyperblob_id == self.num_hyperblobs:
            raise ValueError("Cannot get covariance for outlier hyperblob")
        return np.array(self.retval.hyperblobs_state.hyperblob_covs)[hyperblob_id]
    
    def hyperblob_mixture_weight(self, hyperblob_id: int):
        if hyperblob_id == self.num_hyperblobs:
            return float(self.retval.hypers.outlier_prob)
        weights = np.array(self.retval.hyperblobs_state.hyperblob_weights)
        return float(weights[hyperblob_id] * (1 - self.retval.hypers.outlier_prob))
    
    def hyperblob_trans_vel(self, hyperblob_id: int):
        if hyperblob_id == self.num_hyperblobs:
            raise ValueError("Cannot get translation velocity for outlier hyperblob")
        return np.array(self.retval.hyperblobs_state.hyperblob_trans_vels)[hyperblob_id]
    
    def hyperblob_rot_vel(self, hyperblob_id: int):
        if hyperblob_id == self.num_hyperblobs:
            raise ValueError("Cannot get rotation velocity for outlier hyperblob")
        return np.array(self.retval.hyperblobs_state.hyperblob_rot_vels)[hyperblob_id]
    
    def hyperblob_blobs(self, hyperblob_id: int):
        assignments = np.array(self.retval.blobs_state.hyperblob_assignments)
        return np.where(assignments == hyperblob_id)[0].tolist()
    
    def all_hyperblob_means(self):
        return np.array(self.retval.hyperblobs_state.hyperblob_means)
    
    def all_hyperblob_covariances(self):
        return np.array(self.retval.hyperblobs_state.hyperblob_covs)
    
    def all_hyperblob_mixture_weights(self):
        weights = np.array(self.retval.hyperblobs_state.hyperblob_weights)
        adjusted_weights = weights * (1 - self.retval.hypers.outlier_prob)
        return np.append(adjusted_weights, self.retval.hypers.outlier_prob)
    
    def all_hyperblob_trans_vels(self):
        return np.array(self.retval.hyperblobs_state.hyperblob_trans_vels)
    
    def all_hyperblob_rot_vels(self):
        return np.array(self.retval.hyperblobs_state.hyperblob_rot_vels)
    
    def blob_mean(self, blob_id: int):
        if blob_id == self.num_blobs:
            raise ValueError("Cannot get mean for outlier blob")
        return np.array(self.retval.blobs_state.blob_means)[blob_id]
    
    def blob_covariance(self, blob_id: int):
        if blob_id == self.num_blobs:
            raise ValueError("Cannot get covariance for outlier blob")
        return np.array(self.retval.blobs_state.blob_covs)[blob_id]
    
    def blob_velocity_mean(self, blob_id: int):
        if blob_id == self.num_blobs:
            raise ValueError("Cannot get velocity mean for outlier blob")
        return np.array(self.retval.blobs_state.blob_vel_means)[blob_id]
    
    def blob_velocity_covariance(self, blob_id: int):
        if blob_id == self.num_blobs:
            raise ValueError("Cannot get velocity covariance for outlier blob")
        return np.array(self.retval.blobs_state.blob_vel_covs)[blob_id]
    
    def blob_mixture_weight(self, blob_id: int):
        if blob_id == self.num_blobs:
            return float(self.retval.hypers.outlier_prob)
        weights = np.array(self.retval.blobs_state.blob_weights)
        return float(weights[blob_id] * (1 - self.retval.hypers.outlier_prob))
    
    def blob_hyperblob(self, blob_id: int):
        if blob_id == self.num_blobs:
            return self.num_hyperblobs
        return int(np.array(self.retval.blobs_state.hyperblob_assignments)[blob_id])
    
    def blob_datapoints(self, blob_id: int):
        return np.array(self.retval.datapoints_state.datapoint_positions)[self.blob_datapoints_indices(blob_id)]
    
    def blob_datapoints_indices(self, blob_id: int):
        blob_assignments = np.array(self.retval.datapoints_state.blob_assignments)
        return np.where(blob_assignments == blob_id)[0]
    
    def all_blob_means(self):
        return np.array(self.retval.blobs_state.blob_means)
    
    def all_blob_covariances(self):
        return np.array(self.retval.blobs_state.blob_covs)
    
    def all_blob_velocity_means(self):
        return np.array(self.retval.blobs_state.blob_vel_means)
    
    def all_blob_velocity_covariances(self):
        return np.array(self.retval.blobs_state.blob_vel_covs)
    
    def all_blob_mixture_weights(self):
        weights = np.array(self.retval.blobs_state.blob_weights)
        adjusted_weights = weights * (1 - self.retval.hypers.outlier_prob)
        return np.append(adjusted_weights, self.retval.hypers.outlier_prob)
    
    def all_blob_hyperblob_assignments(self):
        assignments = np.array(self.retval.blobs_state.hyperblob_assignments)
        return np.append(assignments, self.num_hyperblobs)
    
    def datapoint_position(self, datapoint_id: int):
        return np.array(self.retval.datapoints_state.datapoint_positions)[datapoint_id]
    
    def datapoint_velocity(self, datapoint_id: int):
        return np.array(self.retval.datapoints_state.datapoint_vels)[datapoint_id]
    
    def datapoint_blob(self, datapoint_id: int):
        blob_assignment = int(np.array(self.retval.datapoints_state.blob_assignments)[datapoint_id])
        return blob_assignment
    
    def datapoint_hyperblob(self, datapoint_id: int):
        blob_id = self.datapoint_blob(datapoint_id)
        return self.blob_hyperblob(blob_id)
    
    def all_datapoint_positions(self):
        return np.array(self.retval.datapoints_state.datapoint_positions)
    
    def all_datapoint_velocities(self):
        return np.array(self.retval.datapoints_state.datapoint_vels)
    
    def all_datapoint_blob_assignments(self):
        return np.array(self.retval.datapoints_state.blob_assignments)
    
    def all_datapoint_hyperblob_assignments(self):
        blob_assignments = self.all_datapoint_blob_assignments()
        result = np.zeros_like(blob_assignments)
        
        regular_mask = blob_assignments < self.num_blobs
        if np.any(regular_mask):
            hyperblob_assignments = np.array(self.retval.blobs_state.hyperblob_assignments)
            result[regular_mask] = hyperblob_assignments[blob_assignments[regular_mask]]
        
        outlier_mask = blob_assignments == self.num_blobs
        if np.any(outlier_mask):
            result[outlier_mask] = self.num_hyperblobs
            
        return result
    
    @property
    def num_hyperblobs(self):
        return self.retval.hypers.n_hyperblobs
    
    @property
    def num_blobs(self):
        return self.retval.hypers.n_blobs
    
    @property
    def num_datapoints(self):
        return self.retval.hypers.n_datapoints
    
    @property
    def log_likelihood(self):
        log_likelihood_arr = self.log_likelihood_
        if log_likelihood_arr.ndim == 0:
            return log_likelihood_arr
        elif log_likelihood_arr.ndim == 1:
            return jnp.sum(log_likelihood_arr)
        elif log_likelihood_arr.ndim == 2:
            return jnp.sum(log_likelihood_arr, axis=1)
        else:
            raise ValueError("log_likelihood_arr has more than 2 dimensions, not accounted for")
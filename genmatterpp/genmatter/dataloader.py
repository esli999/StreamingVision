import jax.numpy as jnp
import os
import numpy as np


def extract_3d_points_and_motion_vectors_data(data_path, name='camel', extract_colors=False):
    # Try both filename patterns
    motion_file = os.path.join(data_path, f'{name}_3d_motion.npz')
    data_file = os.path.join(data_path, f'{name}_3d_data.npz')
    if os.path.exists(motion_file):
        data = np.load(motion_file)
    elif os.path.exists(data_file):
        data = np.load(data_file)
    else:
        raise FileNotFoundError(f"Neither {motion_file} nor {data_file} found")
    
    positions = data['points_3d']
    motion_vectors = data['motion_vectors_3d']
    if extract_colors:
        colors = data['colors'].reshape(data['colors'].shape[0], -1, 3).astype(jnp.float32)
    
    T, H, W, _ = positions.shape
    img_dims = (H, W)
    positions = positions.reshape(T, H*W, 3)
    motion_vectors = motion_vectors.reshape(T, H*W, 3)
    large_motion_indices = np.where(np.linalg.norm(motion_vectors, axis=2) > 0.35)
    row_indices = large_motion_indices[0][1:]
    col_indices = large_motion_indices[1][1:]
    motion_vectors[row_indices, col_indices] = 0
    
    num_tsteps = positions.shape[0]
    
    if extract_colors:
        return positions, motion_vectors, colors, num_tsteps, img_dims
    else:
        return positions, motion_vectors, num_tsteps, img_dims
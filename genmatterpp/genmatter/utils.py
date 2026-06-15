
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans
from genjax import ChoiceMapBuilder as C

def sample_covariance_matrix_numpy(data):
    if data.shape[0] <= 1:
        return np.eye(data.shape[1])
    
    mean = np.mean(data, axis=0)
    centered = data - mean
    
    cov = (centered.T @ centered) / (data.shape[0] - 1)
    
    epsilon = 1e-6
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    clipped_eigenvalues = np.maximum(eigenvalues, epsilon)
    cov = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T
    
    return cov

def spherical_cap_grid_slow_numpy(theta_max_deg, theta_step_deg):
    
    theta_max = np.deg2rad(theta_max_deg)
    theta_step = np.deg2rad(theta_step_deg)

    thetas = np.arange(0, theta_max + theta_step, theta_step)
    directions = []

    for theta in thetas:
        n_phi = np.maximum(1, np.ceil(2 * np.pi * np.sin(theta) / theta_step).astype(int))
        phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        x = np.sin(theta) * np.cos(phis)
        y = np.sin(theta) * np.sin(phis)
        z = np.full_like(x, np.cos(theta))
        directions.append(np.stack([x, y, z], axis=-1))

    return np.concatenate(directions, axis=0)

@jax.jit
def rotation_matrix_from_z_to_vec(vec):
    vec = vec / jnp.linalg.norm(vec)
    z = jnp.array([0.0, 0.0, 1.0])

    def case_z():
        return jnp.eye(3)

    def case_neg_z():
        return jnp.diag(jnp.array([1.0, -1.0, -1.0]))

    def general_case():
        v = jnp.cross(z, vec)
        s = jnp.linalg.norm(v)
        c = jnp.dot(z, vec)

        vx = jnp.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]
        ])
        return jnp.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    return jax.lax.cond(
        jnp.allclose(vec, z),
        case_z,
        lambda: jax.lax.cond(jnp.allclose(vec, -z), case_neg_z, general_case)
    )

def generate_rotation_grid(theta_max_deg=45, theta_step_deg=5):
    directions = spherical_cap_grid_slow_numpy(theta_max_deg, theta_step_deg)
    rotation_matrices = jax.vmap(rotation_matrix_from_z_to_vec)(directions)
    return directions, rotation_matrices

def kabsch_algorithm_local_rotation_and_translation(P, Q):
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    H = P_centered.T @ Q_centered
    
    U, S, Vt = np.linalg.svd(H)
    
    V = Vt.T
    
    d = np.linalg.det(V @ U.T)
    
    diag = np.eye(3)
    if d < 0:
        diag[2, 2] = -1
    R = V @ diag @ U.T
    
    t = centroid_Q - centroid_P

    return R, t

def make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(
    tracked_points,
    num_blobs,
    num_hyperblobs,
    segmentation_mask,
    subsampled_indices=None,
    roi_blob_ratio=1.0,
    frame_idx=0,
    motion_time_offset=1,
    assume_correspondences=False,
    motion_vectors=None,
    verbose=0,
    num_roi_hyperblobs=1,
    num_roi_blobs=None,  # <-- ADDED: Optionally fix the number of roi blobs
):
    """
    If num_roi_blobs is not None, use fixed number of blobs for ROI region, otherwise use ratio-based allocation.
    """
    assert num_roi_hyperblobs <= num_hyperblobs, "num_roi_hyperblobs must be <= num_hyperblobs"
    assert num_hyperblobs - num_roi_hyperblobs >= 0, "num_hyperblobs - num_roi_hyperblobs must be >= 0"
    if num_roi_blobs is not None:
        assert num_blobs >= num_roi_blobs, "num_blobs must be >= num_roi_blobs"

    if assume_correspondences:
        assert motion_time_offset >= 1
        assert frame_idx + motion_time_offset < len(tracked_points)
    else:
        assert motion_vectors is not None, "Motion vectors must be provided when assume_correspondences is False"
        assert motion_vectors.shape == tracked_points.shape, "Motion vectors must have the same shape as tracked_points"

    frame_xyzs = tracked_points[frame_idx]

    if assume_correspondences:
        datapoint_vels = tracked_points[frame_idx + motion_time_offset] - tracked_points[frame_idx]
    else:
        datapoint_vels = motion_vectors[frame_idx]

    # Align ROI mask to subsampled datapoints if subsampling is used
    if subsampled_indices is not None and segmentation_mask is not None and len(segmentation_mask) != len(tracked_points[frame_idx]):
        roi_mask = segmentation_mask[subsampled_indices]
    else:
        roi_mask = segmentation_mask

    roi_points = frame_xyzs[roi_mask]
    non_roi_points = frame_xyzs[~roi_mask]

    roi_vels = datapoint_vels[roi_mask]
    non_roi_vels = datapoint_vels[~roi_mask]

    total_points = len(frame_xyzs)
    roi_points_count = len(roi_points)
    non_roi_points_count = len(non_roi_points)

    # ----------- ROI blob allocation logic --------------
    if num_roi_blobs is not None:
        # Fixed number of ROI blobs, remaining blobs for non-ROI region
        roi_blobs = min(num_roi_blobs, roi_points_count) if roi_points_count > 0 else 0
        non_roi_blobs = num_blobs - roi_blobs
        # Ensure we do not create negative or 0 clusters for non_roi if non_roi_points present
        if non_roi_points_count > 0 and non_roi_blobs < 1:
            non_roi_blobs = 1
            roi_blobs = num_blobs - 1
    else:
        # Use old ratio-based logic
        effective_roi_ratio = (roi_points_count / total_points) * roi_blob_ratio
        effective_total_ratio = effective_roi_ratio + (non_roi_points_count / total_points)
        roi_blobs = max(3, int(num_blobs * effective_roi_ratio / effective_total_ratio))
        non_roi_blobs = max(3, num_blobs - roi_blobs)

    roi_features = np.concatenate([roi_points, roi_vels], axis=1) if roi_points_count > 0 else np.empty((0,6))
    non_roi_features = np.concatenate([non_roi_points, non_roi_vels], axis=1) if non_roi_points_count > 0 else np.empty((0,6))

    # ROI kmeans
    if roi_blobs > 0 and roi_points_count > roi_blobs:
        roi_kmeans = KMeans(n_clusters=roi_blobs, random_state=0, verbose=verbose)
        roi_labels = roi_kmeans.fit_predict(roi_features)
        roi_centroids = roi_kmeans.cluster_centers_[:, :3]
    elif roi_blobs > 0 and roi_points_count > 0:
        roi_labels = np.arange(roi_points_count)
        roi_centroids = roi_points
        roi_blobs = roi_points_count
    else:
        roi_labels = np.array([], dtype=int)
        roi_centroids = np.empty((0,3))
        roi_blobs = 0

    # non-ROI kmeans
    if non_roi_blobs > 0 and non_roi_points_count > non_roi_blobs:
        non_roi_kmeans = KMeans(n_clusters=non_roi_blobs, random_state=0, verbose=verbose)
        non_roi_labels = non_roi_kmeans.fit_predict(non_roi_features)
        non_roi_centroids = non_roi_kmeans.cluster_centers_[:, :3]
    elif non_roi_blobs > 0 and non_roi_points_count > 0:
        non_roi_labels = np.arange(non_roi_points_count)
        non_roi_centroids = non_roi_points
        non_roi_blobs = non_roi_points_count
    else:
        non_roi_labels = np.array([], dtype=int)
        non_roi_centroids = np.empty((0,3))
        non_roi_blobs = 0

    actual_num_blobs = roi_blobs + non_roi_blobs

    blob_labels = np.zeros(len(frame_xyzs), dtype=int)
    if roi_blobs > 0:
        blob_labels[roi_mask] = roi_labels
    if non_roi_blobs > 0:
        blob_labels[~roi_mask] = non_roi_labels + roi_blobs

    if roi_centroids.shape[0] > 0 and non_roi_centroids.shape[0] > 0:
        blob_centroids = np.vstack([roi_centroids, non_roi_centroids])
    elif roi_centroids.shape[0] > 0:
        blob_centroids = roi_centroids
    elif non_roi_centroids.shape[0] > 0:
        blob_centroids = non_roi_centroids
    else:
        blob_centroids = np.empty((0,3))

    blob_empirical_covs = []
    for i in range(actual_num_blobs):
        points_in_blob = frame_xyzs[blob_labels == i]
        if len(points_in_blob) > 3:
            blob_empirical_covs.append(sample_covariance_matrix_numpy(points_in_blob))
        else:
            blob_empirical_covs.append(np.eye(3) * 0.01)
    if len(blob_empirical_covs) > 0:
        blob_empirical_covs = np.stack(blob_empirical_covs)
    else:
        blob_empirical_covs = np.empty((0,3,3))

    blob_mixture_weights = np.array([np.sum(blob_labels == i) / len(frame_xyzs) for i in range(actual_num_blobs)])

    hyperblob_labels = np.zeros(actual_num_blobs, dtype=int)

    roi_blob_indices = np.arange(roi_blobs)
    roi_blob_position_motion_matrix = np.zeros((len(roi_blob_indices), 6))

    for i, blob_idx in enumerate(roi_blob_indices):
        blob_points = frame_xyzs[blob_labels == blob_idx]
        blob_vels = datapoint_vels[blob_labels == blob_idx]

        roi_blob_position_motion_matrix[i, :3] = blob_centroids[blob_idx]
        roi_blob_position_motion_matrix[i, 3:] = np.mean(blob_vels, axis=0) if len(blob_vels) > 0 else np.zeros(3)

    roi_hyperblob_indices = np.arange(num_roi_hyperblobs)
    if roi_blobs > 0 and num_roi_hyperblobs > 0:
        if roi_blobs > num_roi_hyperblobs:
            roi_hyperblob_kmeans = KMeans(n_clusters=num_roi_hyperblobs, random_state=0)
            roi_hyperblob_labels = roi_hyperblob_kmeans.fit_predict(roi_blob_position_motion_matrix)
            hyperblob_labels[roi_blob_indices] = roi_hyperblob_labels
        else:
            for i, blob_idx in enumerate(roi_blob_indices):
                hyperblob_labels[blob_idx] = i % num_roi_hyperblobs

    if non_roi_blobs > 0 and num_hyperblobs > num_roi_hyperblobs:
        non_roi_blob_indices = np.arange(roi_blobs, actual_num_blobs)
        non_roi_blob_position_motion_matrix = np.zeros((len(non_roi_blob_indices), 6))

        for i, blob_idx in enumerate(non_roi_blob_indices):
            blob_points = frame_xyzs[blob_labels == blob_idx]
            blob_vels = datapoint_vels[blob_labels == blob_idx]

            non_roi_blob_position_motion_matrix[i, :3] = blob_centroids[blob_idx]
            non_roi_blob_position_motion_matrix[i, 3:] = np.mean(blob_vels, axis=0) if len(blob_vels) > 0 else np.zeros(3)

        num_non_roi_hyperblobs = num_hyperblobs - num_roi_hyperblobs
        if non_roi_blobs > num_non_roi_hyperblobs and num_non_roi_hyperblobs > 0:
            non_roi_kmeans = KMeans(n_clusters=num_non_roi_hyperblobs, random_state=0)
            non_roi_hyperblob_labels = non_roi_kmeans.fit_predict(non_roi_blob_position_motion_matrix)
            non_roi_hyperblob_labels += num_roi_hyperblobs
            hyperblob_labels[non_roi_blob_indices] = non_roi_hyperblob_labels
        else:
            for i, blob_idx in enumerate(non_roi_blob_indices):
                hyperblob_labels[blob_idx] = (i % num_non_roi_hyperblobs) + num_roi_hyperblobs if num_non_roi_hyperblobs > 0 else 0

    hyperblob_to_blobs = {i: np.where(hyperblob_labels == i)[0] for i in range(num_hyperblobs)}

    hyperblob_centroids = np.zeros((num_hyperblobs, 3))
    for i in range(num_hyperblobs):
        if len(hyperblob_to_blobs[i]) > 0:
            blob_indices = hyperblob_to_blobs[i]
            weights = blob_mixture_weights[blob_indices]
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
            hyperblob_centroids[i] = np.sum(blob_centroids[blob_indices] * weights[:, np.newaxis], axis=0)
        else:
            hyperblob_centroids[i] = np.mean(frame_xyzs, axis=0)

    hyperblob_to_points = {}
    for i in range(num_hyperblobs):
        points = []
        for blob_idx in hyperblob_to_blobs[i]:
            points.append(frame_xyzs[blob_labels == blob_idx])
        if points:
            hyperblob_to_points[i] = np.vstack(points)
        else:
            hyperblob_to_points[i] = np.empty((0, 3))

    hyperblob_empirical_covs = []
    for i in range(num_hyperblobs):
        points_in_hyperblob = hyperblob_to_points[i]
        if len(points_in_hyperblob) > 3:
            hyperblob_empirical_covs.append(sample_covariance_matrix_numpy(points_in_hyperblob))
        else:
            hyperblob_empirical_covs.append(np.eye(3) * 0.01)

    hyperblob_empirical_covs = np.stack(hyperblob_empirical_covs)

    hyperblob_mixture_weights = np.array([len(hyperblob_to_points[i]) / len(frame_xyzs) for i in range(num_hyperblobs)])

    hyperblob_rot_vels = []
    hyperblob_trans_vels = []

    for i in range(num_hyperblobs):
        hyperblob_points_indices = np.concatenate([np.where(blob_labels == blob_idx)[0] 
                                                 for blob_idx in hyperblob_to_blobs[i]])

        if len(hyperblob_points_indices) > 3:
            if assume_correspondences:
                points_current = tracked_points[frame_idx][hyperblob_points_indices]
                points_next = tracked_points[frame_idx + motion_time_offset][hyperblob_points_indices]

                rotation_matrix, translation_vector = kabsch_algorithm_local_rotation_and_translation(points_current, points_next)
            else:
                points_current = tracked_points[frame_idx][hyperblob_points_indices]
                point_motions = motion_vectors[frame_idx][hyperblob_points_indices]

                points_next = points_current + point_motions

                rotation_matrix, translation_vector = kabsch_algorithm_local_rotation_and_translation(points_current, points_next)

            hyperblob_rot_vels.append(rotation_matrix)
            hyperblob_trans_vels.append(translation_vector)
        else:
            hyperblob_rot_vels.append(np.eye(3))
            hyperblob_trans_vels.append(np.zeros(3))

    hyperblob_rot_vels = np.array(hyperblob_rot_vels)
    hyperblob_trans_vels = np.array(hyperblob_trans_vels)

    blob_vel_means = []
    blob_vel_covs = []

    for i in range(actual_num_blobs):
        blob_points_indices = np.where(blob_labels == i)[0]

        blob_point_vels = datapoint_vels[blob_points_indices]

        hyperblob_idx = hyperblob_labels[i]

        hyperblob_rot = hyperblob_rot_vels[hyperblob_idx]
        hyperblob_trans = hyperblob_trans_vels[hyperblob_idx]

        expected_vel = hyperblob_trans + np.einsum('ij,kj->ki', 
                                                  hyperblob_rot - np.eye(3), 
                                                  blob_centroids[i:i+1] - hyperblob_centroids[hyperblob_idx:hyperblob_idx+1])
        expected_vel = expected_vel[0]

        mean_vel = np.mean(blob_point_vels, axis=0) if len(blob_point_vels) > 0 else expected_vel

        vel_cov = sample_covariance_matrix_numpy(blob_point_vels) if len(blob_point_vels) > 3 else np.eye(3) * 0.01

        blob_vel_means.append(mean_vel)
        blob_vel_covs.append(vel_cov)

    blob_vel_means = np.array(blob_vel_means)
    blob_vel_covs = np.stack(blob_vel_covs)

    blob_counts = np.array([np.sum(blob_labels == i) for i in range(actual_num_blobs)])
    valid_indices = blob_counts > 3

    if np.any(valid_indices):
        median_valid_blob_vel_cov = np.median(blob_vel_covs[valid_indices], axis=0)
        blob_vel_covs[~valid_indices] = median_valid_blob_vel_cov

    kmeans_chm = (
        C.n()
        | C['datapoints', 'blob_assignments'].set(jnp.array(blob_labels))
        | C['datapoints', 'datapoint_positions'].set(jnp.array(frame_xyzs))
        | C['datapoints', 'datapoint_vels'].set(jnp.array(datapoint_vels))
        | C['blobs', 'blob_means'].set(jnp.array(blob_centroids))
        | C['blobs', 'blob_covs'].set(jnp.array(blob_empirical_covs))
        | C['blobs', 'blob_weights'].set(jnp.array(blob_mixture_weights))
        | C['blobs', 'hyperblob_assignments'].set(jnp.array(hyperblob_labels))
        | C['blobs', 'blob_vel_means'].set(jnp.array(blob_vel_means))
        | C['blobs', 'blob_vel_covs'].set(jnp.array(blob_vel_covs))
        | C['hyperblobs', 'hyperblob_means'].set(jnp.array(hyperblob_centroids))
        | C['hyperblobs', 'hyperblob_covs'].set(jnp.array(hyperblob_empirical_covs))
        | C['hyperblobs', 'hyperblob_weights'].set(jnp.array(hyperblob_mixture_weights))
        | C['hyperblobs', 'hyperblob_rot_vels'].set(jnp.array(hyperblob_rot_vels))
        | C['hyperblobs', 'hyperblob_trans_vels'].set(jnp.array(hyperblob_trans_vels))
    )

    return kmeans_chm, roi_blob_indices, roi_hyperblob_indices


def make_hierarchical_kmeans_chm_with_SAM_segmentations(
    tracked_points,
    num_blobs,
    segmentation_mask,
    img_dims,
    subsampled_indices=None,
    frame_idx=0,
    motion_time_offset=1,
    assume_correspondences=False,
    motion_vectors=None,
    verbose=0,
):
    """
    Create hierarchical k-means clustering using SAM segmentations.
    Each distinct segmentation becomes its own hyperblob, with remaining unsegmented pixels
    clustered into additional hyperblobs maintaining consistent hyperblob-to-pixel ratios.
    """
    if assume_correspondences:
        assert motion_time_offset >= 1
        assert frame_idx + motion_time_offset < len(tracked_points)
    else:
        assert motion_vectors is not None, "Motion vectors must be provided when assume_correspondences is False"
        assert motion_vectors.shape == tracked_points.shape, "Motion vectors must have the same shape as tracked_points"

    frame_xyzs = tracked_points[frame_idx]

    if assume_correspondences:
        datapoint_vels = tracked_points[frame_idx + motion_time_offset] - tracked_points[frame_idx]
    else:
        datapoint_vels = motion_vectors[frame_idx]

    # Process segmentation mask: convert H x W x 3 to H x W integer mask
    H, W = img_dims
    if segmentation_mask.shape[:2] != (H, W):
        # Resize segmentation mask to img_dims using nearest neighbor interpolation
        from skimage.transform import resize
        segmentation_mask = resize(segmentation_mask, (H, W), preserve_range=True, anti_aliasing=False, order=0).astype(np.uint8)
    
    # Convert RGB segmentation to integer labels
    # Background (255,255,255) becomes 0, other unique colors get sequential labels
    segmentation_flat = segmentation_mask.reshape(-1, 3)
    unique_colors = np.unique(segmentation_flat, axis=0)
    
    # Create mapping from RGB to integer labels
    color_to_label = {}
    label = 0
    for color in unique_colors:
        if np.array_equal(color, [255, 255, 255]):
            color_to_label[tuple(color)] = 0  # Background
        else:
            label += 1
            color_to_label[tuple(color)] = label
    
    # Convert to integer mask and flatten, then align to subsampled datapoints if provided
    integer_mask_full = np.zeros(H * W, dtype=int)
    for i, color in enumerate(segmentation_flat):
        integer_mask_full[i] = color_to_label[tuple(color)]
    if subsampled_indices is not None:
        integer_mask = integer_mask_full[subsampled_indices]
    else:
        integer_mask = integer_mask_full
    
    # Get unique segmentation labels (excluding background 0)
    unique_segments = np.unique(integer_mask)
    unique_segments = unique_segments[unique_segments > 0]  # Remove background
    num_segmented_hyperblobs = len(unique_segments)
    
    # Calculate number of hyperblobs for unsegmented region
    total_points = len(frame_xyzs)
    unsegmented_points_count = np.sum(integer_mask == 0)
    segmented_points_count = total_points - unsegmented_points_count
    
    if unsegmented_points_count > 0 and segmented_points_count > 0:
        # Maintain same hyperblob-to-pixel ratio for unsegmented region
        hyperblob_to_pixel_ratio = num_segmented_hyperblobs / segmented_points_count
        num_unsegmented_hyperblobs = max(1, int(unsegmented_points_count * hyperblob_to_pixel_ratio))
    elif unsegmented_points_count > 0:
        num_unsegmented_hyperblobs = max(1, num_segmented_hyperblobs)  # Default if no segmented points
    else:
        num_unsegmented_hyperblobs = 0
    
    num_hyperblobs = num_segmented_hyperblobs + num_unsegmented_hyperblobs
    
    # Initialize arrays for all hyperblobs
    all_hyperblob_points = {}
    all_hyperblob_vels = {}
    hyperblob_labels = np.zeros(total_points, dtype=int)
    
    # Process each segmented region as its own hyperblob
    hyperblob_idx = 0
    for segment_label in unique_segments:
        segment_mask = integer_mask == segment_label
        segment_points = frame_xyzs[segment_mask]
        segment_vels = datapoint_vels[segment_mask]
        
        all_hyperblob_points[hyperblob_idx] = segment_points
        all_hyperblob_vels[hyperblob_idx] = segment_vels
        hyperblob_labels[segment_mask] = hyperblob_idx
        hyperblob_idx += 1
    
    # Process unsegmented region with k-means if needed
    if num_unsegmented_hyperblobs > 0:
        unsegmented_mask = integer_mask == 0
        unsegmented_points = frame_xyzs[unsegmented_mask]
        unsegmented_vels = datapoint_vels[unsegmented_mask]
        
        if len(unsegmented_points) > 0:
            unsegmented_features = np.concatenate([unsegmented_points, unsegmented_vels], axis=1)
            
            if len(unsegmented_points) > num_unsegmented_hyperblobs:
                unsegmented_kmeans = KMeans(n_clusters=num_unsegmented_hyperblobs, random_state=0, verbose=verbose)
                unsegmented_labels = unsegmented_kmeans.fit_predict(unsegmented_features)
            else:
                unsegmented_labels = np.arange(len(unsegmented_points))
                num_unsegmented_hyperblobs = len(unsegmented_points)
            
            # Assign unsegmented points to hyperblobs
            for i in range(num_unsegmented_hyperblobs):
                points_in_hyperblob = unsegmented_points[unsegmented_labels == i]
                vels_in_hyperblob = unsegmented_vels[unsegmented_labels == i]
                
                all_hyperblob_points[hyperblob_idx] = points_in_hyperblob
                all_hyperblob_vels[hyperblob_idx] = vels_in_hyperblob
                
                # Update hyperblob labels for these points
                point_indices = np.where(unsegmented_mask)[0]
                hyperblob_point_indices = point_indices[unsegmented_labels == i]
                hyperblob_labels[hyperblob_point_indices] = hyperblob_idx
                
                hyperblob_idx += 1
    
    # Update total number of hyperblobs
    num_hyperblobs = hyperblob_idx
    
    # Now perform blob-level clustering within each hyperblob
    all_blob_labels = np.zeros(total_points, dtype=int)
    all_blob_centroids = []
    all_blob_empirical_covs = []
    all_blob_mixture_weights = []
    blob_to_hyperblob = []
    
    blob_idx = 0
    for hb_idx in range(num_hyperblobs):
        if hb_idx not in all_hyperblob_points:
            continue
            
        hb_points = all_hyperblob_points[hb_idx]
        hb_vels = all_hyperblob_vels[hb_idx]
        hb_point_count = len(hb_points)
        
        if hb_point_count == 0:
            continue
        
        # Allocate blobs to this hyperblob based on ratio
        hb_blob_ratio = hb_point_count / total_points
        hb_num_blobs = max(1, int(num_blobs * hb_blob_ratio))
        
        # Perform k-means clustering within this hyperblob
        if hb_num_blobs > 0 and hb_point_count > hb_num_blobs:
            hb_features = np.concatenate([hb_points, hb_vels], axis=1)
            hb_kmeans = KMeans(n_clusters=hb_num_blobs, random_state=0, verbose=verbose)
            hb_blob_labels = hb_kmeans.fit_predict(hb_features)
            hb_blob_centroids = hb_kmeans.cluster_centers_[:, :3]
        elif hb_num_blobs > 0:
            hb_blob_labels = np.arange(hb_point_count)
            hb_blob_centroids = hb_points
            hb_num_blobs = hb_point_count
        else:
            continue
        
        # Map hyperblob point indices back to global indices
        hb_global_indices = np.where(hyperblob_labels == hb_idx)[0]
        
        # Assign blob labels globally
        for i in range(hb_num_blobs):
            local_blob_mask = hb_blob_labels == i
            global_blob_indices = hb_global_indices[local_blob_mask]
            all_blob_labels[global_blob_indices] = blob_idx + i
            
            # Store blob information
            blob_points = hb_points[local_blob_mask]
            all_blob_centroids.append(hb_blob_centroids[i])
            
            if len(blob_points) > 3:
                all_blob_empirical_covs.append(sample_covariance_matrix_numpy(blob_points))
            else:
                all_blob_empirical_covs.append(np.eye(3) * 0.01)
            
            all_blob_mixture_weights.append(len(blob_points) / total_points)
            blob_to_hyperblob.append(hb_idx)
        
        blob_idx += hb_num_blobs
    
    actual_num_blobs = len(all_blob_centroids)
    
    if actual_num_blobs == 0:
        # Fallback case
        all_blob_centroids = [np.mean(frame_xyzs, axis=0)]
        all_blob_empirical_covs = [np.eye(3) * 0.01]
        all_blob_mixture_weights = [1.0]
        blob_to_hyperblob = [0]
        all_blob_labels = np.zeros(total_points, dtype=int)
        actual_num_blobs = 1
    
    # Convert to arrays
    blob_centroids = np.array(all_blob_centroids)
    blob_empirical_covs = np.stack(all_blob_empirical_covs)
    blob_mixture_weights = np.array(all_blob_mixture_weights)
    blob_hyperblob_assignments = np.array(blob_to_hyperblob)
    
    # Calculate hyperblob properties
    hyperblob_centroids = np.zeros((num_hyperblobs, 3))
    hyperblob_empirical_covs = []
    hyperblob_mixture_weights = np.zeros(num_hyperblobs)
    
    for hb_idx in range(num_hyperblobs):
        if hb_idx in all_hyperblob_points:
            hb_points = all_hyperblob_points[hb_idx]
            hyperblob_centroids[hb_idx] = np.mean(hb_points, axis=0) if len(hb_points) > 0 else np.mean(frame_xyzs, axis=0)
            hyperblob_mixture_weights[hb_idx] = len(hb_points) / total_points
            
            if len(hb_points) > 3:
                hyperblob_empirical_covs.append(sample_covariance_matrix_numpy(hb_points))
            else:
                hyperblob_empirical_covs.append(np.eye(3) * 0.01)
        else:
            hyperblob_centroids[hb_idx] = np.mean(frame_xyzs, axis=0)
            hyperblob_mixture_weights[hb_idx] = 0.0
            hyperblob_empirical_covs.append(np.eye(3) * 0.01)
    
    hyperblob_empirical_covs = np.stack(hyperblob_empirical_covs)
    
    # Calculate hyperblob velocities using Kabsch algorithm
    hyperblob_rot_vels = []
    hyperblob_trans_vels = []
    
    for hb_idx in range(num_hyperblobs):
        if hb_idx in all_hyperblob_points:
            hb_point_indices = np.where(hyperblob_labels == hb_idx)[0]
            
            if len(hb_point_indices) > 3:
                if assume_correspondences:
                    points_current = tracked_points[frame_idx][hb_point_indices]
                    points_next = tracked_points[frame_idx + motion_time_offset][hb_point_indices]
                    rotation_matrix, translation_vector = kabsch_algorithm_local_rotation_and_translation(points_current, points_next)
                else:
                    points_current = tracked_points[frame_idx][hb_point_indices]
                    point_motions = motion_vectors[frame_idx][hb_point_indices]
                    points_next = points_current + point_motions
                    rotation_matrix, translation_vector = kabsch_algorithm_local_rotation_and_translation(points_current, points_next)
                
                hyperblob_rot_vels.append(rotation_matrix)
                hyperblob_trans_vels.append(translation_vector)
            else:
                hyperblob_rot_vels.append(np.eye(3))
                hyperblob_trans_vels.append(np.zeros(3))
        else:
            hyperblob_rot_vels.append(np.eye(3))
            hyperblob_trans_vels.append(np.zeros(3))
    
    hyperblob_rot_vels = np.array(hyperblob_rot_vels)
    hyperblob_trans_vels = np.array(hyperblob_trans_vels)
    
    # Calculate blob velocities
    blob_vel_means = []
    blob_vel_covs = []
    
    for i in range(actual_num_blobs):
        blob_points_indices = np.where(all_blob_labels == i)[0]
        blob_point_vels = datapoint_vels[blob_points_indices]
        
        hyperblob_idx = blob_hyperblob_assignments[i]
        hyperblob_rot = hyperblob_rot_vels[hyperblob_idx]
        hyperblob_trans = hyperblob_trans_vels[hyperblob_idx]
        
        expected_vel = hyperblob_trans + np.einsum('ij,kj->ki', 
                                                  hyperblob_rot - np.eye(3), 
                                                  blob_centroids[i:i+1] - hyperblob_centroids[hyperblob_idx:hyperblob_idx+1])
        expected_vel = expected_vel[0]
        
        mean_vel = np.mean(blob_point_vels, axis=0) if len(blob_point_vels) > 0 else expected_vel
        vel_cov = sample_covariance_matrix_numpy(blob_point_vels) if len(blob_point_vels) > 3 else np.eye(3) * 0.01
        
        blob_vel_means.append(mean_vel)
        blob_vel_covs.append(vel_cov)
    
    blob_vel_means = np.array(blob_vel_means)
    blob_vel_covs = np.stack(blob_vel_covs)
    
    # Handle small blob velocity covariances
    blob_counts = np.array([np.sum(all_blob_labels == i) for i in range(actual_num_blobs)])
    valid_indices = blob_counts > 3
    
    if np.any(valid_indices):
        median_valid_blob_vel_cov = np.median(blob_vel_covs[valid_indices], axis=0)
        blob_vel_covs[~valid_indices] = median_valid_blob_vel_cov
    
    # Create the hierarchical mixture model
    kmeans_chm = (
        C.n()
        | C['datapoints', 'blob_assignments'].set(jnp.array(all_blob_labels))
        | C['datapoints', 'datapoint_positions'].set(jnp.array(frame_xyzs))
        | C['datapoints', 'datapoint_vels'].set(jnp.array(datapoint_vels))
        | C['blobs', 'blob_means'].set(jnp.array(blob_centroids))
        | C['blobs', 'blob_covs'].set(jnp.array(blob_empirical_covs))
        | C['blobs', 'blob_weights'].set(jnp.array(blob_mixture_weights))
        | C['blobs', 'hyperblob_assignments'].set(jnp.array(blob_hyperblob_assignments))
        | C['blobs', 'blob_vel_means'].set(jnp.array(blob_vel_means))
        | C['blobs', 'blob_vel_covs'].set(jnp.array(blob_vel_covs))
        | C['hyperblobs', 'hyperblob_means'].set(jnp.array(hyperblob_centroids))
        | C['hyperblobs', 'hyperblob_covs'].set(jnp.array(hyperblob_empirical_covs))
        | C['hyperblobs', 'hyperblob_weights'].set(jnp.array(hyperblob_mixture_weights))
        | C['hyperblobs', 'hyperblob_rot_vels'].set(jnp.array(hyperblob_rot_vels))
        | C['hyperblobs', 'hyperblob_trans_vels'].set(jnp.array(hyperblob_trans_vels))
    )
    
    # Determine ROI indices (segmented hyperblobs are considered ROI)
    roi_hyperblob_indices = np.arange(num_segmented_hyperblobs)
    roi_blob_indices = np.where(np.isin(blob_hyperblob_assignments, roi_hyperblob_indices))[0]
    
    return kmeans_chm, roi_blob_indices, roi_hyperblob_indices, num_hyperblobs

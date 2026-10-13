"""Causal, flow-guided display geometry; immutable inference remains inspectable.

Opacity memory cannot correct jitter in a particle's center or covariance. This
filter transports each displayed center with the observed image flow, then
damps the residual posterior correction. Coherent depth motion is transported
per object; local depth/shape innovations are damped independently. It is a
presentation filter, not a change to inference or a tracking-accuracy claim.
"""
import cv2
import numpy as np


def matrix_log(covariance):
    values, vectors = np.linalg.eigh(covariance)
    return np.einsum('...ij,...j,...kj->...ik', vectors, np.log(np.maximum(values, 1e-8)), vectors)


def matrix_exp(logarithm):
    values, vectors = np.linalg.eigh(logarithm)
    return np.einsum('...ij,...j,...kj->...ik', vectors, np.exp(values), vectors)


def project(means):
    return means[..., :2]/np.maximum(means[..., 2:3], .05)*500.+[320., 180.]


def advect(uv, frame, ids):
    """Invert the saved backward flow, preserving subpixel coordinates.

    Capture grids are sampled at source pixels 4,12,...; their vectors are in
    eight-pixel units. Reject invalid or differently labelled correspondences.
    """
    if 'backward_grid' not in frame:
        return uv.copy(), np.zeros(len(uv), bool)
    backward = np.asarray(frame['backward_grid'], np.float32).transpose(1, 2, 0)*8.
    sample = lambda array, points: cv2.remap(np.asarray(array, np.float32),
        ((points[:, 0]-4)/8).astype(np.float32)[:, None],
        ((points[:, 1]-4)/8).astype(np.float32)[:, None], cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE).reshape(len(points), -1)
    destination = uv.copy()
    for _ in range(4):
        destination = uv-sample(backward, destination)
    residual = destination+sample(backward, destination)-uv
    valid = (sample(frame['reliable_grid'], destination)[:, 0] > .9)
    valid &= np.linalg.norm(residual, axis=1) < 1.
    valid &= np.all((destination >= [4, 4]) & (destination <= [636, 356]), axis=1)
    valid &= np.all((uv >= [4, 4]) & (uv <= [636, 356]), axis=1)
    if 'support' in frame:
        valid &= frame['support'] > .5
    if 'labels' in frame:
        xx=np.clip(np.rint(destination[:, 0]).astype(int),0,639)
        yy=np.clip(np.rint(destination[:, 1]).astype(int),0,359)
        valid &= frame['labels'][yy,xx] == ids
    return destination, valid


def stabilize(means, covs, supports, ids, frame_reader, fps, time_constant=.14):
    """Transport motion immediately; damp center/shape corrections over 140 ms.

    Every update uses only the current frame and previous display state. A true
    constant flow has no smoothing lag. Covariance smoothing is log-Euclidean,
    so all shapes remain positive definite and eigenvector sign flips are inert.
    """
    means, covs = np.asarray(means, float), np.asarray(covs, float)
    output = means.copy(); shapes = covs.copy()
    uv = project(means)
    depth = np.log(np.maximum(means[..., 2], .05))
    log_cov = matrix_log(covs)
    gain = 1-np.exp(-1/(fps*time_constant))
    depth_gain = 1-np.exp(-1/(fps*.10))
    depth_velocity = {label:0. for label in np.unique(ids)}
    flow_valid = np.zeros(supports.shape, bool)
    for i in range(1, len(means)):
        old_uv = project(output[i-1])
        predicted_uv, valid = advect(old_uv, frame_reader(i), ids)
        flow_valid[i] = valid
        depth_motion = np.zeros(len(ids))
        # Each identity supplies a robust fallback. This never mixes an animal
        # with scene context, or substitutes a trajectory from another identity.
        for label in np.unique(ids):
            selected = ids == label
            supported = selected & (supports[i] > .5) & (supports[i-1] > .5)
            if supported.any():
                motion = np.median(uv[i, supported]-uv[i-1, supported], axis=0)
                measured_depth = np.median(depth[i, supported]-depth[i-1, supported])
                # Relative-depth estimation can move an entire object forward
                # and back together. Passing its median displacement through
                # unfiltered preserves that jitter. A causal velocity estimate
                # removes reversals while preserving sustained depth motion.
                depth_velocity[label] += depth_gain*(measured_depth-depth_velocity[label])
                depth_motion[selected] = depth_velocity[label]
            else:
                motion = np.zeros(2)
            missing = selected & ~valid
            predicted_uv[missing] = old_uv[missing]+motion
        predicted_depth = np.log(np.maximum(output[i-1, :, 2], .05))+depth_motion
        filtered_uv = predicted_uv+gain*(uv[i]-predicted_uv)
        filtered_depth = predicted_depth+gain*(depth[i]-predicted_depth)
        z = np.exp(filtered_depth)
        output[i] = np.column_stack([(filtered_uv-[320., 180.])*z[:, None]/500., z])
        # Transport coherent scale changes before smoothing local deformation.
        predicted_shape = matrix_log(shapes[i-1])+2*depth_motion[:, None, None]*np.eye(3)
        shapes[i] = matrix_exp(predicted_shape+gain*(log_cov[i]-predicted_shape))
    return output, shapes, flow_valid


def diagnostics(raw_means, means, raw_covs, covs, ids):
    result = {}
    for name, selected in [('object', ids > 0), ('background', ids == 0)]:
        if not selected.any() or len(means)<3:
            continue
        row = {}
        for prefix, positions, shapes in [('raw',raw_means,raw_covs),('display',means,covs)]:
            acceleration = np.linalg.norm(np.diff(project(positions[:,selected]), n=2, axis=0),axis=-1)
            change = np.linalg.norm(np.diff(shapes[:,selected],axis=0),axis=(-1,-2))
            change /= np.maximum(np.linalg.norm(shapes[:-1,selected],axis=(-1,-2)),1e-8)
            row[prefix] = dict(projected_acceleration_median=float(np.median(acceleration)),
                               projected_acceleration_p95=float(np.percentile(acceleration,95)),
                               depth_acceleration_median=float(np.median(np.abs(np.diff(positions[:,selected,2],n=2,axis=0)))),
                               relative_covariance_change_median=float(np.median(change)))
        result[name] = row
    result['note'] = 'Acceleration includes real motion; this is display stability, not tracking accuracy.'
    return result

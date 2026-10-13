"""Explicit camera geometry and current-to-previous frame correspondences."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    position: np.ndarray
    rotation: np.ndarray  # world -> camera; image y points down
    focal: float
    width: int
    height: int

    def project(self, points):
        p = (np.asarray(points) - self.position) @ self.rotation.T
        z = p[:, 2]
        safe = np.where(z > 1e-4, z, np.nan)
        uv = p[:, :2] * (self.focal / safe[:, None])
        uv += (self.width / 2, self.height / 2)
        return uv, z

    def footprints(self, means, covs, sigma=1.0):
        p = (np.asarray(means) - self.position) @ self.rotation.T
        z = np.maximum(p[:, 2], 1e-4)
        j = np.zeros((len(p), 2, 3), np.float64)
        j[:, 0, 0] = j[:, 1, 1] = self.focal / z
        j[:, :, 2] = -self.focal * p[:, :2] / z[:, None] ** 2
        r = self.rotation
        c = np.einsum('ij,njk,lk->nil', r, covs, r)
        s = np.einsum('nij,njk,nlk->nil', j, c, j) * sigma**2
        s += np.eye(2)[None] * 0.5
        return s


def look_at(position, target, focal, width, height):
    position, target = np.asarray(position, float), np.asarray(target, float)
    f = target - position
    f /= max(np.linalg.norm(f), 1e-8)
    right = np.cross([0., 1., 0.], f)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1., 0., 0.])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return Camera(position, np.stack([right, down, f]), focal, width, height)


def orbit_camera(target, yaw, pitch, distance, focal, width, height):
    y, p = np.deg2rad([yaw, pitch])
    offset = distance * np.array([np.sin(y)*np.cos(p), -np.sin(p), -np.cos(y)*np.cos(p)])
    return look_at(np.asarray(target) + offset, target, focal, width, height)


def backward_maps(backward):
    h, w = backward.shape[1:]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    x, y = xx + backward[0], yy + backward[1]
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w-1) & (y >= 0) & (y <= h-1)
    return x, y, valid


def warp_previous(previous, backward, nearest=False):
    x, y, valid = backward_maps(backward)
    out = cv2.remap(np.asarray(previous, np.float32), x, y,
                    cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, valid


def correspondence_confidence(forward, backward):
    old_f, valid = warp_previous(forward.transpose(1, 2, 0), backward)
    residual = np.linalg.norm(old_f + backward.transpose(1, 2, 0), axis=-1)
    scale = 1.0 + 0.05 * np.linalg.norm(backward, axis=0)
    return valid & (residual < scale)


def align_depth(raw, previous, backward, valid, background, bounds=None):
    """Robust affine disparity alignment on corresponding background pixels only."""
    raw = np.asarray(raw, np.float32)
    if previous is None:
        return raw.copy(), tuple(float(x) for x in np.percentile(raw, [1, 99]))
    reference, in_bounds = warp_previous(previous, backward)
    use = valid & in_bounds & background & np.isfinite(reference) & np.isfinite(raw)
    yy, xx = np.where(use[::4, ::4])
    if len(xx) < 32:
        return raw.copy(), bounds
    x, y = raw[yy*4, xx*4].astype(float), reference[yy*4, xx*4].astype(float)
    scale, offset = 1., float(np.median(y-x))
    for _ in range(3):
        residual = scale*x + offset-y
        mad = 1.4826*np.median(np.abs(residual-np.median(residual))) + 1e-5
        weights = np.minimum(1., 1.345*mad/(np.abs(residual)+1e-8))
        a = np.stack([x, np.ones_like(x)], 1)
        fit = np.linalg.lstsq(a*weights[:, None]**.5, y*weights**.5, rcond=None)[0]
        scale, offset = float(fit[0]), float(fit[1])
    if not np.isfinite([scale, offset]).all() or not 0.5 < scale < 2.:
        return raw.copy(), bounds
    aligned = (scale*raw + offset).astype(np.float32)
    # Smooth only flow-consistent depth, never across an occlusion boundary.
    agree = use & (np.abs(aligned-reference) < np.std(aligned)*0.12)
    aligned[agree] = .8*aligned[agree] + .2*reference[agree]
    return aligned, bounds


def depth_to_xyz(depth, bounds, focal=500.):
    lo, hi = bounds
    z = 1./(.2 + .8*np.clip((depth-lo)/max(hi-lo, 1e-6), 0, 1))
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    return np.stack([(xx-w/2)*z/focal, (yy-h/2)*z/focal, z], -1).astype(np.float32)


def sharpen_object_depth(depth, labels, radius=3):
    """Correct depth bleeding only at an inferred object's boundary.

    A monocular depth pixel straddling foreground/background may put a fur or
    cloth edge on the background plane. Replace only boundary depths outside
    the object's interior range, using the nearest interior observation. This
    is a causal object-shape prior: labels are frame-0 seeds or warped previous
    predictions, never later annotations. Interior depth variation is retained.
    """
    result = np.asarray(depth,np.float32).copy()
    kernel = np.ones((2*radius+1,2*radius+1),np.uint8)
    for label in np.unique(labels):
        if label <= 0:
            continue
        mask = labels == label
        core = cv2.erode(mask.astype(np.uint8),kernel).astype(bool)
        if core.sum() < 16:
            continue
        low,high = np.percentile(depth[core],[5,95])
        margin = max((high-low)*.25,float(np.ptp(depth))*.015,1e-5)
        bad = mask & ~core & ((depth < low-margin)|(depth > high+margin))
        if not bad.any():
            continue
        _,nearest = cv2.distanceTransformWithLabels((~core).astype(np.uint8),cv2.DIST_L2,5,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        values = np.zeros(nearest.max()+1,np.float32)
        values[nearest[core]] = depth[core]
        result[bad] = values[nearest[bad]]
    return result


def measured_motion(current_xyz, previous_xyz, backward, valid):
    if previous_xyz is None:
        return np.zeros_like(current_xyz)
    previous, inside = warp_previous(previous_xyz, backward)
    velocity = current_xyz-previous
    velocity[~(valid & inside)] = 0
    return velocity


def rigid_alignment(current, previous, valid):
    """Robust current-camera -> previous-camera transform; identity on degeneracy."""
    p, q = current[valid][::4], previous[valid][::4]
    if len(p) < 24:
        return np.eye(3), np.zeros(3)
    keep = np.ones(len(p), bool)
    r, t = np.eye(3), np.zeros(3)
    for _ in range(3):
        a, b = p[keep], q[keep]
        # Kabsch requires arithmetic centroids. Robustness comes from trimming
        # correspondences; coordinate-wise medians bias rotations on skewed scenes.
        ca, cb = a.mean(0), b.mean(0)
        u, _, vt = np.linalg.svd((a-ca).T @ (b-cb))
        r = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
        t = cb-r@ca
        residual = np.linalg.norm(p@r.T+t-q, axis=1)
        keep = residual <= np.percentile(residual, 70)
    if np.linalg.norm(t) > .5 or np.trace(r) < 2.7:
        return np.eye(3), np.zeros(3)
    return r, t

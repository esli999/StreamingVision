"""3D covariance matrices to Rerun ellipsoid parameters."""

from __future__ import annotations

import numpy as np


def covariance_to_ellipsoid(
    cov: np.ndarray,
    *,
    sigma_scale: float = 1.0,
    min_eigenvalue: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert 3x3 covariance to axis half-lengths and xyzw quaternion.

    Returns ``(half_sizes, quat_xyzw)`` for ``rr.Ellipsoids3D``.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (3, 3):
        raise ValueError(f"expected (3, 3) covariance, got {cov.shape}")
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    half_sizes = sigma_scale * np.sqrt(eigvals)

    # Rotation matrix -> quaternion (xyzw)
    rot = eigvecs
    quat = _rotation_matrix_to_quaternion_xyzw(rot)
    return half_sizes.astype(np.float32), quat.astype(np.float32)


def covariance_batch_to_ellipsoids(
    covs: np.ndarray,
    *,
    sigma_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch version for ``(N, 3, 3)`` covariances."""
    n = covs.shape[0]
    half_sizes = np.zeros((n, 3), dtype=np.float32)
    quats = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        half_sizes[i], quats[i] = covariance_to_ellipsoid(
            covs[i], sigma_scale=sigma_scale
        )
    return half_sizes, quats


def _rotation_matrix_to_quaternion_xyzw(rot: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    m = rot
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q

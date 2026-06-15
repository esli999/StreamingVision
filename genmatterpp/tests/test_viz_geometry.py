"""Tests for genmatter.viz.geometry."""

from __future__ import annotations

import numpy as np
import pytest

from genmatter.viz.geometry import covariance_to_ellipsoid


def test_covariance_to_ellipsoid_diagonal() -> None:
    cov = np.diag([4.0, 1.0, 0.25])
    half, quat = covariance_to_ellipsoid(cov, sigma_scale=1.0)
    assert half.shape == (3,)
    assert quat.shape == (4,)
    np.testing.assert_allclose(np.sort(half), [0.5, 1.0, 2.0], rtol=1e-5)
    assert np.isfinite(quat).all()


def test_covariance_invalid_shape() -> None:
    with pytest.raises(ValueError):
        covariance_to_ellipsoid(np.eye(2))

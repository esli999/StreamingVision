"""Tests for outlier fraction helper."""

from __future__ import annotations

import numpy as np

from genmatter.tracking.outlier_stats import mean_outlier_fraction_percent


def test_mean_outlier_fraction_percent() -> None:
    tracking = [
        {"blob_assignments": np.array([0, 1, 2], dtype=np.int32), "n_blobs": 2},
        {"blob_assignments": np.array([0, 2, 2], dtype=np.int32), "n_blobs": 2},
    ]
    mean_p, min_p, max_p = mean_outlier_fraction_percent(tracking)
    assert abs(mean_p - 50.0) < 1e-6
    assert abs(min_p - 100.0 / 3.0) < 1e-6
    assert abs(max_p - 200.0 / 3.0) < 1e-6

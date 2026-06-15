"""Tests for genmatter.viz.colors."""

from __future__ import annotations

import numpy as np

from genmatter.viz.colors import (
    blob_mean_rgb_colors,
    distinct_palette,
    features_to_rgb_pca,
    point_colors_from_hyperblob_assignments,
)


def test_distinct_palette_size() -> None:
    p = distinct_palette(5)
    assert p.shape == (5, 3)
    assert p.dtype == np.uint8


def test_features_to_rgb_pca_shape() -> None:
    feat = np.random.randn(3, 100, 10).astype(np.float32)
    rgb = features_to_rgb_pca(feat, seed=0)
    assert rgb.shape == (3, 100, 3)
    assert rgb.dtype == np.uint8


def test_blob_mean_rgb_colors() -> None:
    assign = np.array([0, 0, 1, 1], dtype=np.int32)
    pts = np.array([[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    colors = blob_mean_rgb_colors(assign, pts, n_blobs=2)
    assert colors.shape == (2, 3)
    np.testing.assert_array_equal(colors[0], [255, 0, 0])
    np.testing.assert_array_equal(colors[1], [0, 127, 127])


def test_point_colors_from_hyperblob_assignments() -> None:
    palette = distinct_palette(2, seed=0)
    blob_assign = np.array([0, 1, 1, 2], dtype=np.int32)  # 2 = outlier
    hb_per_blob = np.array([0, 1], dtype=np.int32)
    colors = point_colors_from_hyperblob_assignments(
        blob_assign, hb_per_blob, n_blobs=2, n_hyperblobs=2, palette=palette
    )
    assert colors.shape == (4, 3)
    np.testing.assert_array_equal(colors[0], palette[0])
    np.testing.assert_array_equal(colors[1], palette[1])
    np.testing.assert_array_equal(colors[2], palette[1])
    np.testing.assert_array_equal(colors[3], [40, 40, 40])

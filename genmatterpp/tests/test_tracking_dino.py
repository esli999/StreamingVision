"""Tests for genmatter.tracking.dino helpers."""

from __future__ import annotations

import numpy as np
import pytest

from genmatter.tracking.dino import (
    DinoTrackingResult,
    DinoTrackingTimings,
    sample_datapoints_percentage,
    save_dense_tracking_npz,
)


def test_sample_datapoints_percentage_same_indices() -> None:
    T, N = 4, 1000
    pts = np.random.randn(T, N, 3).astype(np.float32)
    mvs = np.random.randn(T, N, 3).astype(np.float32)
    sp, sm, idx = sample_datapoints_percentage(
        pts, mvs, 78.125, seed=42, same_indices_all_timesteps=True
    )
    expected_keep = max(1, int(N * 78.125 / 100.0))
    assert sp.shape == (T, expected_keep, 3)
    assert len(idx) == expected_keep


def test_save_dense_tracking_npz_roundtrip(tmp_path) -> None:
    T, n_blobs, n_hyper, F = 2, 3, 2, 4
    hw = 6
    frame = {
        "n_blobs": n_blobs,
        "n_hyperblobs": n_hyper,
        "n_datapoints": hw,
        "blob_assignments": np.arange(hw, dtype=np.int32),
        "blob_weights": np.ones(n_blobs, dtype=np.float32),
        "blob_means": np.zeros((n_blobs, 3), dtype=np.float32),
        "blob_covs": np.eye(3, dtype=np.float32)[None].repeat(n_blobs, axis=0),
        "blob_vel_means": np.zeros((n_blobs, 3), dtype=np.float32),
        "blob_vel_covs": np.eye(3, dtype=np.float32)[None].repeat(n_blobs, axis=0),
        "blob_features": np.zeros((n_blobs, F), dtype=np.float32),
        "hyperblob_assignments": np.zeros(n_blobs, dtype=np.int32),
        "hyperblob_weights": np.ones(n_hyper, dtype=np.float32),
        "hyperblob_means": np.zeros((n_hyper, 3), dtype=np.float32),
        "hyperblob_trans_vels": np.zeros((n_hyper, 3), dtype=np.float32),
        "hyperblob_rot_vels": np.zeros((n_hyper, 3, 3), dtype=np.float32),
        "datapoint_positions": np.zeros((hw, 3), dtype=np.float32),
        "datapoint_vels": np.zeros((hw, 3), dtype=np.float32),
        "datapoint_features": np.zeros((hw, F), dtype=np.float32),
    }
    result = DinoTrackingResult(
        video_id="test_vid",
        tracking_data=[frame, frame],
        img_dims=(2, 3),
        subsampled_indices=np.array([0, 1], dtype=np.int32),
        focal_length=520.0,
        timings=DinoTrackingTimings(num_frames=2),
        gaussian_means=np.zeros(F),
        gaussian_stds=np.ones(F),
    )
    out = tmp_path / "tracking_dense.npz"
    size = save_dense_tracking_npz(out, result)
    assert size > 0
    data = np.load(out)
    assert data["blob_assignments"].shape == (2, hw)
    assert (tmp_path / "tracking_dense.meta.json").is_file()

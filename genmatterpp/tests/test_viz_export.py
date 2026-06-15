"""Tests for genmatter.viz.rerun_export with mocked rerun."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from genmatter.custom.config_schema import VizConfig
from genmatter.viz.artifacts import VizArtifacts
from genmatter.viz.rerun_export import export_to_rrd


def _tiny_artifacts() -> VizArtifacts:
    T, H, W, F = 2, 2, 2, 3
    n_blobs_max = 2
    tracking = {
        "datapoint_positions": np.zeros((T, H * W, 3), dtype=np.float32),
        "datapoint_features": np.zeros((T, H * W, F), dtype=np.float32),
        "blob_assignments": np.zeros((T, H * W), dtype=np.int32),
        "blob_means": np.zeros((T, n_blobs_max, 3), dtype=np.float32),
        "blob_covs": np.tile(np.eye(3), (T, n_blobs_max, 1, 1)).astype(np.float32),
        "hyperblob_assignments": np.zeros((T, n_blobs_max), dtype=np.int32),
        "hyperblob_means": np.zeros((T, 1, 3), dtype=np.float32),
        "n_blobs": np.array([1, 1], dtype=np.int32),
        "n_hyperblobs": np.array([1, 1], dtype=np.int32),
        "img_dims": np.array([H, W], dtype=np.int32),
        "focal_length": np.float32(520.0),
    }
    return VizArtifacts(
        video_id="tiny",
        num_frames=T,
        height=H,
        width=W,
        focal_length=520.0,
        rgb_frame_paths=[Path("/tmp/f0.jpg"), Path("/tmp/f1.jpg")],
        points_3d=np.zeros((T, H, W, 3), dtype=np.float64),
        motion_vectors_3d=np.zeros((T, H, W, 3), dtype=np.float64),
        colors=np.zeros((T, H, W, 3), dtype=np.uint8),
        tracking=tracking,
        feature_rgb=np.zeros((T, H * W, 3), dtype=np.uint8),
    )


@patch("genmatter.viz.rerun_export.rr")
@patch("genmatter.viz.rerun_export.cv2.imread")
def test_export_to_rrd_calls_save(mock_imread, mock_rr, tmp_path) -> None:
    mock_imread.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
    mock_rr.Ellipsoids3D = MagicMock(side_effect=lambda *a, **kw: kw)
    mock_rr.Points3D = MagicMock(side_effect=lambda *a, **kw: kw)
    mock_rr.Arrows3D = MagicMock(side_effect=lambda *a, **kw: kw)
    mock_rr.DepthImage = MagicMock(side_effect=lambda *a, **kw: (a, kw))
    mock_rr.Image = MagicMock(side_effect=lambda *a, **kw: (a, kw))
    out = tmp_path / "out.rrd"
    out.write_bytes(b"x")
    mock_rr.save = MagicMock()
    size = export_to_rrd(_tiny_artifacts(), VizConfig(), out)
    mock_rr.init.assert_called_once()
    mock_rr.save.assert_called_once_with(str(out))
    assert size >= 0

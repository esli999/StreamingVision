"""Tests for genmatter.custom.viz."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from genmatter.custom.config_schema import load_config
from genmatter.custom.console import PreprocessConsole
from genmatter.custom.paths import resolve_video_paths
from genmatter.custom.viz import run_viz
from genmatter.viz.artifacts import VizArtifacts


def _write_tracking_stack(paths) -> None:
    paths.tracking_dir.mkdir(parents=True, exist_ok=True)
    T, H, W, F, B, HB = 2, 2, 2, 3, 2, 1
    np.savez(
        paths.tracking_npz,
        datapoint_positions=np.zeros((T, H * W, 3), dtype=np.float32),
        datapoint_features=np.zeros((T, H * W, F), dtype=np.float32),
        blob_assignments=np.zeros((T, H * W), dtype=np.int32),
        blob_means=np.zeros((T, B, 3), dtype=np.float32),
        blob_covs=np.tile(np.eye(3), (T, B, 1, 1)).astype(np.float32),
        hyperblob_assignments=np.zeros((T, B), dtype=np.int32),
        hyperblob_means=np.zeros((T, HB, 3), dtype=np.float32),
        n_blobs=np.array([1, 1], dtype=np.int32),
        n_hyperblobs=np.array([1, 1], dtype=np.int32),
        img_dims=np.array([H, W], dtype=np.int32),
        focal_length=np.float32(520.0),
    )
    paths.track_manifest_path.write_text(
        json.dumps({"stages": {"dense": {"status": "success"}}}),
        encoding="utf-8",
    )


def _write_motion_and_rgb(paths) -> None:
    T, H, W = 2, 2, 2
    paths.npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        paths.npz_path,
        points_3d=np.zeros((T, H, W, 3)),
        motion_vectors_3d=np.zeros((T, H, W, 3)),
        colors=np.zeros((T, H, W, 3), dtype=np.uint8),
        intrinsics={"fx": 520, "fy": 520, "cx": 1, "cy": 1, "width": W, "height": H},
    )
    paths.rgb_frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(T):
        (paths.rgb_frames_dir / f"{i:05d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")


@patch("genmatter.custom.viz.export_to_rrd")
@patch("genmatter.custom.viz.load_viz_artifacts")
def test_run_viz_success(mock_load, mock_export, minimal_config_yaml, tmp_path) -> None:
    cfg = load_config(minimal_config_yaml)
    paths = resolve_video_paths(cfg, "viz_vid")
    _write_tracking_stack(paths)
    _write_motion_and_rgb(paths)

    mock_load.return_value = VizArtifacts(
        video_id="viz_vid",
        num_frames=2,
        height=2,
        width=2,
        focal_length=520.0,
        rgb_frame_paths=[paths.rgb_frames_dir / "00000.jpg", paths.rgb_frames_dir / "00001.jpg"],
        points_3d=np.zeros((2, 2, 2, 3)),
        motion_vectors_3d=np.zeros((2, 2, 2, 3)),
        colors=np.zeros((2, 2, 2, 3), dtype=np.uint8),
        tracking={},
        feature_rgb=np.zeros((2, 4, 3), dtype=np.uint8),
    )
    mock_export.return_value = 1024

    ui = PreprocessConsole()
    out_rrd = tmp_path / "custom.rrd"
    result = run_viz("viz_vid", cfg, ui, output_path=out_rrd)
    assert result.success
    mock_export.assert_called_once()


def test_run_viz_missing_tracking_raises(minimal_config_yaml) -> None:
    cfg = load_config(minimal_config_yaml)
    ui = PreprocessConsole()
    with pytest.raises(FileNotFoundError, match="Missing"):
        run_viz("no_track", cfg, ui)

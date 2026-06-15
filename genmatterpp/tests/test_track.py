"""Tests for genmatter.custom.track."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from genmatter.custom.config_schema import load_config
from genmatter.custom.console import PreprocessConsole
from genmatter.custom.paths import resolve_video_paths
from genmatter.custom.track import run_track
from genmatter.tracking.dino import DinoTrackingResult, DinoTrackingTimings


def _write_preprocess_artifacts(paths) -> None:
    paths.rgb_frames_dir.mkdir(parents=True, exist_ok=True)
    paths.npz_path.parent.mkdir(parents=True, exist_ok=True)
    paths.dino_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W = 2, 4, 4
    np.savez(
        paths.npz_path,
        points_3d=np.zeros((T, H, W, 3)),
        motion_vectors_3d=np.zeros((T, H, W, 3)),
        colors=np.zeros((T, H, W, 3)),
    )
    np.savez(
        paths.dino_path,
        pca_features_unnormalized=np.zeros((T, H, W, 3)),
        gaussian_means=np.zeros(3),
        gaussian_stds=np.ones(3),
    )
    paths.sam_path.parent.mkdir(parents=True, exist_ok=True)
    paths.sam_path.write_bytes(b"placeholder")
    manifest = {
        "video_id": paths.video_id,
        "config_fingerprint": "abc",
        "stages": {
            "frames": {"status": "success"},
            "motion_3d": {"status": "success"},
            "dino": {"status": "success"},
            "sam_frame0": {"status": "success"},
        },
    }
    paths.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@patch("genmatter.custom.track.run_dino_tracking")
def test_run_track_writes_manifest(mock_run, minimal_config_yaml, tmp_path) -> None:
    cfg = load_config(minimal_config_yaml)
    paths = resolve_video_paths(cfg, "vid_track")
    _write_preprocess_artifacts(paths)

    mock_run.return_value = DinoTrackingResult(
        video_id="vid_track",
        tracking_data=[],
        img_dims=(4, 4),
        subsampled_indices=np.array([0]),
        focal_length=520.0,
        timings=DinoTrackingTimings(
            load_seconds=0.1,
            num_frames=2,
            num_tracking_steps=1,
        ),
        gaussian_means=np.zeros(3),
        gaussian_stds=np.ones(3),
    )

    ui = PreprocessConsole()
    with patch("genmatter.custom.track.save_dense_tracking_npz", return_value=1000):
        result = run_track("vid_track", cfg, ui)

    assert result.success
    assert paths.track_manifest_path.is_file()
    data = json.loads(paths.track_manifest_path.read_text())
    assert data["stages"]["validate"]["status"] == "success"
    mock_run.assert_called_once()


def test_run_track_missing_preprocess_raises(minimal_config_yaml) -> None:
    cfg = load_config(minimal_config_yaml)
    ui = PreprocessConsole()
    with pytest.raises(FileNotFoundError, match="Preprocess manifest"):
        run_track("no_such_video", cfg, ui)

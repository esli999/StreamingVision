"""Tests for lightweight helpers in vda_preprocess_video."""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.motion_extraction_3d.vda_preprocess_video import (
    Motion3DTimings,
    is_image_folder,
    is_video_file,
    unproject_points,
)


def test_motion3d_timings_fps() -> None:
    t = Motion3DTimings(
        depth_seconds=2.0,
        depth_post_seconds=0.1,
        load_rgb_seconds=0.2,
        flow_seconds=4.0,
        save_seconds=0.5,
        num_frames=20,
        num_frame_pairs=19,
    )
    assert t.depth_fps == pytest.approx(10.0)
    assert t.flow_fps == pytest.approx(4.75)


def test_motion3d_timings_zero_duration() -> None:
    t = Motion3DTimings(0, 0, 0, 0, 0, 10, 9)
    assert t.depth_fps == 0.0
    assert t.flow_fps == 0.0


def test_unproject_points_center() -> None:
    x = np.array([[0, 1], [0, 1]])
    y = np.array([[0, 0], [1, 1]])
    z = np.ones((2, 2)) * 5.0
    pts = unproject_points(x, y, z, fx=520, fy=520, cx=0.5, cy=0.5)
    assert pts.shape == (2, 2, 3)
    assert pts[0, 0, 2] == pytest.approx(5.0)


def test_is_video_file(tmp_path) -> None:
    f = tmp_path / "a.mp4"
    f.touch()
    assert is_video_file(str(f))
    assert not is_video_file(str(tmp_path))


def test_is_image_folder(frame_dir) -> None:
    assert is_image_folder(str(frame_dir))

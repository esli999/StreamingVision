"""Shared pytest fixtures for GenMatter tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def default_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "custom_default.yaml"


@pytest.fixture
def minimal_config_yaml(tmp_path: Path) -> Path:
    """Small YAML config with a custom output root under tmp_path."""
    data = {
        "paths": {"custom_videos_root": str(tmp_path / "custom_videos")},
        "preprocess": {
            "pipeline": {"skip_existing": False, "force": False},
            "frames": {"max_len": 3, "skip_frames": 1, "max_res": 320},
            "motion_3d": {"max_len": 3, "device": "cpu"},
            "dino": {"device": "cpu"},
            "sam_frame0": {"frame_name": "00000.jpg"},
        },
        "tracking": {"num_blobs": 8},
    }
    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


@pytest.fixture
def tiny_mp4(tmp_path: Path) -> Path:
    """Short synthetic MP4 (5 frames) for frame-extraction tests."""
    path = tmp_path / "clip.mp4"
    w, h = 64, 48
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter could not open mp4v codec")
    for i in range(5):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 2] = min(255, i * 50)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def frame_dir(tmp_path: Path) -> Path:
    """Directory with two RGB JPEGs for DINO loader tests."""
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(2):
        img = np.zeros((28, 28, 3), dtype=np.uint8)
        img[:, :, i % 3] = 200
        cv2.imwrite(str(d / f"{i:05d}.jpg"), img)
    return d

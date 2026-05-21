"""Tests for custom preprocess orchestration (mocked GPU stages)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genmatter.custom.config_schema import load_config
from genmatter.custom.console import PreprocessConsole
from genmatter.custom.preprocess import _should_skip, run_preprocess
from genmatter.preprocessing.dino import DinoResult, DinoStepTimings
from genmatter.preprocessing.frames import ExtractFramesResult
from genmatter.preprocessing.motion_3d import Motion3DResult
from genmatter.preprocessing.sam_frame0 import SamFrame0Result
from preprocessing.motion_extraction_3d.vda_preprocess_video import Motion3DTimings


@pytest.fixture
def cfg(minimal_config_yaml: Path):
    return load_config(minimal_config_yaml)


def test_should_skip_respects_flags(cfg, tmp_path: Path) -> None:
    f = tmp_path / "exists.npz"
    f.write_bytes(b"x")
    cfg.preprocess.pipeline.skip_existing = True
    cfg.preprocess.pipeline.force = False
    assert _should_skip(f, cfg) is True

    cfg.preprocess.pipeline.force = True
    assert _should_skip(f, cfg) is False

    cfg.preprocess.pipeline.force = False
    cfg.preprocess.pipeline.skip_existing = False
    assert _should_skip(f, cfg) is False


@patch("genmatter.custom.preprocess.run_sam_frame0")
@patch("genmatter.custom.preprocess.load_dino_model")
@patch("genmatter.custom.preprocess.run_dino_extraction")
@patch("genmatter.custom.preprocess.run_motion_3d")
@patch("genmatter.custom.preprocess.extract_frames_from_mp4")
def test_run_preprocess_writes_manifest(
    mock_frames,
    mock_motion,
    mock_dino,
    mock_load_dino,
    mock_sam,
    cfg,
    tiny_mp4: Path,
    tmp_path: Path,
) -> None:
    video_id = "mock_clip"
    paths = __import__(
        "genmatter.custom.paths", fromlist=["resolve_video_paths"]
    ).resolve_video_paths(cfg, video_id)

    mock_frames.return_value = ExtractFramesResult(
        output_dir=paths.rgb_frames_dir,
        num_frames=3,
        elapsed_seconds=0.1,
        fps=30.0,
        frame_paths=[],
    )
    timings = Motion3DTimings(1, 0.1, 0.05, 2, 0.1, 3, 2)
    mock_motion.return_value = Motion3DResult(
        output_path=paths.npz_path,
        timings=timings,
        file_size_bytes=100,
    )
    mock_load_dino.return_value = (MagicMock(), MagicMock())
    mock_dino.return_value = DinoResult(
        output_path=paths.dino_path,
        num_frames=3,
        elapsed_seconds=1.0,
        fps=3.0,
        file_size_bytes=200,
        timings=DinoStepTimings(0.1, 0.5, 0.05, 0.35, 3),
    )
    mock_sam.return_value = SamFrame0Result(
        output_path=paths.sam_path,
        elapsed_seconds=0.5,
        file_size_bytes=50,
    )

    paths.rgb_frames_dir.mkdir(parents=True)
    (paths.rgb_frames_dir / "00000.jpg").write_bytes(b"fake-jpeg")

    ui = PreprocessConsole()
    result = run_preprocess(tiny_mp4, cfg, video_id, ui)

    assert result.success
    assert paths.manifest_path.is_file()
    manifest = json.loads(paths.manifest_path.read_text())
    assert manifest["video_id"] == video_id
    assert manifest["stages"]["frames"]["status"] == "success"
    mock_motion.assert_called_once()
    mock_dino.assert_called_once()
    mock_sam.assert_called_once()

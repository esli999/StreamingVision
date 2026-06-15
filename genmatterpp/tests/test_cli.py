"""Tests for genmatter.custom.cli."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from genmatter.custom.cli import main


def test_main_preprocess_missing_mp4(tmp_path) -> None:
    code = main(["preprocess", str(tmp_path / "missing.mp4")])
    assert code == 1


def test_main_track_requires_video_id() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["track"])
    assert exc.value.code == 2


@patch("genmatter.custom.cli.run_track")
@patch("genmatter.custom.cli.load_config")
def test_main_track_success(mock_load, mock_run, minimal_config_yaml) -> None:
    from genmatter.custom.config_schema import load_config
    from genmatter.custom.track import TrackResult
    from genmatter.custom.paths import resolve_video_paths

    cfg = load_config(minimal_config_yaml)
    mock_load.return_value = cfg
    paths = resolve_video_paths(cfg, "clip")
    mock_run.return_value = TrackResult(
        video_id="clip",
        paths=paths,
        manifest_path=paths.track_manifest_path,
        success=True,
    )

    code = main(
        [
            "track",
            "--video-id",
            "clip",
            "--config",
            str(minimal_config_yaml),
        ]
    )
    assert code == 0
    mock_run.assert_called_once()


@patch("genmatter.custom.cli.run_preprocess")
@patch("genmatter.custom.cli.load_config")
def test_main_preprocess_success(mock_load, mock_run, minimal_config_yaml, tiny_mp4) -> None:
    from genmatter.custom.config_schema import load_config
    from genmatter.custom.preprocess import PreprocessResult
    from genmatter.custom.paths import resolve_video_paths

    cfg = load_config(minimal_config_yaml)
    mock_load.return_value = cfg
    paths = resolve_video_paths(cfg, "clip")
    mock_run.return_value = PreprocessResult(
        video_id="clip",
        paths=paths,
        manifest_path=paths.manifest_path,
        success=True,
    )

    code = main(
        [
            "preprocess",
            str(tiny_mp4),
            "--config",
            str(minimal_config_yaml),
            "--video-id",
            "clip",
        ]
    )
    assert code == 0
    mock_run.assert_called_once()


@patch("genmatter.custom.cli.run_viz")
@patch("genmatter.custom.cli.load_config")
def test_main_viz_success(mock_load, mock_run, minimal_config_yaml, tmp_path) -> None:
    from genmatter.custom.config_schema import load_config
    from genmatter.custom.viz import VizResult
    from genmatter.custom.paths import resolve_video_paths

    cfg = load_config(minimal_config_yaml)
    mock_load.return_value = cfg
    paths = resolve_video_paths(cfg, "clip")
    mock_run.return_value = VizResult(
        video_id="clip",
        paths=paths,
        rrd_path=tmp_path / "x.rrd",
        success=True,
        file_size_bytes=100,
    )

    code = main(
        [
            "viz",
            "--video-id",
            "clip",
            "--config",
            str(minimal_config_yaml),
        ]
    )
    assert code == 0
    mock_run.assert_called_once()

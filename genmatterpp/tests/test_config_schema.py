"""Tests for genmatter.custom.config_schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from genmatter.custom.config_schema import (
    CustomConfig,
    load_config,
    _parse_scalar,
)


def test_load_default_config(default_config_path: Path) -> None:
    cfg = load_config(default_config_path)
    assert cfg.preprocess.motion_3d.encoder == "vitl"
    assert cfg.preprocess.dino.target_h == 520
    assert cfg.tracking.num_blobs == 500
    assert cfg.tracking.pipeline.skip_existing is False
    assert cfg.tracking.measure_fps is True
    assert cfg.viz.include_flow is True
    assert cfg.viz.point_radius == 0.015
    assert cfg.pseudo_gt.model == "sam2.1_l.pt"
    assert cfg.preprocess.sam_frame0.model == "sam2.1_l.pt"
    assert cfg.config_path == default_config_path.resolve()


def test_load_config_with_dot_override(minimal_config_yaml: Path) -> None:
    cfg = load_config(
        minimal_config_yaml,
        dot_overrides=["preprocess.frames.max_len=7", "tracking.num_blobs=12"],
    )
    assert cfg.preprocess.frames.max_len == 7
    assert cfg.tracking.num_blobs == 12


def test_dot_override_bool_and_null(minimal_config_yaml: Path) -> None:
    cfg = load_config(
        minimal_config_yaml,
        dot_overrides=["preprocess.pipeline.skip_existing=true"],
    )
    assert cfg.preprocess.pipeline.skip_existing is True


def test_invalid_set_raises(minimal_config_yaml: Path) -> None:
    with pytest.raises(ValueError, match="key=value"):
        load_config(minimal_config_yaml, dot_overrides=["not-a-valid-override"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("false", False),
        ("null", None),
        ("42", 42),
        ("3.5", 3.5),
        ("cuda", "cuda"),
    ],
)
def test_parse_scalar(raw: str, expected) -> None:
    assert _parse_scalar(raw) == expected


def test_config_fingerprint_stable(minimal_config_yaml: Path) -> None:
    a = load_config(minimal_config_yaml)
    b = load_config(minimal_config_yaml)
    assert a.config_fingerprint() == b.config_fingerprint()


def test_resolved_custom_videos_root_override(minimal_config_yaml: Path, tmp_path: Path) -> None:
    cfg = load_config(minimal_config_yaml)
    expected = (tmp_path / "custom_videos").resolve()
    assert cfg.resolved_custom_videos_root() == expected

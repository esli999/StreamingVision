"""Tests for genmatter.custom.paths."""

from __future__ import annotations

from pathlib import Path

from genmatter.custom.config_schema import load_config
from genmatter.custom.paths import ensure_video_dirs, resolve_video_paths


def test_resolve_video_paths(minimal_config_yaml: Path, tmp_path: Path) -> None:
    cfg = load_config(minimal_config_yaml)
    paths = resolve_video_paths(cfg, "my_clip")

    root = (tmp_path / "custom_videos" / "my_clip").resolve()
    assert paths.video_root == root
    assert paths.rgb_frames_dir == root / "rgb_frames" / "my_clip"
    assert paths.npz_path == root / "npzs" / "my_clip_3d_motion.npz"
    assert paths.dino_path == root / "dino" / "my_clip_dino_pca_per_pixel.npz"
    assert paths.sam_path == root / "SAM_frame0" / "my_clip_SAM_frame0.png"
    assert paths.manifest_path == root / "preprocess_manifest.json"
    assert paths.tracking_dir == root / "tracking"
    assert paths.tracking_npz == root / "tracking" / "tracking_dense.npz"
    assert paths.track_manifest_path == root / "tracking" / "track_manifest.json"
    assert paths.rrd_path == root / "tracking" / "my_clip.rrd"


def test_ensure_video_dirs_creates_parents(minimal_config_yaml: Path) -> None:
    cfg = load_config(minimal_config_yaml)
    paths = resolve_video_paths(cfg, "dir_test")
    ensure_video_dirs(paths)
    assert paths.rgb_frames_dir.is_dir()
    assert paths.npz_path.parent.is_dir()
    assert paths.dino_path.parent.is_dir()
    assert paths.sam_path.parent.is_dir()
    assert paths.tracking_dir.is_dir()

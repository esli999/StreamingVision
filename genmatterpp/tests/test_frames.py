"""Tests for genmatter.preprocessing.frames."""

from __future__ import annotations

from pathlib import Path

from genmatter.preprocessing.frames import (
    extract_frames_from_mp4,
    list_frame_files,
    probe_mp4_frame_count,
)


def test_probe_mp4_frame_count(tiny_mp4: Path) -> None:
    count = probe_mp4_frame_count(tiny_mp4)
    assert count >= 5


def test_extract_frames_from_mp4(tiny_mp4: Path, tmp_path: Path) -> None:
    out = tmp_path / "rgb"
    result = extract_frames_from_mp4(
        tiny_mp4,
        out,
        max_len=3,
        skip_frames=1,
        max_res=320,
    )
    assert result.num_frames == 3
    assert result.elapsed_seconds >= 0
    assert result.fps > 0
    assert (out / "00000.jpg").is_file()
    assert (out / "00002.jpg").is_file()
    assert not (out / "00003.jpg").exists()


def test_list_frame_files_sorted(frame_dir: Path) -> None:
    paths = list_frame_files(frame_dir)
    assert len(paths) == 2
    assert paths[0].name == "00000.jpg"
    assert paths[1].name == "00001.jpg"

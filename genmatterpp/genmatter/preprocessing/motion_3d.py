"""Wrapper around VDA+RAFT 3D motion extraction with timing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from preprocessing.motion_extraction_3d.vda_preprocess_video import Motion3DTimings


@dataclass
class Motion3DParams:
    encoder: str = "vitl"
    input_size: int = 518
    max_res: int = 1280
    max_len: int = -1
    target_fps: int = -1
    fp32: bool = False
    device: str = "cuda"
    skip_frames: int = 1
    subsample: float = 1.0


@dataclass
class Motion3DResult:
    output_path: Path
    timings: Motion3DTimings
    file_size_bytes: int


def run_motion_3d(
    input_path: Path,
    output_path: Path,
    params: Motion3DParams,
    *,
    on_step_start: Optional[Callable[[str], None]] = None,
    on_frame_progress: Optional[Callable[[int, int], None]] = None,
    verbose: bool = False,
    quiet: bool = True,
) -> Motion3DResult:
    """Run VDA+RAFT; imports VDA lazily to keep ``genmatter --help`` fast."""
    from preprocessing.motion_extraction_3d.vda_preprocess_video import (
        process_video_to_3d_data,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = process_video_to_3d_data(
        input_path=str(input_path),
        output_path=str(output_path),
        encoder=params.encoder,
        input_size=params.input_size,
        max_res=params.max_res,
        max_len=params.max_len,
        target_fps=params.target_fps,
        fp32=params.fp32,
        device=params.device,
        skip_frames=params.skip_frames,
        subsample=params.subsample,
        output_format="npz",
        verbose=verbose,
        on_step_start=on_step_start,
        on_frame_progress=on_frame_progress,
        return_timings=True,
        quiet=quiet,
    )
    path_str, timings = result
    out = Path(path_str)
    return Motion3DResult(
        output_path=out,
        timings=timings,
        file_size_bytes=out.stat().st_size if out.is_file() else 0,
    )

"""Shared preprocessing utilities for DAVIS benchmarks and custom MP4 pipeline."""

from genmatter.preprocessing.frames import ExtractFramesResult, extract_frames_from_mp4
from genmatter.preprocessing.motion_3d import Motion3DParams, Motion3DResult, run_motion_3d
from genmatter.preprocessing.dino import DinoParams, DinoResult, load_dino_model, run_dino_extraction
from genmatter.preprocessing.sam_frame0 import SamFrame0Params, SamFrame0Result, run_sam_frame0

__all__ = [
    "ExtractFramesResult",
    "extract_frames_from_mp4",
    "Motion3DParams",
    "Motion3DResult",
    "run_motion_3d",
    "DinoParams",
    "DinoResult",
    "load_dino_model",
    "run_dino_extraction",
    "SamFrame0Params",
    "SamFrame0Result",
    "run_sam_frame0",
]

"""Shared CLI flags for DAVIS GenMatter experiment scripts (e.g. ``--gt-init``)."""

from __future__ import annotations

import argparse
from types import ModuleType

import config

_KIND_TO_PAIR = {
    "tracking": (
        config.DAVIS_TRACKING_OUTPUT_DIR_SAM,
        config.DAVIS_TRACKING_OUTPUT_DIR_GT_INIT,
    ),
    "subsampling": (
        config.DAVIS_SUBSAMPLING_OUTPUT_DIR_SAM,
        config.DAVIS_SUBSAMPLING_OUTPUT_DIR_GT_INIT,
    ),
    "ablation": (
        config.DAVIS_ABLATION_OUTPUT_DIR_SAM,
        config.DAVIS_ABLATION_OUTPUT_DIR_GT_INIT,
    ),
}


def add_frame0_init_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gt-init",
        action="store_true",
        help=(
            "Use TAP-Vid frame-0 mask and ratio-based K-means init. "
            "Default is SAM frame-0 PNGs."
        ),
    )


def add_save_3wide_video_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-3wide-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write GenMatter 3-wide visualization MP4s under 3wide_videos/ (default: off). "
            "Pass --save-3wide-video to enable encoding and disk I/O."
        ),
    )


def add_skip_completed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Skip work when json_results/{video}_results.json already exists "
            "(resume after interrupt; tracking: one JSON per video under json_results/; "
            "subsampling & ablation: per subsample_* subfolder)."
        ),
    )


def configure_experiment_module(
    module: ModuleType, args: argparse.Namespace, kind: str
) -> None:
    if kind not in _KIND_TO_PAIR:
        raise ValueError(f"unknown DAVIS experiment kind: {kind!r}")
    sam_dir, gt_init_dir = _KIND_TO_PAIR[kind]
    use_sam = not bool(getattr(args, "gt_init", False))
    module.USE_SAM_FRAME0 = use_sam
    module.EXPERIMENT_SAVE_DIR = str(sam_dir if use_sam else gt_init_dir)
    module.SAVE_3WIDE_VIDEO = bool(getattr(args, "save_3wide_video", False))
    module.SKIP_COMPLETED = bool(getattr(args, "skip_completed", False))

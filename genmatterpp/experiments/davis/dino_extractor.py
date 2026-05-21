#!/usr/bin/env python3
"""Batch DINO feature extraction for TAP-Vid DAVIS RGB frame folders."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config
from genmatter.preprocessing.dino import (
    DEFAULT_N_COMPONENTS,
    DEFAULT_PATCH_SIZE,
    DEFAULT_TARGET_H,
    DEFAULT_TARGET_W,
    DinoParams,
    load_dino_model,
    process_video_dino,
)

BASE_VIDEO_PATH = str(config.DAVIS_RGB_PATH)
OUTPUT_DIR = str(config.DAVIS_DINO_PATH)


def main() -> None:
    params = DinoParams(
        target_h=DEFAULT_TARGET_H,
        target_w=DEFAULT_TARGET_W,
        patch_size=DEFAULT_PATCH_SIZE,
        n_components=DEFAULT_N_COMPONENTS,
    )
    print("Loading DINO model...")
    model, device = load_dino_model(params.device)
    print(f"Model loaded on {device}")

    print(f"Processing {len(config.TAPVID_DAVIS_VIDEO_NAMES)} videos...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    successful = failed = 0
    for i, video_name in enumerate(config.TAPVID_DAVIS_VIDEO_NAMES, 1):
        print(f"[{i}/{len(config.TAPVID_DAVIS_VIDEO_NAMES)}] Processing {video_name}...")
        if process_video_dino(
            video_name, BASE_VIDEO_PATH, model, device, OUTPUT_DIR, params
        ):
            successful += 1
        else:
            failed += 1
        gc.collect()
        print()

    print("=" * 60)
    print("Summary:")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(config.TAPVID_DAVIS_VIDEO_NAMES)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

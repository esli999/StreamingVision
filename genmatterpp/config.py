"""
Central paths and shared experiment constants for GenMatterPlusPlus (DAVIS + custom MP4).

DAVIS inputs live under ``<repo>/assets/tapvid_davis_30_videos_processed/``.
Custom MP4 preprocess outputs live under ``<repo>/assets/custom_videos/``.
See README and docs/EXPERIMENTS.md for layout.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

# SAM / Ultralytics checkpoints (see ``experiments/davis/sam_frame0_extractor.py``)
DEEPLEARNING_WEIGHTS_DIR = (
    Path(os.environ["GENMATTER_DEEPLEARNING_WEIGHTS_DIR"]).expanduser().resolve()
    if os.environ.get("GENMATTER_DEEPLEARNING_WEIGHTS_DIR")
    else (REPO_ROOT / "assets" / "deeplearning_weights").resolve()
)


def _resolve_assets_parent() -> Path:
    """Directory that contains ``tapvid_davis_30_videos_processed/``."""
    if os.environ.get("GENMATTER_DAVIS_DIR"):
        return Path(os.environ["GENMATTER_DAVIS_DIR"])
    local = REPO_ROOT / "assets"
    if local.is_dir() and not local.is_symlink():
        return local.resolve()
    return local


def _resolve_data_dir() -> Path:
    if os.environ.get("GENMATTER_DATA_DIR"):
        return Path(os.environ["GENMATTER_DATA_DIR"])
    if os.environ.get("GENMATTER_ASSETS_DIR"):
        return Path(os.environ["GENMATTER_ASSETS_DIR"])
    return REPO_ROOT


GENMATTER_DATA_DIR = _resolve_data_dir()
GENMATTER_LOCAL_ASSETS = GENMATTER_DATA_DIR / "assets"

# 30 TAP-Vid DAVIS videos (DINO, tracking, CoTracker, populate docs)
TAPVID_DAVIS_VIDEO_NAMES = (
    "blackswan",
    "bike-packing",
    "bmx-trees",
    "breakdance",
    "camel",
    "car-roundabout",
    "car-shadow",
    "cows",
    "dance-twirl",
    "dog",
    "dogs-jump",
    "drift-chicane",
    "drift-straight",
    "goat",
    "gold-fish",
    "horsejump-high",
    "india",
    "judo",
    "kite-surf",
    "lab-coat",
    "libby",
    "loading",
    "mbike-trick",
    "motocross-jump",
    "paragliding-launch",
    "parkour",
    "pigs",
    "scooter-black",
    "shooting",
    "soapbox",
)

RESULTS_DIR = Path(os.environ.get("GENMATTER_RESULTS_DIR", REPO_ROOT / "results"))

# DAVIS: parent directory must contain ``tapvid_davis_30_videos_processed/``
DAVIS_PARENT_DIR = _resolve_assets_parent()
DAVIS_BASE = DAVIS_PARENT_DIR / "tapvid_davis_30_videos_processed"

DAVIS_3D_MOTION_PATH = DAVIS_BASE / "tapvid_davis_npzs"
_TAPVID_DAVIS_MAX_FRAMES: int | None = None


def tapvid_davis_max_frames(
    video_names: tuple[str, ...] | None = None,
    *,
    motion_path: Path | None = None,
) -> int:
    """Longest frame count among TAP-Vid DAVIS motion NPZs (cached)."""
    global _TAPVID_DAVIS_MAX_FRAMES
    if (
        _TAPVID_DAVIS_MAX_FRAMES is not None
        and video_names is None
        and motion_path is None
    ):
        return _TAPVID_DAVIS_MAX_FRAMES

    import numpy as np

    names = video_names or TAPVID_DAVIS_VIDEO_NAMES
    base = motion_path or DAVIS_3D_MOTION_PATH
    max_t = 0
    for vid in names:
        for suffix in ("_3d_motion.npz", "_3d_data.npz"):
            path = base / f"{vid}{suffix}"
            if path.is_file():
                with np.load(path) as z:
                    max_t = max(max_t, int(z["points_3d"].shape[0]))
                break
    if max_t < 1:
        raise FileNotFoundError(
            f"No DAVIS motion NPZs found under {base} for TAPVID_DAVIS_VIDEO_NAMES"
        )
    if video_names is None and motion_path is None:
        _TAPVID_DAVIS_MAX_FRAMES = max_t
    return max_t


DAVIS_SEGMASKS_PATH = DAVIS_BASE / "tapvid_davis_segmasks"
DAVIS_RGB_PATH = DAVIS_BASE / "tapvid_davis_rgb_frames"
DAVIS_DINO_PATH = DAVIS_BASE / "tapvid_davis_dino"
DAVIS_SAM_FRAME0_PATH = DAVIS_BASE / "tapvid_davis_SAM_frame0"

# Custom MP4 pipeline (``genmatter preprocess`` / ``genmatter track``)
CUSTOM_VIDEOS_BASE = (
    Path(os.environ["GENMATTER_CUSTOM_VIDEOS_DIR"]).expanduser().resolve()
    if os.environ.get("GENMATTER_CUSTOM_VIDEOS_DIR")
    else (GENMATTER_LOCAL_ASSETS / "custom_videos")
)

# DAVIS: separate folders for SAM vs GT (TAP-Vid) frame-0 initialization.
DAVIS_TRACKING_OUTPUT_DIR_SAM = RESULTS_DIR / "davis_tracking_sam"
DAVIS_TRACKING_OUTPUT_DIR_GT_INIT = RESULTS_DIR / "davis_tracking_gt_init"
DAVIS_SUBSAMPLING_OUTPUT_DIR_SAM = RESULTS_DIR / "davis_subsampling_sam"
DAVIS_SUBSAMPLING_OUTPUT_DIR_GT_INIT = RESULTS_DIR / "davis_subsampling_gt_init"
DAVIS_ABLATION_OUTPUT_DIR_SAM = RESULTS_DIR / "davis_ablation_sam"
DAVIS_ABLATION_OUTPUT_DIR_GT_INIT = RESULTS_DIR / "davis_ablation_gt_init"

# Legacy names — SAM runs (matches historical ``results/davis_*`` layout).
DAVIS_TRACKING_OUTPUT_DIR = DAVIS_TRACKING_OUTPUT_DIR_SAM
DAVIS_SUBSAMPLING_OUTPUT_DIR = DAVIS_SUBSAMPLING_OUTPUT_DIR_SAM
DAVIS_ABLATION_OUTPUT_DIR = DAVIS_ABLATION_OUTPUT_DIR_SAM
COTRACKER_OUTPUT_DIR = RESULTS_DIR / "cotracker_baseline"
POSTPROCESSING_OUTPUT_DIR = RESULTS_DIR / "postprocessing"

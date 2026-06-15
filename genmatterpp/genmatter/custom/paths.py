"""Output path layout for custom video preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from genmatter.custom.config_schema import CustomConfig


@dataclass(frozen=True)
class VideoPaths:
    video_id: str
    video_root: Path
    rgb_frames_dir: Path
    npz_path: Path
    dino_path: Path
    sam_path: Path
    manifest_path: Path
    tracking_dir: Path
    tracking_npz: Path
    track_manifest_path: Path
    rrd_path: Path
    pseudo_gt_dir: Path
    pseudo_gt_segmasks_dir: Path
    pseudo_gt_manifest: Path
    pseudo_gt_annotations_root: Path


def resolve_video_paths(cfg: CustomConfig, video_id: str) -> VideoPaths:
    root = cfg.resolved_custom_videos_root() / video_id
    pseudo_gt = root / "pseudo_gt_sam"
    return VideoPaths(
        video_id=video_id,
        video_root=root,
        rgb_frames_dir=root / "rgb_frames" / video_id,
        npz_path=root / "npzs" / f"{video_id}_3d_motion.npz",
        dino_path=root / "dino" / f"{video_id}_dino_pca_per_pixel.npz",
        sam_path=root / "SAM_frame0" / f"{video_id}_SAM_frame0.png",
        manifest_path=root / "preprocess_manifest.json",
        tracking_dir=root / "tracking",
        tracking_npz=root / "tracking" / "tracking_dense.npz",
        track_manifest_path=root / "tracking" / "track_manifest.json",
        rrd_path=root / "tracking" / f"{video_id}.rrd",
        pseudo_gt_dir=pseudo_gt,
        pseudo_gt_segmasks_dir=pseudo_gt / "segmasks" / video_id,
        pseudo_gt_manifest=pseudo_gt / "manifest.json",
        pseudo_gt_annotations_root=pseudo_gt / "segmasks",
    )


def ensure_video_dirs(paths: VideoPaths) -> None:
    paths.rgb_frames_dir.mkdir(parents=True, exist_ok=True)
    paths.npz_path.parent.mkdir(parents=True, exist_ok=True)
    paths.dino_path.parent.mkdir(parents=True, exist_ok=True)
    paths.sam_path.parent.mkdir(parents=True, exist_ok=True)
    paths.tracking_dir.mkdir(parents=True, exist_ok=True)

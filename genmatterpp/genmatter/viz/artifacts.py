"""Load preprocess + tracking artifacts for visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np

from genmatter.custom.paths import VideoPaths


@dataclass
class VizArtifacts:
    video_id: str
    num_frames: int
    height: int
    width: int
    focal_length: float
    rgb_frame_paths: list[Path]
    points_3d: np.ndarray
    motion_vectors_3d: np.ndarray
    colors: np.ndarray
    tracking: dict[str, np.ndarray]
    feature_rgb: np.ndarray


def load_viz_artifacts(paths: VideoPaths) -> VizArtifacts:
    """Load motion NPZ, tracking_dense NPZ, and RGB frame list."""
    if not paths.tracking_npz.is_file():
        raise FileNotFoundError(
            f"Tracking NPZ not found: {paths.tracking_npz}\n"
            "Run: uv run genmatter track --video-id " + paths.video_id
        )
    if not paths.npz_path.is_file():
        raise FileNotFoundError(f"Motion NPZ not found: {paths.npz_path}")

    motion = np.load(paths.npz_path)
    points_3d = motion["points_3d"]
    motion_vectors_3d = motion["motion_vectors_3d"]
    colors = motion["colors"]
    motion.close()

    track = np.load(paths.tracking_npz)
    tracking = {k: track[k] for k in track.files}
    track.close()

    T_motion, H, W, _ = points_3d.shape
    T_track = int(tracking["datapoint_positions"].shape[0])
    if T_motion != T_track:
        raise ValueError(
            f"Frame count mismatch: motion NPZ has {T_motion} frames, "
            f"tracking_dense has {T_track}"
        )

    img_dims = tracking["img_dims"]
    Ht, Wt = int(img_dims[0]), int(img_dims[1])
    if (Ht, Wt) != (H, W):
        raise ValueError(
            f"Spatial dims mismatch: motion ({H}, {W}) vs tracking img_dims ({Ht}, {Wt})"
        )

    rgb_paths = sorted(glob(str(paths.rgb_frames_dir / "*.jpg")))
    if len(rgb_paths) < T_track:
        rgb_paths = sorted(glob(str(paths.rgb_frames_dir / "*.png")))
    if len(rgb_paths) < T_track:
        raise FileNotFoundError(
            f"Need {T_track} RGB frames in {paths.rgb_frames_dir}, found {len(rgb_paths)}"
        )
    rgb_paths = [Path(p) for p in rgb_paths[:T_track]]

    focal = float(tracking.get("focal_length", np.array(520.0)))
    features = tracking["datapoint_features"]

    from genmatter.viz.colors import features_to_rgb_pca

    feature_rgb = features_to_rgb_pca(features)

    return VizArtifacts(
        video_id=paths.video_id,
        num_frames=T_track,
        height=H,
        width=W,
        focal_length=focal,
        rgb_frame_paths=rgb_paths,
        points_3d=points_3d,
        motion_vectors_3d=motion_vectors_3d,
        colors=colors,
        tracking=tracking,
        feature_rgb=feature_rgb,
    )


def load_meta(paths: VideoPaths) -> dict:
    meta_path = paths.tracking_npz.with_suffix(".meta.json")
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}

"""Load and validate YAML configuration for the custom GenMatter pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

import config as repo_config


@dataclass
class PathsConfig:
    custom_videos_root: str | None = None
    deeplearning_weights_dir: str | None = None
    video_depth_anything_path: str | None = None


@dataclass
class PipelineConfig:
    skip_existing: bool = False
    force: bool = False


@dataclass
class FramesConfig:
    max_len: int = -1
    target_fps: int = -1
    skip_frames: int = 1
    max_res: int = 1280
    jpg_quality: int = 95
    frame_name_pattern: str = "{:05d}.jpg"


@dataclass
class Motion3DConfig:
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
class DinoConfig:
    target_h: int = 520
    target_w: int = 960
    patch_size: int = 14
    n_components: int = 10
    device: str = "cuda"


@dataclass
class SamFrame0Config:
    model: str = "sam2.1_l.pt"
    min_threshold: float = 0.15
    seed: int = 42
    frame_name: str = "00000.jpg"


@dataclass
class PseudoGtSettings:
    """SAM2 settings for ``genmatter pseudo-gt`` (full-video segmentations)."""

    model: str = "sam2.1_l.pt"
    min_threshold: float = 0.15
    iou_threshold: float = 0.3
    prefer_video: bool = True
    # Inference size for SAM2 video (output masks are always native frame resolution)
    video_imgsz: int = 1024
    max_objects: int | None = 12
    video_chunk_frames: int = 16
    detect_new_objects: bool = True
    new_object_iou_threshold: float | None = None  # default: same as iou_threshold


@dataclass
class PreprocessConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    motion_3d: Motion3DConfig = field(default_factory=Motion3DConfig)
    dino: DinoConfig = field(default_factory=DinoConfig)
    sam_frame0: SamFrame0Config = field(default_factory=SamFrame0Config)


@dataclass
class TrackingHyperparamsConfig:
    sigma_F: float = 2.0
    sigma_F_H: float = 2.0
    outlier_prob: float = 5.0
    outlier_velocity_gamma_shape: float = 5.0
    outlier_velocity_gamma_rate: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    sigma_H: float = 25.0
    sigma_V: float = 1e14
    translation_gaussian_scale: float = 0.2
    translation_max_radius: float = 0.35
    translation_num_radii_cells: int = 15
    translation_theta_step_deg: float = 15.0
    rotation_vmf_kappa: float = 100.0
    rotation_angle_max_deg: float = 25.0
    rotation_angle_step_deg: float = 0.375


@dataclass
class TrackingPipelineConfig:
    skip_existing: bool = False
    force: bool = False


@dataclass
class TrackingConfig:
    pipeline: TrackingPipelineConfig = field(default_factory=TrackingPipelineConfig)
    num_blobs: int = 500
    num_hyperblobs: int = 9
    datapoint_retain_pct: float = 78.125
    random_seed: int = 42
    focal_length: float = 520.0
    use_sam_frame0: bool = True
    init_gibbs_sweeps: int = 15
    tracking_outlier_prob: float = 1e-28
    dense_disable_outlier_prob: bool = False
    calibrate_feature_sigmas: bool = False
    blob_counting_threshold: int = 0
    measure_fps: bool = True
    hyperparams: TrackingHyperparamsConfig = field(default_factory=TrackingHyperparamsConfig)


@dataclass
class VizConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    point_radius: float = 0.015
    flow_arrow_scale: float = 1.0
    ellipsoid_sigma_scale: float = 1.0
    feature_pca_seed: int = 42
    include_assignment_colors: bool = True
    include_hyperblob_assignment_colors: bool = True
    include_flow: bool = True
    include_depth: bool = True


@dataclass
class CustomConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    pseudo_gt: PseudoGtSettings = field(default_factory=PseudoGtSettings)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    config_path: Path | None = None

    def resolved_custom_videos_root(self) -> Path:
        if self.paths.custom_videos_root:
            return Path(self.paths.custom_videos_root).expanduser().resolve()
        return Path(repo_config.CUSTOM_VIDEOS_BASE).resolve()

    def resolved_weights_dir(self) -> Path:
        if self.paths.deeplearning_weights_dir:
            return Path(self.paths.deeplearning_weights_dir).expanduser().resolve()
        return Path(repo_config.DEEPLEARNING_WEIGHTS_DIR).resolve()

    def config_fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _merge_dict(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def _dict_to_dataclass(data: dict) -> CustomConfig:
    paths = PathsConfig(**data.get("paths", {}))
    pre = data.get("preprocess", {})
    preprocess = PreprocessConfig(
        pipeline=PipelineConfig(**pre.get("pipeline", {})),
        frames=FramesConfig(**pre.get("frames", {})),
        motion_3d=Motion3DConfig(**pre.get("motion_3d", {})),
        dino=DinoConfig(**pre.get("dino", {})),
        sam_frame0=SamFrame0Config(**pre.get("sam_frame0", {})),
    )
    tr = data.get("tracking", {})
    tracking = TrackingConfig(
        pipeline=TrackingPipelineConfig(**tr.get("pipeline", {})),
        **{k: v for k, v in tr.items() if k not in ("hyperparams", "pipeline")},
        hyperparams=TrackingHyperparamsConfig(**tr.get("hyperparams", {})),
    )
    vz = data.get("viz", {})
    viz = VizConfig(
        pipeline=PipelineConfig(**vz.get("pipeline", {})),
        **{k: v for k, v in vz.items() if k != "pipeline"},
    )
    pg = data.get("pseudo_gt", {})
    pseudo_gt = PseudoGtSettings(**pg)
    return CustomConfig(
        paths=paths,
        preprocess=preprocess,
        pseudo_gt=pseudo_gt,
        tracking=tracking,
        viz=viz,
    )


def load_config(
    config_path: Path,
    dot_overrides: list[str] | None = None,
) -> CustomConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if dot_overrides:
        for item in dot_overrides:
            if "=" not in item:
                raise ValueError(f"Invalid --set (expected key=value): {item}")
            key_path, _, value_str = item.partition("=")
            keys = key_path.split(".")
            node = raw
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = _parse_scalar(value_str)

    cfg = _dict_to_dataclass(raw)
    cfg.config_path = config_path.resolve()

    if cfg.paths.video_depth_anything_path:
        import os

        os.environ["GENMATTER_VIDEO_DEPTH_ANYTHING_PATH"] = str(
            Path(cfg.paths.video_depth_anything_path).expanduser()
        )

    return cfg


def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() == "null":
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s

"""Build pseudo-GT SAM segmentations + correspondence for custom videos."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from genmatter.custom.config_schema import CustomConfig
from genmatter.custom.paths import VideoPaths, resolve_video_paths
from genmatter.pseudo_gt.colors import (
    build_track_color_palette,
    label_map_to_rgb,
    max_track_id_from_label_maps,
)
from genmatter.pseudo_gt.correspondence import (
    FramePairCorrespondence,
    build_from_sam2_track_ids,
    propagate_track_ids,
)
from genmatter.pseudo_gt.sam_video import segment_sequence


@dataclass
class PseudoGtConfig:
    model: str = "sam2.1_l.pt"
    min_threshold: float = 0.15
    iou_threshold: float = 0.3
    prefer_video: bool = True
    video_imgsz: int = 1024
    max_objects: int | None = 12
    video_chunk_frames: int = 16
    detect_new_objects: bool = True
    new_object_iou_threshold: float | None = None
    force: bool = False


def pseudo_gt_config_from_yaml(cfg: CustomConfig) -> PseudoGtConfig:
    s = cfg.pseudo_gt
    return PseudoGtConfig(
        model=s.model,
        min_threshold=s.min_threshold,
        iou_threshold=s.iou_threshold,
        prefer_video=s.prefer_video,
        video_imgsz=s.video_imgsz,
        max_objects=s.max_objects,
        video_chunk_frames=s.video_chunk_frames,
        detect_new_objects=s.detect_new_objects,
        new_object_iou_threshold=s.new_object_iou_threshold,
    )


@dataclass
class PseudoGtResult:
    video_id: str
    manifest_path: Path
    num_frames: int
    method: str
    success: bool


def _pseudo_gt_root(paths: VideoPaths) -> Path:
    return paths.video_root / "pseudo_gt_sam"


def pseudo_gt_segmasks_dir(paths: VideoPaths) -> Path:
    return _pseudo_gt_root(paths) / "segmasks" / paths.video_id


def pseudo_gt_segmasks_colored_dir(paths: VideoPaths) -> Path:
    return _pseudo_gt_root(paths) / "segmasks_colored" / paths.video_id


def pseudo_gt_manifest_path(paths: VideoPaths) -> Path:
    return _pseudo_gt_root(paths) / "manifest.json"


def pseudo_gt_annotations_root(paths: VideoPaths) -> Path:
    """Root passed to ``get_segmentation_mask`` (parent of per-video folders)."""
    return _pseudo_gt_root(paths) / "segmasks"


def should_skip(paths: VideoPaths, force: bool) -> bool:
    if force:
        return False
    manifest = pseudo_gt_manifest_path(paths)
    return manifest.is_file()


def build_pseudo_gt(
    video_id: str,
    cfg: CustomConfig,
    pg_cfg: PseudoGtConfig | None = None,
    *,
    on_segment_progress: Callable[[int, int], None] | None = None,
) -> PseudoGtResult:
    pg = pg_cfg or pseudo_gt_config_from_yaml(cfg)
    paths = resolve_video_paths(cfg, video_id)
    if not paths.rgb_frames_dir.is_dir():
        raise FileNotFoundError(f"RGB frames missing: {paths.rgb_frames_dir}")

    if should_skip(paths, pg.force):
        manifest = pseudo_gt_manifest_path(paths)
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)
        return PseudoGtResult(
            video_id=video_id,
            manifest_path=manifest,
            num_frames=int(data.get("num_frames", 0)),
            method=str(data.get("method", "skipped")),
            success=True,
        )

    per_frame_masks, method, (h, w), sam2_track_ids = segment_sequence(
        paths.rgb_frames_dir,
        model=pg.model,
        weights_dir=cfg.resolved_weights_dir(),
        min_threshold=pg.min_threshold,
        prefer_video=pg.prefer_video,
        video_imgsz=pg.video_imgsz,
        max_objects=pg.max_objects,
        video_chunk_frames=pg.video_chunk_frames,
        iou_threshold=pg.iou_threshold,
        detect_new_objects=pg.detect_new_objects,
        new_object_iou_threshold=pg.new_object_iou_threshold,
        on_progress=on_segment_progress,
        show_tqdm=on_segment_progress is None,
    )

    if sam2_track_ids is not None:
        label_maps, frame_pairs, tracks = build_from_sam2_track_ids(
            per_frame_masks,
            sam2_track_ids,
            h,
            w,
            iou_threshold=pg.iou_threshold,
        )
    else:
        label_maps, frame_pairs, tracks = propagate_track_ids(
            per_frame_masks,
            iou_threshold=pg.iou_threshold,
        )

    out_dir = pseudo_gt_segmasks_dir(paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    color_dir = pseudo_gt_segmasks_colored_dir(paths)
    color_dir.mkdir(parents=True, exist_ok=True)
    color_seed = int(cfg.preprocess.sam_frame0.seed)
    palette = build_track_color_palette(max_track_id_from_label_maps(label_maps), seed=color_seed)
    for t, lab in enumerate(label_maps):
        # PNG uint16 for instance ids; evaluation treats label > 0 as foreground
        cv2.imwrite(str(out_dir / f"{t:05d}.png"), lab)
        rgb = label_map_to_rgb(lab, palette)
        cv2.imwrite(str(color_dir / f"{t:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    corr_path = _pseudo_gt_root(paths) / "correspondence.json"
    with open(corr_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_id": video_id,
                "frame_pairs": [asdict(fp) for fp in frame_pairs],
                "tracks": tracks,
            },
            f,
            indent=2,
        )

    meta_path = _pseudo_gt_root(paths) / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_id": video_id,
                "img_dims": [h, w],
                "num_frames": len(label_maps),
                "method": method,
                "model": pg.model,
                "iou_threshold": pg.iou_threshold,
            },
            f,
            indent=2,
        )

    manifest_path = pseudo_gt_manifest_path(paths)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_id": video_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "num_frames": len(label_maps),
                "method": method,
                "model": pg.model,
                "segmasks_dir": str(out_dir),
                "segmasks_colored_dir": str(color_dir),
                "color_seed": color_seed,
                "correspondence": str(corr_path),
                "annotations_root": str(pseudo_gt_annotations_root(paths)),
            },
            f,
            indent=2,
        )

    return PseudoGtResult(
        video_id=video_id,
        manifest_path=manifest_path,
        num_frames=len(label_maps),
        method=method,
        success=True,
    )


def colorize_pseudo_gt_segmasks(
    video_id: str,
    cfg: CustomConfig,
    *,
    seed: int | None = None,
) -> Path:
    """Write RGB previews from existing uint16 label PNGs (no re-segmentation)."""
    from genmatter.pseudo_gt.colors import colorize_label_dir

    paths = resolve_video_paths(cfg, video_id)
    label_dir = pseudo_gt_segmasks_dir(paths)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label masks missing: {label_dir}")
    color_dir = pseudo_gt_segmasks_colored_dir(paths)
    color_seed = int(seed if seed is not None else cfg.preprocess.sam_frame0.seed)
    n = colorize_label_dir(label_dir, color_dir, seed=color_seed)
    manifest_path = pseudo_gt_manifest_path(paths)
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        data["segmasks_colored_dir"] = str(color_dir)
        data["color_seed"] = color_seed
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    return color_dir

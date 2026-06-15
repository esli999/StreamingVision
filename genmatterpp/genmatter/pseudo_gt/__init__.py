"""Pseudo ground-truth from SAM2 for custom video evaluation."""

from genmatter.pseudo_gt.build import (
    PseudoGtConfig,
    PseudoGtResult,
    build_pseudo_gt,
    pseudo_gt_annotations_root,
    pseudo_gt_manifest_path,
    pseudo_gt_segmasks_dir,
)

__all__ = [
    "PseudoGtConfig",
    "PseudoGtResult",
    "build_pseudo_gt",
    "pseudo_gt_annotations_root",
    "pseudo_gt_manifest_path",
    "pseudo_gt_segmasks_dir",
]

"""Evaluation wrapper and outlier gate tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from genmatter.evaluation import evaluate_custom_instance_tracking, evaluate_custom_tracking


def test_evaluate_custom_instance_tracking_synthetic(tmp_path) -> None:
    vid = "tiny"
    seg_root = tmp_path / "segmasks"
    seg_dir = seg_root / vid
    seg_dir.mkdir(parents=True)
    h, w = 4, 4
    for t in range(2):
        lab = np.zeros((h, w), dtype=np.uint16)
        lab[1:3, 1:3] = 1
        cv2.imwrite(str(seg_dir / f"{t:05d}.png"), lab)

    n = h * w
    tracking_data = []
    for _t in range(2):
        tracking_data.append(
            {
                "n_blobs": 2,
                "blob_assignments": np.zeros(n, dtype=np.int32),
                "blob_weights": np.array([0.5, 0.5], dtype=np.float32),
            }
        )

    metrics = evaluate_custom_instance_tracking(
        vid,
        tracking_data,
        annotations_path=seg_root,
        img_dims=(h, w),
    )
    assert "avg_mean_gt_iou" in metrics
    assert 0.0 <= metrics["avg_mean_gt_iou"] <= 1.0


def test_evaluate_custom_tracking_synthetic(tmp_path) -> None:
    vid = "tiny"
    seg_root = tmp_path / "segmasks"
    seg_dir = seg_root / vid
    seg_dir.mkdir(parents=True)
    h, w = 4, 4
    for t in range(2):
        lab = np.zeros((h, w), dtype=np.uint16)
        lab[1:3, 1:3] = 1
        cv2.imwrite(str(seg_dir / f"{t:05d}.png"), lab)

    n = h * w
    tracking_data = []
    for t in range(2):
        tracking_data.append(
            {
                "n_blobs": 2,
                "blob_assignments": np.zeros(n, dtype=np.int32),
                "blob_weights": np.array([0.5, 0.5], dtype=np.float32),
            }
        )

    metrics = evaluate_custom_tracking(
        vid,
        tracking_data,
        annotations_path=seg_root,
        img_dims=(h, w),
    )
    assert "avg_matter_weighted_jaccard_fixed" in metrics
    assert 0.0 <= metrics["avg_matter_weighted_jaccard_fixed"] <= 1.0

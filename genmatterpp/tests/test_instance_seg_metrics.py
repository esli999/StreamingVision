"""Tests for multi-instance Jaccard metrics."""

from __future__ import annotations

import numpy as np

from genmatter.instance_seg_metrics import (
    frame_instance_jaccard_metrics,
    hungarian_match_instances,
    instance_iou_matrix,
)


def test_perfect_match_two_objects() -> None:
    gt = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32)
    pred = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32)
    m = frame_instance_jaccard_metrics(gt, pred, score_iou_threshold=0.5)
    assert m["mean_matched_iou"] == 1.0
    assert m["mean_gt_iou"] == 1.0
    assert m["gt_recall_at_thresh"] == 1.0
    assert m["pred_precision_at_thresh"] == 1.0


def test_unmatched_gt_lowers_mean_gt_iou() -> None:
    gt = np.array([[1, 1, 0], [2, 2, 0]], dtype=np.int32)
    pred = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.int32)
    m = frame_instance_jaccard_metrics(gt, pred, score_iou_threshold=0.5)
    assert m["mean_matched_iou"] == 1.0
    assert m["mean_gt_iou"] == 0.5
    assert m["gt_recall_at_thresh"] == 0.5


def test_hungarian_picks_higher_iou_pair() -> None:
    gt = np.zeros((4, 4), dtype=np.int32)
    gt[:, :2] = 1
    gt[:, 2:] = 2
    pred = np.zeros((4, 4), dtype=np.int32)
    pred[:3, :] = 1  # mostly object 1
    pred[3:, 2:] = 2  # small object 2 region
    iou = instance_iou_matrix(gt, pred)
    matches = hungarian_match_instances(iou, iou_threshold=0.1)
    assert len(matches) >= 1
    assert max(m[2] for m in matches) > 0.3

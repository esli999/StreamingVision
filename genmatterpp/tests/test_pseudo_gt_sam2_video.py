"""Tests for SAM2 video tracked pseudo-GT helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from genmatter.pseudo_gt.correspondence import align_chunk_track_ids, build_from_sam2_track_ids
from genmatter.pseudo_gt.sam_video import (
    resize_mask_to_hw,
    select_bboxes_by_area,
    tracked_masks_from_result,
)


def _result_with_masks_and_cls() -> SimpleNamespace:
    m0 = np.zeros((4, 4), dtype=bool)
    m0[0, 0] = True
    m1 = np.zeros((4, 4), dtype=bool)
    m1[1:, 1:] = True
    return SimpleNamespace(
        masks=SimpleNamespace(data=torch.stack([torch.tensor(m0), torch.tensor(m1)])),
        boxes=SimpleNamespace(cls=torch.tensor([1.0, 0.0])),
    )


def test_tracked_masks_from_result_sorts_by_track_id() -> None:
    masks, tids = tracked_masks_from_result(_result_with_masks_and_cls(), 0.15)
    assert tids == [0, 1]
    assert len(masks) == 2


def test_resize_mask_to_hw() -> None:
    small = np.zeros((4, 8), dtype=bool)
    small[1, 2] = True
    out = resize_mask_to_hw(small, 8, 16)
    assert out.shape == (8, 16)
    assert out[2, 4]


def test_select_bboxes_by_area_caps_count() -> None:
    boxes = np.array([[0, 0, 1, 1], [0, 0, 10, 10], [0, 0, 5, 5]], dtype=np.float32)
    kept = select_bboxes_by_area(boxes, 2)
    assert len(kept) == 2
    assert kept[1, 2] == 10


def test_align_chunk_track_ids() -> None:
    m = np.zeros((4, 4), dtype=bool)
    m[0, 0] = True
    chunk_masks = [[m], [m]]
    chunk_tids = [[0], [0]]
    aligned = align_chunk_track_ids(
        [m],
        [7],
        chunk_tids,
        chunk_masks,
        iou_threshold=0.3,
    )
    assert aligned[0][0] == 7


def test_build_from_sam2_track_ids_stable_colors() -> None:
    m = np.zeros((4, 4), dtype=bool)
    m[0, 0] = True
    per_frame_masks = [[m], [m]]
    per_frame_track_ids = [[5], [5]]
    labels, pairs, tracks = build_from_sam2_track_ids(
        per_frame_masks, per_frame_track_ids, 4, 4
    )
    assert labels[0][0, 0] == labels[1][0, 0] == 6  # track_id 5 -> label 6
    assert len(pairs) == 1
    assert pairs[0].matches[0][2] == 1.0
    assert tracks[0]["track_id"] == 5


def test_segment_sequence_prefers_sam2_video(monkeypatch, tmp_path) -> None:
    from genmatter.pseudo_gt import sam_video

    rgb = tmp_path / "rgb"
    rgb.mkdir()
    import cv2

    for i in range(2):
        cv2.imwrite(str(rgb / f"{i:05d}.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))

    def fake_tracked(*_a, **_k):
        masks = [[np.ones((4, 4), dtype=bool)]]
        return [masks[0]], [[0]], "sam2_video_tracked"

    monkeypatch.setattr(sam_video, "segment_sam2_video_tracked", fake_tracked)
    monkeypatch.setattr(
        sam_video,
        "resolve_model_weights",
        lambda *_a, **_k: tmp_path / "w.pt",
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "ultralytics",
        MagicMock(SAM=MagicMock(return_value=MagicMock())),
    )

    masks, method, hw, tids = sam_video.segment_sequence(
        rgb,
        model="sam2.1_l.pt",
        weights_dir=tmp_path,
        min_threshold=0.15,
        prefer_video=True,
        detect_new_objects=False,
        show_tqdm=False,
    )
    assert method == "sam2_video_tracked"
    assert tids == [[0]]
    assert hw == (4, 4)

"""Tests for hybrid SAM2 video + per-frame new-object merge."""

from __future__ import annotations

import numpy as np

from genmatter.pseudo_gt.correspondence import (
    extract_unmatched_sam_masks,
    merge_tracked_with_new_objects,
    propagate_track_ids_lists,
)


def test_extract_unmatched_sam_masks() -> None:
    tracked = [np.zeros((4, 4), dtype=bool)]
    tracked[0][0, 0] = True
    sam_same = [tracked[0].copy()]
    sam_new = [np.zeros((4, 4), dtype=bool)]
    sam_new[0][3, 3] = True
    assert extract_unmatched_sam_masks(tracked, sam_same, iou_threshold=0.5) == []
    unmatched = extract_unmatched_sam_masks(tracked, sam_new, iou_threshold=0.5)
    assert len(unmatched) == 1


def test_merge_tracked_with_new_objects() -> None:
    m_vid = np.zeros((4, 4), dtype=bool)
    m_vid[0, 0] = True
    m_new = np.zeros((4, 4), dtype=bool)
    m_new[3, 3] = True

    video_masks = [[m_vid], [m_vid]]
    video_tids = [[0], [0]]
    sam_masks = [[m_vid, m_new], [m_vid, m_new]]

    merged_masks, merged_tids = merge_tracked_with_new_objects(
        video_masks,
        video_tids,
        sam_masks,
        new_object_iou_threshold=0.5,
        link_iou_threshold=0.3,
    )
    assert len(merged_masks[0]) == 2
    assert merged_tids[0][0] == 0
    assert merged_tids[1][0] == 0
    assert merged_tids[0][1] == merged_tids[1][1] == 1


def test_propagate_track_ids_lists_offset() -> None:
    m = np.zeros((2, 2), dtype=bool)
    m[0, 0] = True
    tids = propagate_track_ids_lists([[m], [m]], 0.3, id_offset=10)
    assert tids == [[10], [10]]

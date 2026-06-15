"""Tests for pseudo-GT correspondence utilities."""

from __future__ import annotations

import numpy as np

from genmatter.pseudo_gt.correspondence import (
    hungarian_link,
    mask_iou,
    mask_iou_matrix,
    propagate_track_ids,
)


def test_mask_iou_disjoint() -> None:
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[0, 0] = True
    b[3, 3] = True
    assert mask_iou(a, b) == 0.0


def test_mask_iou_identical() -> None:
    a = np.zeros((3, 3), dtype=bool)
    a[1:, 1:] = True
    assert mask_iou(a, a) == 1.0


def test_hungarian_link() -> None:
    iou = np.array([[0.9, 0.1], [0.2, 0.85]])
    matches, deaths, births = hungarian_link(iou, iou_threshold=0.5)
    assert len(matches) == 2
    assert deaths == []
    assert births == []


def test_propagate_track_ids_two_frames() -> None:
    m0 = [np.ones((2, 2), dtype=bool)]
    m1 = [np.ones((2, 2), dtype=bool)]
    labels, pairs, tracks = propagate_track_ids([m0, m1], iou_threshold=0.3)
    assert len(labels) == 2
    assert labels[0].sum() > 0
    assert len(tracks) == 1
    assert len(pairs) == 1

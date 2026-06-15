"""Tests for pseudo-GT colorization."""

from __future__ import annotations

import numpy as np

from genmatter.pseudo_gt.colors import (
    build_track_color_palette,
    label_map_to_rgb,
    max_track_id_from_label_maps,
)


def test_consistent_colors_across_frames() -> None:
    palette = build_track_color_palette(2, seed=0)
    f0 = np.array([[0, 1, 2], [1, 1, 0]], dtype=np.uint16)
    f1 = np.array([[2, 0, 1], [0, 2, 1]], dtype=np.uint16)
    rgb0 = label_map_to_rgb(f0, palette)
    rgb1 = label_map_to_rgb(f1, palette)
    c_track0 = palette[0].tolist()
    c_track1 = palette[1].tolist()
    assert rgb0[0, 1].tolist() == c_track0
    assert rgb1[0, 2].tolist() == c_track0
    assert rgb0[0, 2].tolist() == c_track1
    assert rgb1[0, 0].tolist() == c_track1


def test_max_track_id() -> None:
    maps = [np.array([[0, 3]], dtype=np.uint16), np.array([[5, 0]], dtype=np.uint16)]
    assert max_track_id_from_label_maps(maps) == 4

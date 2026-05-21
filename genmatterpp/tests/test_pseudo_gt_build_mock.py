"""Mocked pseudo-GT build (no SAM GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from genmatter.custom.config_schema import load_config
from genmatter.pseudo_gt.build import PseudoGtConfig, build_pseudo_gt
from genmatter.pseudo_gt.sam_video import list_rgb_frames, read_frame_hw


def test_build_pseudo_gt_mocked(tmp_path, monkeypatch) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "custom_default.yaml")
    vid = "mock_vid"
    root = tmp_path / vid
    rgb = root / "rgb_frames" / vid
    rgb.mkdir(parents=True)
    import cv2

    for i in range(3):
        cv2.imwrite(str(rgb / f"{i:05d}.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))

    cfg.paths.custom_videos_root = str(tmp_path)

    def fake_segment(*_a, **_k):
        masks = [[np.ones((8, 8), dtype=bool)] for _ in range(3)]
        return masks, "mock", (8, 8), None

    with patch("genmatter.pseudo_gt.build.segment_sequence", side_effect=fake_segment):
        result = build_pseudo_gt(vid, cfg, PseudoGtConfig(force=True))

    assert result.success
    assert result.num_frames == 3
    seg = root / "pseudo_gt_sam" / "segmasks" / vid
    assert (seg / "00000.png").is_file()
    with open(root / "pseudo_gt_sam" / "manifest.json", encoding="utf-8") as f:
        man = json.load(f)
    assert man["video_id"] == vid

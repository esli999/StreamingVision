"""Tests for genmatter.preprocessing.dino (no model download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from genmatter.preprocessing.dino import (
    apply_pca,
    load_and_preprocess_frames,
)


def test_load_and_preprocess_frames_patch_align(frame_dir: Path) -> None:
    frames, h, w = load_and_preprocess_frames(
        frame_dir, patch_size=14, progress=False
    )
    assert frames is not None
    assert h % 14 == 0
    assert w % 14 == 0
    assert len(frames) == 2


def test_apply_pca_shape() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((100, 384)).astype(np.float32)
    pca, unnorm, rgb = apply_pca(features, n_components=10)
    assert unnorm.shape == (100, 10)
    assert rgb.shape == (100, 3)
    assert pca.n_components_ == 10

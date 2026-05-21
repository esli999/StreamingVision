"""Tests for genmatter.preprocessing.sam_frame0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from genmatter.preprocessing.sam_frame0 import combined_mask, resolve_model_weights


def test_combined_mask_deterministic_with_seed() -> None:
    masks = torch.zeros(2, 4, 4)
    masks[0, 1:3, 1:3] = 0.9
    masks[1, 0:2, 0:2] = 0.8
    a = combined_mask(masks, min_threshold=0.15, seed=42)
    b = combined_mask(masks, min_threshold=0.15, seed=42)
    assert a.shape == (4, 4, 3)
    assert np.array_equal(a, b)
    assert not np.all(a == 255)


def test_combined_mask_higher_priority_wins() -> None:
    masks = torch.zeros(1, 2, 2)
    masks[0, :, :] = 0.5
    out = combined_mask(masks, min_threshold=0.1, seed=0)
    assert out.shape == (2, 2, 3)


def test_resolve_model_weights_existing_file(tmp_path: Path) -> None:
    weights = tmp_path / "custom.pt"
    weights.write_bytes(b"fake")
    resolved = resolve_model_weights(str(weights), tmp_path)
    assert Path(resolved).resolve() == weights.resolve()


def test_resolve_model_weights_missing_absolute_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pt"
    with pytest.raises(FileNotFoundError):
        resolve_model_weights(str(missing), tmp_path)

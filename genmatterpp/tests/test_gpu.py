"""Tests for genmatter.preprocessing.gpu."""

from __future__ import annotations

from genmatter.preprocessing.gpu import gpu_memory_snapshot, release_gpu


def test_release_gpu_does_not_raise() -> None:
    release_gpu()


def test_gpu_memory_snapshot() -> None:
    snap = gpu_memory_snapshot()
    if snap is not None:
        used, total = snap
        assert used >= 0
        assert total > 0
        assert used <= total

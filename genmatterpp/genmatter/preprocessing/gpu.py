"""GPU memory helpers for sequential preprocessing stages."""

from __future__ import annotations

import gc
from typing import Optional

import torch


def release_gpu() -> None:
    """Drop cached CUDA allocations between heavy model stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_memory_snapshot() -> Optional[tuple[float, float]]:
    """Return (used_gb, total_gb) if CUDA is available, else None."""
    if not torch.cuda.is_available():
        return None
    free_b, total_b = torch.cuda.mem_get_info()
    used_b = total_b - free_b
    return used_b / (1024**3), total_b / (1024**3)

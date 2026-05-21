"""Outlier fraction statistics from dense tracking assignments."""

from __future__ import annotations

import numpy as np


def mean_outlier_fraction_percent(tracking_data: list[dict]) -> tuple[float, float, float]:
    """
    Return (mean%, min% per frame, max% per frame) of outlier-labeled pixels.

    Outlier label is ``blob_assignments >= n_blobs`` per frame.
    """
    per_frame: list[float] = []
    for fd in tracking_data:
        assign = np.asarray(fd["blob_assignments"])
        n_blobs = int(fd["n_blobs"])
        per_frame.append(float(np.mean(assign >= n_blobs)) * 100.0)
    if not per_frame:
        return 0.0, 0.0, 0.0
    arr = np.array(per_frame, dtype=np.float64)
    return float(arr.mean()), float(arr.min()), float(arr.max())

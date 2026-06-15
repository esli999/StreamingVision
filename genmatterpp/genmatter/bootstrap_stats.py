"""Percentile bootstrap confidence intervals for the sample mean."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

BOOTSTRAP_N_SAMPLES = max(1, int(os.environ.get("GENMATTER_BOOTSTRAP_B", "10000")))
BOOTSTRAP_RANDOM_SEED = int(os.environ.get("GENMATTER_BOOTSTRAP_SEED", "42"))


def bootstrap_mean_ci_95(values: Sequence[float]) -> tuple[float, float, float]:
    """Mean of *values* and a 95% percentile-bootstrap CI for that mean.

    Resamples the *values* array with replacement ``BOOTSTRAP_N_SAMPLES`` times
    (each resample has the same length as *values*). The interval is the 2.5 and
    97.5 percentiles of the bootstrap distribution of the mean.

    Non-finite entries are dropped. Empty input yields three NaNs. A single
    finite value yields a degenerate CI at that value.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(np.mean(arr))
    if n == 1:
        return mean, mean, mean

    rng = np.random.RandomState(BOOTSTRAP_RANDOM_SEED)
    idx = rng.randint(0, n, size=(BOOTSTRAP_N_SAMPLES, n))
    boot_means = np.mean(arr[idx], axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean, float(lo), float(hi)

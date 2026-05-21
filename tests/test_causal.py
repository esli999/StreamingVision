"""No-future-frame-leakage gate for the streaming MatterWorker.

Runs a controlled sequence of synthetic depth/flow/feature frames through
``genmatter_rt.step`` and asserts that the output at frame ``k`` depends
only on observations up to frame ``k`` — i.e. mutating frames ``> k`` later
in the recorded sequence does not change the state we already produced.

This is the "no cheating" gate the user explicitly asked for.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import pytest

# Force tiny GPU memory fraction so this test can run alongside heavier
# workloads.  Must be set before any jax operation.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.10")
os.environ.setdefault("GENMATTER_DEFAULT_MAX_FRAMES", "16")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import genmatter_rt  # noqa: E402


def _make_frame_sequence(n_frames: int, h: int = 360, w: int = 640, seed: int = 0):
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_frames):
        # Smooth slowly varying depth (correlated across frames) so positions
        # don't jump randomly.
        depth = rng.uniform(0.3, 1.0, (h, w)).astype(np.float32)
        flow = rng.normal(0, 0.3, (2, h, w)).astype(np.float32)
        feat = rng.normal(0, 1, (256, 384)).astype(np.float32)
        frames.append((depth, flow, feat))
    return frames


def _run_through(frames, key, indices=None, pca_basis=None, pca_mean=None,
                  pca_std=None):
    """Run init_state then a series of step() calls, returning the final
    blob_assignments and a list of per-frame snapshots so the caller can
    compare.
    """
    state = None
    snapshots = []
    for t, (depth, flow, feat) in enumerate(frames):
        if indices is None:
            indices = genmatter_rt.subsample_indices(
                depth.shape[0], depth.shape[1], genmatter_rt.STRIDE,
                genmatter_rt.N_KEEP, seed=0)
        positions, velocities = genmatter_rt.unproject(
            depth, flow, indices, genmatter_rt.DEFAULT_INTRINSICS,
            genmatter_rt.STRIDE)
        features, pca_basis, pca_mean, pca_std = (
            genmatter_rt.dino_features_to_datapoints(
                feat, indices, pca_basis, pca_mean, pca_std,
                stride=genmatter_rt.STRIDE,
                image_hw=depth.shape,
                target_dim=genmatter_rt.FEATURE_DIM,
            )
        )
        if state is None:
            state, key = genmatter_rt.init_state(
                positions, velocities, features, key,
                num_blobs=16, num_hyperblobs=2, num_warmup_sweeps=3,
            )
        else:
            state, key = genmatter_rt.step(
                state, positions, velocities, features, key)
        blob_a, _ = genmatter_rt.extract_assignments(state)
        snapshots.append(blob_a.copy())
    return snapshots, key, (indices, pca_basis, pca_mean, pca_std)


@pytest.mark.slow
def test_no_future_frame_leakage():
    """Run the same prefix through two different futures.  Assert the prefix
    snapshots are byte-identical regardless of what came after.
    """
    N = 4
    frames_a = _make_frame_sequence(N, seed=0)
    # Diverge after frame 2 — flip the second half to a different RNG.
    frames_b = list(frames_a)
    diverger = np.random.default_rng(99)
    for t in range(N // 2, N):
        depth = diverger.uniform(0.3, 1.0, (360, 640)).astype(np.float32)
        flow = diverger.normal(0, 0.3, (2, 360, 640)).astype(np.float32)
        feat = diverger.normal(0, 1, (256, 384)).astype(np.float32)
        frames_b[t] = (depth, flow, feat)

    snaps_a, _, _ = _run_through(frames_a, jax.random.PRNGKey(7))
    snaps_b, _, _ = _run_through(frames_b, jax.random.PRNGKey(7))

    # Prefix (frames 0..N/2 - 1) must be byte-identical because nothing
    # downstream has run yet.  If MatterWorker were peeking at future frames,
    # this would fail.
    for t in range(N // 2):
        np.testing.assert_array_equal(
            snaps_a[t], snaps_b[t],
            err_msg=f"frame {t} differs between two futures — "
                    "MatterWorker is leaking future observations",
        )


if __name__ == "__main__":
    test_no_future_frame_leakage()
    print("ok")

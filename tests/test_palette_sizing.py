"""Palette-size invariant gate.

``compute_blob_color_lut`` / ``render_matter_label_grids`` / ``render_matter_tile``
default their blob-id range to the 128-entry ``BLOB_PALETTE``.  A caller running
with MORE blobs (the particle demo uses 256) must pass ``num_blobs`` so ids
>= 128 survive instead of collapsing onto blob 127 — the palette-clamp smear
(a colour no pixel in the scene actually has) fixed in the particle-viz work.

This pins both halves of the invariant: the default path stays palette-sized
(bit-exact for the legacy 128-blob callers), and passing ``num_blobs`` preserves
high ids end-to-end through the LUT, the label grid, and the coloured tile.
"""

from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.10")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genmatter_rt  # noqa: E402  — core constants (subsample_indices, N_KEEP)
import genmatter_viz as viz  # noqa: E402

H, W, STR = 360, 640, 8
NB = 200  # deliberately > the 128-entry BLOB_PALETTE


def _synthetic():
    """Datapoints where blob 199 (a high id) and blob 50 (a low id) both own a
    contiguous chunk, the rest spread over low ids."""
    indices = genmatter_rt.subsample_indices(H, W, STR, genmatter_rt.N_KEEP, seed=0)
    n = indices.shape[0]
    blob_a = np.empty(n, dtype=np.int32)
    blob_a[:100] = 199
    blob_a[100:200] = 50
    blob_a[200:] = np.arange(n - 200) % 40
    hyper_a = (np.arange(n) % 4).astype(np.int32)
    bgr = np.full((H, W, 3), 80, dtype=np.uint8)
    return indices, blob_a, hyper_a, bgr


def test_compute_blob_color_lut_sizing():
    indices, blob_a, _, bgr = _synthetic()
    assert viz.compute_blob_color_lut(blob_a, indices, bgr, num_blobs=NB).shape[0] == NB
    # default falls back to the 128-entry BLOB_PALETTE (legacy / bit-exact)
    default_lut = viz.compute_blob_color_lut(blob_a, indices, bgr)
    assert default_lut.shape[0] == viz.BLOB_PALETTE.shape[0] == 128


def test_label_grid_preserves_high_ids():
    indices, blob_a, hyper_a, _ = _synthetic()
    blob_full, _, _ = viz.render_matter_label_grids(blob_a, hyper_a, indices, H, W, STR, num_blobs=NB)
    assert blob_full.max() >= 128, "high blob ids must survive when num_blobs is passed"
    # default clamps high ids into the palette range (the legacy smear behaviour)
    blob_def, _, _ = viz.render_matter_label_grids(blob_a, hyper_a, indices, H, W, STR)
    assert blob_def.max() <= 127


def test_render_matter_tile_high_id_color():
    indices, blob_a, hyper_a, _ = _synthetic()
    lut = np.zeros((NB, 3), dtype=np.uint8)
    lut[199] = (0, 0, 255)  # blob 199 -> pure red (BGR)
    out, _ = viz.render_matter_tile(blob_a, hyper_a, indices, H, W, STR,
                                    blob_color_lut=lut, num_blobs=NB)
    assert int(np.all(out == (0, 0, 255), axis=-1).sum()) > 0, \
        "blob 199 must paint its own colour when num_blobs is passed"
    # default (None) clamps blob 199 -> 127; lut[127] is black, so no red survives
    out_def, _ = viz.render_matter_tile(blob_a, hyper_a, indices, H, W, STR, blob_color_lut=lut)
    assert int(np.all(out_def == (0, 0, 255), axis=-1).sum()) == 0

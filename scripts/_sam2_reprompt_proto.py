#!/usr/bin/env python
"""SAM2 re-prompt library for tokencut.py's seed augmentation.

Re-prompt SAM2 at VIDEO-DERIVED foreground points to recover the object SAM2's
auto-mask missed, and use the CLEAN mask to augment the frame-0 seed.

The prompt points are derived ENTIRELY from the video (tokencut's normalized-cut
posterior peaks, or the most-salient frame-0 DINO-feature cluster here — central +
moving + minority; a fixed, domain-agnostic prior, NOT GT, NOT per-video tuned).
SAM2 (ultralytics) turns the points into a clean object mask; ``_augment_grid``
paints it into the seed ONLY where SAM left a hole (additive).

Self-contained: tokencut.augment_seed_grid() calls ``_sam2_mask_grid_multi`` +
``_augment_grid``."""
import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))
import genmatter_rt                      # noqa: E402  (STRIDE -> stride-8 grid dims)
from sklearn.cluster import KMeans       # noqa: E402

# Stride-8 perception grid (matches genmatter_rt) + the frame-0 inputs this
# re-prompt reads: the cached-perception npz (features/positions/Z_sam) and the
# DAVIS rgb frame.
GH, GW = 360 // genmatter_rt.STRIDE, 640 // genmatter_rt.STRIDE
_LABELS_DIR = _REPO / "runs/calibrate_consistency/labels"
_DAVIS_RGB = _REPO / "assets/tapvid_davis_30_videos_processed/tapvid_davis_rgb_frames"
NEW_COLOR = np.array([7, 250, 9], dtype=np.uint8)
_SAM = None


def _sam():
    global _SAM
    if _SAM is None:
        from ultralytics import SAM
        _SAM = SAM(str(_REPO / "assets/deeplearning_weights/sam2.1_l.pt"))
    return _SAM


def _salient_point(vid, K=8):
    """Video-derived prompt: centroid (grid coords) of the most-salient frame-0
    feature cluster (central + moving + minority). Returns (row,col) in the GH×GW
    grid + the cluster mask + its uncovered fraction."""
    with np.load(_LABELS_DIR / f"{vid}.npz") as d:
        feat = np.asarray(d["features"])[0]; pos = np.asarray(d["positions"])[0]
        vel = np.asarray(d["velocities"])[0]; z = np.asarray(d["Z_sam"])[0]
        idx = np.asarray(d["indices"]).reshape(-1)
    rows = idx // GW; cols = idx % GW
    ctr = pos[:, :2].mean(0); spread = np.linalg.norm(pos[:, :2].std(0)) + 1e-6
    ego = np.median(vel, 0)
    lab = KMeans(K, n_init=4, random_state=0).fit_predict(feat)
    best = None
    for c in range(K):
        m = lab == c
        if m.sum() < max(0.01 * m.size, 5):
            continue
        cent = 1 - np.linalg.norm(pos[m, :2].mean(0) - ctr) / spread
        mot = np.linalg.norm(vel[m] - ego, axis=1).mean()
        sz = m.mean()
        sal = cent + mot - 2.0 * max(sz - 0.25, 0)
        if best is None or sal > best[1]:
            best = (c, sal, m, sz)
    c, sal, m, sz = best
    r = float(rows[m].mean()); cc_ = float(cols[m].mean())
    unc = float((z[m] == 0).mean())
    return (r, cc_), m, idx, rows, cols, sz, unc


def _sam2_mask_grid(vid, point_rc):
    """Run SAM2 at the video-derived point -> boolean mask, downsampled to GH×GW."""
    import cv2
    with np.load(_LABELS_DIR / f"{vid}.npz") as d:
        fi = int(np.asarray(d["frame_idx"]).reshape(-1)[0])
    rgb_dir = _DAVIS_RGB / vid
    f0 = rgb_dir / f"{fi:05d}.jpg"
    if not f0.is_file():
        f0 = rgb_dir / f"{fi:05d}.png"
    img = cv2.imread(str(f0)); H, W = img.shape[:2]
    px = (point_rc[1] + 0.5) / GW * W; py = (point_rc[0] + 0.5) / GH * H
    res = _sam().predict(str(f0), points=[[float(px), float(py)]], labels=[1], verbose=False)[0]
    if res.masks is None or len(res.masks.data) == 0:
        return None
    mask = res.masks.data[0].cpu().numpy().astype(bool)  # (h,w) at model res
    mask = cv2.resize(mask.astype(np.uint8), (GW, GH), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def _sam2_mask_grid_multi(vid, points_rc):
    """Run SAM2 with MULTIPLE foreground points (thin/elongated objects that a
    single point under-segments) -> boolean mask, downsampled to GH×GW."""
    import cv2
    with np.load(_LABELS_DIR / f"{vid}.npz") as d:
        fi = int(np.asarray(d["frame_idx"]).reshape(-1)[0])
    rgb_dir = _DAVIS_RGB / vid
    f0 = rgb_dir / f"{fi:05d}.jpg"
    if not f0.is_file():
        f0 = rgb_dir / f"{fi:05d}.png"
    img = cv2.imread(str(f0)); H, W = img.shape[:2]
    pts = [[(c + 0.5) / GW * W, (r + 0.5) / GH * H] for (r, c) in points_rc]
    res = _sam().predict(str(f0), points=pts, labels=[1] * len(pts), verbose=False)[0]
    if res.masks is None or len(res.masks.data) == 0:
        return None
    mask = res.masks.data[0].cpu().numpy().astype(bool)
    return cv2.resize(mask.astype(np.uint8), (GW, GH), interpolation=cv2.INTER_NEAREST).astype(bool)


def _augment_grid(base_grid, mask_grid, idx):
    """Paint the SAM2 mask into the seed grid where SAM currently has a hole (white)."""
    grid = base_grid.copy(); flat = grid.reshape(-1, 3)
    cell = flat[idx]; is_white = np.all(cell == 255, axis=1)
    mask_flat = mask_grid.reshape(-1)
    add = is_white & mask_flat[idx]
    flat[idx[add]] = NEW_COLOR
    return flat.reshape(GH, GW, 3), int(add.sum())

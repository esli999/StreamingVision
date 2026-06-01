#!/usr/bin/env python
"""Prototype: re-prompt SAM2 at a VIDEO-DERIVED salient point to recover the object
SAM2's auto-mask missed, and use the CLEAN mask to augment the frame-0 seed.

NO cheating: the prompt point is derived ENTIRELY from the video — the spatial
centroid of the most-salient frame-0 DINO-feature cluster (central + moving +
minority; a fixed, domain-agnostic a-priori prior, NOT GT, NOT per-video tuned).
SAM2 (ultralytics) turns that point into a clean object mask; we paint it into the
seed ONLY where SAM left a hole (additive). Clean masks should be far safer than the
raw feature cluster (which catastrophically broke scooter-black).

Tests broken + working videos to check: do clean masks help the broken ones WITHOUT
hurting the working ones?"""
import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "genmatterpp"))
import calibrate_consistency as cc      # noqa: E402  (sets GENMATTER_DAVIS_DIR)
import genmatter_rt                      # noqa: E402
import pvc_loop                          # noqa: E402
import config as gm_config               # noqa: E402  (repo-assets path, via cc's env)
from sklearn.cluster import KMeans       # noqa: E402

GH, GW = cc.GRID_H, cc.GRID_W
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
    with np.load(cc.LABELS_DIR / f"{vid}.npz") as d:
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
    with np.load(cc.LABELS_DIR / f"{vid}.npz") as d:
        fi = int(np.asarray(d["frame_idx"]).reshape(-1)[0])
    rgb_dir = Path(gm_config.DAVIS_RGB_PATH) / vid
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
    """Run SAM2 with MULTIPLE foreground points (thin/elongated objects the kite
    sail that a single point under-segments) -> boolean mask, downsampled to GH×GW."""
    import cv2
    with np.load(cc.LABELS_DIR / f"{vid}.npz") as d:
        fi = int(np.asarray(d["frame_idx"]).reshape(-1)[0])
    rgb_dir = Path(gm_config.DAVIS_RGB_PATH) / vid
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


def _score(vid, grid):
    labels, gt = pvc_loop._load(vid)
    cfg = genmatter_rt.load_yaml_hypers(_REPO / "configs/streaming_general.yaml")
    tr = cc._run_tracker_on_video(labels, cfg, -1, num_sweeps=1, sam_grid=grid)
    if "error" in tr:
        return float("nan")
    hb = tr["hyperblob_a"]; z = labels["Z_sam"]; Tc = min(hb.shape[0], z.shape[0], len(gt))
    return pvc_loop._region_j(hb, lambda t: (None if gt[t] is None else np.asarray(gt[t]).reshape(-1)), Tc)


def main():
    cc._ensure_jax_setup(); cc._GT_SCORING_ALLOWED = True
    disc = cc.discover_videos()
    VIDS = ["horsejump-high", "libby", "parkour", "kite-surf",      # broken
            "scooter-black", "judo", "car-roundabout", "cows", "gold-fish", "blackswan"]  # working
    print(f"{'video':16s} {'baseJ':>6s} {'reprJ':>6s} {'delta':>7s}  {'pt(r,c)':>9s} unc nadd")
    for vid in VIDS:
        if not (cc.LABELS_DIR / f"{vid}.npz").is_file():
            continue
        labels = cc._load_labels(vid)
        base_grid = cc._frame0_sam_grid(disc[vid], labels)
        if base_grid is None:
            print(f"{vid:16s} no grid"); continue
        (r, c), m, idx, rows, cols, sz, unc = _salient_point(vid)
        mask_grid = _sam2_mask_grid(vid, (r, c))
        baseJ = _score(vid, base_grid)
        if mask_grid is None:
            print(f"{vid:16s} {baseJ:6.3f} {'--':>6s} {'SAM2 no mask':>7s}"); continue
        aug_grid, nadd = _augment_grid(base_grid, mask_grid, idx)
        reprJ = baseJ if nadd < 5 else _score(vid, aug_grid)
        print(f"{vid:16s} {baseJ:6.3f} {reprJ:6.3f} {reprJ-baseJ:+7.3f}  ({r:.0f},{c:.0f}) {unc:.2f} {nadd}", flush=True)


if __name__ == "__main__":
    main()

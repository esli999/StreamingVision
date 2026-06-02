#!/usr/bin/env python3
"""High-quality 2x4 LIVE tracking render — particle + pixel coloring, GT-seeded.

A redesigned, bigger sibling of ``render_live_grid.py`` (the "classic" 1x5,
which is left BYTE-FOR-BYTE intact and still renders). Same real live inferencer
(DepthAnythingV2 + SEA-RAFT + DINOv2 + GenMatter++ Gibbs tracker), same per-frame
tracker configuration (so tracking QUALITY is identical) — only the
VISUALIZATION changes: bigger 16:9 panels and a clean 2-row grid organised by
``column = coloring`` / ``row = representation`` so you can read at a glance that

  * CLUSTERS (hyperblobs) semantically correspond to OBJECTS, and
  * PARTICLES (blobs) track little bits of MATTER,

with the clearest coloring — DINO feature, motion, or average RGB.

Layout (2x4 tiles + a full-width stats row):

    row1  pixels (dense)  :  [ RGB | clusters | particles·<col3> | particles·<col4> ]
    row2  points (3-D)    :  [ RGB | clusters | particles·<col3> | particles·<col4> ]

``--col3`` / ``--col4`` choose what the particle columns are colored by
(``feature`` | ``motion`` | ``rgb`` | ``frozen_rgb``) — the "find what looks
good" knob. Defaults: col3=feature, col4=motion. The SAME per-particle coloring
drives both the dense-pixel tile (row1) and the 3-D point cloud (row2), so the
clearer representation is obvious side-by-side.

Output is H.264 (VSCode-friendly).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))

import run_streaming_live as live          # imports render_demo -> loads depth/flow/dino
import genmatter_rt
from render_demo import _transcode_to_h264
# Reuse the classic renderer's seeding / frame-iteration / stats helpers verbatim
# (none depend on tile size) so the two renderers stay in lock-step on everything
# EXCEPT the visualization, which is the whole point of v2.
import render_live_grid as classic
from render_live_grid import (
    _resolve, _seed_grid, _cap_seed_clusters, _iter_frames_looped,
    _count_seed_instances, _stats_row, MAX_SAM_CLUSTERS,
)

# Bigger, native-16:9 panels (classic is 464x360, horizontally squished). 768x432
# is ~1.65x wider and preserves the source aspect (no squish) -> crisper.
TILE_H, TILE_W = 432, 768
STATS_H = classic.STATS_H                  # reuse the classic stats panel verbatim
NCOL = 4
FPS_OUT = 30.0
SRC_FPS = 30.0
YAW_DEG = float(os.environ.get("PC_YAW", "8.0"))
DEFAULT_VIDEOS = ["car-roundabout", "car-shadow", "blackswan", "judo", "wine_swirl"]
# Particle-coloring modes for the two particle columns.
COLOR_MODES = ("feature", "motion", "rgb", "frozen_rgb")
_MODE_LABEL = {"feature": "feature (DINO)", "motion": "motion",
               "rgb": "avg-RGB", "frozen_rgb": "avg-RGB (frozen)"}


def _log(m): print(f"[render_live_grid_v2 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _label(img, name, fps=None, *, accent=(60, 255, 255)):
    """Top-left title bar (+ optional FPS at right) on a tile."""
    cv2.rectangle(img, (0, 0), (TILE_W, 30), (0, 0, 0), -1)
    cv2.putText(img, name, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    if fps is not None:
        t = f"{fps:4.1f} FPS"
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)
        cv2.putText(img, t, (TILE_W - tw - 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    accent, 1, cv2.LINE_AA)
    return img


def _fit(img):
    return img if img.shape[:2] == (TILE_H, TILE_W) else cv2.resize(img, (TILE_W, TILE_H))


# ---------------------------------------------------------------------------
# Motion-wheel legend: a compact HSV color key (hue = direction, sat = speed)
# stamped into the corner of the motion tiles so the coloring is self-describing.
# ---------------------------------------------------------------------------
_WHEEL_CACHE: dict = {}


def _motion_wheel(r: int = 40):
    cached = _WHEEL_CACHE.get(r)
    if cached is not None:
        return cached
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    rad = np.sqrt(xx * xx + yy * yy)
    ang = np.arctan2(yy, xx)
    hue = ((ang + np.pi) / (2.0 * np.pi) * 179.0).astype(np.uint8)
    sat = (np.clip(rad / r, 0.0, 1.0) * 255.0).astype(np.uint8)
    val = np.where(rad <= r, 255, 0).astype(np.uint8)
    hsv = np.stack([hue, sat, val], axis=2)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    mask = rad <= r
    _WHEEL_CACHE[r] = (bgr, mask)
    return bgr, mask


def _stamp_motion_legend(tile: np.ndarray, r: int = 40, pad: int = 12) -> np.ndarray:
    """Alpha-blend the motion color wheel into the tile's bottom-right corner."""
    wheel, mask = _motion_wheel(r)
    d = 2 * r + 1
    y0 = tile.shape[0] - d - pad - 16
    x0 = tile.shape[1] - d - pad
    if y0 < 0 or x0 < 0:
        return tile
    roi = tile[y0:y0 + d, x0:x0 + d]
    roi[mask] = (0.85 * wheel[mask] + 0.15 * roi[mask]).astype(np.uint8)
    cv2.circle(tile, (x0 + r, y0 + r), r, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(tile, "dir->hue  speed->sat", (x0 - 8, y0 + d + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 235), 1, cv2.LINE_AA)
    return tile


def _color_cluster_tile(hyper_full, outlier_full, num_fg):
    """Color the cluster (hyperblob) label grid: vivid-fg/muted-bg palette when
    the seeded foreground count is known, else the rainbow palette; outliers
    painted OUTLIER_BGR. Matches render_matter_tile's cluster tile exactly."""
    nhyp = genmatter_rt.HYPERBLOB_PALETTE.shape[0]
    pal = (genmatter_rt._cluster_palette_for(num_fg, nhyp) if num_fg and num_fg > 0
           else genmatter_rt.HYPERBLOB_PALETTE)
    hb = pal[hyper_full].astype(np.uint8)
    if outlier_full.any():
        hb[outlier_full] = genmatter_rt.OUTLIER_BGR
    return hb


def _color_particle_tile(mode, blob_full, outlier_full, *, feat_lut, motion_lut,
                         frozen_rgb_lut, rgb_guide, h, w):
    """Color the particle (blob) label grid by ``mode`` — a cheap LUT lookup (or
    per-frame avg RGB) on the SHARED upsampled grid. Painting OUTLIER_BGR on the
    outlier mask reproduces render_matter_tile's particle tile bit-for-bit (see
    the equivalence test), so the only change vs the classic path is speed: the
    expensive NN-fill + edge-aware upsample runs ONCE for all colorings."""
    if mode in ("feature", "motion", "frozen_rgb"):
        lut = np.asarray({"feature": feat_lut, "motion": motion_lut,
                          "frozen_rgb": frozen_rgb_lut}[mode])
        bb = lut[np.clip(blob_full, 0, lut.shape[0] - 1)].astype(np.uint8)
    elif mode == "rgb":
        bb = genmatter_rt._avg_rgb_by_label(blob_full, rgb_guide, h, w)
    else:
        raise ValueError(f"unknown coloring mode {mode!r} (choose from {COLOR_MODES})")
    if outlier_full.any():
        bb[outlier_full] = genmatter_rt.OUTLIER_BGR
    return bb


def render_video(vid: str, out_path: Path, *, yaml_cfg: dict, num_sweeps: int,
                 col3: str = "feature", col4: str = "motion",
                 point_subsample: int = 1, point_size: int = 2,
                 max_frames: int = -1, target_duration: float = 0.0,
                 out_fps: float = 30.0, frame_stride: int = 1) -> dict:
    src, mask_path, is_gt = _resolve(vid)
    if src is None or not Path(src).exists():   # exists(): src may be an RGB-frames dir
        return {"vid": vid, "status": "no_source"}
    stride, n_keep = genmatter_rt.STRIDE, genmatter_rt.N_KEEP
    H, W = live.WORK_HW
    indices = genmatter_rt.subsample_indices(h=H, w=W, stride=stride, n_keep=n_keep, seed=0)
    intr = genmatter_rt.DEFAULT_INTRINSICS
    nb = int(yaml_cfg["tracking"]["num_blobs"])
    nh = int(yaml_cfg["tracking"]["num_hyperblobs"])
    seed_grid = _seed_grid(mask_path, is_gt, H // stride, W // stride)
    if seed_grid is not None and not is_gt:
        seed_grid = _cap_seed_clusters(seed_grid, MAX_SAM_CLUSTERS)
    seed_kind = ("GT" if is_gt else "SAM") if seed_grid is not None else "kmeans"
    num_fg = _count_seed_instances(seed_grid)

    # ---- tracker configuration (IDENTICAL to the classic renderer) ----------
    # Threaded from YAML so the v2 demo tracker behaves exactly like the shipped
    # render (same freeze/damping/anchor numerics) — only the viz differs.
    _trk = yaml_cfg["tracking"]
    feat_final = bool(_trk.get("feature_aware_final_assignment",
                               genmatter_rt._FEATURE_AWARE_FINAL_DEFAULT))
    final_outlier = bool(_trk.get("final_assignment_outlier",
                                  genmatter_rt._FINAL_OUTLIER_DEFAULT))
    freeze_hb = bool(_trk.get("freeze_hyperblob_assignment",
                              genmatter_rt._FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT))
    blob_means_updates = int(_trk.get("blob_means_updates_per_frame",
                                      genmatter_rt._BLOB_MEANS_UPDATES_DEFAULT))
    freeze_blob_features = bool(_trk.get("freeze_blob_features",
                                         genmatter_rt._FREEZE_BLOB_FEATURES_DEFAULT))
    _damp_raw = _trk.get("feature_update_damping", None)
    feature_update_damping = float(_damp_raw) if _damp_raw is not None else None
    use_damp = (feature_update_damping is not None) and (feature_update_damping < 1.0)
    _tau_raw = _trk.get("final_feature_temp", None)
    use_feature_temp_final = (_tau_raw is not None)
    final_feature_temp = float(_tau_raw) if _tau_raw is not None else 1.0
    final_assignment_anchor = bool(_trk.get("final_assignment_anchor", False))
    _need_anchor = use_damp or (use_feature_temp_final and final_assignment_anchor)
    blob_feat_anchor = None
    hb_feat_anchor = None
    _log(f"{vid}: src={Path(src).name} seed={seed_kind} num_blobs={nb} "
         f"col3={col3} col4={col4} pt_sub={point_subsample} pt_sz={point_size} "
         f"freeze_blob_features={freeze_blob_features} "
         f"feature_update_damping={feature_update_damping}")

    import jax
    raw_frames = sum(1 for _ in live.iter_frames(Path(src), max_frames))
    src_frames = (raw_frames + frame_stride - 1) // max(frame_stride, 1)
    target_frames = int(round(target_duration * out_fps)) if target_duration > 0 else 0
    total_frames = max(src_frames, target_frames) if target_frames > 0 else src_frames
    canvas_w, canvas_h = TILE_W * NCOL, TILE_H * 2 + STATS_H
    writer = cv2.VideoWriter(str(out_path) + ".mp4v.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        return {"vid": vid, "status": "writer_fail"}

    key = jax.random.PRNGKey(0)
    state = None; seed_state = None; prev_tensor = None; depth_ema = None
    pca = [None, None, None]
    pc_focal = None
    frozen_rgb_lut = None        # LOCKED frame-0 per-particle mean RGB
    feat_basis = None            # FROZEN frame-0 PCA->RGB basis for the feature LUT
    motion_ref = None            # EMA reference speed for motion saturation scale
    fps_hist = deque(maxlen=12)
    drop_accum = 0.0; wall_accum = 0.0; worst_ms = 0.0; n_written = 0
    need_motion = (col3 == "motion") or (col4 == "motion")
    need_feature = (col3 == "feature") or (col4 == "feature")
    for i, bgr, is_seam in _iter_frames_looped(Path(src), max_frames, target_frames, frame_stride):
        t0 = time.monotonic()
        if is_seam:
            prev_tensor = None
            if seed_state is not None:
                state = seed_state
        d_raw = live._depth_forward(bgr).astype(np.float32)
        depth_ema = d_raw if depth_ema is None else (0.6 * d_raw + 0.4 * depth_ema)
        t_depth = time.monotonic()
        cur_tensor, hw = live._bgr_to_raft_tensor(bgr)
        flow = (np.zeros((2, H, W), np.float32) if prev_tensor is None
                else live._flow_forward(prev_tensor, cur_tensor, hw))
        prev_tensor = cur_tensor
        t_flow = time.monotonic()
        feat_raw, grid_hw = live._features_forward(bgr)
        positions, velocities = genmatter_rt.unproject(depth_ema, flow, indices, intr, stride)
        features, pca[0], pca[1], pca[2] = genmatter_rt.dino_features_to_datapoints(
            feat_raw, indices, pca[0], pca[1], pca[2], stride=stride,
            image_hw=bgr.shape[:2], target_dim=genmatter_rt.FEATURE_DIM, feat_grid_hw=grid_hw)
        t_dino = time.monotonic()
        if state is None:
            state, key = genmatter_rt.init_state(
                positions, velocities, features, key, yaml_cfg=yaml_cfg, num_blobs=nb,
                num_hyperblobs=nh, sam_segmentation=seed_grid, subsample_indices=indices)
            seed_state = state
            if _need_anchor:
                import jax.numpy as _jnp
                blob_feat_anchor = _jnp.asarray(state.blobs_state.blob_features)
                hb_feat_anchor = _jnp.asarray(state.hyperblobs_state.hyperblob_features)
        state, key = genmatter_rt.step_multi_sweep(
            state, positions, velocities, features, key, num_sweeps=num_sweeps,
            feature_aware_final=feat_final, final_outlier=final_outlier,
            freeze_hyperblob_assignment=freeze_hb,
            blob_means_updates=blob_means_updates,
            use_feature_temp_final=use_feature_temp_final,
            final_feature_temp=final_feature_temp,
            final_assignment_anchor=final_assignment_anchor,
            freeze_blob_features=freeze_blob_features,
            blob_feat_anchor=blob_feat_anchor,
            hb_feat_anchor=hb_feat_anchor,
            feature_update_damping=feature_update_damping)
        state.datapoints_state.blob_assignments.block_until_ready()
        blob_a, hyperblob_a = genmatter_rt.extract_assignments(state)
        cluster_lab = hyperblob_a
        outlier_frac = float(np.mean(blob_a < 0))
        t_gibbs = time.monotonic()

        # ---- per-particle color LUTs (additive helpers; see genmatter_rt) ----
        if frozen_rgb_lut is None:
            frozen_rgb_lut = genmatter_rt.compute_blob_color_lut(
                blob_a, indices, bgr, h=H, w=W, stride=stride)
        feat_lut = None
        if need_feature:
            feat_lut, feat_basis = genmatter_rt.compute_blob_feature_lut(
                np.asarray(state.blobs_state.blob_features), basis=feat_basis)
        motion_lut = None
        if need_motion:
            motion_lut, fref = genmatter_rt.compute_blob_motion_lut(
                np.asarray(state.blobs_state.blob_vel_means), ref_mag=motion_ref)
            if fref > 1e-6:   # lock/track the saturation scale once matter moves
                motion_ref = fref if motion_ref is None else 0.85 * motion_ref + 0.15 * fref

        # ---- ROW 1: dense PIXEL tiles ---------------------------------------
        # Build the upsampled blob + cluster label grids ONCE (the expensive
        # NN-fill + edge-aware upsample), then apply the three colorings as cheap
        # LUT lookups — bit-identical to render_matter_tile but ~2x cheaper than
        # calling it once per coloring. num_fg drives the vivid-fg/muted-bg
        # CLUSTER palette so the seeded object pops.
        blob_full, hyper_full, outlier_full = genmatter_rt.render_matter_label_grids(
            blob_a, cluster_lab, indices, h=H, w=W, stride=stride, rgb_guide=bgr)
        px_cluster = _color_cluster_tile(hyper_full, outlier_full, num_fg)
        px_col3 = _color_particle_tile(col3, blob_full, outlier_full, feat_lut=feat_lut,
                                       motion_lut=motion_lut, frozen_rgb_lut=frozen_rgb_lut,
                                       rgb_guide=bgr, h=H, w=W)
        px_col4 = _color_particle_tile(col4, blob_full, outlier_full, feat_lut=feat_lut,
                                       motion_lut=motion_lut, frozen_rgb_lut=frozen_rgb_lut,
                                       rgb_guide=bgr, h=H, w=W)

        # ---- ROW 2: 3-D POINT-CLOUD tiles (same coloring, discrete) ---------
        bgr_work = bgr if bgr.shape[:2] == (H, W) else cv2.resize(bgr, (W, H))
        pc_rgb, pc_cluster, f_used, _proj = genmatter_rt.render_pointcloud_tiles_pair(
            depth_ema, bgr_work, px_cluster, intr, yaw_deg=YAW_DEG, pitch_deg=0.0,
            point_subsample=point_subsample, point_size=point_size,
            out_hw=(TILE_H, TILE_W), focal_length=pc_focal)
        if pc_focal is None and f_used > 0.0:
            pc_focal = f_used
        pc_col3, pc_col4, _f2, _p2 = genmatter_rt.render_pointcloud_tiles_pair(
            depth_ema, px_col3, px_col4, intr, yaw_deg=YAW_DEG, pitch_deg=0.0,
            point_subsample=point_subsample, point_size=point_size,
            out_hw=(TILE_H, TILE_W), focal_length=pc_focal or f_used)
        t_render = time.monotonic()

        # ---- stats accounting (mirrors the classic renderer) ----------------
        dt = t_gibbs - t0
        if i > 0:
            fps_hist.append(1.0 / max(dt, 1e-6))
            wall_accum += dt
            drop_accum += max(0.0, dt * SRC_FPS - 1.0)
            worst_ms = max(worst_ms, dt * 1000.0)
        fps = float(np.mean(fps_hist)) if fps_hist else SRC_FPS
        lag_ms = max(0.0, wall_accum - n_written / SRC_FPS) * 1000.0
        st = {"fps": fps, "play_fps": out_fps, "frame": i, "total": total_frames - 1,
              "dropped": int(round(drop_accum)), "lag_ms": lag_ms, "worst_ms": worst_ms,
              "depth_ms": (t_depth - t0) * 1000, "flow_ms": (t_flow - t_depth) * 1000,
              "dino_ms": (t_dino - t_flow) * 1000, "gibbs_ms": (t_gibbs - t_dino) * 1000,
              "render_ms": (t_render - t_gibbs) * 1000, "seed": seed_kind,
              "outlier_frac": outlier_frac, "n_clusters": int(np.unique(
                  hyperblob_a[hyperblob_a >= 0]).size)}

        # ---- compose the 2x4 grid + stats row -------------------------------
        lab3, lab4 = _MODE_LABEL.get(col3, col3), _MODE_LABEL.get(col4, col4)
        if col3 == "motion":
            px_col3 = _stamp_motion_legend(px_col3)
        if col4 == "motion":
            px_col4 = _stamp_motion_legend(px_col4)
        row1 = np.hstack([
            _label(_fit(bgr.copy()), "RGB", fps),
            _label(_fit(px_cluster), "pixels - clusters (objects)"),
            _label(_fit(px_col3), f"pixels - particles - {lab3}"),
            _label(_fit(px_col4), f"pixels - particles - {lab4}")])
        row2 = np.hstack([
            _label(_fit(pc_rgb), "3D points - RGB"),
            _label(_fit(pc_cluster), "3D points - clusters (objects)"),
            _label(_fit(pc_col3), f"3D points - particles - {lab3}"),
            _label(_fit(pc_col4), f"3D points - particles - {lab4}")])
        frame = np.vstack([row1, row2, _stats_row(canvas_w, st)])
        writer.write(frame); n_written += 1
    writer.release()
    tmp = str(out_path) + ".mp4v.mp4"
    if _transcode_to_h264(tmp, str(out_path)):
        Path(tmp).unlink(missing_ok=True)
    else:
        Path(tmp).rename(out_path)
    return {"vid": vid, "status": "ok", "frames": n_written, "seed": seed_kind,
            "fps": round(float(np.mean(fps_hist)), 1) if fps_hist else None,
            "path": str(out_path)}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    p.add_argument("--config", default=str(_REPO / "configs/streaming_render_v2.yaml"))
    p.add_argument("--out-dir", default=str(_REPO / "runs/calibrate_consistency/viz_v2"))
    p.add_argument("--num-sweeps", type=int, default=None)
    p.add_argument("--col3", choices=COLOR_MODES, default="feature",
                   help="particle coloring for column 3 (default: feature)")
    p.add_argument("--col4", choices=COLOR_MODES, default="motion",
                   help="particle coloring for column 4 (default: motion)")
    p.add_argument("--point-subsample", type=int, default=1,
                   help="keep every Nth pixel for the 3D cloud (1 = densest).")
    p.add_argument("--point-size", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument("--out-fps", type=float, default=30.0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--target-duration", type=float, default=0.0,
                   help="loop each (short) clip up to ~this many seconds (frame-0 "
                        "seed runs once; flow zeroed at each loop seam).")
    args = p.parse_args(argv)
    cfg = genmatter_rt.load_yaml_hypers(Path(args.config))
    cfg["tracking"]["use_sam_frame0"] = True
    sweeps = args.num_sweeps or int(cfg["tracking"].get("num_gibbs_sweeps_per_frame", 4))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"rendering {len(args.videos)} videos (sweeps={sweeps} col3={args.col3} "
         f"col4={args.col4} pt_sub={args.point_subsample} yaw={YAW_DEG} "
         f"target_dur={args.target_duration}s GT/SAM-seed) -> {out_dir}")
    results = []
    for j, vid in enumerate(args.videos, 1):
        out = out_dir / f"{vid}_live.mp4"
        try:
            r = render_video(vid, out, yaml_cfg=cfg, num_sweeps=sweeps,
                             col3=args.col3, col4=args.col4,
                             point_subsample=args.point_subsample,
                             point_size=args.point_size, max_frames=args.max_frames,
                             target_duration=args.target_duration,
                             out_fps=args.out_fps, frame_stride=args.frame_stride)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); r = {"vid": vid, "status": "error", "error": repr(e)}
        results.append(r)
        _log(f"[{j}/{len(args.videos)}] {vid}: {r.get('status')} seed={r.get('seed')} "
             f"frames={r.get('frames')} fps={r.get('fps')} -> {r.get('path','-')}")
    _log("DONE: " + ", ".join(f"{r['vid']}={r['status']}" for r in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

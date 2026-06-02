#!/usr/bin/env python3
"""Focused 2x3 demo: clusters + FAITHFUL particle Gaussians, GT/SAM-seeded.

A pared-down sibling of render_live_grid_v2.py for the focus videos
(wine_swirl, jello_trim, blackswan, gray_jacket, purple_jacket). Same live tracker +
identical per-frame config; the bottom row draws the tracker's OWN latent Gaussians
(`blob_means`/`blob_covs`, no per-frame re-fit) as depth-shaded 3-D particles, so the
VISUALIZATION cannot diverge from the TRACKING.

Layout — SEMANTIC 2x3. Captions name the TYPE of visualization first (2D pixels /
3D particles / 3D point cloud), then the coloring; rows are 2-D (top) and the 3-D
camera-pan render (bottom); ONE camera PANS (yaw sweeps slowly, monotonic). The six tiles:

         col1                         col2                      col3
    2D | RGB camera frame           | 2D pixels, by particle  | 2D pixels, by cluster      |
    3D | 3D particles, by avg color | 3D particles, by cluster| 3D point cloud, by cluster |
    (+ a stats row)

The two 3-D PARTICLE lenses (col1/col2, bottom) are the SAME Gaussians from the SAME
geometry, differing ONLY in tint: col1 by per-blob AVG color (each particle = the average
color of the pixels it covers — "what it is"), col2 highlighted by CLUSTER (vivid
foreground / muted background — "which object it's in", the same palette as the cluster
seg). col3 lifts the depth-projected pixels into 3-D as a point cloud, colored by cluster.
The 2-D top row is the per-pixel assignment of the image to particles (avg color) and to
clusters. Particles are solid depth-shaded balls, painter-occluded (no 2-D outline). The
3-D view is ONE slow shallow pan (no double-back). Several RENDER-ONLY temporal EMAs keep
it calm (blob means/covs + orbit centre kill the particle jitter; a 2-D seg-color EMA
kills boundary flicker), big diffuse background particles are filtered, outliers are dark
red, and per-particle alpha is smoothed off the posterior weight trace so weak particles
fade. Tracking is untouched (bit-identical).

Output is H.264. Run:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 python scripts/render_gaussian_demo.py \
        --target-duration 6 --out-dir runs/calibrate_consistency/viz_gaussian
"""
from __future__ import annotations
import argparse, os, sys, time
from collections import deque
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))

import run_streaming_live as live
import genmatter_rt
from render_demo import _transcode_to_h264
import render_live_grid as classic
from render_live_grid import (_resolve, _seed_grid, _cap_seed_clusters,
                              _count_seed_instances, _iter_frames_looped, _stats_row,
                              MAX_SAM_CLUSTERS)
from render_live_grid_v2 import _color_cluster_tile

TILE_H, TILE_W = 432, 768
STATS_H = classic.STATS_H
NCOL = 3
SRC_FPS = 30.0
# 3-D camera PAN: ONE slow MONOTONIC sweep -amp -> +amp across the whole clip (a
# single gentle pan from a shallow angle; never doubles back). yaw uses the output
# frame index so it's smooth even across loop seams.
PC_PAN_AMP = float(os.environ.get("PC_PAN_AMP", "7.0"))        # pan half-range (deg) — shallow
# Jitter fix (render-only EMAs; tracking state untouched -> bit-identical 0.7013):
PC_CENTROID_ALPHA = float(os.environ.get("PC_CENTROID_ALPHA", "0.15"))  # orbit-center EMA (kills cloud swim)
MARBLE_GEOM_BETA = float(os.environ.get("MARBLE_GEOM_BETA", "0.25"))    # blob mean/cov EMA (kills particle wobble)
MARBLE_SIGMA = float(os.environ.get("MARBLE_SIGMA", "1.3"))   # particle size (chosen by looking)
SEG_EMA = float(os.environ.get("SEG_EMA", "0.3"))            # render-only EMA of the 2D seg tiles (edge-flicker fix; 0.3 ~halves boundary oscillation)
SEG_EDGE_AWARE = os.environ.get("SEG_EDGE_AWARE", "1") == "1"  # 0 => plain (blocky but temporally STABLE) upsample
# Spurious-particle filter (systematic, fg/bg-AGNOSTIC). The robust signal is
# DENSITY = effective datapoint count / 3-D covariance extent: a particle that is
# BIG yet owns FEW datapoints is diffuse/spurious (the "big gray background blob"),
# regardless of whether it was mis-assigned to a foreground cluster. Thresholds are
# frozen at frame 0 (so the cull set doesn't itself flicker) and applied per-frame to
# the EMA'd (smooth) geometry. Env knobs override the auto cuts:
SPURIOUS_EXTENT_FENCE = float(os.environ.get("SPURIOUS_EXTENT_FENCE", "0.0"))  # 0 => auto (Q75+1.5*IQR boxplot fence on extent)
SPURIOUS_DENSITY_PCT = float(os.environ.get("SPURIOUS_DENSITY_PCT", "15.0"))   # density percentile (among alive blobs) for the "diffuse" cut
SPURIOUS_EXTENT_PCT = float(os.environ.get("SPURIOUS_EXTENT_PCT", "70.0"))     # extent percentile gate for "big" (paired with the density cut)
SPURIOUS_EXTENT = float(os.environ.get("SPURIOUS_EXTENT", "0.0"))   # bg backstop: 0 => auto (88th pct of bg extents)
SPURIOUS_HARD_CAP = float(os.environ.get("SPURIOUS_HARD_CAP", "0.0"))  # 0 => disabled
SPURIOUS_DEBUG = os.environ.get("SPURIOUS_DEBUG", "1") == "1"  # print the per-blob extent/count/density distribution at frame 0
WEMA_ALPHA = 0.30                                             # temporal EMA on posterior weights
DARK_RED_BGR = np.array([30, 30, 130], dtype=np.uint8)       # outliers (was light grey OUTLIER_BGR)
CULL_MIN_PTS, CULL_FULL_PTS = 3.0, 11.0                       # smoothstep cull thresholds (datapoints)
DEFAULT_VIDEOS = ["wine_swirl", "jello_trim", "blackswan", "gray_jacket", "purple_jacket"]


def _log(m): print(f"[render_gaussian {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _label(img, name, fps=None):
    # Larger, bolder caption bar so the per-tile type is easy to read.
    cv2.rectangle(img, (0, 0), (TILE_W, 42), (0, 0, 0), -1)
    cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if fps is not None:
        t = f"{fps:4.1f} FPS"
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        cv2.putText(img, t, (TILE_W - tw - 10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 255, 255), 1, cv2.LINE_AA)
    return img


def _fit(img):
    return img if img.shape[:2] == (TILE_H, TILE_W) else cv2.resize(img, (TILE_W, TILE_H))


def render_video(vid, out_path, *, yaml_cfg, num_sweeps, max_frames=-1,
                 target_duration=0.0, out_fps=30.0, frame_stride=1):
    src, mask_path, is_gt = _resolve(vid)
    if src is None or not Path(src).exists():
        return {"vid": vid, "status": "no_source"}
    stride, n_keep = genmatter_rt.STRIDE, genmatter_rt.N_KEEP
    H, W = live.WORK_HW
    indices = genmatter_rt.subsample_indices(h=H, w=W, stride=stride, n_keep=n_keep, seed=0)
    intr = genmatter_rt.DEFAULT_INTRINSICS
    nb = int(yaml_cfg["tracking"]["num_blobs"]); nh = int(yaml_cfg["tracking"]["num_hyperblobs"])
    seed_grid = _seed_grid(mask_path, is_gt, H // stride, W // stride)
    if seed_grid is not None and not is_gt:
        seed_grid = _cap_seed_clusters(seed_grid, MAX_SAM_CLUSTERS)
    seed_kind = ("GT" if is_gt else "SAM") if seed_grid is not None else "kmeans"
    num_fg = _count_seed_instances(seed_grid)

    _trk = yaml_cfg["tracking"]
    feat_final = bool(_trk.get("feature_aware_final_assignment", genmatter_rt._FEATURE_AWARE_FINAL_DEFAULT))
    final_outlier = bool(_trk.get("final_assignment_outlier", genmatter_rt._FINAL_OUTLIER_DEFAULT))
    freeze_hb = bool(_trk.get("freeze_hyperblob_assignment", genmatter_rt._FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT))
    blob_means_updates = int(_trk.get("blob_means_updates_per_frame", genmatter_rt._BLOB_MEANS_UPDATES_DEFAULT))
    freeze_blob_features = bool(_trk.get("freeze_blob_features", genmatter_rt._FREEZE_BLOB_FEATURES_DEFAULT))
    _damp = _trk.get("feature_update_damping", None)
    feature_update_damping = float(_damp) if _damp is not None else None
    use_damp = (feature_update_damping is not None) and (feature_update_damping < 1.0)
    blob_feat_anchor = hb_feat_anchor = None
    _log(f"{vid}: src={Path(src).name} seed={seed_kind} num_blobs={nb} "
         f"freeze_blob_features={freeze_blob_features} damping={feature_update_damping}")

    import jax
    raw = sum(1 for _ in live.iter_frames(Path(src), max_frames))
    src_frames = (raw + frame_stride - 1) // max(frame_stride, 1)
    target_frames = int(round(target_duration * out_fps)) if target_duration > 0 else 0
    total_frames = max(src_frames, target_frames) if target_frames > 0 else src_frames
    cw, ch = TILE_W * NCOL, TILE_H * 2 + STATS_H
    writer = cv2.VideoWriter(str(out_path) + ".mp4v.mp4", cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (cw, ch))
    if not writer.isOpened():
        return {"vid": vid, "status": "writer_fail"}

    key = jax.random.PRNGKey(0)
    state = seed_state = prev = depth_ema = None
    pca = [None, None, None]
    pc_focal = None                  # FROZEN focal (sized at the pan extreme -> no overflow)
    centroid_ema = None              # EMA'd orbit center (jitter fix; threaded into the projection)
    rgb_lut = None                   # LOCKED frame-0 per-particle AVG-RGB BGR (2-D avg-color seg + 3-D color particles)
    w_ema = None                     # temporal EMA of posterior blob weights (cull smoothing)
    bm_ema = bc_ema = None           # EMA'd blob means/covs for RENDER ONLY (kills particle jitter)
    clu_ema = rgb_ema = None         # EMA'd 2D seg color tiles for RENDER ONLY (edge-flicker fix)
    spurious_cuts = None             # FROZEN frame-0 {extent fence, density cut, ...} for the spurious filter
    kill_ema = None                  # EMA of the spurious kill-mask (fades a rejected particle in/out, no pop)
    fps_hist = deque(maxlen=12)
    drop = wall = worst = 0.0; nwrote = 0
    for i, bgr, is_seam in _iter_frames_looped(Path(src), max_frames, target_frames, frame_stride):
        t0 = time.monotonic()
        if is_seam:
            prev = None
            if seed_state is not None:
                state = seed_state
            w_ema = None             # reset the smoothing EMAs at the loop seam
            bm_ema = bc_ema = None
            centroid_ema = None
            clu_ema = rgb_ema = None
            kill_ema = None
        d_raw = live._depth_forward(bgr).astype(np.float32)
        depth_ema = d_raw if depth_ema is None else 0.6 * d_raw + 0.4 * depth_ema
        t_d = time.monotonic()
        cur, hw = live._bgr_to_raft_tensor(bgr)
        flow = np.zeros((2, H, W), np.float32) if prev is None else live._flow_forward(prev, cur, hw)
        prev = cur; t_f = time.monotonic()
        feat_raw, grid_hw = live._features_forward(bgr)
        positions, velocities = genmatter_rt.unproject(depth_ema, flow, indices, intr, stride)
        features, pca[0], pca[1], pca[2] = genmatter_rt.dino_features_to_datapoints(
            feat_raw, indices, pca[0], pca[1], pca[2], stride=stride,
            image_hw=bgr.shape[:2], target_dim=genmatter_rt.FEATURE_DIM, feat_grid_hw=grid_hw)
        t_dn = time.monotonic()
        if state is None:
            state, key = genmatter_rt.init_state(
                positions, velocities, features, key, yaml_cfg=yaml_cfg, num_blobs=nb,
                num_hyperblobs=nh, sam_segmentation=seed_grid, subsample_indices=indices)
            seed_state = state
            if use_damp:
                import jax.numpy as _jnp
                blob_feat_anchor = _jnp.asarray(state.blobs_state.blob_features)
                hb_feat_anchor = _jnp.asarray(state.hyperblobs_state.hyperblob_features)
        state, key = genmatter_rt.step_multi_sweep(
            state, positions, velocities, features, key, num_sweeps=num_sweeps,
            feature_aware_final=feat_final, final_outlier=final_outlier,
            freeze_hyperblob_assignment=freeze_hb, blob_means_updates=blob_means_updates,
            freeze_blob_features=freeze_blob_features, blob_feat_anchor=blob_feat_anchor,
            hb_feat_anchor=hb_feat_anchor, feature_update_damping=feature_update_damping)
        state.datapoints_state.blob_assignments.block_until_ready()
        blob_a, hyperblob_a = genmatter_rt.extract_assignments(state)
        outlier_frac = float(np.mean(blob_a < 0))
        t_g = time.monotonic()

        if rgb_lut is None:
            # Per-blob AVG color (true scene color), LOCKED at frame 0 (stable identity
            # across frames). Feeds BOTH the 2-D avg-color particle seg AND the 3-D color
            # particles, so each particle keeps its real color as it moves.
            rgb_lut = genmatter_rt.compute_blob_color_lut(
                blob_a, indices, bgr, h=H, w=W, stride=stride)

        # Smoothed weight culling from the probabilistic trace -> per-particle alpha.
        # _blob_weights_mean_jit is the DETERMINISTIC Dirichlet posterior mean (no
        # PRNG), so reading it does NOT advance the tracker key stream — the tracking
        # stays bit-identical to the validated run. Temporal EMA + smoothstep means
        # weak particles FADE, never pop (the flicker fix the user asked for).
        w = np.asarray(genmatter_rt._blob_weights_mean_jit(state))
        w_ema = w if w_ema is None else WEMA_ALPHA * w + (1.0 - WEMA_ALPHA) * w_ema
        blob_alpha = genmatter_rt.blob_alpha_from_weights(
            w_ema, total_points=float(genmatter_rt.N_KEEP),
            min_points=CULL_MIN_PTS, full_points=CULL_FULL_PTS)
        bm, bc = genmatter_rt.extract_blob_means_and_covs(state)   # REAL means/covs (no re-fit)
        tint = np.asarray(rgb_lut)[:bm.shape[0]].astype(np.float32)   # 3-D particles: avg color (true scene color)
        # JITTER FIX: per-blob EMA of the particle GEOMETRY (means + covs), RENDER ONLY —
        # `state` is never written back, so tracking stays bit-identical. Blob index i is
        # the same particle every frame, so this index-keyed EMA is valid; a convex combo
        # of PSD covs stays PSD. Particles are drawn from these smoothed copies, not `bm`/`bc`,
        # so they stop wobbling frame-to-frame. (Reset on seam.)
        if bm_ema is None:
            bm_ema, bc_ema = bm.copy(), bc.copy()
        else:
            bm_ema = MARBLE_GEOM_BETA * bm + (1.0 - MARBLE_GEOM_BETA) * bm_ema
            bc_ema = MARBLE_GEOM_BETA * bc + (1.0 - MARBLE_GEOM_BETA) * bc_ema

        # SYSTEMATIC SPURIOUS-PARTICLE FILTER (fg/bg-AGNOSTIC). The robust discriminator is
        # DENSITY = effective datapoint count / 3-D covariance extent. A real object particle
        # is COMPACT for the matter it owns (high density); a spurious blob is BIG yet owns
        # FEW datapoints (low density) — e.g. the big diffuse gray blob smeared over the wine
        # background, even when it gets mis-assigned to a FOREGROUND cluster (the old is_bg-only
        # gate missed exactly that). Robust thresholds are FROZEN at frame 0 (so the cull SET
        # doesn't itself flicker) and applied each frame to the EMA'd (smooth) extent/count, so a
        # blob that drifts big+diffuse is faded out gradually. Render-only; tracking untouched.
        hb_per_blob = np.asarray(state.blobs_state.hyperblob_assignments)[:bm.shape[0]]
        extent = np.linalg.eigvalsh(np.asarray(bc_ema))[:, -1]        # (L,) max eigenvalue of 3D cov
        count = np.asarray(w_ema)[:bm.shape[0]] * float(genmatter_rt.N_KEEP)   # effective datapoints/blob
        density = count / (extent + 1e-6)                            # compact (real) vs diffuse (spurious)
        is_bg = hb_per_blob >= max(int(num_fg), 0)
        if spurious_cuts is None:          # FREEZE robust cuts once at frame 0
            alive = count > CULL_MIN_PTS                             # only characterise DRAWABLE particles
            ext_a = extent[alive] if alive.any() else extent
            den_a = density[alive] if alive.any() else density
            q25, q75 = np.percentile(ext_a, [25, 75])
            e_fence = (SPURIOUS_EXTENT_FENCE if SPURIOUS_EXTENT_FENCE > 0.0
                       else float(q75 + 1.5 * (q75 - q25)))          # boxplot upper-fence size outlier
            e_mid = float(np.percentile(ext_a, SPURIOUS_EXTENT_PCT)) # "big" gate (paired with density)
            d_lo = float(np.percentile(den_a, SPURIOUS_DENSITY_PCT)) # "diffuse" gate
            bg_ext = extent[is_bg]
            bg_cut = (SPURIOUS_EXTENT if SPURIOUS_EXTENT > 0.0
                      else (float(np.percentile(bg_ext, 88)) if bg_ext.size > 4 else np.inf))
            spurious_cuts = {"e_fence": e_fence, "e_mid": e_mid, "d_lo": d_lo, "bg_cut": bg_cut}
            if SPURIOUS_DEBUG:
                pe = np.percentile(ext_a, [50, 90, 99]); pd = np.percentile(den_a, [1, 10, 50])
                _log(f"{vid}: spurious cuts e_fence={e_fence:.4g} e_mid={e_mid:.4g} d_lo={d_lo:.4g} "
                     f"bg_cut={bg_cut:.4g} | extent p50/p90/p99={pe[0]:.3g}/{pe[1]:.3g}/{pe[2]:.3g} "
                     f"| density p1/p10/p50={pd[0]:.3g}/{pd[1]:.3g}/{pd[2]:.3g} (alive={int(alive.sum())})")
        c = spurious_cuts
        spurious = ((extent > c["e_fence"])                          # clear SIZE outlier (any fg/bg)
                    | ((extent > c["e_mid"]) & (density < c["d_lo"])) # BIG and DIFFUSE
                    | (is_bg & (extent > c["bg_cut"])))               # background backstop
        if SPURIOUS_HARD_CAP > 0.0:
            spurious = spurious | (extent > SPURIOUS_HARD_CAP)
        # Fade the kill in/out (temporal EMA of the spurious MASK) so a blob that crosses the
        # frozen threshold ramps to invisible over a few frames instead of POPPING — the last
        # anti-flicker guard. Render-only; reset on seam.
        kill = spurious.astype(np.float32)
        kill_ema = kill if kill_ema is None else 0.3 * kill + 0.7 * kill_ema
        blob_alpha = blob_alpha * (1.0 - kill_ema)
        if SPURIOUS_DEBUG and i == 0:
            _log(f"{vid}: frame0 spurious={int(spurious.sum())}/{int(spurious.size)} "
                 f"[fence={int((extent > c['e_fence']).sum())} "
                 f"diffuse={int(((extent > c['e_mid']) & (density < c['d_lo'])).sum())} "
                 f"bg={int((is_bg & (extent > c['bg_cut'])).sum())}]")

        # 2-D segmentation tiles (top row): particles (AVG-RGB) + clusters (objects), both from
        # the SAME upsampled label grids. The avg-RGB particle seg posterizes each pixel to its
        # blob's true color (shows the Gaussian tessellation in real color). Outliers -> DARK
        # RED (local override so the live demo's OUTLIER_BGR is untouched).
        bf, hf, of = genmatter_rt.render_matter_label_grids(
            blob_a, hyperblob_a, indices, h=H, w=W, stride=stride,
            rgb_guide=(bgr if SEG_EDGE_AWARE else None))
        px_cluster = _color_cluster_tile(hf, of, num_fg)
        rgb_dense = np.asarray(rgb_lut)[np.clip(bf, 0, rgb_lut.shape[0] - 1)].astype(np.uint8)
        if of.any():
            px_cluster[of] = DARK_RED_BGR
            rgb_dense[of] = DARK_RED_BGR
        # EDGE-FLICKER FIX: render-only temporal EMA of the 2-D seg COLOR tiles. The label
        # grids are recomputed fresh each frame (Gibbs assignments + NN-fill + edge-snap all
        # churn at object boundaries); blending toward the previous frame makes a boundary
        # pixel EASE between labels instead of snapping. The EMA'd tiles are what we display
        # AND what the 3-D cloud samples, so both calm down. (Reset on seam.)
        clu_ema = px_cluster.astype(np.float32) if clu_ema is None else \
            SEG_EMA * px_cluster + (1.0 - SEG_EMA) * clu_ema
        rgb_ema = rgb_dense.astype(np.float32) if rgb_ema is None else \
            SEG_EMA * rgb_dense + (1.0 - SEG_EMA) * rgb_ema
        px_cluster = clu_ema.clip(0, 255).astype(np.uint8)
        rgb_dense = rgb_ema.clip(0, 255).astype(np.uint8)

        # BOTTOM ROW = ONE 3-D view whose camera PANS (yaw_i). The three bottom cells share ONE
        # projection this frame: col1/col2 are the two PARTICLE lenses (avg color + cluster-
        # highlight) drawn from the SAME geometry/alpha/proj so they differ ONLY in tint, and col3
        # is the cluster-colored point cloud (the SAME object colors as the 2-D cluster tile,
        # splatted = "3-D scene clusters as the pixels"). Jitter fix: the orbit center is EMA'd
        # (prev_centroid/centroid_alpha) so the cloud doesn't swim, and the focal is frozen at the
        # pan EXTREME (computed once on frame 0) so nothing overflows mid-sweep.
        # ONE slow monotonic pan -amp -> +amp across the whole clip (shallow; no double-back).
        frac = i / max(total_frames - 1, 1)
        yaw_i = -PC_PAN_AMP + 2.0 * PC_PAN_AMP * float(frac)
        if pc_focal is None:                 # frame 0: size the framing for the WIDEST yaw
            *_unused, f_ext, _pj = genmatter_rt._build_pointcloud_projection(
                depth_ema, intr, yaw_deg=PC_PAN_AMP, pitch_deg=0.0, point_subsample=1,
                out_hw=(TILE_H, TILE_W), focal_length=None)
            pc_focal = f_ext if f_ext > 0.0 else None
        (pc_cluster3d,), _f, proj = genmatter_rt.render_pointcloud_tiles_multi(
            depth_ema, (px_cluster,), intr, yaw_deg=yaw_i, pitch_deg=0.0,
            point_subsample=1, point_size=2, out_hw=(TILE_H, TILE_W),
            focal_length=pc_focal, prev_centroid=centroid_ema,
            centroid_alpha=PC_CENTROID_ALPHA)
        if proj is not None:
            centroid_ema = proj["centroid"]
        # Per-blob CLUSTER color (vivid fg / muted bg) — the SAME palette _color_cluster_tile
        # paints the 2-D cluster tile with, so the cluster particles match the object segmentation.
        nhyp = genmatter_rt.HYPERBLOB_PALETTE.shape[0]
        cluster_pal = (genmatter_rt._cluster_palette_for(num_fg, nhyp)
                       if num_fg and num_fg > 0 else genmatter_rt.HYPERBLOB_PALETTE)
        cluster_tint = cluster_pal[np.clip(hb_per_blob, 0, nhyp - 1)].astype(np.float32)
        # Two particle lenses: SAME geometry/alpha/proj, differ ONLY in tint (avg color vs cluster).
        particles_color = genmatter_rt.render_particle_marbles_tile(
            bm_ema, bc_ema, tint, proj, out_hw=(TILE_H, TILE_W),
            alpha_per=blob_alpha, sigma_scale=MARBLE_SIGMA)
        particles_cluster = genmatter_rt.render_particle_marbles_tile(
            bm_ema, bc_ema, cluster_tint, proj, out_hw=(TILE_H, TILE_W),
            alpha_per=blob_alpha, sigma_scale=MARBLE_SIGMA)
        t_r = time.monotonic()

        dt = t_g - t0
        if i > 0:
            fps_hist.append(1.0 / max(dt, 1e-6)); wall += dt
            drop += max(0.0, dt * SRC_FPS - 1.0); worst = max(worst, dt * 1000.0)
        fps = float(np.mean(fps_hist)) if fps_hist else SRC_FPS
        st = {"fps": fps, "play_fps": out_fps, "frame": i, "total": total_frames - 1,
              "dropped": int(round(drop)), "lag_ms": max(0.0, wall - nwrote / SRC_FPS) * 1000.0,
              "worst_ms": worst, "depth_ms": (t_d - t0) * 1000, "flow_ms": (t_f - t_d) * 1000,
              "dino_ms": (t_dn - t_f) * 1000, "gibbs_ms": (t_g - t_dn) * 1000,
              "render_ms": (t_r - t_g) * 1000, "seed": seed_kind, "outlier_frac": outlier_frac,
              "n_clusters": int(np.unique(hyperblob_a[hyperblob_a >= 0]).size)}

        # SEMANTIC 2x3 (user-approved). Captions name the visualization TYPE first
        # (2D pixels / 3D particles / 3D point cloud), then the coloring. Rows = 2-D (top) /
        # 3-D camera-pan (bottom).
        row1 = np.hstack([_label(_fit(bgr.copy()), "RGB camera frame", fps),
                          _label(_fit(rgb_dense), "2D pixels, by particle"),
                          _label(_fit(px_cluster), "2D pixels, by cluster")])
        row2 = np.hstack([_label(_fit(particles_color), "3D particles, by avg color (panning camera)"),
                          _label(_fit(particles_cluster), "3D particles, by cluster (panning camera)"),
                          _label(_fit(pc_cluster3d), "3D point cloud, by cluster (panning camera)")])
        writer.write(np.vstack([row1, row2, _stats_row(cw, st)])); nwrote += 1
    writer.release()
    tmp = str(out_path) + ".mp4v.mp4"
    if _transcode_to_h264(tmp, str(out_path)):
        Path(tmp).unlink(missing_ok=True)
    else:
        Path(tmp).rename(out_path)
    return {"vid": vid, "status": "ok", "frames": nwrote, "seed": seed_kind,
            "fps": round(float(np.mean(fps_hist)), 1) if fps_hist else None, "path": str(out_path)}


_ENV_KNOBS_EPILOG = """
environment-variable knobs (advanced tuning; all render-only, tracking is untouched):
  camera pan / jitter
    PC_PAN_AMP=7.0           camera pan half-range in degrees (the slow monotonic sweep)
    PC_CENTROID_ALPHA=0.15   orbit-center EMA (lower = steadier cloud, less swim)
    MARBLE_GEOM_BETA=0.25    per-particle mean/cov EMA (lower = steadier particles)
    MARBLE_SIGMA=1.3         particle (marble) size, in sigmas
  2-D segmentation tiles
    SEG_EMA=0.3              temporal EMA of the 2-D seg color (lower = less edge flicker)
    SEG_EDGE_AWARE=1         1 = RGB-guided upsample; 0 = blocky-but-stable
  spurious-particle filter (drops big diffuse particles from the 3-D views)
    SPURIOUS_EXTENT_FENCE=0  >0 overrides the auto size-outlier cut (cov max-eigenvalue)
    SPURIOUS_DENSITY_PCT=15  "diffuse" cut: density percentile among drawable particles
    SPURIOUS_EXTENT_PCT=70   "big" gate paired with the density cut (extent percentile)
    SPURIOUS_EXTENT=0        >0 overrides the background size backstop
    SPURIOUS_HARD_CAP=0      >0 = absolute extent cap, kills anything bigger
    SPURIOUS_DEBUG=1         print the per-particle extent/count/density at frame 0

examples:
  # render the 5 focus videos (recommended entry point: scripts/render_particles_demo.sh)
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 python scripts/render_gaussian_demo.py --target-duration 6
  # one DAVIS video, a quick 24-frame smoke
  python scripts/render_gaussian_demo.py --videos blackswan --max-frames 24 --target-duration 0
  # a wider camera pan, bigger particles
  PC_PAN_AMP=12 MARBLE_SIGMA=1.6 python scripts/render_gaussian_demo.py --videos wine_swirl
"""


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, epilog=_ENV_KNOBS_EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS,
                   help="video names to render (DAVIS GT names or assets/custom_videos/<name>); "
                        f"default: {' '.join(DEFAULT_VIDEOS)}")
    p.add_argument("--config", default=str(_REPO / "configs/streaming_render_v2.yaml"),
                   help="tracking YAML (the validated, shipped config; default streaming_render_v2.yaml)")
    p.add_argument("--out-dir", default=str(_REPO / "runs/calibrate_consistency/viz_gaussian"),
                   help="output directory for <video>_live.mp4 files")
    p.add_argument("--num-sweeps", type=int, default=None,
                   help="Gibbs sweeps/frame (default: config num_gibbs_sweeps_per_frame, else 4)")
    p.add_argument("--max-frames", type=int, default=-1,
                   help="cap source frames read (-1 = whole clip); use a small value for a quick smoke")
    p.add_argument("--out-fps", type=float, default=30.0, help="output video frame rate")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="keep every Nth source frame before looping (2 = half the frames)")
    p.add_argument("--target-duration", type=float, default=6.0,
                   help="loop the (short) clip until this many seconds are emitted; 0 = play once")
    args = p.parse_args(argv)
    cfg = genmatter_rt.load_yaml_hypers(Path(args.config))
    cfg["tracking"]["use_sam_frame0"] = True
    sweeps = args.num_sweeps or int(cfg["tracking"].get("num_gibbs_sweeps_per_frame", 4))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"rendering {len(args.videos)} videos (sweeps={sweeps} "
         f"target_dur={args.target_duration}s) -> {out_dir}")
    results = []
    for j, vid in enumerate(args.videos, 1):
        out = out_dir / f"{vid}_live.mp4"
        try:
            r = render_video(vid, out, yaml_cfg=cfg, num_sweeps=sweeps, max_frames=args.max_frames,
                             target_duration=args.target_duration, out_fps=args.out_fps,
                             frame_stride=args.frame_stride)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); r = {"vid": vid, "status": "error", "error": repr(e)}
        results.append(r)
        _log(f"[{j}/{len(args.videos)}] {vid}: {r.get('status')} seed={r.get('seed')} "
             f"frames={r.get('frames')} fps={r.get('fps')} -> {r.get('path','-')}")
    _log("DONE: " + ", ".join(f"{r['vid']}={r['status']}" for r in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

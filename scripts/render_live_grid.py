#!/usr/bin/env python3
"""Sequential LIVE 1×5 tracking render, GROUND-TRUTH-seeded, with a stats row.

Runs the real live inferencer (DepthAnythingV2 + SEA-RAFT + DINOv2 + GenMatter++
Gibbs tracker) frame-by-frame, sequentially — so frame 0 IS frame 0 (unlike the
threaded live demo, which loops during its slow init and so seeds segmentation
on a late frame). Frame 0 is SEEDED from the **ground-truth** segmask for
DAVIS (or the SAM2 pseudo-GT for local videos), then tracked.

Top row — exactly 5 windows, 1×5:
    [ RGB | pixel_by_cluster | pixel_by_particle | pointcloud_by_cluster | pointcloud_by_particle ]
Bottom row — a STATS panel with the original realtime-demo analytics: pipeline
FPS, per-stage latency (depth / flow / DINO / Gibbs), and — paced against a 30 fps
source clock — the frames that would be DROPPED to keep real-time + the LAG.

Point clouds are near ON-AXIS (small yaw). Output is H.264 (VSCode-friendly).
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
import genmatter_viz
from render_demo import _transcode_to_h264

TILE_H, TILE_W = 360, 464                   # 5×464 = 2320 wide
STATS_H = 168
FPS_OUT = 30.0
SRC_FPS = 30.0                              # source clock for the real-time pacing stats
# Centered, undistorted point cloud with a moderate ~8° yaw (no pitch): enough
# parallax to clearly read the 3D structure (the tracked object's volume pops out
# of the background plane) while staying near-frontal (the projection re-centers
# the cloud centroid on the optical axis and frames to the camera aspect — see
# _build_pointcloud_projection). At 0° a pinhole reprojection collapses to the
# flat 2D image; small angles read as nearly-flat (object barely separable from
# the background slab), so 8° balances depth readability against a frontal view.
# Override with PC_YAW.
YAW_DEG = float(os.environ.get("PC_YAW", "8.0"))
# Ceiling on SAM-SEEDED (local-video) instance hyperblobs so the cluster palette
# stays legible. SAM2 can yield dozens of tiny segments; we keep the largest
# MAX_SAM_CLUSTERS and fold the rest back into the (white) background, which the
# init k-means then re-splits into a few bg hyperblobs. GT-seeded DAVIS is
# naturally bounded (object instances + bg-kmeans) and is NOT capped.
MAX_SAM_CLUSTERS = int(os.environ.get("MAX_SAM_CLUSTERS", "14"))
DEFAULT_VIDEOS = ["car-roundabout", "car-shadow", "blackswan", "judo", "wine_swirl"]
_DAVIS_GT = _REPO / "assets/tapvid_davis_30_videos_processed/tapvid_davis_segmasks"
_DAVIS_SRC = _REPO / "runs/calibrate_consistency/_davis_src"
_DAVIS_FRAMES = _REPO / "assets/tapvid_davis_30_videos_processed/tapvid_davis_rgb_frames"
_CUSTOM = _REPO / "assets/custom_videos"


def _log(m): print(f"[render_live_grid {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _resolve(vid: str):
    """Return (source, seed_mask_path, is_ground_truth). DAVIS → GT segmask;
    local → SAM2 pseudo-GT."""
    gt = _DAVIS_GT / vid / "00000.png"
    if gt.is_file():
        src = _DAVIS_SRC / f"{vid}.mp4"
        if src.is_file():
            return src, gt, True
        # Fallback: most held-out DAVIS clips have no pre-cut mp4, only an
        # RGB-frames dir — live.iter_frames reads a directory just like an mp4,
        # so the renderer reaches all 14 videos. mp4 keeps precedence above, so
        # the mp4-backed demos resolve (and render) byte-identically.
        frames = _DAVIS_FRAMES / vid
        return (frames if frames.is_dir() else None), gt, True
    base = _CUSTOM / vid
    for ext in ("source.mp4", "source.mov", "source.MOV"):
        if (base / ext).is_file():
            return base / ext, base / "pseudo_gt_sam" / "segmasks" / vid / "00000.png", False
    return None, None, False


def _seed_grid(mask_path, is_gt: bool, gh: int, gw: int):
    """Build the (gh, gw, 3) RGB seed grid for init_state's SAM-frame-0 branch.
    GT masks are RGB with the object(s) in color + black background → recolor
    black→white (the helper's bg sentinel), keep object colors as instances.
    SAM masks are uint16 id maps → bijective id→color via instance_mask_to_rgb_grid."""
    if mask_path is None or not Path(mask_path).is_file():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED if not is_gt else cv2.IMREAD_COLOR)
    if m is None:
        return None
    if is_gt:
        bg = np.all(m == 0, axis=2)
        m = m.copy(); m[bg] = (255, 255, 255)         # black bg → white sentinel
        return cv2.resize(m, (gw, gh), interpolation=cv2.INTER_NEAREST)
    return genmatter_rt.instance_mask_to_rgb_grid(m, gh, gw)


def _cap_seed_clusters(seed_grid: np.ndarray, max_clusters: int) -> np.ndarray:
    """Cap the number of distinct INSTANCE colors in an RGB seed grid.

    ``seed_grid`` is the ``(gh, gw, 3)`` RGB frame-0 seed (white ``[255,255,255]``
    = background; every other distinct color = one object instance, the encoding
    ``make_hierarchical_kmeans_chm_with_SAM_segmentations`` reads). SAM2 can emit
    far more instances than make a legible palette, so keep the ``max_clusters``
    LARGEST instances (by pixel count) and recolor the rest to the white bg
    sentinel — the init k-means then re-splits that bg into a few hyperblobs, so
    background still gets several clusters (never collapsed to one).
    """
    flat = seed_grid.reshape(-1, 3)
    white = np.array([255, 255, 255], dtype=seed_grid.dtype)
    codes = (flat[:, 0].astype(np.int64) + flat[:, 1].astype(np.int64) * 256
             + flat[:, 2].astype(np.int64) * 65536)
    white_code = 255 + 255 * 256 + 255 * 65536
    uniq, counts = np.unique(codes, return_counts=True)
    inst = [(u, c) for u, c in zip(uniq, counts) if u != white_code]
    if len(inst) <= max_clusters:
        return seed_grid
    inst.sort(key=lambda x: -x[1])                 # largest instances first
    drop = {u for u, _ in inst[max_clusters:]}     # fold the smallest excess to bg
    out = seed_grid.reshape(-1, 3).copy()
    out[np.isin(codes, list(drop))] = white
    _log(f"capped SAM seed: {len(inst)} -> {max_clusters} instances "
         f"(folded {len(drop)} small segments into background)")
    return out.reshape(seed_grid.shape)


def _iter_frames_looped(src: Path, max_frames: int, target_frames: int,
                        frame_stride: int = 1):
    """Yield ``(emit_idx, bgr, is_seam)`` looping the source until ``target_frames``
    have been emitted (or once through if ``target_frames <= 0``).

    ``frame_stride`` subsamples the source (keep every Nth frame) BEFORE looping —
    e.g. stride 2 feeds a 30 fps clip at an effective 15 fps (half the frames), so
    pairing it with a 15 fps writer plays back at the source's real-time length
    and shows the true inference cadence.

    DAVIS clips are short (~34-75 frames); looping lets the demo reach a ~12 s
    playback while ``render_live_grid``'s correct frame-0 GT/SAM seeding (which
    runs once, on ``emit_idx == 0``) is preserved — the tracker just keeps
    running across the loop boundary. ``is_seam`` marks the first frame of each
    repeat so the caller can reset optical flow there (prev=None -> zero flow),
    which avoids a garbage cross-loop motion spike at the seam (clean cut, no
    cluster glitch). ``emit_idx`` is monotonic so the stats/FPS accounting and
    JIT warmup (i==0) behave exactly as the non-looped path."""
    frames = [bgr for _, bgr in live.iter_frames(src, max_frames)]
    if frame_stride > 1:
        frames = frames[::frame_stride]          # keep every Nth frame (half at stride 2)
    if not frames:
        return
    emit = 0
    first = True
    while True:
        for j, bgr in enumerate(frames):
            if target_frames > 0 and emit >= target_frames:
                return
            yield emit, bgr, (j == 0 and not first)
            emit += 1
            first = False
        if target_frames <= 0:
            return


def _count_seed_instances(seed_grid) -> int:
    """Number of distinct OBJECT-instance colors in the ``(gh, gw, 3)`` RGB seed
    grid (every non-white color is one instance; white ``[255,255,255]`` is the
    background sentinel). This is exactly ``num_segmented_hyperblobs`` — the
    foreground hyperblob ids 0..(n-1) that
    ``make_hierarchical_kmeans_chm_with_SAM_segmentations`` assigns FIRST (before
    the background k-means hyperblobs) — so it drives the vivid-fg / muted-bg
    cluster palette split. Returns 0 when there is no seed (flat k-means init)."""
    if seed_grid is None:
        return 0
    flat = np.asarray(seed_grid).reshape(-1, 3).astype(np.int64)
    codes = flat[:, 0] + flat[:, 1] * 256 + flat[:, 2] * 65536
    white_code = 255 + 255 * 256 + 255 * 65536
    uniq = np.unique(codes)
    return int(np.sum(uniq != white_code))


def _label(img, name, fps):
    cv2.rectangle(img, (0, 0), (TILE_W, 26), (0, 0, 0), -1)
    cv2.putText(img, name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if fps is not None:
        t = f"{fps:4.1f} FPS"
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, t, (TILE_W - tw - 6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (60, 255, 255), 1, cv2.LINE_AA)
    return img


def _fit(img):
    return img if img.shape[:2] == (TILE_H, TILE_W) else cv2.resize(img, (TILE_W, TILE_H))


def _stats_row(width, st):
    """Bottom analytics panel (original realtime-demo style)."""
    img = np.full((STATS_H, width, 3), (20, 20, 26), dtype=np.uint8)
    cv2.line(img, (0, 0), (width, 0), (90, 90, 110), 2)
    F = cv2.FONT_HERSHEY_SIMPLEX
    # Big INFERENCE throughput on the left (frames/sec the model can PROCESS —
    # independent of, and may exceed, the playback rate; > play fps = real-time).
    cv2.putText(img, "INFERENCE FPS", (24, 30), F, 0.6, (160, 160, 180), 1, cv2.LINE_AA)
    cv2.putText(img, f"{st['fps']:4.1f}", (24, 92), F, 1.9, (60, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, f"playback {st.get('play_fps', 30):.0f} fps", (24, 120), F, 0.5,
                (150, 210, 160), 1, cv2.LINE_AA)
    cv2.putText(img, f"frame {st['frame']}/{st['total']}", (24, 146), F, 0.55,
                (200, 200, 210), 1, cv2.LINE_AA)
    # Columns of stats.
    def col(x, title, lines):
        cv2.putText(img, title, (x, 30), F, 0.6, (160, 160, 180), 1, cv2.LINE_AA)
        y = 60
        for lab, val, c in lines:
            cv2.putText(img, f"{lab:<10}{val}", (x, y), F, 0.52, c, 1, cv2.LINE_AA); y += 28
    W = (200, 210, 220)
    drop_c = (255, 255, 255) if st['dropped'] == 0 else (60, 200, 255)
    lag_c = (255, 255, 255) if st['lag_ms'] < 100 else (60, 200, 255)
    # Outlier fraction: white at ~0 (the target), amber if it creeps up.
    of = float(st.get('outlier_frac', 0.0))
    of_c = (255, 255, 255) if of < 0.02 else ((120, 230, 160) if of < 0.10 else (60, 200, 255))
    col(int(width * 0.26), "REAL-TIME @30fps SOURCE", [
        ("dropped", f"{st['dropped']}", drop_c),
        ("lag", f"{st['lag_ms']:.0f} ms", lag_c),
        ("worst", f"{st['worst_ms']:.0f} ms/frame", W)])
    col(int(width * 0.55), "PER-STAGE LATENCY (ms)", [
        ("depth", f"{st['depth_ms']:.1f}", W),
        ("flow", f"{st['flow_ms']:.1f}", W),
        ("dino", f"{st['dino_ms']:.1f}", W)])
    col(int(width * 0.78), "TRACKER", [
        ("gibbs", f"{st['gibbs_ms']:.1f} ms", W),
        ("outlier", f"{of*100:.2f}%", of_c),
        ("clusters", f"{st.get('n_clusters', 0)}", W),
        ("seed", st['seed'], (120, 230, 160))])
    return img


def render_video(vid: str, out_path: Path, *, yaml_cfg: dict, num_sweeps: int,
                 max_frames: int = -1, target_duration: float = 0.0,
                 out_fps: float = 30.0, frame_stride: int = 1) -> dict:
    src, mask_path, is_gt = _resolve(vid)
    if src is None or not Path(src).exists():   # exists(): src may be an RGB-frames dir
        return {"vid": vid, "status": "no_source"}
    stride, n_keep = genmatter_rt.STRIDE, genmatter_rt.N_KEEP
    H, W = live.WORK_HW
    indices = genmatter_rt.subsample_indices(h=H, w=W, stride=stride, n_keep=n_keep, seed=0)
    intr = genmatter_rt.DEFAULT_INTRINSICS
    # Global num_blobs (host int, fed to init_state's K-means; NOT jax-traced).
    # ONE global value — no per-video knobs.
    nb = int(yaml_cfg["tracking"]["num_blobs"])
    nh = int(yaml_cfg["tracking"]["num_hyperblobs"])
    seed_grid = _seed_grid(mask_path, is_gt, H // stride, W // stride)
    # SAM-seeded (local) videos can spawn many tiny background hyperblobs; cap the
    # SEEDED cluster count so the palette stays legible. GT-seeded DAVIS is
    # naturally bounded (few object instances + bg-kmeans), so leave it untouched.
    if seed_grid is not None and not is_gt:
        seed_grid = _cap_seed_clusters(seed_grid, MAX_SAM_CLUSTERS)
    seed_kind = ("GT" if is_gt else "SAM") if seed_grid is not None else "kmeans"
    # Foreground instance count = the low hyperblob ids the SAM/GT seed assigns
    # first; drives the vivid-fg / muted-bg CLUSTER palette so the object pops.
    num_fg = _count_seed_instances(seed_grid)
    # "Fixed cluster view" flags: jit-static in genmatter_rt.step_multi_sweep,
    # which DEFAULTS them to False. Read from YAML and thread through, or the
    # frozen-semantic-cluster behavior never reaches the tracker.
    _trk = yaml_cfg["tracking"]
    feat_final = bool(_trk.get("feature_aware_final_assignment",
                               genmatter_rt._FEATURE_AWARE_FINAL_DEFAULT))
    final_outlier = bool(_trk.get("final_assignment_outlier",
                                  genmatter_rt._FINAL_OUTLIER_DEFAULT))
    freeze_hb = bool(_trk.get("freeze_hyperblob_assignment",
                              genmatter_rt._FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT))
    # Per-frame gibbs_blob_means refinement count. ONE global value — no
    # per-video knobs.
    blob_means_updates = int(_trk.get("blob_means_updates_per_frame",
                                      genmatter_rt._BLOB_MEANS_UPDATES_DEFAULT))
    # ANTI-DRIFT feature update: static freeze + the DAMPED generalization
    # (feature_update_damping in [0,1]; <1 blends each per-frame Gibbs feature
    # update back toward the frame-0 anchor captured below). Read from YAML so the
    # render config's damping reaches the demo tracker.
    freeze_blob_features = bool(_trk.get("freeze_blob_features",
                                         genmatter_rt._FREEZE_BLOB_FEATURES_DEFAULT))
    _damp_raw = _trk.get("feature_update_damping", None)
    feature_update_damping = float(_damp_raw) if _damp_raw is not None else None
    use_damp = (feature_update_damping is not None) and (feature_update_damping < 1.0)
    # Inference feature-temperature (tau) on the final assignment.
    _tau_raw = _trk.get("final_feature_temp", None)
    use_feature_temp_final = (_tau_raw is not None)
    final_feature_temp = float(_tau_raw) if _tau_raw is not None else 1.0
    final_assignment_anchor = bool(_trk.get("final_assignment_anchor", False))
    _need_anchor = use_damp or (use_feature_temp_final and final_assignment_anchor)
    blob_feat_anchor = None
    hb_feat_anchor = None
    _log(f"{vid}: src={Path(src).name} seed={seed_kind} num_blobs={nb} "
         f"feat_final={feat_final} final_outlier={final_outlier} freeze_hb={freeze_hb} "
         f"blob_means_updates={blob_means_updates} freeze_blob_features={freeze_blob_features} "
         f"feature_update_damping={feature_update_damping} final_feature_temp={final_feature_temp}")

    import jax
    raw_frames = sum(1 for _ in live.iter_frames(Path(src), max_frames))
    src_frames = (raw_frames + frame_stride - 1) // max(frame_stride, 1)  # after subsample
    # Loop short clips up to ~target_duration seconds (the GT/SAM seed runs once
    # on frame 0; the tracker continues across loop seams, flow zeroed at each).
    target_frames = int(round(target_duration * out_fps)) if target_duration > 0 else 0
    total_frames = max(src_frames, target_frames) if target_frames > 0 else src_frames
    writer = cv2.VideoWriter(str(out_path) + ".mp4v.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (TILE_W * 5, TILE_H + STATS_H))
    if not writer.isOpened():
        return {"vid": vid, "status": "writer_fail"}

    key = jax.random.PRNGKey(0)
    state = None; seed_state = None; prev_tensor = None; depth_ema = None
    pc_color_lut = None   # per-particle colours LOCKED from the first frame
    pca = [None, None, None]
    pc_focal = None
    fps_hist = deque(maxlen=12)
    drop_accum = 0.0; wall_accum = 0.0; worst_ms = 0.0; n_written = 0
    for i, bgr, is_seam in _iter_frames_looped(Path(src), max_frames, target_frames, frame_stride):
        t0 = time.monotonic()
        # At a loop seam, reset flow (prev=None -> zero flow this frame) so the
        # cross-loop jump (last frame -> first frame) doesn't inject a garbage
        # motion spike that would glitch the clusters for one frame.  AND RE-APPLY
        # the frame-0 semantic seed: the object cluster decays across a clip
        # (object blobs absorb background datapoints through warmup + tracking),
        # so without a re-seed later loops degrade relative to the first.  We
        # RESTORE the cached post-init frame-0 state (`seed_state`, immutable JAX
        # pytree) rather than re-running init_state -> the CHM membership + blob
        # geometry snap back to the frame-0 semantic seed INSTANTLY (no per-seam
        # K-means/warmup latency hitch), so every loop is as fresh as the first AND
        # the demo stays real-time.  The seam frame then runs the same per-frame
        # step on the restored seed, exactly like the original frame 0.
        # prev_tensor=None keeps the flow-reset at the seam.
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
            # Cache the pristine post-init frame-0 seed for instant re-seed at
            # every loop seam (see the is_seam branch above).
            seed_state = state
            # Capture the frame-0 appearance anchor for the damped feature update
            # / anchor-referenced final assignment (the seam re-seed restores
            # seed_state, so this anchor stays valid).
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
        t_gibbs = time.monotonic()

        # CLUSTER view = the model's ACTUAL per-datapoint hyperblob assignment.
        # With freeze_hyperblob_assignment the frame-0 semantic seed stays sticky,
        # so the raw hyperblobs are the clusters. The frame-0 GT/SAM seed enters at
        # init_state (sam_segmentation=seed_grid), not here. Hyperblob ids are
        # constant across frames under freeze, so render_matter_tile's palette[id]
        # lookup is stable (no per-frame re-derivation / re-sorting), which avoids
        # cluster-color flicker.
        cluster_lab = hyperblob_a

        # Per-frame outlier fraction for the stats readout (datapoints the Gibbs
        # sampler left unassigned; rendered as the desaturated OUTLIER_BGR overlay
        # in render_matter_tile, not as a saturated cluster block).
        outlier_frac = float(np.mean(blob_a < 0))

        # CLUSTER tile: vivid foreground (num_fg seeded-object hyperblobs) +
        #   muted background, so the object POPS (num_fg_hyperblobs=num_fg).
        # PARTICLE tile: full rainbow (num_fg_hyperblobs left at default None in
        #   the palette path) so every background particle stays visible.
        # Both tiles get edge-aware (rgb_guide=bgr) upsampling so the stride-8
        # boundaries snap to the RGB edges instead of staircasing.
        # PARTICLE tile = each particle's average RGB, LOCKED from the first
        # frame (compute the per-blob colour LUT once, then reuse it every frame
        # so colours stay fixed as particles move — no per-frame colour drift).
        # CLUSTER tile keeps the vivid-fg/muted-bg palette so the object pops.
        if pc_color_lut is None:
            pc_color_lut = genmatter_viz.compute_blob_color_lut(
                blob_a, indices, bgr, h=H, w=W, stride=stride, num_blobs=nb)
        px_particle, px_cluster = genmatter_viz.render_matter_tile(
            blob_a, cluster_lab, indices, h=H, w=W, stride=stride,
            rgb_guide=bgr, num_fg_hyperblobs=(num_fg if num_fg > 0 else None),
            blob_color_lut=pc_color_lut, num_blobs=nb)
        # FREEZE the autofit focal from frame 0 (thread pc_focal back in) so the
        # 3D framing is STABLE across frames — no scale "pop"/jitter. The autofit
        # (margin 0.98 + 90th-pct extent, in _build_pointcloud_projection) fills
        # the tile; freezing frame-0's focal keeps it from re-scaling per frame.
        pc_cluster, pc_particle, f_used, _proj = genmatter_viz.render_pointcloud_tiles_pair(
            depth_ema, px_cluster, px_particle, intr, yaw_deg=YAW_DEG, pitch_deg=0.0,
            point_size=2, out_hw=(TILE_H, TILE_W), focal_length=pc_focal)
        if pc_focal is None and f_used > 0.0:
            pc_focal = f_used
        t_render = time.monotonic()

        dt = t_gibbs - t0                       # inference time (excl. our offline render)
        if i > 0:
            fps_hist.append(1.0 / max(dt, 1e-6))
            wall_accum += dt
            drop_accum += max(0.0, dt * SRC_FPS - 1.0)   # real-time budget at the 30fps source rate
            worst_ms = max(worst_ms, dt * 1000.0)
        fps = float(np.mean(fps_hist)) if fps_hist else SRC_FPS   # TRUE inference throughput (1/dt)
        lag_ms = max(0.0, wall_accum - n_written / SRC_FPS) * 1000.0
        st = {"fps": fps, "play_fps": out_fps, "frame": i, "total": total_frames - 1,
              "dropped": int(round(drop_accum)), "lag_ms": lag_ms, "worst_ms": worst_ms,
              "depth_ms": (t_depth - t0) * 1000, "flow_ms": (t_flow - t_depth) * 1000,
              "dino_ms": (t_dino - t_flow) * 1000, "gibbs_ms": (t_gibbs - t_dino) * 1000,
              "render_ms": (t_render - t_gibbs) * 1000, "seed": seed_kind,
              "outlier_frac": outlier_frac, "n_clusters": int(np.unique(
                  hyperblob_a[hyperblob_a >= 0]).size)}
        tiles = [_label(_fit(bgr.copy()), "RGB", fps),
                 _label(_fit(px_cluster), "pixel_by_cluster", fps),
                 _label(_fit(px_particle), "pixel_by_particle", fps),
                 _label(_fit(pc_cluster), "pointcloud_by_cluster", fps),
                 _label(_fit(pc_particle), "pointcloud_by_particle", fps)]
        frame = np.vstack([np.hstack(tiles), _stats_row(TILE_W * 5, st)])
        writer.write(frame); n_written += 1
    writer.release()
    tmp = str(out_path) + ".mp4v.mp4"
    if _transcode_to_h264(tmp, str(out_path)):
        Path(tmp).unlink(missing_ok=True)
    else:
        Path(tmp).rename(out_path)
    return {"vid": vid, "status": "ok", "frames": n_written, "seed": seed_kind,
            "fps": round(float(np.mean(fps_hist)), 1) if fps_hist else None, "path": str(out_path)}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    p.add_argument("--config", default=str(_REPO / "configs/streaming_render_v2.yaml"))
    p.add_argument("--out-dir", default=str(_REPO / "runs/calibrate_consistency/tracking_videos"))
    p.add_argument("--num-sweeps", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument("--out-fps", type=float, default=30.0,
                   help="Playback fps of the written video. Pair with --frame-stride "
                        "2 + 15 fps to play a 30 fps clip at real-time / true "
                        "inference speed.")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Keep every Nth source frame (2 = half the frames, ~15 fps "
                        "from a 30 fps source).")
    p.add_argument("--target-duration", type=float, default=0.0,
                   help="Loop each (short) clip up to ~this many seconds of "
                        "playback (frame-0 seed runs once; flow is zeroed at "
                        "each loop seam so there's no cross-loop glitch). "
                        "0 = render the source once at its natural length.")
    args = p.parse_args(argv)
    cfg = genmatter_rt.load_yaml_hypers(Path(args.config))
    # Frame-0 semantic seed is the whole point of this renderer (GT for DAVIS /
    # SAM for local); force it on regardless of the config.
    cfg["tracking"]["use_sam_frame0"] = True
    sweeps = args.num_sweeps or int(cfg["tracking"].get("num_gibbs_sweeps_per_frame", 4))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"rendering {len(args.videos)} videos (sweeps={sweeps} yaw={YAW_DEG} "
         f"target_dur={args.target_duration}s GT/SAM-seed)")
    results = []
    for j, vid in enumerate(args.videos, 1):
        out = out_dir / f"{vid}_live.mp4"
        try:
            r = render_video(vid, out, yaml_cfg=cfg, num_sweeps=sweeps,
                             max_frames=args.max_frames,
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

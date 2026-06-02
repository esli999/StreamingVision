#!/usr/bin/env python3
"""Background-bleed diagnostic (READ-ONLY; tracker + config UNCHANGED).

Q (user): in gray_jacket, static background-table datapoints get pulled into the
moving foreground cluster -- "clusters on the background table surface move even
though that's not supposed to happen". Does purple_jacket show it too, and can we
SEE/QUANTIFY where the leak is?

Runs the SAME live tracker + streaming_render_v2.yaml as render_gaussian_demo.py
(NO config / tracker / iteration change -- this only READS state) and per frame
measures four complementary signals:

  bleed_rate  fraction of frame-0 BACKGROUND datapoints now committed to a
              FOREGROUND hyperblob (cluster id 0..num_fg-1). The assignment-level
              leak. (Partly confounded by real object motion INTO a fixed image
              cell, so read it together with bg_speed.)
  bg_speed    mean EGO-MOTION-COMPENSATED speed of BACKGROUND blobs (blobs whose
              datapoints are mostly frame-0 bg). The table is static, so this should
              be ~0; a "background" blob carrying the jacket's motion IS the bleed.
  fg_speed    same for FOREGROUND blobs (reference; should be > bg_speed).
  vel_ll gap  mean per-datapoint velocity log-likelihood of bg datapoints that
              LEAKED into fg clusters vs bg datapoints that STAYED in bg. A big
              negative gap = the leaked points fit their fg blob's velocity poorly
              (grabbed on position/feature, not motion) -- localises the bleed.

Writes runs/bleed_debug/<vid>_bleed.mp4 [RGB | ego-motion | clusters | bleed-hi]
and prints the time series + a per-video summary. NOTHING is written back to state,
so the tracking is bit-identical to the validated run. This is investigation only:
it diagnoses the leak, it does NOT re-tune the train/held-out-validated config.

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 python scripts/diagnose_background_bleed.py \
        --videos gray_jacket purple_jacket --max-frames 60
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))

import run_streaming_live as live
import genmatter_rt
from render_demo import _transcode_to_h264
from render_live_grid import (_resolve, _seed_grid, _cap_seed_clusters,
                              _count_seed_instances, _iter_frames_looped, MAX_SAM_CLUSTERS)
from render_live_grid_v2 import _color_cluster_tile
from _data_loglik import per_datapoint_terms

WHITE_CODE = 255 + 255 * 256 + 255 * 65536
TH, TW = 360, 640


def _log(m): print(f"[bleed {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _seed_bg_mask(seed_grid, indices):
    """(N,) bool: True where the datapoint's frame-0 SAM/GT seed cell is BACKGROUND
    (white sentinel). Datapoints sit at flattened stride-grid cells `indices`."""
    if seed_grid is None:
        return np.zeros(indices.shape[0], dtype=bool)
    flat = np.asarray(seed_grid).reshape(-1, 3).astype(np.int64)
    codes = flat[:, 0] + flat[:, 1] * 256 + flat[:, 2] * 65536
    is_bg_cell = codes == WHITE_CODE                       # (gh*gw,)
    return is_bg_cell[np.asarray(indices).reshape(-1)]      # (N,)


def _ego_comp_speed(bvm):
    """Per-blob ego-motion-compensated 3-D speed (L,), the same median-subtraction
    compute_blob_motion_lut uses: removes the dominant (background) velocity so a
    STATIC blob reads ~0 and only differential motion survives."""
    v = np.asarray(bvm, dtype=np.float32)
    active = np.linalg.norm(v, axis=1) > 1e-6
    if int(active.sum()) >= 8:
        v = v - np.median(v[active], axis=0)
    return np.linalg.norm(v, axis=1)


def _motion_tile(bf, motion_lut, of):
    """Dense ego-motion tile: each pixel coloured by its blob's ego-comp velocity
    (white = still, vivid hue = moving differently from the background)."""
    L = motion_lut.shape[0]
    dense = motion_lut[np.clip(bf, 0, L - 1)].astype(np.uint8)
    if of is not None and of.any():
        dense[of] = (40, 40, 40)
    return dense


def _bleed_hi_tile(bgr, indices, leaked, fg_dp, gh, gw, stride):
    """Dim-RGB base; foreground-cluster datapoints faint cyan; frame-0-bg datapoints
    that LEAKED into a foreground cluster painted BRIGHT RED -- the bleed, in place."""
    base = (bgr.astype(np.float32) * 0.32).astype(np.uint8)
    fg_grid = np.zeros(gh * gw, dtype=bool); fg_grid[indices[fg_dp]] = True
    lk_grid = np.zeros(gh * gw, dtype=bool); lk_grid[indices[leaked]] = True
    fg_up = cv2.resize(fg_grid.reshape(gh, gw).astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    lk_up = cv2.resize(lk_grid.reshape(gh, gw).astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    base[fg_up] = (0.5 * base[fg_up] + np.array([110, 110, 0], np.float32) * 0.5).astype(np.uint8)
    base[lk_up] = (60, 60, 255)
    return base


def _fit(img):
    return img if img.shape[:2] == (TH, TW) else cv2.resize(img, (TW, TH))


def _label(img, txt):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run_video(vid, cfg, *, max_frames, out_dir):
    src, mask_path, is_gt = _resolve(vid)
    if src is None or not Path(src).exists():
        _log(f"{vid}: NO SOURCE"); return None
    stride, n_keep = genmatter_rt.STRIDE, genmatter_rt.N_KEEP
    H, W = live.WORK_HW
    gh, gw = H // stride, W // stride
    indices = genmatter_rt.subsample_indices(h=H, w=W, stride=stride, n_keep=n_keep, seed=0)
    intr = genmatter_rt.DEFAULT_INTRINSICS
    nb = int(cfg["tracking"]["num_blobs"]); nh = int(cfg["tracking"]["num_hyperblobs"])
    seed_grid = _seed_grid(mask_path, is_gt, gh, gw)
    if seed_grid is not None and not is_gt:
        seed_grid = _cap_seed_clusters(seed_grid, MAX_SAM_CLUSTERS)
    seed_kind = ("GT" if is_gt else "SAM") if seed_grid is not None else "kmeans"
    num_fg = _count_seed_instances(seed_grid)
    dp_bg = _seed_bg_mask(seed_grid, indices)              # (N,) frame-0 background datapoints
    n_bg = int(dp_bg.sum())
    _log(f"{vid}: src={Path(src).name} seed={seed_kind} num_fg={num_fg} "
         f"datapoints N={indices.shape[0]} bg={n_bg} ({100.0*n_bg/max(indices.shape[0],1):.0f}%)")

    _trk = cfg["tracking"]
    feat_final = bool(_trk.get("feature_aware_final_assignment", genmatter_rt._FEATURE_AWARE_FINAL_DEFAULT))
    final_outlier = bool(_trk.get("final_assignment_outlier", genmatter_rt._FINAL_OUTLIER_DEFAULT))
    freeze_hb = bool(_trk.get("freeze_hyperblob_assignment", genmatter_rt._FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT))
    blob_means_updates = int(_trk.get("blob_means_updates_per_frame", genmatter_rt._BLOB_MEANS_UPDATES_DEFAULT))
    freeze_blob_features = bool(_trk.get("freeze_blob_features", genmatter_rt._FREEZE_BLOB_FEATURES_DEFAULT))
    _damp = _trk.get("feature_update_damping", None)
    feature_update_damping = float(_damp) if _damp is not None else None
    use_damp = (feature_update_damping is not None) and (feature_update_damping < 1.0)
    num_sweeps = int(cfg["tracking"].get("num_gibbs_sweeps_per_frame", 4))

    import jax
    key = jax.random.PRNGKey(0)
    state = prev = depth_ema = None
    blob_feat_anchor = hb_feat_anchor = None      # mirror render_gaussian_demo so the blend matches
    pca = [None, None, None]
    out = out_dir / f"{vid}_bleed.mp4"
    cw, chh = TW * 4, TH
    writer = cv2.VideoWriter(str(out) + ".mp4v.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (cw, chh))
    series = []
    for i, bgr, is_seam in _iter_frames_looped(Path(src), max_frames, 0, 1):
        if is_seam:
            prev = None
        d_raw = live._depth_forward(bgr).astype(np.float32)
        depth_ema = d_raw if depth_ema is None else 0.6 * d_raw + 0.4 * depth_ema
        cur, hw = live._bgr_to_raft_tensor(bgr)
        flow = np.zeros((2, H, W), np.float32) if prev is None else live._flow_forward(prev, cur, hw)
        prev = cur
        feat_raw, grid_hw = live._features_forward(bgr)
        positions, velocities = genmatter_rt.unproject(depth_ema, flow, indices, intr, stride)
        features, pca[0], pca[1], pca[2] = genmatter_rt.dino_features_to_datapoints(
            feat_raw, indices, pca[0], pca[1], pca[2], stride=stride,
            image_hw=bgr.shape[:2], target_dim=genmatter_rt.FEATURE_DIM, feat_grid_hw=grid_hw)
        if state is None:
            state, key = genmatter_rt.init_state(
                positions, velocities, features, key, yaml_cfg=cfg, num_blobs=nb,
                num_hyperblobs=nh, sam_segmentation=seed_grid, subsample_indices=indices)
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
        blob_a, hyperblob_a = genmatter_rt.extract_assignments(state)   # (N,), (N,)

        # --- bleed metrics -------------------------------------------------------
        valid = blob_a >= 0
        fg_cluster_dp = valid & (hyperblob_a >= 0) & (hyperblob_a < max(num_fg, 0))
        leaked = dp_bg & fg_cluster_dp                                   # bg seed -> fg cluster
        stayed = dp_bg & valid & ~fg_cluster_dp
        bleed_rate = float(leaked.sum()) / max(n_bg, 1)

        bvm = np.asarray(state.blobs_state.blob_vel_means)               # (L, 3)
        speed = _ego_comp_speed(bvm)                                     # (L,) ego-comp
        L = bvm.shape[0]
        # classify each blob bg/fg by the majority frame-0 seed of its datapoints
        blob_bg_frac = np.full(L, np.nan, np.float32)
        ba = blob_a[valid]; bgv = dp_bg[valid]
        cnt = np.bincount(ba, minlength=L).astype(np.float32)
        bgc = np.bincount(ba, weights=bgv.astype(np.float32), minlength=L)
        has = cnt >= 3
        blob_bg_frac[has] = bgc[has] / cnt[has]
        bg_blob = has & (blob_bg_frac > 0.5)
        fg_blob = has & (blob_bg_frac <= 0.5)
        bg_speed = float(np.mean(speed[bg_blob])) if bg_blob.any() else 0.0
        fg_speed = float(np.mean(speed[fg_blob])) if fg_blob.any() else 0.0
        # fastest-moving "background" blob (the worst offender) + how many bg blobs move
        bg_speed_p95 = float(np.percentile(speed[bg_blob], 95)) if bg_blob.any() else 0.0
        moving_bg = int((bg_blob & (speed > max(fg_speed * 0.5, 1e-6))).sum())

        terms = per_datapoint_terms(state)
        vel_ll = np.asarray(terms["vel_ll"])                            # (N,)
        vll_leak = float(np.mean(vel_ll[leaked])) if leaked.any() else float("nan")
        vll_stay = float(np.mean(vel_ll[stayed])) if stayed.any() else float("nan")

        series.append((i, bleed_rate, bg_speed, fg_speed, bg_speed_p95, moving_bg, vll_leak, vll_stay))
        if i % 10 == 0:
            _log(f"{vid} f{i:3d}: bleed={bleed_rate:.3f} bg_speed={bg_speed:.4f} "
                 f"fg_speed={fg_speed:.4f} (bg_p95={bg_speed_p95:.4f} moving_bg={moving_bg}) "
                 f"vel_ll leak={vll_leak:.2f} stay={vll_stay:.2f}")

        # --- diagnostic tiles ----------------------------------------------------
        bf, hf, of = genmatter_rt.render_matter_label_grids(
            blob_a, hyperblob_a, indices, h=H, w=W, stride=stride, rgb_guide=bgr)
        motion_lut, _ref = genmatter_rt.compute_blob_motion_lut(bvm, subtract_median=True)
        px_cluster = _color_cluster_tile(hf, of, num_fg)
        mot = _motion_tile(bf, motion_lut, of)
        hi = _bleed_hi_tile(bgr, indices, leaked, fg_cluster_dp, gh, gw, stride)
        row = np.hstack([
            _label(_fit(bgr.copy()), "RGB"),
            _label(_fit(mot), "ego-motion (white=still)"),
            _label(_fit(px_cluster), f"clusters (fg<{num_fg})"),
            _label(_fit(hi), f"bleed: bg->fg (red)  rate={bleed_rate:.2f}")])
        writer.write(row)
        if i in (5, 20, 40):
            cv2.imwrite(str(out_dir / f"{vid}_f{i}.png"), row)
    writer.release()
    tmp = str(out) + ".mp4v.mp4"
    if _transcode_to_h264(tmp, str(out)):
        Path(tmp).unlink(missing_ok=True)
    else:
        Path(tmp).rename(out)

    arr = np.array([s[1:] for s in series], dtype=np.float64)            # (T, 7)
    if arr.shape[0] == 0:
        return None
    mean = arr.mean(axis=0)
    _log(f"{vid} SUMMARY over {arr.shape[0]} frames: "
         f"bleed_rate={mean[0]:.3f}  bg_speed={mean[1]:.4f}  fg_speed={mean[2]:.4f}  "
         f"ratio bg/fg={mean[1]/max(mean[2],1e-6):.2f}  bg_speed_p95={mean[3]:.4f}  "
         f"moving_bg={mean[4]:.1f}  vel_ll leak={mean[5]:.2f} stay={mean[6]:.2f} "
         f"gap={mean[5]-mean[6]:.2f}")
    return {"vid": vid, "frames": int(arr.shape[0]), "bleed_rate": float(mean[0]),
            "bg_speed": float(mean[1]), "fg_speed": float(mean[2]),
            "bg_fg_ratio": float(mean[1] / max(mean[2], 1e-6)),
            "moving_bg": float(mean[4]), "vel_ll_gap": float(mean[5] - mean[6]),
            "path": str(out)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="+", default=["gray_jacket", "purple_jacket"])
    p.add_argument("--config", default=str(_REPO / "configs/streaming_render_v2.yaml"))
    p.add_argument("--out-dir", default=str(_REPO / "runs/bleed_debug"))
    p.add_argument("--max-frames", type=int, default=60)
    args = p.parse_args(argv)
    cfg = genmatter_rt.load_yaml_hypers(Path(args.config))
    cfg["tracking"]["use_sam_frame0"] = True
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"diagnosing background bleed on {args.videos} -> {out_dir}")
    rows = []
    for vid in args.videos:
        try:
            r = run_video(vid, cfg, max_frames=args.max_frames, out_dir=out_dir)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); r = {"vid": vid, "error": repr(e)}
        if r:
            rows.append(r)
    _log("=== COMPARISON ===")
    for r in rows:
        if "error" in r:
            _log(f"  {r['vid']}: ERROR {r['error']}"); continue
        _log(f"  {r['vid']:14s} bleed_rate={r['bleed_rate']:.3f}  "
             f"bg_speed={r['bg_speed']:.4f}  bg/fg={r['bg_fg_ratio']:.2f}  "
             f"moving_bg={r['moving_bg']:.1f}  vel_ll_gap={r['vel_ll_gap']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

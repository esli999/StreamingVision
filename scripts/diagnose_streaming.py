"""Workstream D — automated debug harness for the streaming tracker.

Offline driver per video: load the cached 3D motion (positions/velocities) +
RGB frames, recompute the chosen featurizer, run the *new* streaming tracker
(``genmatter_rt.init_state`` / ``step``), and score the live blob/hyperblob
assignments against three self-supervised references (feature, motion, SAM) plus
the cached pseudo-GT, per frame.  It also runs the plan's root-cause probes and
emits a panel MP4 + a JSON/Markdown report with per-hypothesis verdicts.

Each invocation evaluates ONE tracker configuration (set by the flags).  To
A/B two configs (e.g. baseline vs dense+calibrated+sam), run twice and pass the
first run's ``report.json`` as ``--baseline-report`` to the second for a delta.

LIMITATION (important): positions/velocities come from the cached 3D-motion npz,
which for some videos (e.g. ``test``) is a near-planar Z=const / |vz|~0 field.
The velocity-driven outlier path that dominates the LIVE streaming demo
(SEA-RAFT flow + per-frame ``_depth_to_Z`` + looping-during-init) therefore does
NOT fire here, so this harness's ``outlier_frac`` and any sigma_F tuning derived
from it do not transfer to the live regime — the depth-velocity probe flags this
explicitly.  To diagnose / tune the live outlier behaviour, the harness needs a
live depth/flow source (DepthAnythingV2 + SEA-RAFT), which is left as the clear
next step.  It is faithful for feature/cluster *semantics* (the DINO featurizer
is recomputed at streaming settings), just not for the motion/outlier regime.

Run from repo root, e.g.:

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python scripts/diagnose_streaming.py \
        --videos test --featurizer dense --calibrate-sigmas on --sam-init on \
        --out-dir runs/diagnose/full --max-frames 24
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "genmatterpp"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402

import genmatter_rt as G  # noqa: E402
import streaming_dino as sdino  # noqa: E402
import streaming_eval as seval  # noqa: E402

from genmatter.custom.config_schema import load_config  # noqa: E402
from genmatter.custom.paths import resolve_video_paths  # noqa: E402
from genmatter.dataloader import extract_3d_points_and_motion_vectors_data  # noqa: E402
from genmatter.instance_seg_metrics import frame_instance_jaccard_metrics  # noqa: E402

WORK_W, WORK_H = 640, 360
GH = WORK_H // G.STRIDE      # 45
GW = WORK_W // G.STRIDE      # 80


# ---------------- data loading ----------------

def _resize_hwc(img_hwc: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Resize an (H, W, C) float array to (gh, gw, C) (cv2 in <=4ch blocks)."""
    c = img_hwc.shape[2]
    out = np.empty((gh, gw, c), dtype=np.float32)
    src = img_hwc.astype(np.float32)
    for c0 in range(0, c, 4):
        c1 = min(c0 + 4, c)
        out[:, :, c0:c1] = cv2.resize(src[:, :, c0:c1], (gw, gh), interpolation=cv2.INTER_AREA)
    return out


def _load_motion(paths, video):
    pos, mvs, nT, (Hm, Wm) = extract_3d_points_and_motion_vectors_data(
        str(paths.npz_path.parent), video)
    return pos.reshape(nT, Hm, Wm, 3), mvs.reshape(nT, Hm, Wm, 3), nT


def _load_coarse_dino(paths):
    """Cached offline 10-comp PCA featurizer (T, 520, 960, 10), z-normalized."""
    d = np.load(paths.dino_path)
    feats = d["pca_features_unnormalized"].astype(np.float32)
    mean = d["gaussian_means"].astype(np.float32)
    std = d["gaussian_stds"].astype(np.float32)
    d.close()
    return (feats - mean) / np.where(std > 1e-6, std, 1.0)


# ---------------- per-config tracker run ----------------

def _build_cfg(base_cfg, args):
    cfg = json.loads(json.dumps(base_cfg))  # deep copy (plain dict)
    tr = cfg["tracking"]
    hp = tr["hyperparams"]
    tr["use_sam_frame0"] = (args.sam_init == "on")
    if args.feature_term == "off":
        # Disable feature *discrimination* without disturbing the inlier/outlier
        # balance: keep sigma_F at its normal value (so the per-point feature
        # log-lik magnitude is unchanged) but feed constant features (see
        # trk_features), so every blob's feature term is identical and only
        # position/velocity drive the assignment.  Calibration is moot here.
        tr["calibrate_feature_sigmas"] = False
    else:
        tr["calibrate_feature_sigmas"] = (args.calibrate_sigmas == "on")
        if args.sigma_F is not None:
            hp["sigma_F"] = args.sigma_F
        if args.sigma_F_H is not None:
            hp["sigma_F_H"] = args.sigma_F_H
    return cfg


def _feature_ref_grid(feats_all, k, method):
    return seval.cluster_labels(feats_all, k, method=method).reshape(GH, GW)


def diagnose_video(video, cfg, cfg_obj, args, model, palette):
    paths = resolve_video_paths(cfg_obj, video)
    pos_all, mv_all, nT = _load_motion(paths, video)
    rgb_files = sorted(paths.rgb_frames_dir.glob("*.jpg")) + sorted(paths.rgb_frames_dir.glob("*.png"))
    seg_files = sorted(paths.pseudo_gt_segmasks_dir.glob("*.png"))
    T = min(nT, len(rgb_files), len(seg_files))
    if args.max_frames > 0:
        T = min(T, args.max_frames)

    sam0_full = seval.sam_png_to_instances(paths.sam_path)
    K = max(2, seval.num_instances(sam0_full))
    sam_grid_rgb = cv2.resize(
        cv2.cvtColor(cv2.imread(str(paths.sam_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
        (GW, GH), interpolation=cv2.INTER_NEAREST)
    indices = G.subsample_indices(WORK_H, WORK_W, G.STRIDE, G.N_KEEP, seed=0)

    coarse = _load_coarse_dino(paths) if (args.featurizer == "coarse"
                                          or args.probe_featurizer) else None
    print(f"\n[{video}] T={T} K={K} featurizer={args.featurizer} "
          f"sam_init={args.sam_init} calibrate={args.calibrate_sigmas} "
          f"feature_term={args.feature_term}", flush=True)

    # Precompute per-frame data + references.
    rgb_bgr, dense_all, coarse_all = [], [], []
    feat_ref, motion_ref, sam_ref = [], [], []
    pca = (None, None, None)
    all_idx = np.arange(GH * GW, dtype=np.int32)
    probe_i = None
    for t in range(T):
        bgr = cv2.resize(cv2.imread(str(rgb_files[t]), cv2.IMREAD_COLOR),
                         (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        rgb_bgr.append(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        patches, grid_hw = sdino.dino_patches(model, rgb, "cuda")
        feats_all, basis, mean, std = G.dino_features_to_datapoints(
            patches, all_idx, pca[0], pca[1], pca[2], stride=G.STRIDE,
            image_hw=(WORK_H, WORK_W), target_dim=G.FEATURE_DIM, feat_grid_hw=grid_hw)
        if pca[0] is None:
            pca = (basis, mean, std)
        dense_all.append(feats_all)
        feat_ref.append(_feature_ref_grid(feats_all, K, args.cluster))

        mfeat = np.concatenate([
            _resize_hwc(pos_all[t][..., :2], GH, GW).reshape(-1, 2),
            _resize_hwc(mv_all[t][..., :2], GH, GW).reshape(-1, 2)], axis=1)
        mfeat = (mfeat - mfeat.mean(0)) / np.where(mfeat.std(0) > 1e-6, mfeat.std(0), 1.0)
        motion_ref.append(seval.cluster_labels(mfeat, K, method=args.cluster).reshape(GH, GW))

        sam_ref.append(seval.downsample_label_map(
            cv2.imread(str(seg_files[t]), cv2.IMREAD_UNCHANGED), GH, GW))

        if coarse is not None:
            cf = _resize_hwc(coarse[min(t, coarse.shape[0] - 1)], GH, GW).reshape(GH * GW, -1)
            coarse_all.append(cf)

    # Probe (i): coarse-vs-dense feature-ref quality on frame 0 (no tracker).
    if args.probe_featurizer and coarse_all:
        d_lab = seval.cluster_labels(dense_all[0], K, args.cluster)
        c_lab = seval.cluster_labels(coarse_all[0], K, args.cluster)
        sam0_idx = sam_ref[0].reshape(-1)
        probe_i = {
            "dense_silhouette": seval.safe_silhouette(dense_all[0], d_lab),
            "coarse_silhouette": seval.safe_silhouette(coarse_all[0], c_lab),
            "dense_ari_vs_sam": seval.agreement(d_lab, sam0_idx)["ari"],
            "coarse_ari_vs_sam": seval.agreement(c_lab, sam0_idx)["ari"],
        }

    # Tracker features per frame (dense subset, or coarse subset).  With the
    # feature term ablated, feed constant (zero) features so the term is
    # identical across blobs and can't pull assignments.
    def trk_features(t):
        src = dense_all[t] if args.featurizer == "dense" else coarse_all[t]
        feats = np.asarray(src[indices], dtype=np.float32)
        if args.feature_term == "off":
            feats = np.zeros_like(feats)
        return feats

    def frame_motion(t):
        pg = _resize_hwc(pos_all[t], GH, GW).reshape(-1, 3)[indices]
        vg = _resize_hwc(mv_all[t], GH, GW).reshape(-1, 3)[indices]
        return pg.astype(np.float32), vg.astype(np.float32)

    # Init + per-frame tracking.
    key = jax.random.PRNGKey(0)
    pos0, vel0 = frame_motion(0)
    state, key = G.init_state(
        pos0, vel0, trk_features(0), key, yaml_cfg=cfg,
        num_blobs=int(cfg["tracking"]["num_blobs"]),
        num_hyperblobs=int(cfg["tracking"]["num_hyperblobs"]),
        sam_segmentation=sam_grid_rgb if args.sam_init == "on" else None,
        subsample_indices=indices, verbose=args.verbose)

    per_frame = []
    frames_mp4 = []
    prev_hb_assign = None
    z_means = []
    for t in range(T):
        pos_t, vel_t = frame_motion(t)
        feat_t = trk_features(t)
        state, key = G.step(state, pos_t, vel_t, feat_t, key)
        state.datapoints_state.blob_assignments.block_until_ready()
        blob_a, hyper_a = G.extract_assignments(state)
        inlier = (blob_a >= 0) & (hyper_a >= 0)

        fr = feat_ref[t].reshape(-1)[indices]
        mr = motion_ref[t].reshape(-1)[indices]
        sr = sam_ref[t].reshape(-1)[indices]
        clu_feat = seval.agreement(hyper_a, fr, mask=inlier)
        clu_sam = seval.agreement(hyper_a, sr, mask=inlier)
        part_motion = seval.agreement(blob_a, mr, mask=inlier)

        # External check: live hyperblob dense map IoU vs pseudo-GT (grid res).
        live_hb_grid = G.labels_to_filled_grid(hyper_a, indices, GH, GW)
        jac = frame_instance_jaccard_metrics(sam_ref[t], live_hb_grid + 1)

        churn = float("nan")
        hb_assign = np.asarray(state.blobs_state.hyperblob_assignments)
        if prev_hb_assign is not None and prev_hb_assign.shape == hb_assign.shape:
            churn = float(np.mean(prev_hb_assign != hb_assign))
        prev_hb_assign = hb_assign
        z_means.append(float(np.mean(pos_t[:, 2])))

        rec = {
            "frame": t,
            "clusters_vs_feature_ari": clu_feat["ari"], "clusters_vs_feature_nmi": clu_feat["nmi"],
            "clusters_vs_sam_ari": clu_sam["ari"], "clusters_vs_sam_nmi": clu_sam["nmi"],
            "particles_vs_motion_ari": part_motion["ari"],
            "pseudo_gt_iou": jac["mean_matched_iou"], "pseudo_gt_pixjac": jac["pixel_jaccard"],
            "silhouette": seval.safe_silhouette(dense_all[t][indices][inlier], hyper_a[inlier])
            if inlier.sum() > 10 else float("nan"),
            "outlier_frac": float(np.mean(blob_a < 0)), "churn": churn,
            "n_hyperblobs_live": int(len(np.unique(hyper_a[hyper_a >= 0]))),
        }
        per_frame.append(rec)
        if not args.no_mp4:
            frames_mp4.append(_panel(rgb_bgr[t], blob_a, hyper_a, indices,
                                     motion_ref[t], feat_ref[t], sam_ref[t], palette, rec))

    return _summarize(video, cfg, args, per_frame, probe_i, z_means, pos_all, mv_all, T), frames_mp4


# ---------------- summary / verdicts ----------------

def _summarize(video, cfg, args, per_frame, probe_i, z_means, pos_all, mv_all, T):
    keys = [k for k in per_frame[0] if k != "frame"]
    means = {k: float(np.nanmean([f[k] for f in per_frame])) for k in keys}

    # Probe (iii): churn classification.
    churn = means["churn"]
    churn_verdict = ("frozen (~0, clusters not adapting)" if churn < 0.02
                     else "healthy (small, clusters adapting)" if churn < 0.30
                     else "thrashing (high, unstable)")

    # Probe (iv): depth scale drift + XY-vs-Z velocity sanity (cached motion Z).
    z_drift = float(np.std(z_means))
    z_span = float(np.ptp(z_means)) + float(np.ptp(mv_all[:T, ..., 2]))
    vxy = float(np.mean(np.linalg.norm(mv_all[:T, ..., :2], axis=-1)))
    vz = float(np.mean(np.abs(mv_all[:T, ..., 2])))
    z_ratio = vz / vxy if vxy > 1e-9 else float("inf")
    if z_drift < 1e-4 and vz < 1e-4:
        # The cached "test" npz is a flat plane (Z const, vz=0): the velocity-
        # outlier path that dominates the LIVE demo never fires here, so this
        # run does NOT reproduce the live outlier regime.  Tune live params with
        # live depth/flow, not this harness.
        depth_verdict = ("DEGENERATE near-planar cached motion (Z const, |vz|~0) "
                         "— NOT representative of the live depth/velocity regime")
    else:
        depth_verdict = (f"Z-dominated velocity (|vz|/|vxy|={z_ratio:.2f}) — suspicious"
                         if z_ratio > 2.0 else f"XY-dominated velocity (|vz|/|vxy|={z_ratio:.2f}) — ok")

    verdicts = {
        "churn": churn_verdict,
        "depth_velocity": depth_verdict,
        "depth_scale": f"Z mean-per-frame drift std={z_drift:.3f}",
    }
    if probe_i is not None:
        better = "dense" if probe_i["dense_ari_vs_sam"] >= probe_i["coarse_ari_vs_sam"] else "coarse"
        verdicts["featurizer"] = (
            f"{better} feature-ref agrees more with SAM "
            f"(dense ARI={probe_i['dense_ari_vs_sam']:.3f} sil={probe_i['dense_silhouette']:.3f} vs "
            f"coarse ARI={probe_i['coarse_ari_vs_sam']:.3f} sil={probe_i['coarse_silhouette']:.3f})")

    return {
        "video": video, "frames": T,
        "config": {"featurizer": args.featurizer, "feature_term": args.feature_term,
                   "calibrate_sigmas": args.calibrate_sigmas, "sam_init": args.sam_init,
                   "sigma_F": cfg["tracking"]["hyperparams"]["sigma_F"],
                   "sigma_F_H": cfg["tracking"]["hyperparams"]["sigma_F_H"]},
        "means": means, "probe_featurizer": probe_i,
        "depth_velocity": {"z_drift_std": z_drift, "vxy": vxy, "vz": vz, "z_ratio": z_ratio},
        "verdicts": verdicts, "per_frame": per_frame,
    }


# ---------------- visualization ----------------

def _label(img, txt):
    cv2.putText(img, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _panel(rgb_bgr, blob_a, hyper_a, indices, motion_g, feat_g, sam_g, palette, rec):
    part_bgr, clu_bgr = G.render_matter_tile(blob_a, hyper_a, indices, WORK_H, WORK_W, G.STRIDE)
    mot = seval.upscale_grid_bgr(seval.colorize_labels(motion_g, palette, bg_id=None), (WORK_H, WORK_W))
    fea = seval.upscale_grid_bgr(seval.colorize_labels(feat_g, palette, bg_id=None), (WORK_H, WORK_W))
    sam = seval.upscale_grid_bgr(seval.colorize_labels(sam_g, palette, bg_id=0), (WORK_H, WORK_W))
    row1 = np.hstack([_label(rgb_bgr.copy(), "rgb"),
                      _label(part_bgr, "live particles"),
                      _label(mot, "motion-ref")])
    row2 = np.hstack([_label(clu_bgr, f"live clusters ({rec['n_hyperblobs_live']})"),
                      _label(fea, "feature-ref"),
                      _label(sam, "SAM-ref")])
    grid = np.vstack([row1, row2])
    hdr = (f"frame {rec['frame']:3d}  clu-SAM ARI {rec['clusters_vs_sam_ari']:.2f}  "
           f"clu-feat ARI {rec['clusters_vs_feature_ari']:.2f}  IoU {rec['pseudo_gt_iou']:.2f}  "
           f"outlier {rec['outlier_frac']:.2f}  churn {rec['churn']:.2f}")
    return _label(grid, hdr)


def _write_mp4(frames, path, fps=8):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    tmp = str(path) + ".mpeg4.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(path)],
                       check=True)
        os.remove(tmp)
    else:
        os.rename(tmp, str(path))


# ---------------- report ----------------

def _write_report(out_dir, reports, baseline):
    (out_dir / "report.json").write_text(json.dumps(reports, indent=2))
    lines = ["# Streaming tracker diagnose (D)", ""]
    for r in reports:
        c = r["config"]
        lines += [f"## {r['video']}  ({r['frames']} frames)",
                  f"config: featurizer={c['featurizer']} feature_term={c['feature_term']} "
                  f"calibrate={c['calibrate_sigmas']} sam_init={c['sam_init']} "
                  f"sigma_F={c['sigma_F']:.4g} sigma_F_H={c['sigma_F_H']:.4g}", "",
                  "| metric | mean |", "|---|---|"]
        for k, v in r["means"].items():
            lines.append(f"| {k} | {v:.4f} |")
        lines += ["", "**verdicts:**"]
        for hk, hv in r["verdicts"].items():
            lines.append(f"- {hk}: {hv}")
        if baseline:
            b = next((x for x in baseline if x["video"] == r["video"]), None)
            if b:
                lines += ["", "**delta vs baseline:**"]
                for k in ("clusters_vs_sam_ari", "clusters_vs_feature_ari",
                          "pseudo_gt_iou", "outlier_frac"):
                    lines.append(f"- {k}: {r['means'][k] - b['means'][k]:+.4f} "
                                 f"({b['means'][k]:.4f} -> {r['means'][k]:.4f})")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", default="test")
    ap.add_argument("--featurizer", default="dense", choices=["dense", "coarse"])
    ap.add_argument("--feature-term", default="on", choices=["on", "off"])
    ap.add_argument("--calibrate-sigmas", default="off", choices=["on", "off"])
    ap.add_argument("--sam-init", default="off", choices=["on", "off"])
    ap.add_argument("--sigma-F", type=float, default=None, dest="sigma_F")
    ap.add_argument("--sigma-F-H", type=float, default=None, dest="sigma_F_H")
    ap.add_argument("--cluster", default="kmeans", choices=["kmeans", "hdbscan"])
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--probe-featurizer", action="store_true",
                    help="also cluster the cached coarse featurizer for the dense-vs-coarse probe")
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--baseline-report", default=None, help="report.json to delta against")
    ap.add_argument("--config", default=str(REPO / "configs" / "streaming_default.yaml"))
    ap.add_argument("--out-dir", default=str(REPO / "runs" / "diagnose"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    base_cfg = G.load_yaml_hypers(Path(args.config))
    cfg = _build_cfg(base_cfg, args)
    # resolve_video_paths needs a CustomConfig, not the plain dict.
    cfg_obj = load_config(Path(args.config))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = seval.make_palette(64, seed=0)
    baseline = json.loads(Path(args.baseline_report).read_text()) if args.baseline_report else None

    print("loading DINOv2-S ...", flush=True)
    model = sdino.load_dino("cuda")

    reports = []
    for video in videos:
        report, frames = diagnose_video(video, cfg, cfg_obj, args, model, palette)
        reports.append(report)
        m = report["means"]
        print(f"  [{video}] clu-SAM ARI={m['clusters_vs_sam_ari']:.3f} "
              f"clu-feat ARI={m['clusters_vs_feature_ari']:.3f} "
              f"pseudoGT IoU={m['pseudo_gt_iou']:.3f} outlier={m['outlier_frac']:.3f} "
              f"churn={m['churn']:.3f}", flush=True)
        for hk, hv in report["verdicts"].items():
            print(f"    [{hk}] {hv}", flush=True)
        if not args.no_mp4 and frames:
            mp4 = out_dir / f"{video}_diagnose.mp4"
            _write_mp4(frames, mp4)
            print(f"    wrote {mp4}", flush=True)

    _write_report(out_dir, reports, baseline)
    print(f"\nwrote {out_dir/'report.json'} and {out_dir/'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

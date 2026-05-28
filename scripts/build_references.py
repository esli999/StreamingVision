"""Workstream C3 — self-supervised reference label grids.

Per video, emit ``(T, GH, GW)`` int label grids on the streaming stride-8 grid,
for the diagnose harness to score the live tracker against:

  * feature reference (semantics) — dense streaming DINO recomputed per frame
    (frozen frame-0 PCA basis, matching the tracker) -> KMeans(K).
  * motion reference (particle-consistency) — cluster ``[pos_XY, 2D velocity]``
    from the cached 3D-motion npz with the jittery Z dropped, standardized,
    KMeans(K).
  * SAM reference — cached per-frame ``pseudo_gt_sam/segmasks`` instance maps.

``K`` defaults to the number of SAM-frame-0 instances so all three references
carve the scene at a comparable object-level granularity.

Run from repo root, e.g.:

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
        python scripts/build_references.py --videos test --out runs/references
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "genmatterpp"))

import cv2
import numpy as np

import genmatter_rt  # noqa: E402
import streaming_dino as sdino  # noqa: E402
import streaming_eval as seval  # noqa: E402

from genmatter.custom.config_schema import load_config  # noqa: E402
from genmatter.custom.paths import resolve_video_paths  # noqa: E402
from genmatter.dataloader import extract_3d_points_and_motion_vectors_data  # noqa: E402

WORK_W, WORK_H = 640, 360
GH = WORK_H // genmatter_rt.STRIDE        # 45
GW = WORK_W // genmatter_rt.STRIDE        # 80


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    return (x - mu) / np.where(sd > 1e-6, sd, 1.0)


def _motion_features(pos_frame: np.ndarray, vel_frame: np.ndarray) -> np.ndarray:
    """[pos_XY, vel_XY] on the grid, standardized.  Z dropped (jittery)."""
    posg = cv2.resize(pos_frame.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
    velg = cv2.resize(vel_frame.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
    feat = np.concatenate([posg[..., :2].reshape(-1, 2),
                           velg[..., :2].reshape(-1, 2)], axis=1)  # (GH*GW, 4)
    return _standardize(feat)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", default="test", help="comma-separated video ids")
    ap.add_argument("--feature-dim", type=int, default=genmatter_rt.FEATURE_DIM)
    ap.add_argument("--cluster", default="kmeans", choices=["kmeans", "hdbscan"])
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--config", default=str(REPO / "configs" / "streaming_default.yaml"))
    ap.add_argument("--out", default=str(REPO / "runs" / "references"))
    args = ap.parse_args()

    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    cfg = load_config(Path(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = seval.make_palette(64, seed=0)

    print("loading DINOv2-S ...", flush=True)
    model = sdino.load_dino("cuda")
    summary = {}

    for video in videos:
        paths = resolve_video_paths(cfg, video)
        rgb_files = sorted(paths.rgb_frames_dir.glob("*.jpg")) + sorted(paths.rgb_frames_dir.glob("*.png"))
        seg_files = sorted(paths.pseudo_gt_segmasks_dir.glob("*.png"))
        positions, motion_vectors, n_motion, img_dims = extract_3d_points_and_motion_vectors_data(
            str(paths.npz_path.parent), video)
        Hm, Wm = img_dims
        positions = positions.reshape(n_motion, Hm, Wm, 3)
        motion_vectors = motion_vectors.reshape(n_motion, Hm, Wm, 3)

        T = min(len(rgb_files), n_motion, len(seg_files))
        if args.max_frames > 0:
            T = min(T, args.max_frames)
        if T == 0:
            print(f"[{video}] no overlapping frames (rgb={len(rgb_files)} "
                  f"motion={n_motion} seg={len(seg_files)}) — skipping", flush=True)
            continue

        # K from the SAM-frame-0 instance count (object-level granularity).
        sam0 = seval.sam_png_to_instances(paths.sam_path)
        K = max(2, seval.num_instances(sam0))
        print(f"\n[{video}] T={T} (rgb={len(rgb_files)} motion={n_motion} seg={len(seg_files)}) K={K}",
              flush=True)

        feature_ref = np.zeros((T, GH, GW), dtype=np.int32)
        motion_ref = np.zeros((T, GH, GW), dtype=np.int32)
        sam_ref = np.zeros((T, GH, GW), dtype=np.int32)
        all_idx = np.arange(GH * GW, dtype=np.int32)
        pca = (None, None, None)  # frozen on frame 0
        sil0 = {}

        for t in range(T):
            bgr = cv2.resize(cv2.imread(str(rgb_files[t]), cv2.IMREAD_COLOR),
                             (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            patches, grid_hw = sdino.dino_patches(model, rgb, "cuda")
            feats, basis, mean, std = genmatter_rt.dino_features_to_datapoints(
                patches, all_idx, pca[0], pca[1], pca[2],
                stride=genmatter_rt.STRIDE, image_hw=(WORK_H, WORK_W),
                target_dim=args.feature_dim, feat_grid_hw=grid_hw)
            if pca[0] is None:
                pca = (basis, mean, std)
            f_lab = seval.cluster_labels(feats, K, method=args.cluster)
            feature_ref[t] = f_lab.reshape(GH, GW)

            mfeat = _motion_features(positions[t], motion_vectors[t])
            m_lab = seval.cluster_labels(mfeat, K, method=args.cluster)
            motion_ref[t] = m_lab.reshape(GH, GW)

            seg = cv2.imread(str(seg_files[t]), cv2.IMREAD_UNCHANGED)
            sam_ref[t] = seval.downsample_label_map(seg, GH, GW)

            if t == 0:
                sil0 = {"feature": seval.safe_silhouette(feats, f_lab),
                        "motion": seval.safe_silhouette(mfeat, m_lab)}

        npz_path = out_dir / f"{video}_references.npz"
        np.savez_compressed(
            npz_path, feature_ref=feature_ref, motion_ref=motion_ref, sam_ref=sam_ref,
            grid_hw=np.array([GH, GW], np.int32), K=np.int32(K),
            cluster=np.array(args.cluster), feature_dim=np.int32(args.feature_dim))

        # [frame-0 distinct labels, distinct labels over all frames]; feature /
        # motion clusters have no background, SAM counts non-bg instances.
        counts = {
            "feature": [int(np.unique(feature_ref[0]).size), int(np.unique(feature_ref).size)],
            "motion": [int(np.unique(motion_ref[0]).size), int(np.unique(motion_ref).size)],
            "sam": [int(seval.num_instances(sam_ref[0])), int(seval.num_instances(sam_ref))],
        }
        summary[video] = {"T": int(T), "K": int(K), "silhouette_frame0": sil0,
                          "label_counts_frame0_total": counts, "npz": str(npz_path)}
        print(f"  feature sil={sil0['feature']:.3f} motion sil={sil0['motion']:.3f} "
              f"sam instances f0={seval.num_instances(sam_ref[0])}", flush=True)
        print(f"  wrote {npz_path}", flush=True)

        # Frame-0 montage: rgb | feature | motion | SAM
        bgr0 = cv2.resize(cv2.imread(str(rgb_files[0]), cv2.IMREAD_COLOR),
                          (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        tiles = [bgr0]
        for name, grid, bg in (("feature", feature_ref[0], None),
                               ("motion", motion_ref[0], None),
                               ("SAM", sam_ref[0], 0)):
            tile = seval.upscale_grid_bgr(seval.colorize_labels(grid, palette, bg_id=bg), (WORK_H, WORK_W))
            cv2.putText(tile, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        cv2.imwrite(str(out_dir / f"{video}_references_frame0.png"), np.hstack(tiles))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

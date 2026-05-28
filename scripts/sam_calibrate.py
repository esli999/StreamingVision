"""Workstream C1 — SAM-frame-0 calibration gate (Step 1 of the calibration).

Fast, single-frame, no tracker.  For each ``(dino_res, feature_dim)`` in a small
grid, on frame 0 of each video:

  1. compute dense streaming DINO at that resolution,
  2. gather to the stride-8 (80x45) tracking grid + PCA to ``feature_dim`` and
     z-score (``genmatter_rt.dino_features_to_datapoints``),
  3. KMeans / HDBSCAN into K (= number of SAM-frame-0 instances) clusters,
  4. score the cluster map vs the SAM-frame-0 instance map (downsampled to the
     grid) with mean matched IoU + ARI + NMI + silhouette,
  5. emit the Step-2 within-cluster feature-variance (a ``sigma_F`` estimate).

A 224x224 (16x16-patch) cell is included as the pre-densification baseline so the
report shows the dense featurizer beating it directly.  Pick the knee: the
highest SAM agreement at the lowest (res, dim) cost.

Run from repo root, e.g.:

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
        python scripts/sam_calibrate.py --videos test --out runs/sam_calib
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

import genmatter_rt  # noqa: E402 — also puts genmatterpp/ on sys.path
import streaming_dino as sdino  # noqa: E402
import streaming_eval as seval  # noqa: E402

from genmatter.custom.config_schema import load_config  # noqa: E402
from genmatter.custom.paths import resolve_video_paths  # noqa: E402
from genmatter.instance_seg_metrics import frame_instance_jaccard_metrics  # noqa: E402

WORK_W, WORK_H = 640, 360                 # streaming working resolution
GH = WORK_H // genmatter_rt.STRIDE        # 45
GW = WORK_W // genmatter_rt.STRIDE        # 80
DEFAULT_RES = ["224x224", "462x266", "644x364", "840x476"]
DEFAULT_DIMS = [16, 32, 64]


def _parse_res(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def _load_frame0_rgb(paths) -> np.ndarray:
    """Frame 0 as RGB, resized to the streaming working resolution (mirrors the
    FrameSource INTER_AREA resize the live pipeline does before DINO)."""
    cands = sorted(paths.rgb_frames_dir.glob("*.jpg")) + sorted(paths.rgb_frames_dir.glob("*.png"))
    if not cands:
        raise FileNotFoundError(f"no rgb frames under {paths.rgb_frames_dir}")
    bgr = cv2.imread(str(cands[0]), cv2.IMREAD_COLOR)
    bgr = cv2.resize(bgr, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _feature_grid(patches, grid_hw, dim):
    """Dense DINO patches -> (GH*GW, dim) z-scored features on the stride-8 grid."""
    all_idx = np.arange(GH * GW, dtype=np.int32)
    feats, _, _, _ = genmatter_rt.dino_features_to_datapoints(
        patches, all_idx, stride=genmatter_rt.STRIDE, image_hw=(WORK_H, WORK_W),
        target_dim=dim, feat_grid_hw=grid_hw)
    return feats


def _evaluate_cell(feats, sam_grid, k, method):
    """Cluster features and score against the SAM grid.  Returns a metrics dict."""
    labels = seval.cluster_labels(feats, k, method=method)
    cluster_grid = labels.reshape(GH, GW)
    sam_flat = sam_grid.reshape(-1)
    # ARI/NMI over all grid cells (bg included); IoU treats cluster 0 as a real
    # instance so offset by +1 (SAM grid already uses 0=bg).
    agr = seval.agreement(sam_flat, labels)
    jac = frame_instance_jaccard_metrics(sam_grid, cluster_grid + 1)
    return {
        "mean_matched_iou": jac["mean_matched_iou"],
        "pixel_jaccard": jac["pixel_jaccard"],
        "ari": agr["ari"],
        "nmi": agr["nmi"],
        "silhouette": seval.safe_silhouette(feats, labels),
        "sigma_F_within_sam": seval.within_label_variance(feats, sam_flat),
        "sigma_F_within_kmeans": seval.within_label_variance(feats, labels),
        "n_clusters": int(len(np.unique(labels))),
    }, cluster_grid


def _montage(rgb, sam_grid, cluster_grid, palette):
    """[rgb | SAM instances | feature clusters] side-by-side, all WORK-sized."""
    tile_hw = (WORK_H, WORK_W)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    sam_bgr = seval.upscale_grid_bgr(seval.colorize_labels(sam_grid, palette), tile_hw)
    clu_bgr = seval.upscale_grid_bgr(
        seval.colorize_labels(cluster_grid + 1, palette, bg_id=None), tile_hw)
    for img, txt in ((rgb_bgr, "rgb"), (sam_bgr, "SAM"), (clu_bgr, "feature clusters")):
        cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack([rgb_bgr, sam_bgr, clu_bgr])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", default="test", help="comma-separated video ids")
    ap.add_argument("--dino-res", default=",".join(DEFAULT_RES),
                    help="comma-separated WxH list (multiples of 14)")
    ap.add_argument("--feature-dim", default=",".join(map(str, DEFAULT_DIMS)),
                    help="comma-separated PCA dims")
    ap.add_argument("--cluster", default="kmeans", choices=["kmeans", "hdbscan"])
    ap.add_argument("--config", default=str(REPO / "configs" / "streaming_default.yaml"))
    ap.add_argument("--out", default=str(REPO / "runs" / "sam_calib"))
    args = ap.parse_args()

    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    res_list = [_parse_res(r) for r in args.dino_res.split(",") if r.strip()]
    dim_list = [int(d) for d in args.feature_dim.split(",") if d.strip()]
    cfg = load_config(Path(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = seval.make_palette(64, seed=0)

    print(f"loading DINOv2-S ...", flush=True)
    model = sdino.load_dino("cuda")

    report = {"videos": {}, "grid": {"res": args.dino_res, "dim": dim_list,
                                       "cluster": args.cluster}}
    for video in videos:
        paths = resolve_video_paths(cfg, video)
        rgb0 = _load_frame0_rgb(paths)
        sam_full = seval.sam_png_to_instances(paths.sam_path)
        sam_grid = seval.downsample_label_map(sam_full, GH, GW)
        n_sam = seval.num_instances(sam_full)
        k = max(2, n_sam)
        print(f"\n[{video}] SAM instances={n_sam} -> K={k}", flush=True)

        cells = []
        best_montage = None
        best_key = None
        for (dw, dh) in res_list:
            patches, grid_hw = sdino.dino_patches(model, rgb0, "cuda", dino_h=dh, dino_w=dw)
            for dim in dim_list:
                if dim > patches.shape[1]:
                    continue
                feats = _feature_grid(patches, grid_hw, dim)
                metrics, cluster_grid = _evaluate_cell(feats, sam_grid, k, args.cluster)
                cell = {"res": f"{dw}x{dh}", "patch_grid": list(grid_hw), "dim": dim,
                        "is_baseline": (dw == 224 and dh == 224), **metrics}
                cells.append(cell)
                tag = f"{dw}x{dh}/d{dim}"
                print(f"  {tag:18s} mIoU={metrics['mean_matched_iou']:.3f} "
                      f"ARI={metrics['ari']:.3f} NMI={metrics['nmi']:.3f} "
                      f"sil={metrics['silhouette']:.3f} "
                      f"sigF~{metrics['sigma_F_within_sam']:.3f}", flush=True)
                # Keep a montage for the streaming-default-ish cell (644x364, d32)
                # or, failing that, the best mIoU cell seen so far.
                is_default = (dw == 644 and dh == 364 and dim == 32)
                if is_default or (best_montage is None):
                    if is_default or best_key is None:
                        best_montage = _montage(rgb0, sam_grid, cluster_grid, palette)
                        best_key = tag

        # Pick the knee: highest mean_matched_iou (tie-break ARI), and report the
        # dense-vs-baseline delta on the shared metrics.
        dense = [c for c in cells if not c["is_baseline"]]
        base = [c for c in cells if c["is_baseline"]]
        best = max(cells, key=lambda c: (c["mean_matched_iou"], c["ari"]))
        report["videos"][video] = {
            "n_sam_instances": n_sam, "K": k, "cells": cells, "best": best,
        }
        if base:
            b = max(base, key=lambda c: c["mean_matched_iou"])
            d = max(dense, key=lambda c: c["mean_matched_iou"]) if dense else b
            print(f"  [{video}] baseline(16x16) mIoU={b['mean_matched_iou']:.3f} "
                  f"ARI={b['ari']:.3f} sil={b['silhouette']:.3f}  |  "
                  f"best-dense mIoU={d['mean_matched_iou']:.3f} ARI={d['ari']:.3f} "
                  f"sil={d['silhouette']:.3f}  ({d['res']}/d{d['dim']})", flush=True)
        if best_montage is not None:
            mp = out_dir / f"{video}_montage_{best_key.replace('/', '_')}.png"
            cv2.imwrite(str(mp), best_montage)
            print(f"  wrote {mp}", flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    _write_markdown(out_dir / "report.md", report)
    print(f"\nwrote {out_dir/'report.json'} and {out_dir/'report.md'}")
    return 0


def _write_markdown(path: Path, report: dict) -> None:
    lines = ["# SAM-frame-0 calibration gate (C1)", "",
             f"cluster={report['grid']['cluster']}  dims={report['grid']['dim']}", ""]
    for video, vr in report["videos"].items():
        lines += [f"## {video}  (SAM instances={vr['n_sam_instances']}, K={vr['K']})", "",
                  "| res | patch | dim | mIoU | pixJac | ARI | NMI | silhouette | sigF~SAM | sigF~km |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c in sorted(vr["cells"], key=lambda c: (-c["mean_matched_iou"], -c["ari"])):
            flag = " (baseline)" if c["is_baseline"] else ""
            lines.append(
                f"| {c['res']}{flag} | {c['patch_grid'][0]}x{c['patch_grid'][1]} | {c['dim']} "
                f"| {c['mean_matched_iou']:.3f} | {c['pixel_jaccard']:.3f} | {c['ari']:.3f} "
                f"| {c['nmi']:.3f} | {c['silhouette']:.3f} | {c['sigma_F_within_sam']:.3f} "
                f"| {c['sigma_F_within_kmeans']:.3f} |")
        b = vr["best"]
        lines += ["", f"**best:** {b['res']} d{b['dim']} — mIoU={b['mean_matched_iou']:.3f}, "
                  f"ARI={b['ari']:.3f}, recommended sigma_F~={b['sigma_F_within_sam']:.3f} "
                  f"(<< the YAML default of 2-9)", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())

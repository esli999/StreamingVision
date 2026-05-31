"""SAM2 video propagation over all 30 DAVIS videos → per-frame uint16 instance-id PNGs.

The user wants self-supervised pseudo-labels for the entire calibration set, NOT the
DAVIS GT segmasks. This script runs `segment_sequence` (which does SAM2 video
propagation + per-frame new-object detection) on each DAVIS RGB-frames directory and
saves the resulting masks under
``assets/tapvid_davis_30_videos_processed/sam2_propagated/<vid>/<i:05d>.png``
as uint16 PNGs (same format as the local custom-video pseudo_gt_sam caches).

Resumable: skips videos that already have a complete output directory.

Measured per-video cost (RTX 5090, sam2.1_l.pt, video_imgsz=1024, hybrid):
- FHD (1920×1080, e.g. blackswan, 50 frames): ~100 s.
- 4K (3840×2160, e.g. judo, 34 frames): ~130 s.
Expected total: ~50-55 min for 30 videos.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "genmatterpp"))
os.environ.setdefault("GENMATTER_DAVIS_DIR", str(_REPO / "assets"))

import config as gm_config  # noqa: E402  — must come after env var set
from genmatter.pseudo_gt import sam_video as _sam_video_mod  # noqa: E402 — monkeypatch target
from genmatter.pseudo_gt.sam_video import segment_sequence  # noqa: E402
from genmatter.pseudo_gt.correspondence import label_map_from_masks  # noqa: E402


OUT_ROOT = _REPO / "assets" / "tapvid_davis_30_videos_processed" / "sam2_propagated"
WEIGHTS_DIR = _REPO / "assets" / "deeplearning_weights"


# ---------------------------------------------------------------------------
# Self-supervised SALIENCE prior for SAM2 frame-0 object selection (Part 1).
# The vendored `segment_sam2_video_tracked` ranks the frame-0 auto-mask bboxes by
# AREA (`select_bboxes_by_area`) and keeps the largest `max_objects`. On
# small-subject videos (kite-surf, libby, horsejump-high) the salient subject is
# small → dropped → the propagated mask tracks the WRONG object (SAM-vs-GT frame-0
# IoU = 0.000), capping the self-supervised region-J on exactly the low-GT videos
# that drag the held-out aggregate. We replace area-ranking with a FIXED a-priori
# salience score
#     salience(bbox) = w_c * centrality + w_m * motion
# both terms self-supervised and domain-agnostic (NOT per-video, NOT tuned to any
# metric): the salient subject is central AND moving; static background (water,
# sky) scores low. Implemented by MONKEYPATCHing the module symbol
# `sam_video.select_bboxes_by_area` (no vendored edit — VENDORED.md); the patch
# reads a per-video context (`_SALIENCE_CTX`) set in `propagate_one`, and on ANY
# error falls back to the original area ranking (best-effort improvement only).
# ---------------------------------------------------------------------------
_SALIENCE_ORIG = _sam_video_mod.select_bboxes_by_area
_SALIENCE_CTX: dict = {}


def _farneback_motion_per_bbox(bboxes: np.ndarray) -> Optional[np.ndarray]:
    """Mean optical-flow magnitude (frame 0→1) inside each xyxy bbox.

    Returns None when the second frame / flow is unavailable so the caller falls
    back to centrality only. cv2.calcOpticalFlowFarneback — no GPU / model needed.
    """
    f0 = _SALIENCE_CTX.get("frame0")
    f1 = _SALIENCE_CTX.get("frame1")
    if f0 is None or f1 is None:
        return None
    g0 = cv2.imread(str(f0), cv2.IMREAD_GRAYSCALE)
    g1 = cv2.imread(str(f1), cv2.IMREAD_GRAYSCALE)
    if g0 is None or g1 is None or g0.shape != g1.shape:
        return None
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    H, W = mag.shape
    out = np.zeros(len(bboxes), dtype=np.float64)
    for i, b in enumerate(bboxes):
        xi0 = max(0, int(np.floor(b[0]))); yi0 = max(0, int(np.floor(b[1])))
        xi1 = min(W, int(np.ceil(b[2])));  yi1 = min(H, int(np.ceil(b[3])))
        if xi1 > xi0 and yi1 > yi0:
            out[i] = float(mag[yi0:yi1, xi0:xi1].mean())
    return out


def _centrality_per_bbox(bboxes: np.ndarray) -> np.ndarray:
    """1 - ||bbox_center - frame_center|| / half_diagonal, clipped to [0, 1]."""
    hw = _SALIENCE_CTX.get("native_hw")
    if hw is not None:
        H, W = float(hw[0]), float(hw[1])
    else:  # frame size unknown → approximate from the bbox extent
        W = float(bboxes[:, 2].max()); H = float(bboxes[:, 3].max())
    cx, cy = W / 2.0, H / 2.0
    half_diag = 0.5 * float(np.hypot(W, H)) + 1e-6
    bx = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
    by = 0.5 * (bboxes[:, 1] + bboxes[:, 3])
    dist = np.hypot(bx - cx, by - cy)
    return np.clip(1.0 - dist / half_diag, 0.0, 1.0)


def _select_bboxes_salient(bboxes: np.ndarray, max_objects):
    """Salience-ranked replacement for `select_bboxes_by_area` (monkeypatched).

    Keeps the top-`max_objects` bboxes by `w_c*centrality + w_m*motion`. Identical
    contract to the original: returns all boxes unchanged when `max_objects is None`
    or `len(bboxes) <= max_objects`. Any failure falls back to the area ranking.
    """
    try:
        bboxes = np.asarray(bboxes)
        if max_objects is None or len(bboxes) <= int(max_objects):
            return bboxes
        w_c = float(_SALIENCE_CTX.get("w_c", 0.5))
        w_m = float(_SALIENCE_CTX.get("w_m", 0.5))
        centrality = _centrality_per_bbox(bboxes)
        motion = _farneback_motion_per_bbox(bboxes)
        if motion is None or float(np.ptp(motion)) < 1e-9:
            motion_norm = np.zeros(len(bboxes), dtype=np.float64)
            w_m = 0.0  # no usable motion signal → centrality only
        else:
            p90 = float(np.percentile(motion, 90)) + 1e-9
            motion_norm = np.clip(motion / p90, 0.0, 1.0)
        denom = (w_c + w_m) or 1.0
        score = (w_c * centrality + w_m * motion_norm) / denom
        keep = np.sort(np.argsort(score)[-int(max_objects):])  # top-K, frame-0 order
        return bboxes[keep]
    except Exception:  # never let salience break a regen — fall back to area
        return _SALIENCE_ORIG(bboxes, max_objects)


def _apply_salience_context(rgb_dir: Path, salience: bool,
                            salience_weights: tuple) -> str:
    """Set the per-video salience context and (de)activate the monkeypatch.

    Returns the active selection mode ("salience" | "area") for logging.
    """
    if not salience:
        _sam_video_mod.select_bboxes_by_area = _SALIENCE_ORIG
        return "area"
    frames = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
    bgr0 = cv2.imread(str(frames[0])) if frames else None
    _SALIENCE_CTX.clear()
    _SALIENCE_CTX.update({
        "frame0": frames[0] if frames else None,
        "frame1": frames[1] if len(frames) > 1 else None,
        "native_hw": (bgr0.shape[0], bgr0.shape[1]) if bgr0 is not None else None,
        "w_c": float(salience_weights[0]),
        "w_m": float(salience_weights[1]),
    })
    _sam_video_mod.select_bboxes_by_area = _select_bboxes_salient
    return "salience"


def _save_label_maps(out_dir: Path, per_frame: List[List[np.ndarray]],
                     track_ids: Optional[List[List[int]]],
                     h: int, w: int) -> int:
    """Rasterize per-frame masks to (H, W) uint16 instance-id PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, masks in enumerate(per_frame):
        tids = track_ids[i] if track_ids else list(range(len(masks)))
        label_map = label_map_from_masks(masks, tids, h, w)
        # cv2.imwrite handles uint16 PNG natively
        ok = cv2.imwrite(str(out_dir / f"{i:05d}.png"), label_map.astype(np.uint16))
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {out_dir / f'{i:05d}.png'}")
        n += 1
    return n


def _video_already_done(out_dir: Path, rgb_dir: Path) -> bool:
    if not out_dir.is_dir():
        return False
    n_in = len(list(rgb_dir.glob("*.jpg")))
    n_out = len(list(out_dir.glob("*.png")))
    return n_in > 0 and n_out == n_in


def _backup_existing(out_dir: Path) -> Optional[Path]:
    """Move existing masks aside to a sibling ``<vid>`` under sam2_propagated_backup
    so a regen is reversible (the plan: keep the old cache as a checkpoint and
    compare per-video self-supervised region-J before committing). Idempotent: if
    a backup already exists it is left intact (never clobber the first checkpoint)."""
    import shutil
    if not out_dir.is_dir() or not any(out_dir.glob("*.png")):
        return None
    backup = out_dir.parent.parent / "sam2_propagated_backup" / out_dir.name
    if backup.is_dir() and any(backup.glob("*.png")):
        return backup   # first checkpoint already taken
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(out_dir, backup, dirs_exist_ok=True)
    return backup


def propagate_one(vid: str, *, force: bool = False, backup: bool = False,
                  model: str = "sam2.1_l.pt", min_threshold: float = 0.08,
                  max_objects: int = 20, video_imgsz: int = 1024,
                  video_chunk_frames: int = 16, iou_threshold: float = 0.3,
                  salience: bool = False,
                  salience_weights: tuple = (0.5, 0.5)) -> dict:
    info = {"vid": vid}
    rgb_dir = Path(gm_config.DAVIS_RGB_PATH) / vid
    out_dir = OUT_ROOT / vid
    info["rgb_dir"] = str(rgb_dir)
    info["out_dir"] = str(out_dir)
    if not rgb_dir.is_dir() or not any(rgb_dir.iterdir()):
        info["status"] = "missing_frames"
        return info
    if _video_already_done(out_dir, rgb_dir) and not force:
        info["status"] = "already_done"
        info["frames"] = len(list(out_dir.glob("*.png")))
        return info

    # Checkpoint the prior masks before overwriting (reversible regen).
    if backup:
        bk = _backup_existing(out_dir)
        if bk is not None:
            info["backup_dir"] = str(bk)

    # Part 1: activate the self-supervised salience prior (centrality + motion) in
    # place of area-ranking for frame-0 object selection (monkeypatch, reversible).
    info["select"] = _apply_salience_context(rgb_dir, salience, salience_weights)

    t0 = time.monotonic()
    # Higher-quality kwargs (the plan, Phase E): lower min_threshold (more recall),
    # raise max_objects (capture more instances), keep detect_new_objects. All are
    # EXISTING segment_sequence kwargs — no vendored edit.
    per_frame, method, (h, w), track_ids = segment_sequence(
        rgb_dir,
        model=model,
        weights_dir=WEIGHTS_DIR,
        min_threshold=min_threshold,
        prefer_video=True,
        video_imgsz=video_imgsz,
        max_objects=max_objects,
        video_chunk_frames=video_chunk_frames,
        iou_threshold=iou_threshold,
        detect_new_objects=True,
        show_tqdm=False,
    )
    t_sam = time.monotonic() - t0

    n_saved = _save_label_maps(out_dir, per_frame, track_ids, h, w)
    info["wall_sec"] = t_sam
    info["frames"] = n_saved
    info["sec_per_frame"] = t_sam / max(n_saved, 1)
    info["native_hw"] = (h, w)
    info["method"] = method
    info["n_objects"] = int(max((len(m) for m in per_frame), default=0))
    info["status"] = "ok"
    return info


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", type=str, default=None,
                   help="Comma-separated subset of TAPVID_DAVIS_VIDEO_NAMES; default = all 30.")
    p.add_argument("--force", action="store_true",
                   help="Re-propagate even if output dir already populated.")
    p.add_argument("--backup", action="store_true",
                   help="Checkpoint existing masks to sam2_propagated_backup/<vid> before "
                        "overwriting (reversible regen; the first checkpoint is never clobbered).")
    # Higher-quality SAM2 kwargs (Phase E). Defaults are the IMPROVED values
    # (lower threshold + more objects); pass the old values to reproduce v1.
    p.add_argument("--model", type=str, default="sam2.1_l.pt")
    p.add_argument("--min-threshold", type=float, default=0.08,
                   help="SAM2 mask confidence floor (was 0.15; lower = more recall).")
    p.add_argument("--max-objects", type=int, default=20,
                   help="Max tracked instances (was 12; higher = capture more objects).")
    p.add_argument("--video-imgsz", type=int, default=1024,
                   help="SAM2 video propagation image size (raise for more detail, more VRAM).")
    p.add_argument("--video-chunk-frames", type=int, default=16)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    # Self-supervised salience prior for frame-0 object selection. OPT-IN: the
    # offline prototype (scripts/_salience_filter_proto.py) showed a salience
    # rerank/threshold is not a robust, generalizing win (no global tau is safe;
    # it helps some videos and collapses others), so the script DEFAULT keeps the
    # vendored area ranking (the known-good pseudo-labels). --salience is retained
    # for experiments; weights are FIXED a-priori (not tuned to any metric).
    p.add_argument("--salience", action="store_true",
                   help="Enable the centrality+motion salience prior for frame-0 selection (experimental).")
    p.add_argument("--salience-centrality-weight", type=float, default=0.5,
                   help="A-priori weight on bbox centrality (default 0.5).")
    p.add_argument("--salience-motion-weight", type=float, default=0.5,
                   help="A-priori weight on bbox optical-flow motion (default 0.5).")
    args = p.parse_args(argv)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    expected = list(gm_config.TAPVID_DAVIS_VIDEO_NAMES)
    if args.videos:
        wanted = set(v.strip() for v in args.videos.split(","))
        expected = [v for v in expected if v in wanted]

    sel = (f"salience(c={args.salience_centrality_weight},m={args.salience_motion_weight})"
           if args.salience else "area")
    print(f"[sam2_davis_propagate] running on {len(expected)} video(s); out={OUT_ROOT}; "
          f"min_threshold={args.min_threshold} max_objects={args.max_objects} "
          f"video_imgsz={args.video_imgsz} backup={args.backup} select={sel}", flush=True)
    total_sec = 0.0
    n_done = 0
    for i, vid in enumerate(expected, 1):
        print(f"[sam2_davis_propagate] {i}/{len(expected)}: {vid}", flush=True)
        info = propagate_one(vid, force=args.force, backup=args.backup, model=args.model,
                             min_threshold=args.min_threshold, max_objects=args.max_objects,
                             video_imgsz=args.video_imgsz,
                             video_chunk_frames=args.video_chunk_frames,
                             iou_threshold=args.iou_threshold,
                             salience=args.salience,
                             salience_weights=(args.salience_centrality_weight,
                                               args.salience_motion_weight))
        if info["status"] == "ok":
            print(f"  ok: {info['frames']} frames ({info.get('n_objects', '?')} objs) in "
                  f"{info['wall_sec']:.1f}s ({info['sec_per_frame']:.2f}s/frame) @ {info['native_hw']}"
                  f" [select={info.get('select', '?')}]"
                  + (f" [backed up → {info['backup_dir']}]" if info.get("backup_dir") else ""),
                  flush=True)
            total_sec += info["wall_sec"]
            n_done += 1
        elif info["status"] == "already_done":
            print(f"  skip: {info['frames']} frames already present", flush=True)
        else:
            print(f"  FAIL: {info['status']}", flush=True)

    print(f"[sam2_davis_propagate] done — {n_done} newly propagated, "
          f"{total_sec/60:.1f} min total work, ~{total_sec/max(n_done, 1):.0f}s/vid avg",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

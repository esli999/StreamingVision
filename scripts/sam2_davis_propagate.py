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
from genmatter.pseudo_gt.sam_video import segment_sequence  # noqa: E402
from genmatter.pseudo_gt.correspondence import label_map_from_masks  # noqa: E402


OUT_ROOT = _REPO / "assets" / "tapvid_davis_30_videos_processed" / "sam2_propagated"
WEIGHTS_DIR = _REPO / "assets" / "deeplearning_weights"


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
                  video_chunk_frames: int = 16, iou_threshold: float = 0.3) -> dict:
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
    args = p.parse_args(argv)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    expected = list(gm_config.TAPVID_DAVIS_VIDEO_NAMES)
    if args.videos:
        wanted = set(v.strip() for v in args.videos.split(","))
        expected = [v for v in expected if v in wanted]

    print(f"[sam2_davis_propagate] running on {len(expected)} video(s); out={OUT_ROOT}; "
          f"min_threshold={args.min_threshold} max_objects={args.max_objects} "
          f"video_imgsz={args.video_imgsz} backup={args.backup}", flush=True)
    total_sec = 0.0
    n_done = 0
    for i, vid in enumerate(expected, 1):
        print(f"[sam2_davis_propagate] {i}/{len(expected)}: {vid}", flush=True)
        info = propagate_one(vid, force=args.force, backup=args.backup, model=args.model,
                             min_threshold=args.min_threshold, max_objects=args.max_objects,
                             video_imgsz=args.video_imgsz,
                             video_chunk_frames=args.video_chunk_frames,
                             iou_threshold=args.iou_threshold)
        if info["status"] == "ok":
            print(f"  ok: {info['frames']} frames ({info.get('n_objects', '?')} objs) in "
                  f"{info['wall_sec']:.1f}s ({info['sec_per_frame']:.2f}s/frame) @ {info['native_hw']}"
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

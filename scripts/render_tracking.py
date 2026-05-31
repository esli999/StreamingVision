#!/usr/bin/env python3
"""Render GenMatter++ tracking overlays for the local calibration videos.

For each local video, run the calibrated config (configs/streaming_general.yaml)
through the cache-based Gibbs tracker WITH SAM2 frame-0 init — the same strong
inference the calibration used — then composite, side by side,

    [ source RGB | blob-id overlay | hyperblob (semantic) overlay ]

into runs/calibrate_consistency/tracking_videos/<vid>.mp4.

Pure cv2 + the cached perception .npz: no depth/flow/DINO models load, so this is
fast and bit-exact to the calibration tracker. Headless-safe (mp4v writer).

Frame alignment: the cache skips video frame 0 (no flow), so tracked frame ``t``
corresponds to source video frame ``frame_idx[t]`` (stored in the .npz). We pair
each tracked frame with that exact source frame, so the overlay and the RGB are
in lock-step.

Usage:
    python scripts/render_tracking.py                       # all 8 local videos
    python scripts/render_tracking.py --videos test wine_swirl
    python scripts/render_tracking.py --config configs/streaming_general.yaml \
        --num-sweeps 25
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import calibrate_consistency as cc      # noqa: E402  (sets env; does NOT load torch)
import genmatter_rt                     # noqa: E402

WORK_H, WORK_W = 360, 640
FPS = 30.0
_TILE_LABELS = ("source", "blobs (fine)", "hyperblobs (semantic)")


def _log(msg: str) -> None:
    print(f"[render_tracking {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _transcode_to_h264(in_path: str, out_path: str) -> bool:
    """Re-encode the mp4v temp to H.264 yuv420p for universal playback (VS Code
    / browsers can't decode cv2's MPEG-4 Simple Profile). Returns False if
    ffmpeg is unavailable (caller keeps the mp4v file)."""
    if not shutil.which("ffmpeg"):
        _log("ffmpeg not found — keeping mp4v (not VS Code-compliant)")
        return False
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-movflags", "+faststart", out_path], check=True)
    return True


def _read_source_frames(source: Path, need: set) -> Dict[int, np.ndarray]:
    """Read the source frames (resized to WORK_HW, BGR) for the indices in
    ``need``, matching run_streaming_live.iter_frames' order + INTER_AREA resize.
    Handles both mp4/mov files and directories of image frames."""
    frames: Dict[int, np.ndarray] = {}
    if not need:
        return frames
    max_need = max(need)
    if source.is_dir():
        files = sorted([p for p in source.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        for i, p in enumerate(files):
            if i > max_need:
                break
            if i in need:
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if bgr is not None:
                    frames[i] = cv2.resize(bgr, (WORK_W, WORK_H),
                                           interpolation=cv2.INTER_AREA)
        return frames
    cap = cv2.VideoCapture(str(source))
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in need:
            frames[i] = cv2.resize(bgr, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        if i >= max_need:
            break
        i += 1
    cap.release()
    return frames


def _annotate(combo: np.ndarray) -> np.ndarray:
    """Put a small label at the top-left of each of the three tiles."""
    for k, label in enumerate(_TILE_LABELS):
        x = k * WORK_W + 8
        cv2.putText(combo, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combo, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return combo


def render_video(vid: str, entry, labels: dict, *, yaml_cfg: dict,
                 num_blobs: int, num_hyperblobs: int, num_sweeps: int,
                 use_sam_frame0: bool, out_path: Path,
                 max_frames: int = -1) -> dict:
    pos = labels["positions"]
    vel = labels["velocities"]
    feat = labels["features"]
    indices = labels["indices"]
    frame_idx = np.asarray(labels["frame_idx"]).reshape(-1)
    if max_frames is not None and max_frames > 0:
        pos, vel, feat = pos[:max_frames], vel[:max_frames], feat[:max_frames]
        frame_idx = frame_idx[:max_frames]

    sam_grid = cc._frame0_sam_grid(entry, labels) if use_sam_frame0 else None
    t0 = time.monotonic()
    tr = genmatter_rt.run_tracker_from_cache(
        pos, vel, feat, indices, yaml_cfg=yaml_cfg,
        num_blobs=num_blobs, num_hyperblobs=num_hyperblobs,
        num_sweeps=num_sweeps, capture_blob_weights=False,
        sam_segmentation=sam_grid)
    if "error" in tr:
        return {"vid": vid, "status": "error", "error": tr["error"]}
    blob_a = tr["blob_a"]            # (T, N) int32, -1 = outlier
    hb_a = tr["hyperblob_a"]         # (T, N) int32, -1 = outlier
    T = int(blob_a.shape[0])

    need = {int(frame_idx[t]) for t in range(min(T, frame_idx.shape[0]))}
    src_frames = _read_source_frames(entry.source, need)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # cv2's mp4v writer emits MPEG-4 Simple Profile, which VS Code / browsers
    # can't decode. Write to a temp mp4v file, then transcode to H.264 yuv420p.
    tmp_path = str(out_path) + ".mp4v.mp4"
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (WORK_W * 3, WORK_H))
    if not writer.isOpened():
        return {"vid": vid, "status": "error", "error": "VideoWriter failed to open"}

    n_written = 0
    blob_color_lut = None   # per-particle colours LOCKED from the first frame
    blank = np.zeros((WORK_H, WORK_W, 3), dtype=np.uint8)
    for t in range(T):
        fi = int(frame_idx[t]) if t < frame_idx.shape[0] else -1
        src = src_frames.get(fi, blank)
        # PARTICLE (blobs) tile = each particle's average RGB, LOCKED from the
        # first frame (compute the per-blob colour LUT once, reuse every frame so
        # colours stay fixed as particles move). rgb_guide also enables
        # edge-aware upsampling for both tiles.
        if blob_color_lut is None:
            blob_color_lut = genmatter_rt.compute_blob_color_lut(
                blob_a[t], indices, src, h=WORK_H, w=WORK_W, stride=genmatter_rt.STRIDE)
        blob_bgr, hyper_bgr = genmatter_rt.render_matter_tile(
            blob_a[t], hb_a[t], indices, h=WORK_H, w=WORK_W,
            stride=genmatter_rt.STRIDE, rgb_guide=src, blob_color_lut=blob_color_lut)
        combo = _annotate(np.hstack([src, blob_bgr, hyper_bgr]))
        writer.write(combo)
        n_written += 1
    writer.release()
    if _transcode_to_h264(tmp_path, str(out_path)):
        Path(tmp_path).unlink(missing_ok=True)
    else:
        Path(tmp_path).rename(out_path)   # fallback: keep mp4v if no ffmpeg
    return {"vid": vid, "status": "ok", "frames": n_written,
            "n_blobs": tr.get("n_blobs"), "path": str(out_path),
            "wall_s": round(time.monotonic() - t0, 1)}


def render_all_local(out_root: Path, *, config_path: Path,
                     max_frames: int = -1,
                     num_sweeps: int = cc.DEFAULT_NUM_GIBBS_SWEEPS,
                     use_sam_frame0: bool = cc.INFER_USE_SAM_FRAME0,
                     num_blobs: int = cc.INFER_NUM_BLOBS,
                     num_hyperblobs: int = cc.INFER_NUM_HYPERBLOBS,
                     init_gibbs_sweeps: int = cc.INFER_INIT_GIBBS_SWEEPS,
                     videos: Optional[List[str]] = None) -> dict:
    """Render tracking mp4s for the local calibration videos. Returns a summary
    dict. Uses the calibrated config at ``config_path`` (falls back to
    streaming_default if missing), overlaying the inference-strategy knobs so the
    render reflects exactly the requested strong-inference setup."""
    cc._ensure_jax_setup()
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        _log(f"config {cfg_path} missing — falling back to {cc.DEFAULT_YAML}")
        cfg_path = cc.DEFAULT_YAML
    yaml_cfg = genmatter_rt.load_yaml_hypers(cfg_path)
    # Force the inference knobs (the calibrated YAML already carries them, but be
    # explicit so a standalone render with overrides behaves predictably).
    yaml_cfg["tracking"]["use_sam_frame0"] = bool(use_sam_frame0)
    yaml_cfg["tracking"]["num_blobs"] = int(num_blobs)
    yaml_cfg["tracking"]["num_hyperblobs"] = int(num_hyperblobs)
    yaml_cfg["tracking"]["init_gibbs_sweeps"] = int(init_gibbs_sweeps)
    # Structural "fixed cluster view" flags so a standalone render matches the
    # demo. setdefault so an explicit YAML value wins (these default False in code).
    _trk = yaml_cfg.setdefault("tracking", {})
    _trk.setdefault("feature_aware_final_assignment", True)
    _trk.setdefault("final_assignment_outlier", False)
    _trk.setdefault("freeze_hyperblob_assignment", True)
    _trk.setdefault("pure_object_seed", True)

    all_videos = cc.discover_videos()
    want = list(videos) if videos else list(cc.LOCAL_VIDEOS)
    out_dir = Path(out_root) / "tracking_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"rendering {len(want)} videos → {out_dir} "
         f"(use_sam_frame0={use_sam_frame0} num_blobs={num_blobs} sweeps={num_sweeps})")

    results: List[dict] = []
    for j, vid in enumerate(want, 1):
        entry = all_videos.get(vid)
        if entry is None:
            _log(f"[{j}/{len(want)}] {vid}: not discovered — skipping")
            results.append({"vid": vid, "status": "missing"})
            continue
        labels_path = cc.LABELS_DIR / f"{vid}.npz"
        if not labels_path.is_file():
            _log(f"[{j}/{len(want)}] {vid}: no cached labels — run --phase pseudo_labels")
            results.append({"vid": vid, "status": "no_labels"})
            continue
        labels = cc._load_labels(vid)
        out_path = out_dir / f"{vid}.mp4"
        try:
            r = render_video(vid, entry, labels, yaml_cfg=yaml_cfg,
                             num_blobs=num_blobs, num_hyperblobs=num_hyperblobs,
                             num_sweeps=num_sweeps, use_sam_frame0=use_sam_frame0,
                             out_path=out_path, max_frames=max_frames)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"vid": vid, "status": "error", "error": repr(e)}
        results.append(r)
        _log(f"[{j}/{len(want)}] {vid}: {r.get('status')} "
             f"frames={r.get('frames')} n_blobs={r.get('n_blobs')} "
             f"wall={r.get('wall_s')}s → {r.get('path', '-')}")

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    summary = {"phase": "render_local", "n_ok": n_ok, "n_total": len(want),
               "out_dir": str(out_dir), "results": results}
    cc._save_json(Path(out_root) / "render_local.json", summary)
    _log(f"render_local done: {n_ok}/{len(want)} videos written to {out_dir}")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=str, default=str(cc.OUT_ROOT),
                   help="Run root (mp4s go to <out>/tracking_videos/).")
    p.add_argument("--config", type=str, default=str(cc.GENERAL_YAML),
                   help="Calibrated YAML to render with.")
    p.add_argument("--videos", nargs="+", default=None,
                   help="Subset of local video names (default: all 8).")
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument("--num-sweeps", type=int, default=cc.DEFAULT_NUM_GIBBS_SWEEPS)
    p.add_argument("--num-blobs", type=int, default=cc.INFER_NUM_BLOBS)
    p.add_argument("--num-hyperblobs", type=int, default=cc.INFER_NUM_HYPERBLOBS)
    p.add_argument("--init-gibbs-sweeps", type=int, default=cc.INFER_INIT_GIBBS_SWEEPS)
    p.add_argument("--no-sam-frame0", dest="use_sam_frame0", action="store_false",
                   default=cc.INFER_USE_SAM_FRAME0)
    args = p.parse_args(argv)
    render_all_local(
        Path(args.out).resolve(), config_path=Path(args.config),
        max_frames=args.max_frames, num_sweeps=args.num_sweeps,
        use_sam_frame0=args.use_sam_frame0, num_blobs=args.num_blobs,
        num_hyperblobs=args.num_hyperblobs, init_gibbs_sweeps=args.init_gibbs_sweeps,
        videos=args.videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

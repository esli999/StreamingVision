#!/usr/bin/env python3
"""
DAVIS TAP-Vid: build ``{video}_3d_motion.npz`` from RGB frame folders using Video-Depth-Anything
+ RAFT. First import clones VDA and fetches checkpoints (see README).

Usage::

    uv run python run_experiments.py davis-extract-3d-motion

Forwarded args (after ``--``) match ``process_video_to_3d_data``. Step progress is printed by default;
use ``--verbose`` / ``-v`` for extra detail (depth/flow logs) and the RAFT tqdm bar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: E402

from preprocessing.motion_extraction_3d import vda_preprocess_video as gp_vda  # noqa: E402

VDA_AVAILABLE = gp_vda.VDA_AVAILABLE
process_video_to_3d_data = gp_vda.process_video_to_3d_data


def main() -> None:
    p = argparse.ArgumentParser(description="DAVIS 3D motion NPZ (VDA + RAFT).")
    p.add_argument(
        "--rgb-base",
        type=Path,
        default=config.DAVIS_RGB_PATH,
        help="Parent of per-video frame folders (default: tapvid_davis_rgb_frames)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=config.DAVIS_3D_MOTION_PATH,
        help="Directory for {video}_3d_motion.npz (default: tapvid_davis_npzs)",
    )
    p.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Subset of video names (default: TAPVID_DAVIS_VIDEO_NAMES with existing folders)",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip if output npz already exists")
    p.add_argument("--encoder", choices=["vits", "vitb", "vitl"], default="vitl")
    p.add_argument("--input-size", type=int, default=518)
    p.add_argument("--max-res", type=int, default=1280)
    p.add_argument("--max-len", type=int, default=-1)
    p.add_argument("--target-fps", type=int, default=-1)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--skip-frames", type=int, default=1)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Extra detail: depth/flow logs and per-frame RAFT tqdm bar (default already prints [1/n]… steps)",
    )
    args = p.parse_args()

    if not (0.0 < args.subsample <= 1.0):
        raise SystemExit("--subsample must be in (0, 1]")

    if not VDA_AVAILABLE:
        print(
            "Video-Depth-Anything failed to import after setup. Check external/video-depth-anything\n"
            "  or GENMATTER_VIDEO_DEPTH_ANYTHING_PATH, or remove a broken clone and rerun.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    rgb_base = args.rgb_base.resolve()
    out_dir = args.output_dir.resolve()
    if not rgb_base.is_dir():
        raise SystemExit(f"RGB base not found: {rgb_base}")

    names = list(args.videos) if args.videos else [
        v for v in config.TAPVID_DAVIS_VIDEO_NAMES if (rgb_base / v).is_dir()
    ]
    if not names:
        raise SystemExit(f"No video folders under {rgb_base}")

    out_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    for name in tqdm(names, desc="davis-3d-motion", mininterval=1.0):
        outp = out_dir / f"{name}_3d_motion.npz"
        if args.skip_existing and outp.is_file():
            continue
        inp = rgb_base / name
        if not inp.is_dir():
            failed.append(f"{name} (missing folder)")
            continue
        try:
            process_video_to_3d_data(
                input_path=str(inp),
                output_path=str(outp),
                encoder=args.encoder,
                input_size=args.input_size,
                max_res=args.max_res,
                max_len=args.max_len,
                target_fps=args.target_fps,
                fp32=args.fp32,
                device=args.device,
                skip_frames=args.skip_frames,
                subsample=args.subsample,
                output_format="npz",
                verbose=args.verbose,
                user_log=tqdm.write,
            )
        except Exception as e:
            failed.append(f"{name}: {e!s}")

    if failed:
        print("Failed:", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

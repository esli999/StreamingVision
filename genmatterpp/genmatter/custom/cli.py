#!/usr/bin/env python3
"""GenMatter custom video CLI: ``genmatter preprocess``, ``genmatter track`` (later)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from genmatter.custom.config_schema import load_config
from genmatter.custom.console import PreprocessConsole
from genmatter.custom.preprocess import run_preprocess
from genmatter.custom.pseudo_gt_cmd import run_pseudo_gt
from genmatter.custom.track import run_track
from genmatter.custom.viz import run_viz

DEFAULT_CONFIG = _REPO_ROOT / "configs" / "custom_default.yaml"
DEFAULT_BAYESOPT_CONFIG = _REPO_ROOT / "configs" / "custom_bayesopt.yaml"


def _cmd_preprocess(args: argparse.Namespace) -> int:
    mp4 = Path(args.mp4).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    cfg = load_config(config_path, dot_overrides=args.set or [])
    video_id = args.video_id or mp4.stem

    ui = PreprocessConsole()
    try:
        result = run_preprocess(mp4, cfg, video_id, ui)
    except Exception as e:
        ui.console.print_exception()
        return 1

    return 0 if result.success else 1


def _cmd_track(args: argparse.Namespace) -> int:
    if not args.video_id:
        print("genmatter track requires --video-id", file=sys.stderr)
        return 2

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    cfg = load_config(config_path, dot_overrides=args.set or [])
    ui = PreprocessConsole()
    try:
        result = run_track(args.video_id, cfg, ui)
    except Exception:
        ui.console.print_exception()
        return 1

    return 0 if result.success else 1


def _cmd_viz(args: argparse.Namespace) -> int:
    if not args.video_id:
        print("genmatter viz requires --video-id", file=sys.stderr)
        return 2

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    cfg = load_config(config_path, dot_overrides=args.set or [])
    output = Path(args.output).expanduser().resolve() if args.output else None

    ui = PreprocessConsole()
    try:
        result = run_viz(args.video_id, cfg, ui, output_path=output)
    except Exception:
        ui.console.print_exception()
        return 1

    return 0 if result.success else 1


def _cmd_pseudo_gt(args: argparse.Namespace) -> int:
    if not args.video_id:
        print("genmatter pseudo-gt requires --video-id", file=sys.stderr)
        return 2
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    ui = PreprocessConsole()
    try:
        result = run_pseudo_gt(
            args.video_id,
            cfg,
            ui,
            force=args.force,
            colorize_only=args.colorize_only,
        )
    except Exception:
        ui.console.print_exception()
        return 1
    return 0 if result.success else 1


def _cmd_bayesopt(args: argparse.Namespace) -> int:
    if not args.video_id:
        print("genmatter bayesopt requires --video-id", file=sys.stderr)
        return 2
    try:
        from genmatter.custom.bayesopt_cmd import run_bayesopt
    except ImportError as e:
        print(
            "BayesOpt dependencies missing. Install with: uv sync --extra bayesopt",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1
    run_bayesopt(
        args.video_id,
        tracking_config=Path(args.config).expanduser().resolve(),
        bayesopt_config=Path(args.bayesopt_config).expanduser().resolve(),
        run_id=args.run_id,
        tracking_set=args.tracking_set or [],
        bayesopt_set=args.bo_set or [],
        resume=not args.no_resume,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="genmatter",
        description="GenMatter — probabilistic 3D tracking on custom MP4 videos.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser(
        "preprocess",
        help="Preprocess one MP4 (frames, 3D motion, DINO, SAM frame 0).",
    )
    p_pre.add_argument("mp4", type=str, help="Path to input .mp4 file")
    p_pre.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    p_pre.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Output folder name (default: MP4 stem)",
    )
    p_pre.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config (dot path), e.g. preprocess.motion_3d.device=cpu",
    )
    p_pre.set_defaults(func=_cmd_preprocess)

    p_track = sub.add_parser(
        "track",
        help="Run GenMatter DINO tracking on a preprocessed custom video.",
    )
    p_track.add_argument(
        "--video-id",
        type=str,
        required=True,
        help="Video ID folder under assets/custom_videos/",
    )
    p_track.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    p_track.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config (dot path), e.g. tracking.pipeline.skip_existing=true",
    )
    p_track.set_defaults(func=_cmd_track)

    p_viz = sub.add_parser(
        "viz",
        help="Export tracking results to a Rerun .rrd recording.",
    )
    p_viz.add_argument(
        "--video-id",
        type=str,
        required=True,
        help="Video ID folder under assets/custom_videos/",
    )
    p_viz.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .rrd path (default: tracking/<video_id>.rrd)",
    )
    p_viz.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    p_viz.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config, e.g. viz.pipeline.skip_existing=true",
    )
    p_viz.set_defaults(func=_cmd_viz)

    p_pg = sub.add_parser(
        "pseudo-gt",
        help="Build SAM2 pseudo-GT segmentations + correspondence for evaluation.",
    )
    p_pg.add_argument("--video-id", type=str, required=True)
    p_pg.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p_pg.add_argument("--force", action="store_true")
    p_pg.add_argument(
        "--colorize-only",
        action="store_true",
        help="Only write RGB previews from existing uint16 segmasks (no SAM).",
    )
    p_pg.set_defaults(func=_cmd_pseudo_gt)

    p_bo = sub.add_parser(
        "bayesopt",
        help="Bayesian optimization of tracking hyperparameters (64 Sobol + LCB).",
    )
    p_bo.add_argument("--video-id", type=str, required=True)
    p_bo.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p_bo.add_argument(
        "--bayesopt-config",
        type=str,
        default=str(DEFAULT_BAYESOPT_CONFIG),
    )
    p_bo.add_argument("--run-id", type=str, default=None)
    p_bo.add_argument("--no-resume", action="store_true")
    p_bo.add_argument("--set", action="append", default=[], dest="tracking_set")
    p_bo.add_argument("--bo-set", action="append", default=[], dest="bo_set")
    p_bo.set_defaults(func=_cmd_bayesopt)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

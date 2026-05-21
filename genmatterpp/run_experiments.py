#!/usr/bin/env python3
"""CLI dispatcher for GenMatterPlusPlus DAVIS experiments and postprocessing."""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

# Value: script path relative to repo root, or (path, default argv prefix)
COMMANDS = {
    "davis-extract-dino": "experiments/davis/dino_extractor.py",
    "davis-extract-sam-frame0": "experiments/davis/sam_frame0_extractor.py",
    "davis-extract-3d-motion": "preprocessing/motion_extraction_3d/run_davis_3d_motion.py",
    "download-tapvid-davis": "scripts/download_tapvid_davis.py",
    "davis-preprocess": "scripts/davis_preprocess.py",
    # DAVIS: 3 experiment types × SAM vs GT-init (six runners). Legacy names = SAM (same as ``*-sam``).
    "davis-tracking-sam": (
        "experiments/davis/run_davis_tracking.py",
        [],
    ),
    "davis-tracking-gt-init": (
        "experiments/davis/run_davis_tracking.py",
        ["--gt-init"],
    ),
    "davis-tracking": (
        "experiments/davis/run_davis_tracking.py",
        [],
    ),
    "davis-subsampling-sam": (
        "experiments/davis/run_davis_subsampling.py",
        [],
    ),
    "davis-subsampling-gt-init": (
        "experiments/davis/run_davis_subsampling.py",
        ["--gt-init"],
    ),
    "davis-subsampling": (
        "experiments/davis/run_davis_subsampling.py",
        [],
    ),
    "davis-ablation-sam": (
        "experiments/davis/run_davis_ablation.py",
        [],
    ),
    "davis-ablation-gt-init": (
        "experiments/davis/run_davis_ablation.py",
        ["--gt-init"],
    ),
    "davis-ablation": (
        "experiments/davis/run_davis_ablation.py",
        [],
    ),
    "cotracker": "experiments/baselines/run_cotracker.py",
    "postprocess-davis": "postprocessing/postprocess_davis.py",
}


def _command_script_and_defaults(spec):
    if isinstance(spec, tuple):
        return spec[0], list(spec[1])
    return spec, []


def main():
    parser = argparse.ArgumentParser(
        description="Run GenMatterPlusPlus DAVIS experiments or postprocessing steps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available commands:\n" + "\n".join(f"  {k}" for k in COMMANDS),
    )
    parser.add_argument(
        "command",
        choices=COMMANDS.keys(),
        help="Experiment, data setup, or postprocessing step to run",
    )
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments forwarded to the script")
    args = parser.parse_args()

    rel_path, default_args = _command_script_and_defaults(COMMANDS[args.command])
    script = REPO_ROOT / rel_path
    if not script.exists():
        print(f"Error: script {script} not found", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(script)] + default_args + args.extra
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

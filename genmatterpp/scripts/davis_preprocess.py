#!/usr/bin/env python3
"""Run DAVIS one-time preprocessing in order: 3D motion → DINO → SAM frame 0.

For step-specific flags, call ``davis-extract-3d-motion``, ``davis-extract-dino``, or
``davis-extract-sam-frame0`` separately.

Invoked via ``uv run python run_experiments.py davis-preprocess``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_STEPS: list[tuple[str, str]] = [
    ("3d motion (VDA + RAFT)", "preprocessing/motion_extraction_3d/run_davis_3d_motion.py"),
    ("DINO features", "experiments/davis/dino_extractor.py"),
    ("SAM frame 0", "experiments/davis/sam_frame0_extractor.py"),
]


def main() -> None:
    exe = sys.executable
    for label, rel in _STEPS:
        script = REPO_ROOT / rel
        if not script.is_file():
            print(f"Error: missing {script}", file=sys.stderr)
            sys.exit(1)
        print(f"\n--- davis-preprocess: {label} ---\n", flush=True)
        r = subprocess.run([exe, str(script)], cwd=str(REPO_ROOT))
        if r.returncode != 0:
            sys.exit(r.returncode)
    print("\n--- davis-preprocess: done ---\n", flush=True)


if __name__ == "__main__":
    main()

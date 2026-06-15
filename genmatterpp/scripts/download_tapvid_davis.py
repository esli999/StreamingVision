#!/usr/bin/env python3
"""Run ``download_tapvid_davis.sh``: DAVIS 2017 full-res zip → TAP-Vid RGB + seg (see ``davis_tapvid_fetch.py``).

Invoked via ``uv run python run_experiments.py download-tapvid-davis``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SH = REPO_ROOT / "scripts" / "download_tapvid_davis.sh"


def main() -> None:
    if not _SH.is_file():
        print(f"Error: missing {_SH}", file=sys.stderr)
        sys.exit(1)
    raise SystemExit(
        subprocess.call(["bash", str(_SH), *sys.argv[1:]], cwd=str(REPO_ROOT))
    )


if __name__ == "__main__":
    main()

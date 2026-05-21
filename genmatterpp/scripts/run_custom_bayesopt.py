#!/usr/bin/env python3
"""Run forever BayesOpt on a custom preprocessed video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genmatter.custom.bayesopt_cmd import run_bayesopt

DEFAULT_TRACKING_CONFIG = REPO_ROOT / "configs" / "custom_default.yaml"
DEFAULT_BAYESOPT_CONFIG = REPO_ROOT / "configs" / "custom_bayesopt.yaml"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-id", required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_TRACKING_CONFIG)
    p.add_argument("--bayesopt-config", type=Path, default=DEFAULT_BAYESOPT_CONFIG)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--set", action="append", default=[], dest="tracking_set")
    p.add_argument("--bo-set", action="append", default=[], dest="bo_set")
    args = p.parse_args()
    run_bayesopt(
        args.video_id,
        tracking_config=args.config,
        bayesopt_config=args.bayesopt_config,
        run_id=args.run_id,
        tracking_set=args.tracking_set,
        bayesopt_set=args.bo_set,
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

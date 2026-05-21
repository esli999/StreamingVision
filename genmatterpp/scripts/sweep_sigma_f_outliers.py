#!/usr/bin/env python3
"""Sweep sigma_F (orders of magnitude) and report mean dense outlier fraction.

Runs full DINO tracking per value in one process so JAX JIT stays warm after the
first compile. Does not write tracking_dense.npz (metrics only).

Usage:
  uv run python scripts/sweep_sigma_f_outliers.py
  uv run python scripts/sweep_sigma_f_outliers.py --video-id original_244622072067
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genmatter.custom.config_schema import load_config
from genmatter.custom.paths import resolve_video_paths
from genmatter.tracking.dino import (
    DinoTrackingInputs,
    configure_jax_cache,
    dino_params_from_config,
    run_dino_tracking,
)
from genmatter.tracking.outlier_stats import mean_outlier_fraction_percent

DEFAULT_VIDEO_ID = "original_244622072067"
# Orders of magnitude around default sigma_F=0.2 (2e-1)
DEFAULT_SIGMA_F_VALUES = (0.002, 0.02, 0.2, 2.0, 20.0, 200.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "custom_default.yaml",
    )
    parser.add_argument(
        "--sigma-f",
        type=float,
        nargs="+",
        default=list(DEFAULT_SIGMA_F_VALUES),
        help="sigma_F values to try (default: six steps in log10 around 0.2)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write results JSON",
    )
    args = parser.parse_args()

    configure_jax_cache(str(REPO_ROOT / ".jax_cache"))
    cfg = load_config(args.config)
    paths = resolve_video_paths(cfg, args.video_id)
    inputs = DinoTrackingInputs(
        video_id=args.video_id,
        motion_npz=paths.npz_path,
        dino_npz=paths.dino_path,
        sam_frame0_png=paths.sam_path if cfg.tracking.use_sam_frame0 else None,
    )

    base_params = dino_params_from_config(cfg.tracking)
    results: list[dict] = []

    print(f"Video: {args.video_id}")
    print(f"sigma_F sweep: {args.sigma_f}")
    print(f"JAX cache: {REPO_ROOT / '.jax_cache'}")
    print()

    for i, sigma_f in enumerate(args.sigma_f):
        hp = replace(base_params.hyperparams, sigma_F=float(sigma_f))
        params = replace(
            base_params,
            hyperparams=hp,
            measure_fps=(i == 0),
        )
        t0 = time.perf_counter()
        print(f"[{i + 1}/{len(args.sigma_f)}] sigma_F={sigma_f:g} — tracking...", flush=True)
        result = run_dino_tracking(inputs, params)
        elapsed = time.perf_counter() - t0
        mean_pct, min_pct, max_pct = mean_outlier_fraction_percent(result.tracking_data)
        row = {
            "sigma_F": sigma_f,
            "mean_outlier_pct": mean_pct,
            "min_outlier_pct_per_frame": min_pct,
            "max_outlier_pct_per_frame": max_pct,
            "num_frames": len(result.tracking_data),
            "n_blobs_last_frame": int(result.tracking_data[-1]["n_blobs"]),
            "elapsed_seconds": elapsed,
            "jit_warmup": i == 0,
        }
        results.append(row)
        print(
            f"    mean outlier % = {mean_pct:.2f}  "
            f"(per-frame min={min_pct:.2f}, max={max_pct:.2f})  "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    print()
    print("Summary (sigma_F → mean outlier %)")
    print("-" * 40)
    for row in results:
        print(f"  {row['sigma_F']:>10g}  →  {row['mean_outlier_pct']:7.2f}%")

    out_path = args.output_json or (
        paths.video_root
        / "tracking"
        / f"sigma_f_sweep_{args.video_id}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"video_id": args.video_id, "results": results}, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

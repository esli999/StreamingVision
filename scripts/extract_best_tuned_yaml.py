#!/usr/bin/env python3
"""Pick the best valid trial from a bayesopt run and emit a streaming_tuned.yaml.

The vendored bayesopt produces:
  assets/custom_videos/<id>/bayesopt/<run-id>/trials_summary.csv
  assets/custom_videos/<id>/bayesopt/<run-id>/trials/trial_NNNNN_params.yaml

This script reads streaming_default.yaml as the base, overlays the winning
trial's ``params`` block onto ``tracking.hyperparams``, also threads
``tracking.init_gibbs_sweeps`` and ``tracking.tracking_outlier_prob`` if
they're in the trial params, and writes streaming_tuned.yaml.

Usage:
  python scripts/extract_best_tuned_yaml.py [--run-id RUN_ID] [--video-id test]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def find_latest_run(video_id: str) -> Path:
    base = REPO / "assets" / "custom_videos" / video_id / "bayesopt"
    if not base.is_dir():
        raise FileNotFoundError(f"No bayesopt runs under {base}")
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run directories under {base}")
    return runs[-1]


def pick_best_trial(run_dir: Path) -> int:
    """Return the trial_index of the highest-objective valid trial."""
    csv_path = run_dir / "trials_summary.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"trials_summary.csv missing in {run_dir}")
    best_idx = None
    best_obj = -1.0
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        idx_status = header.index("status")
        idx_obj = header.index("objective")
        idx_trial = header.index("trial_index")
        for line in f:
            row = line.strip().split(",")
            if row[idx_status] != "completed":
                continue
            obj = float(row[idx_obj])
            if obj > best_obj:
                best_obj = obj
                best_idx = int(row[idx_trial])
    if best_idx is None:
        raise RuntimeError(f"No valid (status=completed) trials in {csv_path}")
    print(f"Best valid trial: #{best_idx} with objective={best_obj:.4f}", file=sys.stderr)
    return best_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-id", default="test")
    p.add_argument("--run-id", default=None,
                   help="Specific bayesopt run dir name (default: latest)")
    p.add_argument("--base-config", default=str(REPO / "configs" / "streaming_default.yaml"))
    p.add_argument("--out", default=str(REPO / "configs" / "streaming_tuned.yaml"))
    args = p.parse_args()

    if args.run_id:
        run_dir = REPO / "assets" / "custom_videos" / args.video_id / "bayesopt" / args.run_id
    else:
        run_dir = find_latest_run(args.video_id)
    print(f"Bayesopt run: {run_dir}", file=sys.stderr)

    best_idx = pick_best_trial(run_dir)
    params_yaml = run_dir / "trials" / f"trial_{best_idx:05d}_params.yaml"
    with open(params_yaml, encoding="utf-8") as f:
        trial = yaml.safe_load(f)
    params = trial["params"]

    with open(args.base_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 14 hyperparameters live under tracking.hyperparams.
    hp_keys = (
        "sigma_F", "outlier_prob", "outlier_velocity_gamma_shape",
        "outlier_velocity_gamma_rate", "alpha", "beta", "sigma_H", "sigma_V",
        "translation_gaussian_scale", "translation_max_radius",
        "translation_num_radii_cells", "translation_theta_step_deg",
        "rotation_vmf_kappa", "rotation_angle_max_deg", "rotation_angle_step_deg",
    )
    for k in hp_keys:
        if k in params:
            cfg["tracking"]["hyperparams"][k] = params[k]
    # 3 more live one level up.
    for k in ("tracking_outlier_prob", "init_gibbs_sweeps"):
        if k in params:
            cfg["tracking"][k] = params[k]

    # Add a provenance note.
    cfg.setdefault("_meta", {})
    cfg["_meta"]["source"] = f"bayesopt {run_dir.name} trial {best_idx}"
    cfg["_meta"]["objective_avg_mean_gt_iou"] = trial.get("objective_score")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pick the best valid trial from a bayesopt run and emit a streaming_tuned.yaml.

The vendored bayesopt produces:
  assets/custom_videos/<id>/bayesopt/<run-id>/trials.jsonl
  assets/custom_videos/<id>/bayesopt/<run-id>/trials/trial_NNNNN.yaml

This script reads streaming_default.yaml as the base, overlays the winning
trial's ``params`` block onto ``tracking.hyperparams``, also threads
``tracking.init_gibbs_sweeps`` and ``tracking.tracking_outlier_prob`` if
they're in the trial params, and writes streaming_tuned.yaml.

Multi-objective runs leave the trial scoring ambiguous: the user picks
how to collapse the per-video Pareto front.  ``--pick-by min`` (default)
finds the trial whose worst-performing video did best — matches the v2
accuracy gate's per-video floor.  ``--pick-by softmin`` / ``mean`` fall
back to the scalar aggregations recorded in trial meta.

Usage:
  python scripts/extract_best_tuned_yaml.py [--run-id RUN_ID] [--video-id test] [--pick-by min|softmin|mean]
"""

from __future__ import annotations

import argparse
import json
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


def _score_row(row: dict, mode: str) -> float:
    """Score a trial under the chosen Pareto-collapse rule."""
    per_video = row.get("per_video") or []
    if mode == "min":
        if not per_video:
            return float(row.get("objective", 0.0))
        return min(float(pv.get("avg_persistent_iou", 0.0)) for pv in per_video)
    if mode == "softmin":
        return float(row.get("softmin_primary", row.get("objective", 0.0)))
    if mode == "mean":
        if "mean_primary" in row:
            return float(row["mean_primary"])
        if per_video:
            vs = [float(pv.get("avg_persistent_iou", 0.0)) for pv in per_video]
            return sum(vs) / len(vs)
        return float(row.get("objective", 0.0))
    raise ValueError(f"unknown --pick-by mode: {mode}")


def pick_best_trial(run_dir: Path, mode: str = "min") -> tuple[int, dict]:
    """Return ``(trial_index, row)`` of the best valid trial under ``mode``.

    Reads trials.jsonl (not trials_summary.csv) because the summary
    doesn't carry the per-video block we need for min-per-video scoring.
    """
    jsonl_path = run_dir / "trials.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"trials.jsonl missing in {run_dir}")
    best_row: dict | None = None
    best_score = -1.0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") != "completed":
                continue
            s = _score_row(row, mode)
            if s > best_score:
                best_score = s
                best_row = row
    if best_row is None:
        raise RuntimeError(f"No completed trials in {jsonl_path}")
    best_idx = int(best_row["trial_index"])
    print(
        f"Best valid trial (pick-by {mode}): #{best_idx} with score={best_score:.4f}",
        file=sys.stderr,
    )
    return best_idx, best_row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-id", default="test")
    p.add_argument("--run-id", default=None,
                   help="Specific bayesopt run dir name (default: latest)")
    p.add_argument("--base-config", default=str(REPO / "configs" / "streaming_default.yaml"))
    p.add_argument("--out", default=str(REPO / "configs" / "streaming_tuned.yaml"))
    p.add_argument("--pick-by", choices=("min", "softmin", "mean"), default="min",
                   help="How to collapse the multi-objective Pareto front to "
                        "a single best trial.  'min' (default) picks the trial "
                        "whose worst-performing video had the highest IoU.")
    args = p.parse_args()

    if args.run_id:
        run_dir = REPO / "assets" / "custom_videos" / args.video_id / "bayesopt" / args.run_id
    else:
        run_dir = find_latest_run(args.video_id)
    print(f"Bayesopt run: {run_dir}", file=sys.stderr)

    best_idx, trial = pick_best_trial(run_dir, mode=args.pick_by)
    # `trial` already has params + per-video + meta from trials.jsonl.
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

    # Add a provenance note.  The bayesopt v3 multi-objective runner does
    # not aggregate; surface the per-video min/mean/softmin so the tuned
    # yaml is self-documenting about how the winner was selected.
    cfg.setdefault("_meta", {})
    cfg["_meta"]["source"] = f"bayesopt {run_dir.name} trial {best_idx}"
    cfg["_meta"]["pick_by"] = args.pick_by
    per_video = trial.get("per_video") or []
    if per_video:
        ious = [float(pv.get("avg_persistent_iou", 0.0)) for pv in per_video]
        cfg["_meta"]["per_video_min_persistent_iou"] = min(ious)
        cfg["_meta"]["per_video_max_persistent_iou"] = max(ious)
        cfg["_meta"]["mean_persistent_iou"] = sum(ious) / len(ious)
        cfg["_meta"]["per_video_persistent_iou"] = {
            pv["video_id"]: float(pv.get("avg_persistent_iou", 0.0))
            for pv in per_video
        }
    if "softmin_primary" in trial:
        cfg["_meta"]["softmin_persistent_iou"] = trial["softmin_primary"]
    if "softmin_tau" in trial:
        cfg["_meta"]["softmin_tau"] = trial["softmin_tau"]
    if not per_video and "softmin_primary" not in trial:
        # Legacy fallback: pre-v3 single-objective run.
        cfg["_meta"]["objective_persistent_iou"] = trial.get("objective")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Quantify the accuracy impact of dropping tracking_outlier_prob from 1e-28
to 1e-60 on test.mp4 via the offline SAM2 pseudo-GT scoring pipeline.

This is the same objective the bayesopt cohort used. Reads
configs/streaming_tuned.yaml, then runs the offline tracker twice (same
sigma_F, sigma_H, etc.), changing only ``tracking_outlier_prob``. Compares
``avg_persistent_iou``, ``avg_mean_gt_iou``, ``avg_pixel_jaccard``, and
``mean_outlier_pct``.

Run from repo root:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python scripts/eval_outlier_prob.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "genmatterpp"))

from genmatter.custom.config_schema import load_config
from genmatter.bayesopt.objective import build_objective_context, TrackingObjective


def main() -> int:
    cfg_path = REPO / "configs" / "streaming_tuned.yaml"
    cfg = load_config(cfg_path)
    video_id = "test"

    ctx = build_objective_context(cfg, video_id)
    obj = TrackingObjective(ctx, max_outlier_pct=100.0)

    print(f"Eval cfg: {cfg_path}")
    print(f"Video:    {video_id} ({ctx.img_dims[1]}x{ctx.img_dims[0]})")
    print()

    rows = []
    for top in (1.0e-28, 1.0e-60):
        # All other hyperparams come from the loaded cfg; we only override
        # tracking_outlier_prob via trial_params so the objective stays
        # apples-to-apples with the cohort scoring math.
        trial_params = {"tracking_outlier_prob": top}
        ax_obj, meta, valid = obj.evaluate(trial_params, measure_fps=False)
        rows.append((top, meta))
        print(f"--- tracking_outlier_prob = {top:.1e} ---")
        print(f"  avg_persistent_iou   = {meta['avg_persistent_iou']:.4f}")
        print(f"  avg_mean_gt_iou      = {meta['avg_mean_gt_iou']:.4f}")
        print(f"  avg_mean_matched_iou = {meta['avg_mean_matched_iou']:.4f}")
        print(f"  avg_pixel_jaccard    = {meta['avg_pixel_jaccard']:.4f}")
        print(f"  avg_gt_recall@0.5    = {meta['avg_gt_recall_at_thresh']:.4f}")
        print(f"  mean_outlier_pct     = {meta['mean_outlier_pct']:.4f}%")
        print(f"  num_frames           = {meta['num_frames']}")
        print(f"  elapsed_seconds      = {meta['elapsed_seconds']:.1f}")
        print()

    print("Summary (after - before):")
    a, b = rows[1][1], rows[0][1]
    for k in ("avg_persistent_iou", "avg_mean_gt_iou", "avg_mean_matched_iou",
             "avg_pixel_jaccard", "avg_gt_recall_at_thresh", "mean_outlier_pct"):
        delta = a[k] - b[k]
        print(f"  Δ {k:<26} = {delta:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

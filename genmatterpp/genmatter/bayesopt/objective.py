"""BO objective: jitted tracking + per-instance mean GT IoU (all pseudo-GT objects per frame)."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from genmatter.custom.config_schema import CustomConfig
from genmatter.custom.paths import VideoPaths
from genmatter.evaluation import evaluate_custom_instance_tracking
from genmatter.tracking.dino import DinoTrackingInputs, configure_jax_cache, run_dino_tracking
from genmatter.tracking.outlier_stats import mean_outlier_fraction_percent

from genmatter.bayesopt.search_space import build_tracking_params

DEFAULT_MAX_OUTLIER_PCT = 10.0
# Soft-penalty weight applied to outlier-cap overshoot when computing the
# GP-facing `ax_objective`.  Earlier code hard-clamped invalid trials to 0.0,
# which destroyed information: an invalid trial at 22% outliers / 0.57 IoU
# and a fully-broken trial at 0.05 IoU both landed at the same training-set
# value, making the GP fit a discontinuous surface across the constraint
# boundary.  Penalty = OUTLIER_PENALTY_WEIGHT * overshoot_frac keeps the
# raw IoU signal intact while still ordering invalid trials below valid ones.
# At the 20% cap, the max possible penalty is (1.0 - 0.20) * 2.0 = 1.6, which
# is comfortably larger than any realistic valid-IoU gap.
OUTLIER_PENALTY_WEIGHT = 2.0
# Optimization objective: persistent-identity IoU (Hungarian-match blob_id ↔
# gt_id ONCE on frame 0, then reuse that mapping for every subsequent
# frame's IoU computation).  This rewards the model's actual capability —
# blob IDs are latent identities meant to be preserved across frames; if
# they silently swap mid-video, the fixed mapping makes the post-swap IoU
# collapse, which is exactly the failure mode we want to penalize.
#
# The previous `avg_mean_gt_iou` (per-frame Hungarian, re-matching every
# frame) hid identity drift by re-assigning matches each frame.  The
# `avg_pixel_jaccard` (instance-agnostic binary foreground IoU) collapses
# to ~1.0 because SAM2 GT covers most pixels.
PRIMARY_METRIC_KEY = "avg_persistent_iou"


@dataclass
class ObjectiveContext:
    cfg: CustomConfig
    paths: VideoPaths
    inputs: DinoTrackingInputs
    img_dims: tuple[int, int]
    annotations_path: str
    video_id: str
    match_iou_threshold: float = 0.0
    score_iou_threshold: float = 0.5


def build_objective_context(
    cfg: CustomConfig,
    video_id: str,
    *,
    bo_cfg: dict[str, Any] | None = None,
) -> ObjectiveContext:
    from genmatter.custom.paths import resolve_video_paths

    paths = resolve_video_paths(cfg, video_id)
    if not paths.pseudo_gt_manifest.is_file():
        raise FileNotFoundError(
            f"Pseudo-GT manifest missing: {paths.pseudo_gt_manifest}\n"
            "Run: uv run genmatter pseudo-gt --video-id " + video_id
        )
    import json

    with open(paths.pseudo_gt_manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    meta_path = paths.pseudo_gt_dir / "meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    h, w = int(meta["img_dims"][0]), int(meta["img_dims"][1])

    obj = (bo_cfg or {}).get("objective", {})
    match_iou = float(obj.get("match_iou_threshold", 0.0))
    score_iou = float(obj.get("score_iou_threshold", 0.5))

    inputs = DinoTrackingInputs(
        video_id=video_id,
        motion_npz=paths.npz_path,
        dino_npz=paths.dino_path,
        sam_frame0_png=paths.sam_path if cfg.tracking.use_sam_frame0 else None,
    )
    return ObjectiveContext(
        cfg=cfg,
        paths=paths,
        inputs=inputs,
        img_dims=(h, w),
        annotations_path=str(paths.pseudo_gt_annotations_root),
        video_id=video_id,
        match_iou_threshold=match_iou,
        score_iou_threshold=score_iou,
    )


class TrackingObjective:
    def __init__(
        self,
        ctx: ObjectiveContext,
        *,
        max_outlier_pct: float = DEFAULT_MAX_OUTLIER_PCT,
    ) -> None:
        self.ctx = ctx
        self.max_outlier_pct = float(max_outlier_pct)
        self._trial_index = 0
        configure_jax_cache()

    def evaluate(
        self,
        trial_params: dict[str, float],
        *,
        measure_fps: bool,
    ) -> tuple[float, dict[str, Any], bool]:
        params = build_tracking_params(
            self.ctx.cfg,
            trial_params,
            measure_fps=measure_fps,
        )
        t0 = time.perf_counter()
        result = run_dino_tracking(self.ctx.inputs, params)
        elapsed = time.perf_counter() - t0
        metrics = evaluate_custom_instance_tracking(
            self.ctx.video_id,
            result.tracking_data,
            annotations_path=self.ctx.annotations_path,
            img_dims=result.img_dims,
            match_iou_threshold=self.ctx.match_iou_threshold,
            score_iou_threshold=self.ctx.score_iou_threshold,
        )
        objective_score = float(metrics[PRIMARY_METRIC_KEY])
        mean_out_pct, min_out_pct, max_out_pct = mean_outlier_fraction_percent(
            result.tracking_data
        )
        valid = mean_out_pct <= self.max_outlier_pct
        meta = {
            "elapsed_seconds": elapsed,
            PRIMARY_METRIC_KEY: objective_score,
            "avg_persistent_iou": float(metrics.get("avg_persistent_iou", 0)),
            "avg_mean_matched_iou": float(metrics.get("avg_mean_matched_iou", 0)),
            "avg_mean_gt_iou": float(metrics.get("avg_mean_gt_iou", 0)),
            "avg_pixel_jaccard": float(metrics.get("avg_pixel_jaccard", 0)),
            "avg_gt_recall_at_thresh": float(metrics.get("avg_gt_recall_at_thresh", 0)),
            "avg_pred_precision_at_thresh": float(
                metrics.get("avg_pred_precision_at_thresh", 0)
            ),
            "num_frames": len(result.tracking_data),
            "mean_outlier_pct": mean_out_pct,
            "min_outlier_pct_per_frame": min_out_pct,
            "max_outlier_pct_per_frame": max_out_pct,
            "max_outlier_pct_allowed": self.max_outlier_pct,
            "valid": valid,
            "instance_match_iou_threshold": self.ctx.match_iou_threshold,
            "instance_score_iou_threshold": self.ctx.score_iou_threshold,
        }
        self._trial_index += 1
        overshoot = max(0.0, mean_out_pct - self.max_outlier_pct) / 100.0
        ax_objective = objective_score - OUTLIER_PENALTY_WEIGHT * overshoot
        return ax_objective, meta, valid


class MultiVideoTrackingObjective:
    """Evaluate a trial against multiple preprocessed videos and aggregate the
    primary metric with a softmin-style score that puts most of the weight on
    the worst-performing video.  A trial is invalid if ANY video's mean
    outlier % exceeds ``max_outlier_pct`` — stable hyperparameters must be
    globally well-behaved.

    Mirrors `TrackingObjective.evaluate`'s return shape so the ax_runner loop
    is interchangeable.
    """

    def __init__(
        self,
        ctxs: Sequence[ObjectiveContext],
        *,
        max_outlier_pct: float = DEFAULT_MAX_OUTLIER_PCT,
        softmin_tau: float = 4.0,
        subsets_by_phase: dict[str, list[str]] | None = None,
        parallel_videos: int = 1,
    ) -> None:
        if not ctxs:
            raise ValueError("MultiVideoTrackingObjective needs at least one context")
        # Keep every originally-passed context so set_phase() can swap the
        # active subset without rebuilding (and re-loading pseudo-GT / DINO
        # npzs for) every video.
        self._all_ctxs: dict[str, ObjectiveContext] = {c.video_id: c for c in ctxs}
        self.ctxs: list[ObjectiveContext] = list(ctxs)
        self.max_outlier_pct = float(max_outlier_pct)
        self._softmin_tau = float(softmin_tau)
        self._subsets_by_phase: dict[str, list[str]] = dict(subsets_by_phase or {})
        self._parallel_videos = max(1, int(parallel_videos))
        self._trial_index = 0
        configure_jax_cache()

    def set_phase(self, phase: str) -> None:
        """Restrict the active context list to the videos named for ``phase``.

        When ``subsets_by_phase[phase]`` is missing or empty, every context
        passed to the constructor is used.  Phase A (Sobol) typically runs
        on a fast 4-video subset; Phase B (LCB) typically opens up to all
        videos so the softmin aggregator sees the hard ones.
        """
        ids = self._subsets_by_phase.get(phase)
        if ids:
            missing = [v for v in ids if v not in self._all_ctxs]
            if missing:
                raise ValueError(
                    f"phase {phase!r} subset references unknown video ids: {missing}"
                )
            self.ctxs = [self._all_ctxs[v] for v in ids]
        else:
            self.ctxs = list(self._all_ctxs.values())

    def _eval_one_video(
        self,
        ctx: ObjectiveContext,
        trial_params: dict[str, float],
        measure_fps: bool,
    ) -> dict[str, Any]:
        """Evaluate a single video.  Pure-ish: only reads ``self.max_outlier_pct``
        on top of its arguments, so safe to run from worker threads."""
        params = build_tracking_params(ctx.cfg, trial_params, measure_fps=measure_fps)
        result = run_dino_tracking(ctx.inputs, params)
        metrics = evaluate_custom_instance_tracking(
            ctx.video_id,
            result.tracking_data,
            annotations_path=ctx.annotations_path,
            img_dims=result.img_dims,
            match_iou_threshold=ctx.match_iou_threshold,
            score_iou_threshold=ctx.score_iou_threshold,
        )
        primary = float(metrics[PRIMARY_METRIC_KEY])
        mean_out, _min, _max = mean_outlier_fraction_percent(result.tracking_data)
        return {
            "video_id": ctx.video_id,
            PRIMARY_METRIC_KEY: primary,
            "avg_mean_gt_iou": float(metrics.get("avg_mean_gt_iou", 0)),
            "avg_persistent_iou": float(metrics.get("avg_persistent_iou", 0)),
            "avg_pixel_jaccard": float(metrics.get("avg_pixel_jaccard", 0)),
            "mean_outlier_pct": mean_out,
            "valid": mean_out <= self.max_outlier_pct,
        }

    def evaluate(
        self,
        trial_params: dict[str, float],
        *,
        measure_fps: bool,
    ) -> tuple[float, dict[str, Any], bool]:
        primary_scores: list[float] = []
        outlier_pcts: list[float] = []
        all_valid = True
        t0 = time.perf_counter()

        # Per-video evaluation is the wall-clock bottleneck of each trial.
        # Each video's `run_dino_tracking` is GIL-releasing JAX work; running
        # them through a ThreadPoolExecutor lets concurrent GPU launches
        # overlap (JAX queues them on the same CUDA stream) and cuts trial
        # wall-clock 3-4× on a 7-video set.  pool.map preserves submission
        # order, so per_video / primary_scores stay aligned with self.ctxs.
        if self._parallel_videos > 1 and len(self.ctxs) > 1:
            with ThreadPoolExecutor(max_workers=self._parallel_videos) as pool:
                per_video = list(pool.map(
                    lambda ctx: self._eval_one_video(ctx, trial_params, measure_fps),
                    self.ctxs,
                ))
        else:
            per_video = [
                self._eval_one_video(ctx, trial_params, measure_fps)
                for ctx in self.ctxs
            ]
        for r in per_video:
            primary_scores.append(r[PRIMARY_METRIC_KEY])
            outlier_pcts.append(r["mean_outlier_pct"])
            all_valid = all_valid and r["valid"]

        elapsed = time.perf_counter() - t0
        # Aggregate: softmin (worst-case-weighted) primary score across the
        # active video set.  Keeping the plain mean alongside it lets the
        # accuracy gate report both — and lets the extract script record
        # what was actually optimized.
        mean_primary = float(sum(primary_scores) / len(primary_scores))
        tau = self._softmin_tau
        weights = [math.exp(-tau * s) for s in primary_scores]
        wsum = sum(weights) or 1.0
        softmin_primary = float(
            sum(w * s for w, s in zip(weights, primary_scores)) / wsum
        )
        max_outlier = float(max(outlier_pcts))

        meta = {
            "elapsed_seconds": elapsed,
            PRIMARY_METRIC_KEY: softmin_primary,
            "softmin_primary": softmin_primary,
            "softmin_tau": tau,
            "mean_primary": mean_primary,
            "per_video": per_video,
            "num_videos": len(self.ctxs),
            "video_ids": [c.video_id for c in self.ctxs],
            "min_video_primary": float(min(primary_scores)),
            "max_video_primary": float(max(primary_scores)),
            "mean_outlier_pct": float(sum(outlier_pcts) / len(outlier_pcts)),
            "max_outlier_pct_per_video": max_outlier,
            "max_outlier_pct_allowed": self.max_outlier_pct,
            "valid": all_valid,
        }
        self._trial_index += 1
        overshoot = max(0.0, max_outlier - self.max_outlier_pct) / 100.0
        ax_objective = softmin_primary - OUTLIER_PENALTY_WEIGHT * overshoot
        return ax_objective, meta, all_valid


def build_multi_objective_contexts(
    cfg: CustomConfig,
    video_ids: Iterable[str],
    *,
    bo_cfg: dict[str, Any] | None = None,
) -> list[ObjectiveContext]:
    """Build one ObjectiveContext per video_id.  Used by the multi-video
    bayesopt runner to score a single set of trial params on every video and
    average the result."""
    return [build_objective_context(cfg, vid, bo_cfg=bo_cfg) for vid in video_ids]

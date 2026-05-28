"""Ax BayesOpt runner: Sobol then BoTorch-Modular (qNEHVI for multi-objective).

Multi-video runs surface every video as an independent Ax objective so the
optimizer sees the Pareto front of per-video IoUs directly — no scalarized
softmin/mean aggregation as a proxy for "all videos must do well".
"""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _rss_gb() -> float:
    """Resident RAM of this process, in GB.  Linux /proc only — returns 0
    on platforms without it (so the call site stays branch-free)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 0.0

from genmatter.bayesopt.config import load_bayesopt_config
from genmatter.bayesopt.objective import (
    PRIMARY_METRIC_KEY,
    MultiVideoTrackingObjective,
    ObjectiveContext,
    TrackingObjective,
    build_multi_objective_contexts,
    build_objective_context,
)
from genmatter.bayesopt.search_space import (
    OBJECTIVE_NAME,
    full_resolved_params_dict,
    specs_from_config,
)
from genmatter.bayesopt.terminal import BayesOptConsole
from genmatter.bayesopt.trial_store import (
    RunState,
    append_jsonl,
    backfill_trial_artifacts,
    load_state,
    read_jsonl,
    save_state,
    save_trial_artifacts,
    trials_summary_path,
    update_best,
)
from genmatter.custom.config_schema import load_config
from genmatter.custom.paths import bayesopt_run_dir, resolve_video_paths


def _build_generation_strategy(bo_cfg: dict[str, Any], sobol_seed: int) -> Any:
    from ax.adapter.registry import Generators
    from ax.generation_strategy.generation_strategy import GenerationStep, GenerationStrategy

    n_sobol = int(bo_cfg["phase"]["phase_a_sobol_evals"])
    return GenerationStrategy(
        steps=[
            GenerationStep(
                generator=Generators.SOBOL,
                num_trials=n_sobol,
                max_parallelism=1,
                generator_kwargs={"seed": int(sobol_seed), "deduplicate": True},
            ),
            GenerationStep(
                generator=Generators.BOTORCH_MODULAR,
                num_trials=-1,
                max_parallelism=1,
            ),
        ]
    )


def _create_ax_client(
    bo_cfg: dict[str, Any],
    specs: list,
    run_id: str,
    objective_names: list[str],
    *,
    per_video_iou_threshold: float = 0.3,
) -> Any:
    """Build the Ax client.  With >1 objective name, ``BOTORCH_MODULAR``
    auto-detects multi-objective and switches the acquisition to qNEHVI.

    ``per_video_iou_threshold`` is the qNEHVI hypervolume reference point —
    any per-video IoU below this value contributes zero to hypervolume.
    Pick low enough that v1-broken trials don't poison the bookkeeping
    (0.3 vs the 0.55 accuracy-gate target).
    """
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties

    sobol_seed = int(bo_cfg["run"]["master_seed"])
    gs = _build_generation_strategy(bo_cfg, sobol_seed)
    client = AxClient(generation_strategy=gs, verbose_logging=False)
    parameters = [s.to_ax_dict() for s in specs]
    if len(objective_names) == 1:
        objectives = {objective_names[0]: ObjectiveProperties(minimize=False)}
    else:
        objectives = {
            name: ObjectiveProperties(minimize=False, threshold=float(per_video_iou_threshold))
            for name in objective_names
        }
    client.create_experiment(
        name=run_id,
        parameters=parameters,
        objectives=objectives,
    )
    return client


def _replay_trials(
    ax_client: Any,
    rows: list[dict[str, Any]],
    objective_names: list[str],
) -> None:
    replayable = sorted(
        [r for r in rows if r.get("status") in ("completed", "invalid")],
        key=lambda r: int(r["trial_index"]),
    )
    for row in replayable:
        params = dict(row["params"])
        raw_data = _row_per_video_objectives(row, objective_names)
        _, tid = ax_client.attach_trial(parameters=params)
        ax_client.complete_trial(trial_index=tid, raw_data=raw_data)


def _warm_start_attach(
    ax_client: Any,
    cfg: Any,
    *,
    warm_start_cfg: dict[str, Any],
    objective_names: list[str],
    trials_path: Path,
    events_path: Path,
    run_root: Path,
    param_names: list[str],
    resolved_model_params_dict: Any,
    ui: Any,
) -> None:
    """Attach a previously-evaluated trial (the v1 winner) as Ax's first arm.

    Reads the trial record yaml emitted by `save_trial_artifacts`, replays
    the params + per-video scores, and writes a synthetic row to
    `trials.jsonl` so a subsequent resume sees the warm-start point as
    trial 0.  Falls back gracefully (with a UI warning) if the file is
    missing or malformed — the long run can still proceed without warm
    start.
    """
    base_root = Path(cfg.paths.custom_videos_root) if hasattr(cfg, "paths") else Path("assets/custom_videos")
    src_video = str(warm_start_cfg.get("video_id"))
    src_run = str(warm_start_cfg.get("run_id"))
    src_trial = int(warm_start_cfg.get("trial_index"))
    record = base_root / src_video / "bayesopt" / src_run / "trials" / f"trial_{src_trial:05d}.yaml"
    if not record.is_file():
        ui.print(f"[yellow]warm_start record not found:[/] {record} — skipping")
        return
    with open(record, encoding="utf-8") as f:
        prev = yaml.safe_load(f) or {}
    params = dict(prev.get("params", {}))
    if not params:
        ui.print(f"[yellow]warm_start trial has no params:[/] {record} — skipping")
        return
    raw_data = _row_per_video_objectives(prev, objective_names)
    _, tid = ax_client.attach_trial(parameters=params)
    ax_client.complete_trial(trial_index=tid, raw_data=raw_data)

    # Persist a synthetic row so resume sees this warm-start.  Copy the
    # per-video block + diagnostics from the source so the run_dir behaves
    # like a normal trial in every other code path.
    row = {
        "trial_index": tid,
        "phase": "warm_start",
        "params": params,
        "objective": float(prev.get("objective", 0.0)),
        "ax_objective": float(prev.get("ax_objective", prev.get("objective", 0.0))),
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "warm_start_source": {"video_id": src_video, "run_id": src_run, "trial_index": src_trial},
        "per_video": prev.get("per_video", []),
        "softmin_primary": prev.get("softmin_primary"),
        "mean_primary": prev.get("mean_primary"),
        "softmin_tau": prev.get("softmin_tau"),
        "min_video_primary": prev.get("min_video_primary"),
        "max_video_primary": prev.get("max_video_primary"),
        "mean_outlier_pct": prev.get("mean_outlier_pct"),
        "max_outlier_pct_per_video": prev.get("max_outlier_pct_per_video"),
        "elapsed_seconds": 0.0,
    }
    append_jsonl(trials_path, row)
    append_jsonl(events_path, {"event": "warm_start", **row})
    save_trial_artifacts(
        run_root,
        row,
        param_names=param_names,
        resolved_hyperparams=resolved_model_params_dict(params),
    )
    ui.print(f"[green]warm_start attached:[/] {record.name} → trial {tid}")


def _lcb_generation_node_name(gs: Any) -> str | None:
    for name in gs.nodes_by_name:
        if "BoTorch" in name or "Botorch" in name:
            return name
    names = list(gs.nodes_by_name.keys())
    return names[1] if len(names) > 1 else None


def _sync_generation_strategy(ax_client: Any, *, valid_completed: int, n_sobol: int) -> None:
    """attach_trial replay does not advance Ax's GS; jump to LCB node after Sobol."""
    if valid_completed < n_sobol:
        return
    gs = ax_client.generation_strategy
    target = _lcb_generation_node_name(gs)
    if target is not None and gs._curr.name != target:
        gs._curr = gs.nodes_by_name[target]


def _resolved_model_params_dict(
    cfg: Any, trial_params: dict[str, float | bool]
) -> dict[str, Any]:
    from genmatter.bayesopt.search_space import build_tracking_params

    params = build_tracking_params(cfg, trial_params, measure_fps=False)
    resolved = full_resolved_params_dict(params)
    return {
        k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v) if isinstance(v, (int, float)) else v)
        for k, v in resolved.items()
        if not isinstance(v, (list, dict))
    }


def _phase_name(completed: int, n_sobol: int) -> str:
    return "sobol" if completed < n_sobol else "bo_moo"


def _objective_names(video_ids: list[str]) -> list[str]:
    """Per-video Ax objective names used for multi-objective optimization.

    Single-video runs collapse to ``[OBJECTIVE_NAME]`` so the existing
    single-objective TrackingObjective path keeps working.
    """
    if len(video_ids) <= 1:
        return [OBJECTIVE_NAME]
    return [f"iou_{vid}" for vid in video_ids]


def _row_per_video_objectives(row: dict[str, Any], objective_names: list[str]) -> dict[str, tuple[float, float]]:
    """Resolve per-objective values for an already-evaluated trial row.

    Reads per-video IoU from ``row["per_video"]`` (written by the
    MultiVideoTrackingObjective) and applies the same outlier penalty
    used at evaluation time so resumed Ax sees the original BO targets.
    On invalid trials, the per-video score collapses to 0.0 (each video's
    contribution to hypervolume is zero).
    """
    if len(objective_names) == 1:
        return {objective_names[0]: (float(row.get("ax_objective", row.get("objective", 0.0))), 0.0)}
    per_video = row.get("per_video", []) or []
    by_vid = {pv["video_id"]: pv for pv in per_video}
    out: dict[str, tuple[float, float]] = {}
    for name in objective_names:
        vid = name[len("iou_"):]
        pv = by_vid.get(vid)
        if pv is None:
            out[name] = (0.0, 0.0)
            continue
        score = float(pv.get("avg_persistent_iou", 0.0)) if pv.get("valid", True) else 0.0
        out[name] = (score, 0.0)
    return out


def run_bayesopt_forever(
    video_id: str,
    *,
    tracking_config_path: Path,
    bayesopt_config_path: Path,
    run_id: str | None = None,
    tracking_overrides: list[str] | None = None,
    bo_overrides: list[str] | None = None,
    resume: bool = True,
) -> None:
    paths_cfg = load_config(tracking_config_path, dot_overrides=tracking_overrides or [])
    # StreamingVision patch: video_id may be a comma-separated list of video
    # ids — if so, evaluate every trial on every video and average the
    # primary metric.  Run artifacts live under the FIRST video's bayesopt
    # directory; per-video sub-scores are recorded in the trial meta.
    video_ids = [v.strip() for v in video_id.split(",") if v.strip()]
    primary_video_id = video_ids[0]
    multi_video = len(video_ids) > 1
    paths = resolve_video_paths(paths_cfg, primary_video_id)
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = bayesopt_run_dir(paths, rid)
    run_root.mkdir(parents=True, exist_ok=True)

    trials_path = run_root / "trials.jsonl"
    state_path = run_root / "state.json"
    events_path = run_root / "events.jsonl"
    frozen_cfg_path = run_root / "run_config.yaml"

    if resume and frozen_cfg_path.is_file():
        with open(frozen_cfg_path, encoding="utf-8") as f:
            bo_cfg = yaml.safe_load(f) or {}
        bo_cfg["video_id"] = video_id
    else:
        bo_cfg = load_bayesopt_config(
            bayesopt_config_path, video_id=video_id, dot_overrides=bo_overrides
        )
    cfg = paths_cfg
    specs = specs_from_config(bo_cfg)
    n_sobol = int(bo_cfg["phase"]["phase_a_sobol_evals"])

    if not resume or not trials_path.is_file():
        with open(run_root / "run_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(bo_cfg, f, sort_keys=False)
        with open(run_root / "search_space_snapshot.json", "w", encoding="utf-8") as f:
            json.dump([s.to_ax_dict() for s in specs], f, indent=2)

    rows = read_jsonl(trials_path) if resume else []
    param_names = [s.name for s in specs]
    objective_names = _objective_names(video_ids)
    phase_cfg_early = bo_cfg.get("phase", {})
    ax_client = _create_ax_client(
        bo_cfg,
        specs,
        rid,
        objective_names,
        per_video_iou_threshold=float(phase_cfg_early.get("per_video_iou_threshold", 0.3)),
    )
    if rows:
        _replay_trials(ax_client, rows, objective_names)
        valid_completed = len([r for r in rows if r.get("status") == "completed"])
        _sync_generation_strategy(ax_client, valid_completed=valid_completed, n_sobol=n_sobol)
        backfill_trial_artifacts(
            run_root,
            rows,
            param_names=param_names,
            resolve_hyperparams=lambda p: _resolved_model_params_dict(cfg, p),
            force=True,
        )

    max_outlier_pct = float(bo_cfg.get("objective", {}).get("max_outlier_pct", 10.0))
    phase_cfg = bo_cfg.get("phase", {})
    parallel_videos = int(phase_cfg.get("parallel_videos", 1))
    if multi_video:
        ctxs = build_multi_objective_contexts(cfg, video_ids, bo_cfg=bo_cfg)
        subsets_by_phase: dict[str, list[str]] = {}
        sobol_subset = phase_cfg.get("sobol_video_subset")
        if sobol_subset:
            subsets_by_phase["sobol"] = list(sobol_subset)
        lcb_subset = phase_cfg.get("lcb_video_subset")
        if lcb_subset:
            subsets_by_phase["bo_moo"] = list(lcb_subset)
        objective = MultiVideoTrackingObjective(
            ctxs,
            max_outlier_pct=max_outlier_pct,
            softmin_tau=float(phase_cfg.get("softmin_tau", 4.0)),
            subsets_by_phase=subsets_by_phase or None,
            parallel_videos=parallel_videos,
        )
    else:
        ctx: ObjectiveContext = build_objective_context(cfg, primary_video_id, bo_cfg=bo_cfg)
        objective = TrackingObjective(ctx, max_outlier_pct=max_outlier_pct)

    ui = BayesOptConsole()
    ui.start()

    state = load_state(state_path)
    if state is None and rows:
        best = max((r for r in rows if r.get("status") == "completed"), key=lambda r: r["objective"], default=None)
        if best:
            state = RunState(
                best_objective=float(best["objective"]),
                best_params=dict(best["params"]),
                trial_count=len([r for r in rows if r.get("status") == "completed"]),
                last_phase=str(best.get("phase", "")),
            )

    valid_completed = len([r for r in rows if r.get("status") == "completed"])
    trial_counter = len([r for r in rows if r.get("status") in ("completed", "invalid", "failed")])
    invalid_count = len([r for r in rows if r.get("status") == "invalid"])

    # Warm-start: attach a previously-evaluated trial (typically the v1
    # winner) as Ax's first arm so the surrogate has a known-good Pareto
    # point before Sobol begins.  No-op on resume — the existing
    # trials.jsonl already carries it.
    if not rows:
        warm_start_cfg = phase_cfg.get("warm_start")
        if warm_start_cfg:
            _warm_start_attach(
                ax_client,
                cfg,
                warm_start_cfg=warm_start_cfg,
                objective_names=objective_names,
                trials_path=trials_path,
                events_path=events_path,
                run_root=run_root,
                param_names=param_names,
                resolved_model_params_dict=lambda p: _resolved_model_params_dict(cfg, p),
                ui=ui,
            )
            # Refresh counters now that a row was appended.
            rows = read_jsonl(trials_path)
            valid_completed = len([r for r in rows if r.get("status") == "completed"])
            trial_counter = len([r for r in rows if r.get("status") in ("completed", "invalid", "failed")])

    ui.print(f"[bold]BayesOpt run[/] {run_root}")
    ui.print(
        f"Trial log: {trials_path} | per-trial params: {run_root / 'trials'}/ | "
        f"summary: {trials_summary_path(run_root)}"
    )
    moo_tag = "multi-objective qNEHVI" if multi_video else "single-objective EI"
    ui.print(
        f"Video{'s' if multi_video else ''}: {','.join(video_ids)} | "
        f"objectives={len(objective_names)} ({moo_tag}) | "
        f"dims={len(specs)} ({', '.join(param_names)}) | "
        f"Sobol={n_sobol} then BoTorch | "
        f"parallel_videos={parallel_videos} | "
        f"max outlier %={max_outlier_pct:.1f} (Ctrl+C to stop)"
    )

    try:
        while True:
            phase = _phase_name(valid_completed, n_sobol)
            if hasattr(objective, "set_phase"):
                objective.set_phase(phase)
            params, trial_index = ax_client.get_next_trial()

            # measure_fps would corrupt timings across concurrent workers,
            # so disable whenever the trial fans out into multiple videos.
            measure_fps = (trial_counter == 0) and (parallel_videos <= 1)
            ui.update(
                {
                    "phase": phase,
                    "trial_index": trial_index,
                    "current_params": params,
                    "best_objective": state.best_objective if state else float("nan"),
                    "best_params": state.best_params if state else {},
                }
            )

            try:
                ax_obj, meta, valid = objective.evaluate(params, measure_fps=measure_fps)
                score = float(meta[PRIMARY_METRIC_KEY])
                if valid:
                    status = "completed"
                else:
                    status = "invalid"
                    invalid_count += 1
                    ui.print(
                        f"[yellow]Trial {trial_index} invalid:[/] "
                        f"mean outlier {meta['mean_outlier_pct']:.1f}% > {max_outlier_pct:.1f}% "
                        f"({PRIMARY_METRIC_KEY}={score:.4f})"
                    )
            except Exception as exc:
                ax_obj = 0.0
                score = 0.0
                meta = {"error": str(exc)}
                status = "failed"
                ui.print(f"[red]Trial {trial_index} failed:[/] {exc}")

            if status in ("completed", "invalid", "failed"):
                # Build per-objective raw_data so multi-video runs feed each
                # video's IoU to Ax separately; the per-video penalty mirrors
                # the scalar ax_objective in the single-video case.
                row_for_ax = {"per_video": meta.get("per_video", []),
                               "ax_objective": ax_obj, "objective": score}
                raw_data = _row_per_video_objectives(row_for_ax, objective_names)
                ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
                trial_counter += 1

            if status == "completed":
                valid_completed += 1
                state = update_best(state, score, params, phase=phase)
                save_state(state_path, state)

            row = {
                "trial_index": trial_index,
                "phase": phase,
                "params": params,
                "objective": score,
                "ax_objective": ax_obj,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **meta,
            }
            append_jsonl(trials_path, row)
            append_jsonl(events_path, {"event": "trial_done", **row})
            save_trial_artifacts(
                run_root,
                row,
                param_names=param_names,
                resolved_hyperparams=_resolved_model_params_dict(cfg, params),
            )

            trial_status = {
                "phase": phase,
                "trial_index": trial_index,
                "last_objective": score,
                "last_outlier_pct": meta.get("mean_outlier_pct"),
                "last_status": status,
                "invalid_count": invalid_count,
                "max_outlier_pct": max_outlier_pct,
                "last_elapsed_s": meta.get("elapsed_seconds", 0),
                "best_objective": state.best_objective if state else score,
                "best_params": state.best_params if state else params,
                "current_params": params,
            }
            ui.update(trial_status)
            ui.log_trial(trial_status)

            # Memory-leak mitigation.  Without this, RSS climbs ~0.7 GB/trial
            # on a 7-video MO run (Ax holds per-trial data + JAX in-memory HLO
            # caches accumulate), hitting host RAM OOM around trial 35 on a
            # 62 GB box.  gc.collect() reclaims dangling Python refs; clearing
            # JAX's in-memory cache forces re-resolution from the on-disk
            # persistent cache (.jax_cache/) — ~5 s vs ~90 s cold compile.
            rss_before = _rss_gb()
            gc.collect()
            if trial_counter % 5 == 0:
                try:
                    import jax
                    jax.clear_caches()
                except Exception:
                    pass
            rss_after = _rss_gb()
            ui.print(
                f"[dim]post-gc RSS {rss_before:.1f} → {rss_after:.1f} GB "
                f"(trial {trial_counter}, jax_cache_clear={trial_counter % 5 == 0})[/]"
            )
    except KeyboardInterrupt:
        ui.print("\n[yellow]Stopped by user.[/]")
    finally:
        ui.stop()
        if state:
            ui.print(f"[green]Best {PRIMARY_METRIC_KEY}[/] {state.best_objective:.4f}")
            ui.print(f"Best params: {state.best_params}")

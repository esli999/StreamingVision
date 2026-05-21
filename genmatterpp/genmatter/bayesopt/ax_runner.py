"""Ax BayesOpt runner: 64 Sobol then LCB forever."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from genmatter.bayesopt.config import load_bayesopt_config
from genmatter.bayesopt.lcb import propose_lcb
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


def _create_ax_client(bo_cfg: dict[str, Any], specs: list, run_id: str) -> Any:
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties

    sobol_seed = int(bo_cfg["run"]["master_seed"])
    gs = _build_generation_strategy(bo_cfg, sobol_seed)
    client = AxClient(generation_strategy=gs, verbose_logging=False)
    parameters = [s.to_ax_dict() for s in specs]
    client.create_experiment(
        name=run_id,
        parameters=parameters,
        objectives={OBJECTIVE_NAME: ObjectiveProperties(minimize=False)},
    )
    return client


def _replay_trials(ax_client: Any, rows: list[dict[str, Any]]) -> None:
    replayable = sorted(
        [r for r in rows if r.get("status") in ("completed", "invalid")],
        key=lambda r: int(r["trial_index"]),
    )
    for row in replayable:
        params = dict(row["params"])
        ax_obj = float(row.get("ax_objective", row["objective"]))
        _, tid = ax_client.attach_trial(parameters=params)
        ax_client.complete_trial(trial_index=tid, raw_data={OBJECTIVE_NAME: ax_obj})


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
    return "sobol" if completed < n_sobol else "bo_lcb"


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
    ax_client = _create_ax_client(bo_cfg, specs, rid)
    if rows:
        _replay_trials(ax_client, rows)
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
    if multi_video:
        ctxs = build_multi_objective_contexts(cfg, video_ids, bo_cfg=bo_cfg)
        objective = MultiVideoTrackingObjective(ctxs, max_outlier_pct=max_outlier_pct)
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
    lcb_seed_offset = 0
    invalid_count = len([r for r in rows if r.get("status") == "invalid"])

    ui.print(f"[bold]BayesOpt run[/] {run_root}")
    ui.print(
        f"Trial log: {trials_path} | per-trial params: {run_root / 'trials'}/ | "
        f"summary: {trials_summary_path(run_root)}"
    )
    ui.print(
        f"Video{'s' if multi_video else ''}: {','.join(video_ids)} | "
        f"objective={PRIMARY_METRIC_KEY}{'(mean across videos)' if multi_video else ''} | "
        f"dims={len(specs)} ({', '.join(param_names)}) | "
        f"Sobol={n_sobol} then LCB | max outlier %={max_outlier_pct:.1f} (Ctrl+C to stop)"
    )

    try:
        while True:
            phase = _phase_name(valid_completed, n_sobol)
            if phase == "sobol":
                params, trial_index = ax_client.get_next_trial()
            else:
                params, trial_index = propose_lcb(
                    ax_client,
                    specs,
                    pool_size=int(bo_cfg["phase"].get("candidate_pool_size", 4096)),
                    beta=float(bo_cfg["phase"].get("lcb_beta", 2.0)),
                    seed=int(bo_cfg["run"]["master_seed"]) + lcb_seed_offset,
                    objective_name=OBJECTIVE_NAME,
                )
                lcb_seed_offset += 1

            measure_fps = trial_counter == 0
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
                ax_client.complete_trial(
                    trial_index=trial_index, raw_data={OBJECTIVE_NAME: ax_obj}
                )
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
    except KeyboardInterrupt:
        ui.print("\n[yellow]Stopped by user.[/]")
    finally:
        ui.stop()
        if state:
            ui.print(f"[green]Best {PRIMARY_METRIC_KEY}[/] {state.best_objective:.4f}")
            ui.print(f"Best params: {state.best_params}")

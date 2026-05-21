"""Append-only trial metadata for BayesOpt runs."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class RunState:
    best_objective: float
    best_params: dict[str, float]
    trial_count: int
    last_phase: str


def load_state(path: Path) -> RunState | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return RunState(
        best_objective=float(data["best_objective"]),
        best_params=dict(data["best_params"]),
        trial_count=int(data["trial_count"]),
        last_phase=str(data.get("last_phase", "")),
    )


def save_state(path: Path, state: RunState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_objective": state.best_objective,
                "best_params": state.best_params,
                "trial_count": state.trial_count,
                "last_phase": state.last_phase,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )


_TRIAL_SUMMARY_FIELDS = (
    "trial_index",
    "phase",
    "status",
    "objective",
    "ax_objective",
    "mean_outlier_pct",
    "elapsed_seconds",
    "timestamp",
)


def trial_params_path(run_root: Path, trial_index: int) -> Path:
    return run_root / "trials" / f"trial_{trial_index:05d}_params.yaml"


def trial_record_path(run_root: Path, trial_index: int) -> Path:
    return run_root / "trials" / f"trial_{trial_index:05d}.yaml"


def trials_summary_path(run_root: Path) -> Path:
    return run_root / "trials_summary.csv"


def _summary_fieldnames(param_names: list[str]) -> list[str]:
    return list(_TRIAL_SUMMARY_FIELDS) + [f"param_{k}" for k in param_names]


def _row_to_summary_dict(row: dict[str, Any], param_names: list[str]) -> dict[str, Any]:
    params = dict(row.get("params", {}))
    flat = {k: row.get(k, "") for k in _TRIAL_SUMMARY_FIELDS}
    for name in param_names:
        flat[f"param_{name}"] = params.get(name, "")
    return flat


def rewrite_trials_summary(run_root: Path, rows: list[dict[str, Any]], param_names: list[str]) -> None:
    summary_path = trials_summary_path(run_root)
    fieldnames = _summary_fieldnames(param_names)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: int(r["trial_index"])):
            writer.writerow(_row_to_summary_dict(row, param_names))


def save_trial_artifacts(
    run_root: Path,
    row: dict[str, Any],
    *,
    param_names: list[str],
    resolved_hyperparams: dict[str, Any] | None = None,
) -> None:
    """Persist per-trial params and a rolling CSV summary for auditing."""
    trial_index = int(row["trial_index"])
    trials_dir = run_root / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    params = dict(row.get("params", {}))
    with trial_params_path(run_root, trial_index).open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "trial_index": trial_index,
                "phase": row.get("phase"),
                "status": row.get("status"),
                "params": params,
                "resolved_hyperparams": resolved_hyperparams or {},
            },
            f,
            sort_keys=False,
        )

    record = dict(row)
    if resolved_hyperparams is not None:
        record["resolved_hyperparams"] = resolved_hyperparams
    with trial_record_path(run_root, trial_index).open("w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, sort_keys=False)

    summary_path = trials_summary_path(run_root)
    write_header = not summary_path.is_file()
    fieldnames = _summary_fieldnames(param_names)
    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(_row_to_summary_dict(row, param_names))


def backfill_trial_artifacts(
    run_root: Path,
    rows: list[dict[str, Any]],
    *,
    param_names: list[str],
    resolve_hyperparams: Any | None = None,
    force: bool = False,
) -> None:
    """Write per-trial YAML/CSV files for rows loaded from trials.jsonl on resume."""
    for row in rows:
        tid = int(row["trial_index"])
        if not force and trial_record_path(run_root, tid).is_file():
            continue
        resolved = None
        if resolve_hyperparams is not None:
            resolved = resolve_hyperparams(dict(row.get("params", {})))
        save_trial_artifacts(
            run_root,
            row,
            param_names=param_names,
            resolved_hyperparams=resolved,
        )
    if rows:
        rewrite_trials_summary(run_root, rows, param_names)


def update_best(
    state: RunState | None,
    objective: float,
    params: dict[str, float],
    *,
    phase: str = "",
) -> RunState:
    count = (state.trial_count if state else 0) + 1
    if state is None or objective > state.best_objective:
        return RunState(
            best_objective=float(objective),
            best_params=dict(params),
            trial_count=count,
            last_phase=phase,
        )
    return RunState(
        best_objective=state.best_objective,
        best_params=state.best_params,
        trial_count=count,
        last_phase=phase or state.last_phase,
    )

"""Trial store JSONL tests."""

from __future__ import annotations

from genmatter.bayesopt.trial_store import (
    RunState,
    append_jsonl,
    backfill_trial_artifacts,
    load_state,
    read_jsonl,
    save_state,
    save_trial_artifacts,
    trial_params_path,
    trials_summary_path,
    update_best,
)


def test_jsonl_roundtrip(tmp_path) -> None:
    p = tmp_path / "trials.jsonl"
    append_jsonl(p, {"trial_index": 0, "objective": 0.5})
    append_jsonl(p, {"trial_index": 1, "objective": 0.7})
    rows = read_jsonl(p)
    assert len(rows) == 2
    assert rows[1]["objective"] == 0.7


def test_trial_artifacts(tmp_path) -> None:
    row = {
        "trial_index": 3,
        "phase": "sobol",
        "status": "completed",
        "objective": 0.42,
        "ax_objective": 0.42,
        "params": {"sigma_F": 1.5, "sigma_H": 10.0},
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    save_trial_artifacts(
        tmp_path,
        row,
        param_names=["sigma_F", "sigma_H"],
        resolved_hyperparams={"sigma_F": 1.5},
    )
    assert trial_params_path(tmp_path, 3).is_file()
    assert trials_summary_path(tmp_path).is_file()
    backfill_trial_artifacts(tmp_path, [row], param_names=["sigma_F", "sigma_H"])
    assert trial_params_path(tmp_path, 3).is_file()


def test_update_best() -> None:
    s = update_best(None, 0.3, {"sigma_F": 1.0}, phase="sobol")
    assert s.best_objective == 0.3
    s2 = update_best(s, 0.25, {"sigma_F": 2.0}, phase="sobol")
    assert s2.best_objective == 0.3
    s3 = update_best(s2, 0.9, {"sigma_F": 3.0}, phase="bo_lcb")
    assert s3.best_objective == 0.9

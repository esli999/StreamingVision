"""LCB surrogate fit against Ax 1.2.x GenerationNode API."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ax")


def test_fit_surrogate_after_sobol_replay(tmp_path) -> None:
    from genmatter.bayesopt.ax_runner import (
        _create_ax_client,
        _replay_trials,
        _sync_generation_strategy,
    )
    from genmatter.bayesopt.lcb import fit_surrogate, propose_lcb
    from genmatter.bayesopt.search_space import OBJECTIVE_NAME, specs_from_config
    from genmatter.bayesopt.trial_store import read_jsonl

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    src = (
        Path(__file__).resolve().parents[1]
        / "assets/custom_videos/original_244622072067/bayesopt/20260520_032146/trials.jsonl"
    )
    if not src.is_file():
        pytest.skip("fixture trials.jsonl not present")

    rows = read_jsonl(src)[:20]
    frozen = (
        Path(__file__).resolve().parents[1]
        / "assets/custom_videos/original_244622072067/bayesopt/20260520_032146/run_config.yaml"
    )
    if not frozen.is_file():
        pytest.skip("fixture run_config.yaml not present")
    import yaml

    with open(frozen, encoding="utf-8") as f:
        bo_cfg = yaml.safe_load(f)
    bo_cfg["phase"]["phase_a_sobol_evals"] = 5
    specs = specs_from_config(bo_cfg)
    client = _create_ax_client(bo_cfg, specs, "lcb_test")
    _replay_trials(client, rows)
    completed = len([r for r in rows if r.get("status") == "completed"])
    _sync_generation_strategy(
        client, valid_completed=max(completed, 5), n_sobol=5
    )
    adapter, _ = fit_surrogate(client)
    assert hasattr(adapter, "predict")
    params, tid = propose_lcb(
        client, specs, pool_size=8, beta=2.0, seed=0, objective_name=OBJECTIVE_NAME
    )
    assert isinstance(tid, int)
    assert set(params) == {s.name for s in specs}

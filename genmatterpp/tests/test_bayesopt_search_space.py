"""Search space unit/value mapping tests."""

from __future__ import annotations

from genmatter.bayesopt.config import load_bayesopt_config
from genmatter.bayesopt.search_space import (
    HyperparamSpec,
    build_tracking_params,
    full_resolved_params_dict,
    sample_sobol_candidates,
    specs_from_config,
)
from genmatter.custom.config_schema import load_config
from pathlib import Path


def test_log_scale_roundtrip() -> None:
    spec = HyperparamSpec("sigma_F", 0.02, 20.0, log_scale=True)
    v = 2.0
    u = spec.value_to_unit(v)
    v2 = spec.unit_to_value(u)
    assert abs(v2 - v) / v < 0.01


def test_sobol_candidates_shape() -> None:
    specs = [
        HyperparamSpec("a", 0.0, 1.0, log_scale=False),
        HyperparamSpec("b", 0.1, 10.0, log_scale=True),
    ]
    cands = sample_sobol_candidates(specs, 8, seed=0)
    assert len(cands) == 8
    assert "a" in cands[0] and "b" in cands[0]


def test_int_and_bool_specs() -> None:
    specs = [
        HyperparamSpec("n", 8, 24, value_type="int"),
        HyperparamSpec("flag", 0, 1, value_type="bool"),
    ]
    cands = sample_sobol_candidates(specs, 4, seed=1)
    assert isinstance(cands[0]["n"], int)
    assert isinstance(cands[0]["flag"], bool)


def test_build_tracking_params_includes_tracking_outlier() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "custom_default.yaml")
    bo_cfg = load_bayesopt_config(repo / "configs" / "custom_bayesopt.yaml", video_id="v")
    trial = {
        s.name: ((s.lower + s.upper) / 2 if s.value_type != "bool" else True)
        for s in specs_from_config(bo_cfg)
    }
    trial["tracking_outlier_prob"] = 0.01
    trial["init_gibbs_sweeps"] = 12
    trial["num_blobs"] = 300  # ignored — fixed from config
    params = build_tracking_params(cfg, trial, measure_fps=False)
    assert params.tracking_outlier_prob == 0.01
    assert params.init_gibbs_sweeps == 12
    assert params.num_blobs == cfg.tracking.num_blobs
    flat = full_resolved_params_dict(params)
    assert flat["tracking_outlier_prob"] == 0.01
    assert "sigma_F" in flat

"""Ax + mock objective smoke test (no JAX)."""

from __future__ import annotations

import pytest

pytest.importorskip("ax")


def test_ax_sobol_two_trials_mock(tmp_path) -> None:
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties

    from genmatter.bayesopt.search_space import OBJECTIVE_NAME, HyperparamSpec

    specs = [HyperparamSpec("x", 0.0, 1.0, log_scale=False)]
    client = AxClient(verbose_logging=False)
    client.create_experiment(
        name="smoke",
        parameters=[s.to_ax_dict() for s in specs],
        objectives={OBJECTIVE_NAME: ObjectiveProperties(minimize=False)},
    )
    for i in range(2):
        params, tid = client.get_next_trial()
        obj = float(params["x"])
        client.complete_trial(trial_index=tid, raw_data={OBJECTIVE_NAME: obj})
    assert len(client.experiment.trials) == 2

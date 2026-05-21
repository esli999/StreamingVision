"""LCB candidate-pool proposal for Phase B (maximize objective)."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from genmatter.bayesopt.search_space import HyperparamSpec, sample_sobol_candidates


def _supports_predict(adapter: Any) -> bool:
    return adapter is not None and hasattr(adapter, "predict")


def _fitted_adapter_from_node(node: Any) -> Any:
    if hasattr(node, "_fitted_adapter"):
        adapter = node._fitted_adapter
        if adapter is not None:
            return adapter
    if hasattr(node, "_fitted_model"):
        adapter = node._fitted_model
        if adapter is not None:
            return adapter
    if hasattr(node, "model") and node.model is not None:
        return node.model
    if hasattr(node, "_pick_fitted_adapter_to_gen_from"):
        spec = node._pick_fitted_adapter_to_gen_from()
        return spec._fitted_adapter
    return None


def fit_surrogate(ax_client: Any) -> tuple[Any, float]:
    gs = ax_client.generation_strategy
    node = gs._curr
    data = ax_client.experiment.lookup_data()
    t0 = perf_counter()
    if hasattr(gs, "_fit_current_model"):
        gs._fit_current_model(data=data)
    elif hasattr(node, "_fit"):
        node._fit(experiment=ax_client.experiment, data=data)
    elif hasattr(node, "fit"):
        node.fit(
            experiment=ax_client.experiment,
            data=data,
            search_space=ax_client.experiment.search_space,
            optimization_config=ax_client.experiment.optimization_config,
        )
    else:
        raise RuntimeError("Could not fit surrogate on current generation node.")
    adapter = _fitted_adapter_from_node(node)
    if not _supports_predict(adapter):
        raise RuntimeError("Fitted Ax model does not support predict().")
    return adapter, perf_counter() - t0


def score_lcb_maximize(
    adapter: Any,
    candidates: list[dict[str, float]],
    *,
    objective_name: str,
    beta: float,
) -> int:
    from ax.core.observation import ObservationFeatures

    features = [ObservationFeatures(parameters=c) for c in candidates]
    means, covs = adapter.predict(features)
    mu = np.asarray(means[objective_name], dtype=float)
    var = np.asarray(covs[objective_name][objective_name], dtype=float)
    sigma = np.sqrt(np.maximum(var, 0.0))
    scores = mu + float(beta) * sigma
    return int(np.argmax(scores))


def propose_lcb(
    ax_client: Any,
    specs: list[HyperparamSpec],
    *,
    pool_size: int,
    beta: float,
    seed: int,
    objective_name: str,
) -> tuple[dict[str, float], int]:
    adapter, _ = fit_surrogate(ax_client)
    candidates = sample_sobol_candidates(specs, pool_size, seed)
    winner = score_lcb_maximize(adapter, candidates, objective_name=objective_name, beta=beta)
    params = candidates[winner]
    _, trial_index = ax_client.attach_trial(parameters=params)
    return params, trial_index

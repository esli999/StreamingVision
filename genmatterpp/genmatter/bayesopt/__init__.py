"""Bayesian optimization for custom-video tracking hyperparameters."""

__all__ = ["run_bayesopt_forever"]


def run_bayesopt_forever(*args, **kwargs):
    from genmatter.bayesopt.ax_runner import run_bayesopt_forever as _run

    return _run(*args, **kwargs)

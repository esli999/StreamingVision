"""CLI entry for custom-video BayesOpt."""

from __future__ import annotations

from pathlib import Path

from genmatter.bayesopt.ax_runner import run_bayesopt_forever
from genmatter.custom.config_schema import load_config


def run_bayesopt(
    video_id: str,
    *,
    tracking_config: Path,
    bayesopt_config: Path,
    run_id: str | None = None,
    tracking_set: list[str] | None = None,
    bayesopt_set: list[str] | None = None,
    resume: bool = True,
) -> None:
    cfg = load_config(tracking_config, dot_overrides=tracking_set or [])
    run_bayesopt_forever(
        video_id,
        tracking_config_path=tracking_config,
        bayesopt_config_path=bayesopt_config,
        run_id=run_id,
        tracking_overrides=tracking_set,
        bo_overrides=bayesopt_set,
        resume=resume,
    )

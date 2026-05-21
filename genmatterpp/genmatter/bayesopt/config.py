"""Load Bayesian optimization YAML configuration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_bayesopt_config(
    config_path: Path,
    *,
    video_id: str,
    dot_overrides: list[str] | None = None,
) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if dot_overrides:
        for item in dot_overrides:
            if "=" not in item:
                raise ValueError(f"Invalid --set: {item}")
            key_path, _, value_str = item.partition("=")
            keys = key_path.split(".")
            node = raw
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = _parse_scalar(value_str)

    run = raw.setdefault("run", {})
    out_tpl = str(run.get("output_root", "assets/custom_videos/{video_id}/bayesopt"))
    run["output_root"] = out_tpl.format(video_id=video_id)
    raw["video_id"] = video_id
    raw["_config_fingerprint"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return raw


def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def merge_tracking_overrides(bo_cfg: dict[str, Any], tracking_overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge fixed tracking settings from bayesopt config into overrides dict."""
    fixed = bo_cfg.get("tracking_fixed", {})
    return _deep_merge(fixed, tracking_overrides)


def _deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

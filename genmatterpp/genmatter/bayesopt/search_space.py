"""BO search space: all tunable DinoTrackingParams + DinoTrackingHyperparams."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Literal

import numpy as np

from genmatter.custom.config_schema import CustomConfig, TrackingHyperparamsConfig
from genmatter.tracking.dino import DinoTrackingHyperparams, DinoTrackingParams, dino_params_from_config


OBJECTIVE_NAME = "mean_gt_iou"
ValueType = Literal["float", "int", "bool"]

# Top-level tracking fields held fixed from custom_default.yaml (not in search_space).
_FIXED_TRACKING_FIELDS = frozenset(
    {
        "num_blobs",
        "num_hyperblobs",
        "datapoint_retain_pct",
        "dense_disable_outlier_prob",
        "random_seed",
        "use_sam_frame0",
        "focal_length",
    }
)

# Fields on DinoTrackingParams that BO may override from trial_params.
_TRACKING_FIELD_NAMES = frozenset(
    f.name
    for f in fields(DinoTrackingParams)
    if f.name not in ("hyperparams", "measure_fps") and f.name not in _FIXED_TRACKING_FIELDS
)

_HYPER_FIELD_NAMES = frozenset(f.name for f in fields(DinoTrackingHyperparams))


@dataclass(frozen=True)
class HyperparamSpec:
    name: str
    lower: float
    upper: float
    log_scale: bool = False
    value_type: ValueType = "float"

    def to_ax_dict(self) -> dict[str, Any]:
        if self.value_type == "bool":
            return {
                "name": self.name,
                "type": "choice",
                "values": [False, True],
                "value_type": "bool",
            }
        out: dict[str, Any] = {
            "name": self.name,
            "type": "range",
            "bounds": [float(self.lower), float(self.upper)],
            "log_scale": self.log_scale,
        }
        if self.value_type == "int":
            out["value_type"] = "int"
        return out

    def value_to_unit(self, value: float | bool) -> float:
        if self.value_type == "bool":
            return 1.0 if bool(value) else 0.0
        lo, hi = float(self.lower), float(self.upper)
        v = float(value)
        if self.log_scale:
            lo, hi = math.log(lo), math.log(hi)
            v = math.log(max(v, 1e-30))
        return float(np.clip((v - lo) / (hi - lo + 1e-30), 0.0, 1.0))

    def unit_to_value(self, unit: float) -> float | bool:
        u = float(np.clip(unit, 0.0, 1.0))
        if self.value_type == "bool":
            return u >= 0.5
        lo, hi = float(self.lower), float(self.upper)
        if self.log_scale:
            v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        else:
            v = lo + u * (hi - lo)
        if self.value_type == "int":
            return int(round(v))
        return float(v)


def specs_from_config(bo_cfg: dict[str, Any]) -> list[HyperparamSpec]:
    out: list[HyperparamSpec] = []
    for item in bo_cfg.get("search_space", []):
        vt = str(item.get("value_type", "float"))
        if vt not in ("float", "int", "bool"):
            raise ValueError(f"Unsupported value_type {vt!r} for {item.get('name')}")
        out.append(
            HyperparamSpec(
                name=str(item["name"]),
                lower=float(item["lower"]),
                upper=float(item["upper"]),
                log_scale=bool(item.get("log_scale", False)),
                value_type=vt,  # type: ignore[arg-type]
            )
        )
    return out


def _coerce_tracking_value(name: str, value: Any) -> Any:
    if name in ("dense_disable_outlier_prob",):
        return bool(value)
    if name in ("num_blobs", "num_hyperblobs", "init_gibbs_sweeps"):
        return int(round(float(value)))
    return float(value)


def params_to_hyperparams(
    params: dict[str, float | bool],
    base: TrackingHyperparamsConfig,
) -> DinoTrackingHyperparams:
    """Apply BO trial dict onto base tracking hyperparams."""
    hp = DinoTrackingHyperparams(
        sigma_F=base.sigma_F,
        outlier_prob=base.outlier_prob,
        outlier_velocity_gamma_shape=base.outlier_velocity_gamma_shape,
        outlier_velocity_gamma_rate=base.outlier_velocity_gamma_rate,
        alpha=base.alpha,
        beta=base.beta,
        sigma_H=base.sigma_H,
        sigma_V=base.sigma_V,
        translation_gaussian_scale=base.translation_gaussian_scale,
        translation_max_radius=base.translation_max_radius,
        translation_num_radii_cells=base.translation_num_radii_cells,
        translation_theta_step_deg=base.translation_theta_step_deg,
        rotation_vmf_kappa=base.rotation_vmf_kappa,
        rotation_angle_max_deg=base.rotation_angle_max_deg,
        rotation_angle_step_deg=base.rotation_angle_step_deg,
    )
    for k, v in params.items():
        if k in _HYPER_FIELD_NAMES:
            if k == "translation_num_radii_cells":
                setattr(hp, k, int(round(float(v))))
            else:
                setattr(hp, k, float(v))
    return hp


def build_tracking_params(
    cfg: CustomConfig,
    trial_params: dict[str, float | bool],
    *,
    measure_fps: bool,
) -> DinoTrackingParams:
    base = dino_params_from_config(cfg.tracking)
    hp = params_to_hyperparams(trial_params, cfg.tracking.hyperparams)

    overrides: dict[str, Any] = {}
    for name in _TRACKING_FIELD_NAMES:
        if name in trial_params:
            overrides[name] = _coerce_tracking_value(name, trial_params[name])

    return DinoTrackingParams(
        num_blobs=overrides.get("num_blobs", base.num_blobs),
        num_hyperblobs=overrides.get("num_hyperblobs", base.num_hyperblobs),
        datapoint_retain_pct=overrides.get("datapoint_retain_pct", base.datapoint_retain_pct),
        random_seed=base.random_seed,
        focal_length=overrides.get("focal_length", base.focal_length),
        use_sam_frame0=base.use_sam_frame0,
        init_gibbs_sweeps=overrides.get("init_gibbs_sweeps", base.init_gibbs_sweeps),
        tracking_outlier_prob=overrides.get(
            "tracking_outlier_prob", base.tracking_outlier_prob
        ),
        dense_disable_outlier_prob=overrides.get(
            "dense_disable_outlier_prob", base.dense_disable_outlier_prob
        ),
        measure_fps=measure_fps,
        hyperparams=hp,
    )


def full_resolved_params_dict(params: DinoTrackingParams) -> dict[str, Any]:
    """Flat dict of every model input used by run_dino_tracking (for trial logs)."""
    from dataclasses import asdict

    out: dict[str, Any] = asdict(params.hyperparams)
    for name in _TRACKING_FIELD_NAMES:
        out[name] = getattr(params, name)
    out["random_seed"] = params.random_seed
    out["use_sam_frame0"] = params.use_sam_frame0
    return out


def sample_sobol_candidates(
    specs: list[HyperparamSpec],
    count: int,
    seed: int,
) -> list[dict[str, float | bool]]:
    from scipy.stats import qmc

    d = len(specs)
    if d == 0:
        return []
    sampler = qmc.Sobol(d=d, scramble=True, seed=int(seed))
    n = int(2 ** math.ceil(math.log2(max(count, 1))))
    unit = sampler.random(n)[:count]
    out: list[dict[str, float | bool]] = []
    for row in unit:
        params = {
            spec.name: spec.unit_to_value(float(u)) for spec, u in zip(specs, row, strict=True)
        }
        out.append(params)
    return out

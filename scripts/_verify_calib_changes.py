"""Throwaway verification of the velvet-knitting calibration changes.

Pure-function unit checks (no GPU/tracker needed): composite objective math,
EmState round-trip with int group_frozen, sigma_V M-step proxy, split discipline,
finite-sigma_V emission, and the no-leakage guard. Run:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python scripts/_verify_calib_changes.py
"""
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "genmatterpp"))

import calibrate_consistency as cc          # noqa: E402
import _calibrate_helpers as ch             # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ok  " if cond else " FAIL ") + name)


# ---- Phase A: finite sigma_V seed (config), default.yaml unchanged ----
check("DAVIS_DINO_DEFAULTS sigma_V is finite (1e-1-ish)",
      np.isfinite(cc.DAVIS_DINO_DEFAULTS["sigma_V"]) and cc.DAVIS_DINO_DEFAULTS["sigma_V"] < 1.0e3)
check("_initial_em_state seeds finite sigma_V",
      np.isfinite(cc._initial_em_state().sigma_V) and cc._initial_em_state().sigma_V < 1e3)

# ---- Phase B: motion group + floors + EM constants ----
check("sigma_V in GROUPS['motion']", "sigma_V" in cc.GROUPS["motion"])
check("SANITY_FLOORS has sigma_V bounded", cc.SANITY_FLOORS.get("sigma_V") == (1.0e-4, 1.0e2))
check("EM_MAX_OUTER_ITERS deeper (>=10)", cc.EM_MAX_OUTER_ITERS >= 10)
check("EM_MAX_ALPHA_RETRIES finer (>=3)", cc.EM_MAX_ALPHA_RETRIES >= 3)
check("EM_GROUP_COOLDOWN defined", getattr(cc, "EM_GROUP_COOLDOWN", 0) >= 1)

# EmState round-trip with int group_frozen + legacy bool coercion
em = cc._initial_em_state()
check("group_frozen are ints (0)", all(isinstance(v, int) for v in em.group_frozen.values()))
d = em.to_dict()
d2 = dict(d); d2["group_frozen"] = {"cluster": True, "particle": False, "motion": True}  # legacy bools
em2 = cc._em_from_dict(d2)
check("legacy bool group_frozen coerces to int 1/0",
      em2.group_frozen["cluster"] == 1 and em2.group_frozen["particle"] == 0)

# _damp_em moves sigma_V toward a proposal
em_d = cc._damp_em(em, {"sigma_V": 1.0e-3}, 0.5)
check("_damp_em damps sigma_V toward proposal",
      abs(em_d.sigma_V - (0.5 * 1e-3 + 0.5 * em.sigma_V)) < 1e-9)

# sigma_V M-step proxy: trace(scatter)/3 over pooled datapoints, prior-centered
defaults = cc._defaults_for_priors()
check("_defaults_for_priors carries sigma_V", "sigma_V" in defaults)
# Build synthetic pooled velocity scatter: 3x3 with known trace, n datapoints.
scatter = np.diag([0.012, 0.012, 0.012]).astype(np.float32)   # trace 0.036
pooled = {"velocity": {"pooled_scatter": scatter, "pooled_n": 1000.0, "sizes": [1000]}}
out = ch.aggregate_posteriors({"v": pooled}, {"v": pooled}, defaults)
# map_variance(trace, 3*n, default): (100*0.1 + 0.036)/(100 + 3000) ~ 0.00324
exp = (100.0 * defaults["sigma_V"] + np.trace(scatter)) / (100.0 + 3.0 * 1000.0)
check("aggregate_posteriors emits finite sigma_V proxy",
      np.isfinite(out["sigma_V"]) and abs(out["sigma_V"] - exp) < 1e-4)

# ---- Phase C: composite objective math ----
# Synthetic: T=4 frames, N=20 datapoints (instances above the size-5 noise floor).
# cluster ids perfectly equal the SAM>0 foreground on a single instance.
T, N = 4, 20
z_sam = np.zeros((T, N), np.int32); z_sam[:, :10] = 1           # instance 1 on first 10 dps
hb = np.full((T, N), -1, np.int32); hb[:, :10] = 7             # cluster 7 == that instance
blob = np.full((T, N), -1, np.int32); blob[:, :10] = 3
labels = {"Z_sam": z_sam, "Z_motion": np.zeros((T, N), np.int32),
          "Z_dino": z_sam.copy(), "positions": np.zeros((T, N, 3), np.float32),
          "velocities": np.zeros((T, N, 3), np.float32),
          "features": np.zeros((T, N, 32), np.float32)}
rj = cc._score_video_region_J("syn", hb, np.arange(N), labels, reference="sam")["region_J"]
check("region_J == 1.0 on a perfect single-instance match", abs(rj - 1.0) < 1e-6)
inst = cc._instance_matched_J(hb, z_sam, T)
check("inst_J == 1.0 on a perfect single-instance match", abs(inst - 1.0) < 1e-6)
temp = cc._temporal_id_continuity(hb, z_sam, T)
check("temporal continuity == 1.0 (ids stable across frames)", abs(temp - 1.0) < 1e-6)
comp = cc._score_video_composite("syn", hb, blob, np.arange(N), labels)
check("composite finite in [0,1] on synthetic", np.isfinite(comp["composite"]) and 0 <= comp["composite"] <= 1)
check("composite components present",
      all(k in comp for k in ("region_J", "inst_J", "temporal", "motion_ari", "blob_J")))
# tiny instances (< min_inst_size) -> inst_J NaN -> composite renormalizes (robust)
small = cc._instance_matched_J(np.full((T, 8), 7, np.int32),
                               np.concatenate([np.ones((T, 4), np.int32), np.zeros((T, 4), np.int32)], 1), T)
check("inst_J returns NaN for sub-threshold instances (noise-robust)", not np.isfinite(small))
# instance-merge penalty: two equal instances collapsed into ONE cluster -> inst_J < region_J
z2 = np.zeros((T, N), np.int32); z2[:, :10] = 1; z2[:, 10:] = 2  # two instances of size 10
hb2 = np.full((T, N), 7, np.int32)                              # ONE cluster covers both
rj2 = cc._score_video_region_J("syn", hb2, np.arange(N), {**labels, "Z_sam": z2}, reference="sam")["region_J"]
inst2 = cc._instance_matched_J(hb2, z2, T)
check("inst_J penalizes instance MERGE (inst_J < region_J)", inst2 < rj2 - 1e-6)

# _self_sup_score honors _OBJECTIVE_MODE
cc._OBJECTIVE_MODE = "composite"
s_comp = cc._self_sup_score("syn", hb, blob, np.arange(N), labels)
check("objective composite -> score == composite", s_comp["score"] == s_comp["composite"])
cc._OBJECTIVE_MODE = "region_J"
s_rj = cc._self_sup_score("syn", hb, blob, np.arange(N), labels)
check("objective region_J -> score == region_J (kill-switch)", s_rj["score"] == s_rj["region_J"])
cc._OBJECTIVE_MODE = "composite"

# ---- Phase D / discipline ----
try:
    cc._assert_split_discipline(); ok = True
except AssertionError:
    ok = False
check("_assert_split_discipline passes", ok)
check("SELECT_VAL disjoint from CROSSVAL (M-step-pool discipline)",
      not (set(cc.SELECT_VAL_VIDEOS) & set(cc.CROSSVAL_VIDEOS)))

# ---- No-leakage: GT read raises outside phase_validate ----
leaked = False
try:
    cc._gt_dp_at_datapoints("blackswan", 1, np.arange(10))   # _GT_SCORING_ALLOWED False
except AssertionError:
    leaked = True
check("GT access asserts outside validate (no leakage)", leaked)
# reference='gt' also guarded
guarded = False
try:
    cc._score_video_region_J("blackswan", hb, np.arange(N), labels, reference="gt")
except AssertionError:
    guarded = True
check("reference='gt' region-J guarded outside validate", guarded)

# ---- Finite sigma_V reaches the emitted YAML; default.yaml semantics ----
em_emit = cc._initial_em_state()
infer = cc._infer_cfg(use_sam_frame0=True, num_blobs=128, num_hyperblobs=10, init_gibbs_sweeps=5)
cfg = cc._make_yaml_cfg(em_emit, infer)
check("_make_yaml_cfg writes finite sigma_V",
      np.isfinite(cfg["tracking"]["hyperparams"]["sigma_V"]) and
      cfg["tracking"]["hyperparams"]["sigma_V"] < 1e3)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES:", FAIL)
    sys.exit(1)
print("ALL CHECKS PASSED")

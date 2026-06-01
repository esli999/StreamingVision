"""Self-supervised per-Gibbs-layer hyperparameter calibration.

Hierarchical Bayesian EM with conjugate posteriors. Plan:
`~/.claude/plans/piped-discovering-pebble.md`.

Versus the previous calibrator (gentle-twirling-sedgewick.md):
- Replaces MLE M-step with weakly-informative conjugate priors (Normal-Inv-Gamma
  for variances, Inv-Wishart for scale matrices, empirical-Bayes CRP for alpha
  and beta), centered at GenMatter DAVIS DINO experiment defaults.
- Plumbs Psi_B/H/V, nu_B/H/V, mu_H through `genmatter_rt._build_hypers` via the
  `tracking.use_calibrated_priors` YAML flag, so calibrated values reach the
  tracker instead of being overwritten by the K-means seed.
- Loosens floors to numerical-stability only; raw proposals outside plausible
  ranges veto for that iter; the per-group accept-test provides the regression
  guard.
- 3-group hierarchical accept-test (cluster / particle / motion) replaces the
  all-or-nothing global accept.
- Tracker outliers (blob_a == -1) — not Z_sam == 0 — drive γ posterior.
- JAX-native sufficient-statistics via `jax.ops.segment_sum` + `jax.vmap` over
  T frames (in `_calibrate_helpers`); ARI in JAX with K_MAX-bound contingency
  segment_sum.

Phases:
  preflight       env / GPU / disk / DAVIS presence + synthetic JIT warmup
  pseudo_labels   per-video × every frame: depth+flow+DINO forward + Z_sam load
                  + within-instance DINO KMeans + motion KMeans → .npz/video
  plumbing_smoke  capture K-means seed Psi_*/nu_*/mu_H on test.mp4 → write
                  configs/streaming_general_smoke.yaml (Section 1 verifier)
  em              outer loop ≤ 5 iters: run tracker on all 38 vids → per-vid
                  JAX-vmap sufficient stats → MAP posterior → veto raw → damp
                  → per-group crossval accept-test
  validate        baseline (streaming_default.yaml) + chosen on all 38 vids →
                  S_fine/S_coarse/S_semantic vs cached pseudo-labels → gate →
                  emit configs/streaming_general.yaml + report.md
  all             chains the above
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_GENMATTERPP_ROOT = _REPO_ROOT / "genmatterpp"
if str(_GENMATTERPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_GENMATTERPP_ROOT))

# genmatterpp/config.py defaults DAVIS_RGB_PATH to genmatterpp/assets/... Point
# it at the repo-root assets/ where the data actually lives — set BEFORE any
# genmatter module is imported.
os.environ.setdefault("GENMATTER_DAVIS_DIR", str(_REPO_ROOT / "assets"))

sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from _calibrate_helpers import (  # noqa: E402
    aggregate_posteriors,
    consistency_score,
    gather_z_sam_at_grid,
    load_z_sam_for_frame,
    motion_kmeans,
    per_video_sufficient_stats,
    within_instance_dino_kmeans,
)


# ----------------------------------------------------------------------
# Calibration set + constants
# ----------------------------------------------------------------------

LOCAL_VIDEOS = (
    "gray_jacket", "jello_trim", "new_eagle_trim", "purple_jacket",
    "snake_trim", "test", "two_blocks_centered", "wine_swirl",
)

# ----------------------------------------------------------------------
# TRAIN / HELD-OUT split (the methodological core: NO test-set tuning).
#
# Everything that LEARNS or SELECTS — the EM main pass, the conjugate M-step, the
# SAEM accept-test, AND the phase_select scalar grid — sees ONLY TRAIN_VIDEOS,
# scored self-supervised (region-J of the CLUSTER foreground vs cached Z_sam>0,
# never GT). HELDOUT_VIDEOS are touched exactly ONCE, in phase_validate, as the
# generalization gate against true DAVIS GT. All 5 demo videos sit in HELD-OUT,
# so their rendered quality measures generalization — not memorization. ONE
# global config comes out; there are no per-video knobs anywhere.
#
# 38 cached videos = 30 DAVIS + 8 local. Split: TRAIN 23 (16 DAVIS + 7 local),
# HELD-OUT 15 (14 DAVIS + wine_swirl). DAVIS lists are diverse on both sides
# (rigid vehicles, deformable animals, humans, water) so each split is
# representative.
# ----------------------------------------------------------------------
TRAIN_VIDEOS = (
    # 16 DAVIS (learning + selection only)
    "bmx-trees", "breakdance", "camel", "dance-twirl", "dogs-jump",
    "drift-straight", "goat", "india", "lab-coat", "loading", "mbike-trick",
    "motocross-jump", "paragliding-launch", "pigs", "shooting", "soapbox",
    # 7 local
    "gray_jacket", "jello_trim", "new_eagle_trim", "purple_jacket",
    "snake_trim", "test", "two_blocks_centered",
)
HELDOUT_VIDEOS = (
    # 4 DAVIS demos (final GT gate only)
    "blackswan", "car-roundabout", "car-shadow", "judo",
    # 10 more held-out DAVIS
    "bike-packing", "cows", "dog", "drift-chicane", "gold-fish",
    "horsejump-high", "kite-surf", "libby", "parkour", "scooter-black",
    # 1 local demo (no GT — held out for the demo render, excluded from the GT gate)
    "wine_swirl",
)
# The 5 streaming-demo videos — ALL must be held out so demo quality == generalization.
DEMO_VIDEOS = ("car-roundabout", "car-shadow", "blackswan", "judo", "wine_swirl")

# SAEM accept-test subsample (8 videos, ⊆ TRAIN): 6 diverse-motion DAVIS + 2 local.
# Gates each per-group SAEM step on the SAME self-supervised region-J objective as
# the EM loop; ⊆ TRAIN so the accept-test never sees held-out data.
CROSSVAL_VIDEOS = ("camel", "drift-straight", "dance-twirl", "motocross-jump",
                   "breakdance", "mbike-trick", "test", "two_blocks_centered")

# Inner self-supervised model-selection scoring subset (⊆ TRAIN, DISJOINT from
# CROSSVAL so the selection scorer and the SAEM accept-test never share a video).
# phase_select scores each grid combo's learned hypers by median self-supervised
# region-J on these (reference="sam", never GT, never HELD-OUT).
SELECT_VAL_VIDEOS = ("bmx-trees", "dogs-jump", "goat", "india", "lab-coat",
                     "loading", "pigs", "soapbox", "new_eagle_trim", "snake_trim")


def _assert_split_discipline() -> None:
    """Fail fast if the train/held-out discipline is ever violated. Guarantees:
    TRAIN ∩ HELD-OUT = ∅; all 5 demos ∈ HELD-OUT; CROSSVAL ⊆ TRAIN;
    SELECT_VAL ⊆ TRAIN and SELECT_VAL ∩ CROSSVAL = ∅; 38 unique videos total."""
    train, held = set(TRAIN_VIDEOS), set(HELDOUT_VIDEOS)
    assert not (train & held), f"TRAIN∩HELDOUT must be empty: {sorted(train & held)}"
    assert set(DEMO_VIDEOS) <= held, \
        f"all demo videos must be HELD-OUT: {sorted(set(DEMO_VIDEOS) - held)} are not"
    assert set(CROSSVAL_VIDEOS) <= train, \
        f"CROSSVAL must be ⊆ TRAIN: {sorted(set(CROSSVAL_VIDEOS) - train)} are not"
    assert set(SELECT_VAL_VIDEOS) <= train, \
        f"SELECT_VAL must be ⊆ TRAIN: {sorted(set(SELECT_VAL_VIDEOS) - train)} are not"
    assert not (set(SELECT_VAL_VIDEOS) & set(CROSSVAL_VIDEOS)), \
        "SELECT_VAL and CROSSVAL must be disjoint (separate selection/accept videos)"
    assert len(train) == len(TRAIN_VIDEOS) and len(held) == len(HELDOUT_VIDEOS), \
        "duplicate video in a split tuple"
    assert len(train | held) == len(TRAIN_VIDEOS) + len(HELDOUT_VIDEOS) == 38, \
        f"split must cover 38 unique videos, got {len(train | held)}"


# ----------------------------------------------------------------------
# Hard no-leakage guard: true DAVIS GT may be read ONLY inside phase_validate
# (the held-out generalization gate). Any GT scoring reached from learning or
# model selection (phase_em / phase_select / _accept_groups) raises immediately.
# ----------------------------------------------------------------------
_GT_SCORING_ALLOWED = False


@contextlib.contextmanager
def _gt_scoring_enabled():
    """Scope inside which reference='gt' region-J + DAVIS_SEGMASKS reads are
    permitted (phase_validate only). Outside it, any GT access asserts."""
    global _GT_SCORING_ALLOWED
    prev = _GT_SCORING_ALLOWED
    _GT_SCORING_ALLOWED = True
    try:
        yield
    finally:
        _GT_SCORING_ALLOWED = prev


# ----------------------------------------------------------------------
# Held-out GT lock (the durable anti-hacking guard). `_GT_SCORING_ALLOWED`
# alone is too coarse: it is a single on/off flag any caller can flip, so a dev
# script could read HELD-OUT GT to inform a DESIGN choice — methodological
# leakage even when no knob is fit to it. We therefore split the permission:
#   * TRAIN GT  — readable whenever `_GT_SCORING_ALLOWED` is set (development
#                 diagnostics + the TRAIN design re-derivation may use it).
#   * HELD-OUT GT — readable ONLY inside the single sanctioned generalization
#                 gate (`_heldout_gate_enabled()`, entered by phase_validate*).
# The check fails CLOSED: any GT read of a HELDOUT_VIDEOS member outside that
# scope raises, so the held-out numbers can be produced exactly once, through
# the gate, and the design can never be (even implicitly) tuned to them.
# ----------------------------------------------------------------------
_HELDOUT_GATE_ACTIVE = False


@contextlib.contextmanager
def _heldout_gate_enabled():
    """Scope inside which HELD-OUT video GT may be read — the one sanctioned
    generalization gate (phase_validate / phase_validate_augmentation). Outside
    it, reading a HELDOUT_VIDEOS GT mask asserts even when `_GT_SCORING_ALLOWED`
    is True. Nest inside `_gt_scoring_enabled()`."""
    global _HELDOUT_GATE_ACTIVE
    prev = _HELDOUT_GATE_ACTIVE
    _HELDOUT_GATE_ACTIVE = True
    try:
        yield
    finally:
        _HELDOUT_GATE_ACTIVE = prev


def _assert_gt_readable(vid: str) -> None:
    """The single chokepoint every true-DAVIS-GT read passes through. Fail
    CLOSED: any GT read requires `_GT_SCORING_ALLOWED`; a HELD-OUT video's GT
    additionally requires the sanctioned gate scope. This makes held-out GT
    physically unreadable outside phase_validate* — design + diagnostics are
    confined to TRAIN."""
    assert _GT_SCORING_ALLOWED, (
        "true DAVIS GT accessed outside a GT-scoring scope — train/held-out "
        "leakage. Learning + selection are self-supervised (reference='sam') only.")
    if vid in HELDOUT_VIDEOS:
        assert _HELDOUT_GATE_ACTIVE, (
            f"HELD-OUT GT for {vid!r} read outside the sanctioned gate "
            "(_heldout_gate_enabled, entered only by phase_validate*). Held-out "
            "GT is gate-only; design + diagnostics must use TRAIN videos.")

# Initialization: GenMatter DAVIS DINO experiment defaults (see
# genmatterpp/genmatter/tracking/dino.py:DinoTrackingHyperparams). The
# material difference from streaming_default.yaml is sigma_F: 2.0 -> 0.2.
#
# sigma_V (the velocity-prior variance) is seeded FINITE here — NOT at the
# streaming_default.yaml 1e14. At 1e14 the velocity anchor in gibbs_blob_means
# (inference.py:671-685, the `lhs_affine/sigmaV` + `rhs_affine/sigmaV` terms) and
# the constant-velocity prior in gibbs_blob_vel_means (:536) are dead, so the
# per-frame predict (`next_blob_means = blob_vel_means + blob_means`) is washed
# out by gibbs_blob_means re-fitting from scratch → blob means chase leaked
# background ("bleed"). A finite seed makes the velocity anchor LIVE from EM
# iter 0 (it also unfreezes the motion group, whose proposals used to regress
# while the velocity model was degenerate). The M-step then learns sigma_V under
# the accept-test (motion group), and phase_select can fall back to ~1e14 if the
# anchor regresses. SANITY_FLOORS["sigma_V"] bounds it. The finite value ships
# ONLY in the emitted streaming_general.yaml / streaming_render_v2.yaml; the
# vendored production live path keeps streaming_default.yaml's sigma_V=1e14, so
# that path stays byte-for-byte identical (verified bit-exact).
_SIGMA_V_SEED = 1.0e-1   # finite velocity-prior variance seed (was 1e14 = anchor off)
DAVIS_DINO_DEFAULTS = {
    "sigma_F": 0.2,
    "sigma_F_H": 2.0,
    "sigma_V": _SIGMA_V_SEED,
    "outlier_prob": 5.0,
    "gamma_shape": 5.0,
    "gamma_rate": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "sigma_H": 25.0,
    "translation_max_radius": 0.35,
    "translation_gaussian_scale": 0.2,
    # Prior centering for the Wishart MAP. nu_0=4 makes it weakly informative
    # so the posterior tracks data when n is large.
    "Psi_B": np.eye(3, dtype=np.float32) * 0.01,
    "Psi_H": np.eye(3, dtype=np.float32) * 1.0,
    "Psi_V": np.eye(3, dtype=np.float32) * 0.001,
    "mu_H": np.array([0.0, 0.0, 2.5], dtype=np.float32),
    "nu_B": 45.0,   # ≈ num_datapoints / num_blobs = 2925 / 64
    "nu_H": 16.0,   # ≈ num_blobs / num_hyperblobs = 64 / 4
    "nu_V": 45.0,
}

# Sanity floors: NUMERICAL-STABILITY only (plan §4). Much looser than the
# old "floor at streaming_default" — the per-group accept-test catches
# regressions, so floors only guard against degenerate proposals. A RAW
# proposal outside [lo, hi] gets NaN'd → damping short-circuits → that layer
# freezes for the iter.
SANITY_FLOORS: Dict[str, Tuple[float, float]] = {
    "sigma_F":      (0.05, 50.0),
    "sigma_F_H":    (0.05, 50.0),
    "outlier_prob": (0.5, 30.0),
    "gamma_shape":  (0.5, 20.0),
    "gamma_rate":   (0.05, 20.0),
    "alpha":        (0.1, 30.0),
    "beta":         (0.1, 30.0),
    "sigma_H":      (1.0, 100.0),
    # sigma_V: the learnable velocity-prior variance (motion group). Floor stops
    # a degenerate M-step from driving the anchor to 0 (rank-deficient P_post);
    # ceiling 1e2 keeps a finite anchor (≫ that is effectively "off"). The
    # phase_select grid can still pick the 1e14-equivalent fallback explicitly.
    "sigma_V":      (1.0e-4, 1.0e2),
    "nu_B":         (1.0, 1e5),
    "nu_H":         (1.0, 1e3),
    "nu_V":         (1.0, 1e5),
    "translation_max_radius":    (0.01, 5.0),
    "translation_gaussian_scale": (0.01, 2.0),
}
PSI_DIAG_FLOOR = (1e-4, 1e2)        # per-axis diagonal range for Psi_*
MU_H_RANGE = (-5.0, 5.0)             # per-axis range for mu_H

# Per-group key sets. Plan §3, with three estimators dropped after mini-EM
# revealed scale mismatches:
#   • outlier_prob: empirical fraction (~0.2) is on a probability scale, but
#     the model uses outlier_prob as a multiplicative weight (default 5.0).
#   • gamma_shape/gamma_rate: MoM on outlier-velocity magnitudes is unstable
#     because the outlier-velocity distribution isn't gamma (heavy-tailed
#     mixture). Yields implausibly low shape (~0.3).
#   • sigma_H: empirical centroid std (~0.88 m) is on physical-distance units,
#     but the model's sigma_H=25.0 is an effectively-flat prior (semantically
#     similar to sigma_V=1e14). Tightening regresses the live demo.
# These four stay at GenMatter DAVIS DINO defaults. Their MAP proposals are
# still computed and logged for diagnostics, just not applied via accept-test.
GROUPS: Dict[str, Tuple[str, ...]] = {
    "cluster":  ("sigma_F", "Psi_B", "nu_B", "alpha"),
    "particle": ("sigma_F_H", "Psi_H", "nu_H", "beta", "mu_H"),
    # sigma_V joins the motion group: now that it is seeded finite (the velocity
    # anchor is live), its closed-form M-step (aggregate_posteriors, the
    # trace(pooled_velocity_scatter)/3 isotropic proxy) is accept-tested
    # alongside Psi_V / nu_V / translation. This is THE #1 accuracy lever — it
    # learns how hard the per-frame blob mean anchors to its constant-velocity
    # prediction (less bleed).
    "motion":   ("sigma_V", "Psi_V", "nu_V",
                 "translation_max_radius", "translation_gaussian_scale"),
}

# EM hyperparameters. Deeper + less conservative than v1 (the motion group is now
# learnable, so EM needs room to converge it): more outer iters and a finer α
# ladder. The per-group accept-test is the regression guard, so running longer is
# safe — a group that fluctuates is re-attempted after a short cooldown rather
# than permanently frozen (see _accept_groups + EM_GROUP_COOLDOWN).
EM_ALPHA = 0.3
EM_ALPHA_RETRY_DIVISOR = 2.0
EM_MAX_OUTER_ITERS = 10      # was 5 — deeper EM to converge the live motion group
EM_MAX_ALPHA_RETRIES = 3     # was 2 — finer α ladder: 0.3, 0.15, 0.075, 0.0375
EM_PLATEAU_TOL = 0.005
EM_GROUP_COOLDOWN = 2        # iters a regressing group sits out before re-attempt
EM_VAL_PATIENCE = 3          # outer iters w/o val-objective improvement → early stop

# Stride-8 grid the tracker / J-mean evaluator operate on: WORK_HW (360,640) //
# STRIDE 8 = (45, 80). Hardcoded so the em/validate path never imports the live
# runner (which would pull in the torch perception models). Matches the
# img_dims=(45,80) the GenMatter++ evaluator is called with.
GRID_H, GRID_W = 45, 80

# ---- Inference-strategy defaults (the "much stronger inference" — ONE global
# config). SAM2 is a video segmenter whose per-frame masks are the dense
# self-supervised target, AND its frame-0 mask seeds a semantic init; the bigger
# blob budget + much deeper Gibbs E-step de-bias the conjugate M-step (the
# SAEM/MCEM lever). All CLI-overridable. num_blobs stays <= K_MAX-2 (=254) so the
# M-step segment_sum never drops a tracker blob.
INFER_USE_SAM_FRAME0 = True
INFER_NUM_BLOBS = 128
INFER_NUM_HYPERBLOBS = 4        # k-means fallback only; SAM init sets it dynamically
INFER_INIT_GIBBS_SWEEPS = 30    # frame-0 warmup sweeps (converge the SAM-seeded init)
DEFAULT_NUM_GIBBS_SWEEPS = 25   # per-frame Level-1 measurement-update depth

# ---- Architectural-structure flags (the v2 cluster-freeze fix). These are
# GLOBAL STRUCTURE, not per-video knobs. They default ON here so the conjugate
# M-step re-fits the hypers UNDER the structure we actually ship, and so they get
# baked into EVERY _make_yaml_cfg output (EM, accept-test, validate-chosen pass,
# emitted YAML) — closing the calibration/shipping divergence (the EM used to
# silently run the vendored structure while the demo ran the fixed one). The
# vendored production live path still defaults them OFF
# (genmatter_rt._*_DEFAULT), so that path stays bit-exact. See
# memory: project-cluster-freeze-fix.
INFER_FEATURE_AWARE_FINAL = True   # final blob assign uses full pos+feature likelihood
INFER_FINAL_OUTLIER = False        # no outlier injection on the final assignment step
INFER_FREEZE_HYPERBLOB = True      # keep the frame-0 seeded blob->hyperblob map (THE fix)
INFER_PURE_OBJECT_SEED = True      # rebuild CHM so each object hyperblob is pure at frame 0
INFER_BLOB_MEANS_UPDATES = 15      # per-frame gibbs_blob_means refinement count (one global)
# Anti-drift feature update (Phase A re-optimization). feature_update_damping in
# [0, 1] blends each per-frame Gibbs DINO-feature update back toward the frame-0
# anchor (d=0 == freeze the appearance, d=1 == vendored pure-Gibbs). This is the
# DAMPED generalization of the freeze_blob_features fix; it is SELECTED on TRAIN in
# phase_select and learned-under in phase_em. 1.0 here keeps the default bit-exact;
# the legacy static freeze flag is kept for back-compat (off by default — damping
# supersedes it).
INFER_FREEZE_BLOB_FEATURES = False
INFER_FEATURE_UPDATE_DAMPING = 1.0
# INFERENCE feature-TEMPERATURE on the FINAL assignment (Phase 1 re-optimization).
# final_feature_temp (tau) scales the DINO-feature term at the step that WRITES the
# labels: logits = pos + vel + tau*feat + log_mix. tau=1 reproduces the vendored
# feature-aware-final bit-for-bit; tau>1 up-weights the expert features (toward the
# frozen-centroid classifier that beats the drifting tracker). SELECTED on TRAIN in
# phase_select; default 1.0 keeps the shipped numerics. final_assignment_anchor scores
# the re-impl feature term vs the FROZEN frame-0 anchor (no-op at damping d=0).
INFER_FINAL_FEATURE_TEMP = 1.0
INFER_FINAL_ASSIGNMENT_ANCHOR = False
# TokenCut self-supervised seed augmentation (scripts/tokencut). Default-off keeps
# the cfg bit-exact; --tokencut-seed-augment turns it ON for the WHOLE pipeline so
# the conjugate hypers are LEARNED-UNDER and SELECTED-UNDER the augmented seeds
# (the coherent methodology, not a bolt-on). Self-supervised (TokenCut+SAM2, NO GT)
# → legal in phase_em/phase_select; GT is still read only in phase_validate. The
# augmented grids are cached to disk (deterministic per vid+knobs) so the EM stays
# fast. Knobs come from the self-supervised select (runs/.../tokencut_knobs.json).
INFER_TOKENCUT_SEED_AUGMENT = False
INFER_TOKENCUT_KNOBS = None

OUT_ROOT = _REPO_ROOT / "runs" / "calibrate_consistency"
LABELS_DIR = OUT_ROOT / "labels"
GENERAL_YAML = _REPO_ROOT / "configs" / "streaming_general.yaml"
GENERAL_SMOKE_YAML = _REPO_ROOT / "configs" / "streaming_general_smoke.yaml"
DEFAULT_YAML = _REPO_ROOT / "configs" / "streaming_default.yaml"
# Demo-facing config (the streaming grid render reads this). Same calibrated
# hypers + GLOBAL structural flags as streaming_general.yaml, with the demo's
# richer k-means-fallback num_hyperblobs. Emitted on a passing gate — ONE global
# config, NO per-video keys.
RENDER_V2_YAML = _REPO_ROOT / "configs" / "streaming_render_v2.yaml"
RENDER_V2_NUM_HYPERBLOBS = 10   # demo k-means fallback richness (SAM/GT seed overrides dynamically)
JAX_CACHE_DIR = str(_REPO_ROOT / ".jax_cache")
CUSTOM_VIDEOS_DIR = _REPO_ROOT / "assets" / "custom_videos"


# ----------------------------------------------------------------------
# Logging helpers (copied from calibrate_general.py)
# ----------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[calibrate_consistency {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Video discovery (38: 8 local + 30 DAVIS)
# ----------------------------------------------------------------------

@dataclass
class VideoEntry:
    vid: str
    kind: str             # "local" | "davis"
    source: Path
    sam_dir: Path         # where Z_sam PNGs live


def _resolve_local_source(vid: str) -> Optional[Path]:
    root = CUSTOM_VIDEOS_DIR / vid
    if not root.is_dir():
        return None
    for name in ("source.mp4", "source.mov", "source.MOV"):
        p = root / name
        if p.is_file():
            return p
    rf = root / "rgb_frames" / vid
    if rf.is_dir():
        return rf
    return None


def _resolve_local_sam_dir(vid: str) -> Path:
    return CUSTOM_VIDEOS_DIR / vid / "pseudo_gt_sam" / "segmasks" / vid


def _resolve_davis_source(vid: str) -> Optional[Path]:
    import config as gm_config
    rgb_dir = Path(gm_config.DAVIS_RGB_PATH) / vid
    if rgb_dir.is_dir() and any(rgb_dir.iterdir()):
        return rgb_dir
    return None


def _resolve_davis_sam_dir(vid: str) -> Path:
    return (_REPO_ROOT / "assets" / "tapvid_davis_30_videos_processed" /
            "sam2_propagated" / vid)


def discover_videos() -> Dict[str, VideoEntry]:
    out: Dict[str, VideoEntry] = {}
    for v in LOCAL_VIDEOS:
        src = _resolve_local_source(v)
        if src is None:
            _log(f"discover: local '{v}' source missing — skipping")
            continue
        out[v] = VideoEntry(vid=v, kind="local", source=src,
                             sam_dir=_resolve_local_sam_dir(v))
    try:
        import config as gm_config
        davis_names = gm_config.TAPVID_DAVIS_VIDEO_NAMES
    except Exception:
        davis_names = ()
    for v in davis_names:
        src = _resolve_davis_source(v)
        if src is None:
            _log(f"discover: DAVIS '{v}' frames missing — skipping")
            continue
        sam_dir = _resolve_davis_sam_dir(v)
        if not sam_dir.is_dir():
            _log(f"discover: DAVIS '{v}' SAM2 propagated dir missing — skipping")
            continue
        out[v] = VideoEntry(vid=v, kind="davis", source=src, sam_dir=sam_dir)
    return out


# ----------------------------------------------------------------------
# SAM2 self-supervision plumbing (annotations paths + frame-0 init grid)
# ----------------------------------------------------------------------

def _sam_annotations_path(entry: VideoEntry) -> str:
    """Parent dir the GenMatter++ evaluator joins with ``vid`` to read the SAM2
    per-frame masks (the self-supervised J-mean target).
    ``get_segmentation_mask`` does ``os.path.join(annotations_path, vid, f"{k:05d}.png")``,
    so we return the parent of ``entry.sam_dir``'s ``{vid}`` leaf — for DAVIS
    ``.../sam2_propagated`` and for local ``.../pseudo_gt_sam/segmasks``. The
    evaluator binarizes ``frame_seg >= 1`` so the uint16 multi-instance SAM2
    masks collapse to a single foreground (matter-vs-background) target."""
    return str(entry.sam_dir.parent)


def _davis_gt_path(vid: str) -> str:
    """Real DAVIS GT segmasks dir — used ONLY for the final validation report
    (the true benchmark), never for the self-supervised EM objective. ``vid`` is
    required so the held-out lock can verify the caller is inside the sanctioned
    gate (the matter-J GT read at the validate call site)."""
    _assert_gt_readable(vid)   # GT scope required; held-out GT gate-only
    import config as gm_config
    return str(gm_config.DAVIS_SEGMASKS_PATH)


def _frame0_sam_grid(entry: VideoEntry, labels_TN: dict):
    """Build the (GRID_H, GRID_W, 3) RGB SAM mask for the cache's FIRST frame, to
    seed ``init_state``'s semantic SAM-frame-0 branch. The cache skips video
    frame 0 (no flow), so its first positions are at ``frame_idx[0]`` (==1
    typically); we load the SAM mask for that exact frame so the init mask and
    init positions are 0-frame aligned (strictly better than the live demo's
    1-frame offset). Returns None if the mask is missing → init_state falls back
    to flat k-means."""
    import genmatter_rt
    fi = labels_TN.get("frame_idx")
    frame0 = int(np.asarray(fi).reshape(-1)[0]) if fi is not None and np.size(fi) else 1
    mask = load_z_sam_for_frame(entry.vid, frame0, kind=entry.kind, repo_root=_REPO_ROOT)
    if mask is None:
        return None
    return genmatter_rt.instance_mask_to_rgb_grid(mask, GRID_H, GRID_W)


def _build_sam_grids(videos: Dict[str, VideoEntry],
                     labels_cache: Dict[str, dict],
                     use_sam_frame0: bool) -> Dict[str, object]:
    """Precompute each video's frame-0 SAM RGB grid once (cache-invariant across
    EM iters / accept-test trials). Empty dict when SAM init is disabled."""
    if not use_sam_frame0:
        return {}
    grids: Dict[str, object] = {}
    n_ok = 0
    for vid, labels in labels_cache.items():
        grid = _frame0_sam_grid(videos[vid], labels)
        grids[vid] = grid
        n_ok += int(grid is not None)
    _log(f"SAM frame-0 init: built grids for {n_ok}/{len(labels_cache)} videos "
         f"(missing → k-means fallback)")
    return grids


def _infer_cfg(*, use_sam_frame0: bool, num_blobs: int, num_hyperblobs: int,
               init_gibbs_sweeps: int,
               feature_aware_final: bool = INFER_FEATURE_AWARE_FINAL,
               final_outlier: bool = INFER_FINAL_OUTLIER,
               freeze_hyperblob: bool = INFER_FREEZE_HYPERBLOB,
               pure_object_seed: bool = INFER_PURE_OBJECT_SEED,
               blob_means_updates: int = INFER_BLOB_MEANS_UPDATES,
               freeze_blob_features: bool = INFER_FREEZE_BLOB_FEATURES,
               feature_update_damping: float = INFER_FEATURE_UPDATE_DAMPING,
               final_feature_temp: float = INFER_FINAL_FEATURE_TEMP,
               final_assignment_anchor: bool = INFER_FINAL_ASSIGNMENT_ANCHOR,
               tokencut_seed_augment=None, tokencut_knobs=None) -> dict:
    """Bundle the inference-strategy + architectural-structure knobs threaded into
    _make_yaml_cfg + the tracker. ONE global config — the same dict drives EM,
    accept-test, validate (chosen pass), and the emitted YAML.

    The architectural-structure flags (feature_aware_final / final_outlier /
    freeze_hyperblob / pure_object_seed / blob_means_updates) are the v2
    cluster-freeze fix; they default ON so calibration LEARNS the conjugate hypers
    UNDER the structure we ship (the previously-missing step). They are GLOBAL —
    never per-video. The vendored live path defaults them OFF (bit-exact)."""
    return {"use_sam_frame0": bool(use_sam_frame0),
            "num_blobs": int(num_blobs),
            "num_hyperblobs": int(num_hyperblobs),
            "init_gibbs_sweeps": int(init_gibbs_sweeps),
            "feature_aware_final": bool(feature_aware_final),
            "final_outlier": bool(final_outlier),
            "freeze_hyperblob": bool(freeze_hyperblob),
            "pure_object_seed": bool(pure_object_seed),
            "blob_means_updates": int(blob_means_updates),
            "freeze_blob_features": bool(freeze_blob_features),
            "feature_update_damping": float(feature_update_damping),
            "final_feature_temp": float(final_feature_temp),
            "final_assignment_anchor": bool(final_assignment_anchor),
            # Default to the module globals (set once from the CLI in main) so EM /
            # select / validate-chosen all build cfgs UNDER the augmentation; the
            # shipping-baseline path passes tokencut_seed_augment EXPLICITLY from the
            # baseline YAML (reverted → off) so the validate 'before' stays un-augmented.
            "tokencut_seed_augment": bool(INFER_TOKENCUT_SEED_AUGMENT
                                          if tokencut_seed_augment is None else tokencut_seed_augment),
            "tokencut_knobs": (INFER_TOKENCUT_KNOBS if tokencut_knobs is None else tokencut_knobs)}


# Inference config for the SHIPPING baseline (validate's honest "before"): the
# VENDORED structure exactly as the prior config deploys — flat k-means init, 64
# blobs, the default 15-sweep warmup, 1 sweep/frame, AND the architectural-
# structure flags OFF (feature_aware_final=False, final_outlier=True,
# freeze_hyperblob=False, pure_object_seed=False) so the baseline reflects the
# pre-fix cluster behaviour. (phase_validate overrides this with the actual
# currently-shipping streaming_general.yaml when that file is present.)
SHIPPING_INFER = _infer_cfg(use_sam_frame0=False, num_blobs=64,
                            num_hyperblobs=4, init_gibbs_sweeps=15,
                            feature_aware_final=False, final_outlier=True,
                            freeze_hyperblob=False, pure_object_seed=False,
                            blob_means_updates=15)


# ----------------------------------------------------------------------
# Lazy live-runner import
# ----------------------------------------------------------------------

_HAVE_LIVE = False


def _import_live_runner():
    global _HAVE_LIVE
    if _HAVE_LIVE:
        import run_streaming_live as live  # type: ignore
        return live
    from genmatter.tracking.dino import configure_jax_cache
    configure_jax_cache(JAX_CACHE_DIR)
    import run_streaming_live as live  # type: ignore  (loads depth/flow/DINO)
    _HAVE_LIVE = True
    return live


def _ensure_jax_setup() -> None:
    """Configure the persistent JAX compile cache WITHOUT loading the perception
    models. The em/validate phases run the tracker from cached perception outputs
    (the pseudo-label .npz), so they never need depth/flow/DINO — skipping that
    import also frees several GB of GPU memory for XLA."""
    from genmatter.tracking.dino import configure_jax_cache
    configure_jax_cache(JAX_CACHE_DIR)


# ----------------------------------------------------------------------
# Phase 1: preflight
# ----------------------------------------------------------------------

def phase_preflight(out_root: Path) -> dict:
    """Env / GPU / disk / DAVIS presence + JAX warmup (~40-60 s warm cache)."""
    import torch
    info: dict = {"phase": "preflight"}
    info["python"] = sys.version.split(" ", 1)[0]
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        info["cuda_mem_free_gb"] = float(free / 1024 ** 3)
        info["cuda_mem_total_gb"] = float(total / 1024 ** 3)
    stat = shutil.disk_usage(str(_REPO_ROOT))
    info["disk_free_gb"] = float(stat.free / 1024 ** 3)
    info["disk_total_gb"] = float(stat.total / 1024 ** 3)
    info["jax_cache_dir"] = JAX_CACHE_DIR

    try:
        import config as gm_config
        rgb_root = Path(gm_config.DAVIS_RGB_PATH)
        sam_root = _REPO_ROOT / "assets" / "tapvid_davis_30_videos_processed" / "sam2_propagated"
        info["davis_rgb_root"] = str(rgb_root)
        info["davis_sam2_root"] = str(sam_root)
        rgb_present = 0
        sam_present = 0
        for v in gm_config.TAPVID_DAVIS_VIDEO_NAMES:
            if (rgb_root / v / "00000.jpg").is_file():
                rgb_present += 1
            if (sam_root / v / "00000.png").is_file():
                sam_present += 1
        info["davis_rgb_present"] = rgb_present
        info["davis_sam2_present"] = sam_present
        info["davis_expected"] = len(gm_config.TAPVID_DAVIS_VIDEO_NAMES)
    except Exception as e:
        info["davis_inventory_error"] = repr(e)

    videos = discover_videos()
    info["videos_discovered"] = {v: {"kind": e.kind, "source": str(e.source),
                                       "sam_dir": str(e.sam_dir)}
                                   for v, e in videos.items()}
    info["n_videos"] = len(videos)

    _log("preflight: synthetic init+step JIT warm-up (primes .jax_cache)")
    t0 = time.monotonic()
    try:
        import genmatter_rt
        import jax
        rng = np.random.default_rng(0)
        n = genmatter_rt.N_KEEP
        positions = (rng.normal(size=(n, 3)).astype(np.float32) *
                     np.array([2.0, 1.0, 0.5]) + np.array([0.0, 0.0, 3.0]))
        velocities = rng.normal(size=(n, 3)).astype(np.float32) * 0.05
        features = rng.normal(size=(n, genmatter_rt.FEATURE_DIM)).astype(np.float32)
        key = jax.random.PRNGKey(0)
        yaml_cfg = genmatter_rt.load_yaml_hypers(DEFAULT_YAML)
        yaml_cfg["tracking"]["num_blobs"] = 64
        yaml_cfg["tracking"]["num_hyperblobs"] = 4
        yaml_cfg["tracking"]["use_sam_frame0"] = False
        state, key = genmatter_rt.init_state(positions, velocities, features, key,
                                              yaml_cfg=yaml_cfg, num_blobs=64,
                                              num_hyperblobs=4, verbose=False)
        state, key = genmatter_rt.step(state, positions, velocities, features, key)
        state.datapoints_state.blob_assignments.block_until_ready()
        info["warmup_sec"] = float(time.monotonic() - t0)
        info["warmup_ok"] = True
    except Exception as e:
        info["warmup_ok"] = False
        info["warmup_error"] = repr(e)
        traceback.print_exc()

    # Train/held-out perception-cache inventory. The select/em/validate fast path
    # reads these .npz; they are hyper-invariant and are NEVER regenerated by
    # those phases (rebuild via --phase pseudo_labels if missing).
    _assert_split_discipline()
    missing_train = [v for v in TRAIN_VIDEOS if not (LABELS_DIR / f"{v}.npz").is_file()]
    missing_held = [v for v in HELDOUT_VIDEOS if not (LABELS_DIR / f"{v}.npz").is_file()]
    info["labels_dir"] = str(LABELS_DIR)
    info["train_npz_present"] = len(TRAIN_VIDEOS) - len(missing_train)
    info["heldout_npz_present"] = len(HELDOUT_VIDEOS) - len(missing_held)
    info["train_npz_missing"] = missing_train
    info["heldout_npz_missing"] = missing_held
    info["split_ok"] = True
    if missing_train or missing_held:
        _log(f"preflight WARNING: missing perception cache .npz — TRAIN missing "
             f"{missing_train}; HELD-OUT missing {missing_held}. Run "
             f"`--phase pseudo_labels` to (re)build before select/em/validate.")

    _save_json(out_root / "preflight.json", info)
    _log(f"preflight done: warmup_sec={info.get('warmup_sec', float('nan')):.1f} "
         f"DAVIS_rgb={info.get('davis_rgb_present', 0)}/{info.get('davis_expected', 0)} "
         f"DAVIS_sam2={info.get('davis_sam2_present', 0)}/{info.get('davis_expected', 0)} "
         f"total_videos={info['n_videos']}; "
         f"cache TRAIN {info['train_npz_present']}/{len(TRAIN_VIDEOS)} "
         f"HELD-OUT {info['heldout_npz_present']}/{len(HELDOUT_VIDEOS)}")
    return info


# ----------------------------------------------------------------------
# Phase 2: pseudo-labels (per-video .npz cache)
# ----------------------------------------------------------------------

def _run_perception(source: Path, max_frames: int = -1):
    """Drive depth + flow + DINO on a video without the tracker. Yields
    (frame_idx, positions, velocities, features). Reuses the live-runner's
    perception kernels and PCA basis fitted on the first valid frame.
    """
    import jax  # noqa: F401  (ensures jax is loaded)
    live = _import_live_runner()
    import genmatter_rt

    stride = genmatter_rt.STRIDE
    n_keep = genmatter_rt.N_KEEP
    work_hw = live.WORK_HW
    indices = genmatter_rt.subsample_indices(h=work_hw[0], w=work_hw[1],
                                              stride=stride, n_keep=n_keep, seed=0)

    prev_tensor = None
    depth_ema = None
    pca_basis = pca_mean = pca_std = None
    intrinsics = genmatter_rt.DEFAULT_INTRINSICS

    for i, bgr in live.iter_frames(Path(source), max_frames):
        d_raw = live._depth_forward(bgr).astype(np.float32)
        depth_ema = d_raw if depth_ema is None else (
            0.6 * d_raw + 0.4 * depth_ema)
        cur_tensor, hw = live._bgr_to_raft_tensor(bgr)
        if prev_tensor is None:
            prev_tensor = cur_tensor
            continue
        flow_np = live._flow_forward(prev_tensor, cur_tensor, hw)
        prev_tensor = cur_tensor
        feat_raw, grid_hw = live._features_forward(bgr)
        positions, velocities = genmatter_rt.unproject(
            depth_ema, flow_np, indices, intrinsics, stride)
        features, pca_basis, pca_mean, pca_std = genmatter_rt.dino_features_to_datapoints(
            feat_raw, indices, pca_basis, pca_mean, pca_std,
            stride=stride, image_hw=bgr.shape[:2],
            target_dim=genmatter_rt.FEATURE_DIM,
            feat_grid_hw=grid_hw,
        )
        yield i, positions, velocities, features, indices


def _frame_count(source: Path) -> int:
    """Best-effort count: dir of frames → glob count, mp4 → cv2."""
    if source.is_dir():
        return len([p for p in source.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    import cv2
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return -1
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _process_one_video_pseudo_labels(entry: VideoEntry, *, work_hw: Tuple[int, int],
                                      stride: int, max_frames: int = -1,
                                      k_motion: int = 8) -> dict:
    """Drive perception + Z_sam load + Z_dino + Z_motion on one video. Saves
    `LABELS_DIR/<vid>.npz` and returns a small summary dict."""
    pos_list: List[np.ndarray] = []
    vel_list: List[np.ndarray] = []
    feat_list: List[np.ndarray] = []
    z_sam_list: List[np.ndarray] = []
    frame_indices: List[int] = []

    indices = None
    n_frames_seen = 0
    z_sam_missing = 0
    for frame_idx, pos, vel, feat, idx in _run_perception(entry.source, max_frames):
        if indices is None:
            indices = idx
        mask_native = load_z_sam_for_frame(entry.vid, frame_idx,
                                            kind=entry.kind, repo_root=_REPO_ROOT)
        if mask_native is None:
            z_sam_missing += 1
            continue
        z_sam = gather_z_sam_at_grid(mask_native, indices, work_hw, stride)
        pos_list.append(pos)
        vel_list.append(vel)
        feat_list.append(feat)
        z_sam_list.append(z_sam)
        frame_indices.append(int(frame_idx))
        n_frames_seen += 1

    if not pos_list:
        return {"status": "no_frames", "z_sam_missing": z_sam_missing}

    positions = np.stack(pos_list, axis=0).astype(np.float32)         # (T, N, 3)
    velocities = np.stack(vel_list, axis=0).astype(np.float32)        # (T, N, 3)
    features = np.stack(feat_list, axis=0).astype(np.float32)         # (T, N, D)
    z_sam_TN = np.stack(z_sam_list, axis=0).astype(np.int32)          # (T, N)
    frame_idx_arr = np.asarray(frame_indices, dtype=np.int32)

    # Per-frame within-instance DINO sub-clustering → Z_dino
    z_dino_TN = np.empty_like(z_sam_TN)
    for t in range(positions.shape[0]):
        z_dino_TN[t] = within_instance_dino_kmeans(features[t], z_sam_TN[t])

    # Per-frame motion KMeans → Z_motion
    z_motion_TN = np.empty_like(z_sam_TN)
    for t in range(positions.shape[0]):
        z_motion_TN[t] = motion_kmeans(velocities[t], k=k_motion, seed=0)

    out_path = LABELS_DIR / f"{entry.vid}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        positions=positions, velocities=velocities, features=features,
        Z_sam=z_sam_TN, Z_dino=z_dino_TN, Z_motion=z_motion_TN,
        frame_idx=frame_idx_arr, indices=indices.astype(np.int32),
    )
    return {
        "status": "ok",
        "frames": int(n_frames_seen),
        "z_sam_missing": int(z_sam_missing),
        "z_sam_k_range": [int(z_sam_TN.min()), int(z_sam_TN.max())],
        "z_dino_k_range": [int(z_dino_TN.min()), int(z_dino_TN.max())],
        "z_motion_k_range": [int(z_motion_TN.min()), int(z_motion_TN.max())],
        "npz_path": str(out_path),
    }


def _refresh_z_sam_one_video(entry: VideoEntry, *, stride: int) -> dict:
    """CHEAP Z_sam refresh (Phase E): reload the existing perception .npz, re-gather
    Z_sam from the (regenerated, higher-quality) SAM PNGs onto the stride-8 grid,
    recompute Z_dino (depends on the new Z_sam), KEEP everything else
    (positions/velocities/features/Z_motion/frame_idx/indices), re-savez_compressed.
    NO depth/flow/DINO recompute (~1.6 s/frame saved) — perception is invariant to
    the pseudo-labels. work_hw is derived from the grid so no live-runner import."""
    path = LABELS_DIR / f"{entry.vid}.npz"
    if not path.is_file():
        return {"status": "no_cache"}
    with np.load(path) as d:
        data = {k: np.asarray(d[k]) for k in d.files}
    indices = data["indices"]
    frame_idx = np.asarray(data["frame_idx"]).reshape(-1)
    features = data["features"]                     # (T, N, D)
    old_z_sam = np.asarray(data["Z_sam"])
    work_hw = (GRID_H * stride, GRID_W * stride)     # (360, 640) — matches live WORK_HW
    z_sam_list: List[np.ndarray] = []
    n_missing = 0
    for t, fi in enumerate(frame_idx):
        mask = load_z_sam_for_frame(entry.vid, int(fi), kind=entry.kind, repo_root=_REPO_ROOT)
        if mask is None:
            n_missing += 1
            z_sam_list.append(old_z_sam[t])          # keep prior labels for this frame
            continue
        z_sam_list.append(gather_z_sam_at_grid(mask, indices, work_hw, stride))
    z_sam_TN = np.stack(z_sam_list, axis=0).astype(np.int32)
    z_dino_TN = np.empty_like(z_sam_TN)
    for t in range(z_sam_TN.shape[0]):
        z_dino_TN[t] = within_instance_dino_kmeans(features[t], z_sam_TN[t])
    np.savez_compressed(
        path,
        positions=data["positions"], velocities=data["velocities"], features=features,
        Z_sam=z_sam_TN, Z_dino=z_dino_TN, Z_motion=data["Z_motion"],
        frame_idx=data["frame_idx"], indices=data["indices"],
    )
    old_k = [int(old_z_sam.min()), int(old_z_sam.max())]
    return {"status": "ok", "frames": int(z_sam_TN.shape[0]),
            "z_sam_missing": int(n_missing),
            "z_sam_k_range_old": old_k,
            "z_sam_k_range": [int(z_sam_TN.min()), int(z_sam_TN.max())],
            "z_dino_k_range": [int(z_dino_TN.min()), int(z_dino_TN.max())]}


def _phase_regen_z_sam(out_root: Path) -> dict:
    """Cheap Z_sam/Z_dino refresh over every cached .npz from regenerated SAM PNGs
    (no perception recompute). Run AFTER scripts/sam2_davis_propagate.py rebuilds
    the higher-quality masks. Re-run `--phase select` afterwards."""
    import genmatter_rt  # noqa: F401 — only for STRIDE (no torch import)
    stride = genmatter_rt.STRIDE
    videos = discover_videos()
    info: dict = {"phase": "regen_z_sam_only", "per_video": {}}
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for i, (vid, entry) in enumerate(videos.items(), 1):
        t0 = time.monotonic()
        try:
            res = _refresh_z_sam_one_video(entry, stride=stride)
            res["wall_sec"] = float(time.monotonic() - t0)
            info["per_video"][vid] = res
            if res.get("status") == "ok":
                n_ok += 1
                _log(f"regen_z_sam[{i}/{len(videos)}] {vid}: ok ({res['frames']} frames, "
                     f"missing={res['z_sam_missing']}, Z_sam k {res['z_sam_k_range_old']}→"
                     f"{res['z_sam_k_range']}) wall={res['wall_sec']:.1f}s")
            else:
                _log(f"regen_z_sam[{i}/{len(videos)}] {vid}: {res.get('status')}")
        except Exception as e:
            info["per_video"][vid] = {"status": "error", "error": repr(e)}
            _log(f"regen_z_sam[{i}/{len(videos)}] {vid}: FAILED: {e}")
            traceback.print_exc()
    info["n_ok"] = n_ok
    _save_json(out_root / "regen_z_sam.json", info)
    _log(f"regen_z_sam done: refreshed {n_ok}/{len(videos)} caches (Z_sam + Z_dino; "
         f"perception untouched). Re-run --phase select next.")
    return info


def phase_pseudo_labels(out_root: Path, *, force: bool = False,
                         max_frames_per_video: int = -1, k_motion: int = 8,
                         regen_z_sam_only: bool = False) -> dict:
    """Per-video × every frame: perception + Z_sam load + Z_dino + Z_motion.

    ``regen_z_sam_only`` takes the CHEAP path (Phase E): refresh Z_sam + Z_dino in
    every cached .npz from regenerated SAM PNGs, NO perception recompute."""
    if regen_z_sam_only:
        return _phase_regen_z_sam(out_root)

    info_path = out_root / "pseudo_labels.json"
    cached = _load_json(info_path)
    if cached and not force:
        _log("pseudo_labels: reusing cached")
        return cached

    info: dict = {"phase": "pseudo_labels", "max_frames_per_video": max_frames_per_video,
                  "k_motion": k_motion, "per_video": {}}
    live = _import_live_runner()
    work_hw = live.WORK_HW

    import genmatter_rt
    stride = genmatter_rt.STRIDE

    videos = discover_videos()
    info["n_videos"] = len(videos)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    for i, (vid, entry) in enumerate(videos.items(), 1):
        npz_path = LABELS_DIR / f"{vid}.npz"
        if npz_path.is_file() and not force:
            try:
                with np.load(npz_path) as d:
                    info["per_video"][vid] = {
                        "status": "cached",
                        "frames": int(d["positions"].shape[0]),
                        "npz_path": str(npz_path),
                    }
                _log(f"pseudo_labels[{i}/{len(videos)}] {vid}: cached "
                     f"({info['per_video'][vid]['frames']} frames)")
                continue
            except Exception:
                pass  # fall through and recompute
        t0 = time.monotonic()
        try:
            res = _process_one_video_pseudo_labels(
                entry, work_hw=work_hw, stride=stride,
                max_frames=max_frames_per_video, k_motion=k_motion)
            res["wall_sec"] = float(time.monotonic() - t0)
            info["per_video"][vid] = res
            _log(f"pseudo_labels[{i}/{len(videos)}] {vid}: {res.get('status', '?')} "
                 f"frames={res.get('frames', '-')} wall={res['wall_sec']:.1f}s")
        except Exception as e:
            info["per_video"][vid] = {"status": "error", "error": repr(e),
                                       "wall_sec": float(time.monotonic() - t0)}
            _log(f"pseudo_labels[{i}/{len(videos)}] {vid}: FAILED: {e}")
            traceback.print_exc()

    _save_json(info_path, info)
    n_ok = sum(1 for v in info["per_video"].values()
               if v.get("status") in ("ok", "cached"))
    _log(f"pseudo_labels done: {n_ok}/{len(videos)} videos ok")
    return info


# ----------------------------------------------------------------------
# Phase 3: EM-with-safeguards outer loop
# ----------------------------------------------------------------------

@dataclass
class EmState:
    """Carries the current hyperparameter values + per-group frozen status.

    Initialized to GenMatter DAVIS DINO experiment defaults; on the first EM
    iter the use_calibrated_priors flag flips on for any group that's accepted,
    swapping in the YAML-supplied Psi_*/nu_*/mu_H instead of the K-means seed.
    """
    use_calibrated_priors: bool
    # Cluster group (cluster priors)
    sigma_F: float
    Psi_B: np.ndarray            # (3, 3)
    nu_B: float
    alpha: float
    # Particle group (hyperblob priors)
    sigma_F_H: float
    Psi_H: np.ndarray            # (3, 3)
    nu_H: float
    beta: float
    sigma_H: float
    mu_H: np.ndarray             # (3,)
    # Motion group (velocity + outlier + translation priors)
    sigma_V: float
    Psi_V: np.ndarray            # (3, 3)
    nu_V: float
    outlier_prob: float
    gamma_shape: float
    gamma_rate: float
    translation_max_radius: float
    translation_gaussian_scale: float
    # Per-group COOLDOWN counter (iters a regressing group sits out). 0 == active;
    # >0 == frozen for that many more iters, then re-attempted (was a permanent
    # bool freeze; the accept-test is the real regression guard, so a group that
    # merely fluctuated should be retried — especially the motion group once
    # sigma_V is finite). Old JSON caches (bool true/false) coerce via int().
    group_frozen: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "use_calibrated_priors": bool(self.use_calibrated_priors),
            "sigma_F": float(self.sigma_F),
            "Psi_B": np.asarray(self.Psi_B).tolist(),
            "nu_B": float(self.nu_B),
            "alpha": float(self.alpha),
            "sigma_F_H": float(self.sigma_F_H),
            "Psi_H": np.asarray(self.Psi_H).tolist(),
            "nu_H": float(self.nu_H),
            "beta": float(self.beta),
            "sigma_H": float(self.sigma_H),
            "mu_H": np.asarray(self.mu_H).tolist(),
            "sigma_V": float(self.sigma_V),
            "Psi_V": np.asarray(self.Psi_V).tolist(),
            "nu_V": float(self.nu_V),
            "outlier_prob": float(self.outlier_prob),
            "gamma_shape": float(self.gamma_shape),
            "gamma_rate": float(self.gamma_rate),
            "translation_max_radius": float(self.translation_max_radius),
            "translation_gaussian_scale": float(self.translation_gaussian_scale),
            "group_frozen": dict(self.group_frozen),
        }


def _clone_em(em: EmState) -> EmState:
    return EmState(
        use_calibrated_priors=em.use_calibrated_priors,
        sigma_F=em.sigma_F, Psi_B=np.asarray(em.Psi_B).copy(), nu_B=em.nu_B,
        alpha=em.alpha,
        sigma_F_H=em.sigma_F_H, Psi_H=np.asarray(em.Psi_H).copy(), nu_H=em.nu_H,
        beta=em.beta, sigma_H=em.sigma_H, mu_H=np.asarray(em.mu_H).copy(),
        sigma_V=em.sigma_V, Psi_V=np.asarray(em.Psi_V).copy(), nu_V=em.nu_V,
        outlier_prob=em.outlier_prob, gamma_shape=em.gamma_shape,
        gamma_rate=em.gamma_rate,
        translation_max_radius=em.translation_max_radius,
        translation_gaussian_scale=em.translation_gaussian_scale,
        group_frozen=dict(em.group_frozen),
    )


def _em_from_dict(d: dict) -> EmState:
    """Reconstruct an EmState from a JSON-decoded dict (e.g. em.json["best_hypers"])."""
    return EmState(
        use_calibrated_priors=bool(d.get("use_calibrated_priors", False)),
        sigma_F=float(d["sigma_F"]),
        Psi_B=np.asarray(d["Psi_B"], dtype=np.float32),
        nu_B=float(d["nu_B"]),
        alpha=float(d["alpha"]),
        sigma_F_H=float(d["sigma_F_H"]),
        Psi_H=np.asarray(d["Psi_H"], dtype=np.float32),
        nu_H=float(d["nu_H"]),
        beta=float(d["beta"]),
        sigma_H=float(d["sigma_H"]),
        mu_H=np.asarray(d["mu_H"], dtype=np.float32),
        sigma_V=float(d["sigma_V"]),
        Psi_V=np.asarray(d["Psi_V"], dtype=np.float32),
        nu_V=float(d["nu_V"]),
        outlier_prob=float(d["outlier_prob"]),
        gamma_shape=float(d["gamma_shape"]),
        gamma_rate=float(d["gamma_rate"]),
        translation_max_radius=float(d["translation_max_radius"]),
        translation_gaussian_scale=float(d["translation_gaussian_scale"]),
        # Coerce legacy bool freezes (True/False) → cooldown ints (1/0).
        group_frozen={k: int(v) for k, v in
                      d.get("group_frozen", {k: 0 for k in GROUPS}).items()},
    )


def _load_labels(vid: str) -> dict:
    """Load and return numpy arrays for one video's pseudo-labels."""
    path = LABELS_DIR / f"{vid}.npz"
    with np.load(path) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def _initial_em_state() -> EmState:
    """GenMatter DAVIS DINO experiment defaults (the calibration prior).

    `use_calibrated_priors=False` so iter-0 main pass uses the streaming
    tracker's K-means seed for Psi_*/nu_*/mu_H; the placeholder matrix
    values here are only used as the conjugate-prior centering and
    cross-iter damping reference until iter 0's M-step produces real
    posteriors.
    """
    return EmState(
        use_calibrated_priors=False,
        sigma_F=DAVIS_DINO_DEFAULTS["sigma_F"],
        Psi_B=DAVIS_DINO_DEFAULTS["Psi_B"].copy(),
        nu_B=DAVIS_DINO_DEFAULTS["nu_B"],
        alpha=DAVIS_DINO_DEFAULTS["alpha"],
        sigma_F_H=DAVIS_DINO_DEFAULTS["sigma_F_H"],
        Psi_H=DAVIS_DINO_DEFAULTS["Psi_H"].copy(),
        nu_H=DAVIS_DINO_DEFAULTS["nu_H"],
        beta=DAVIS_DINO_DEFAULTS["beta"],
        sigma_H=DAVIS_DINO_DEFAULTS["sigma_H"],
        mu_H=DAVIS_DINO_DEFAULTS["mu_H"].copy(),
        sigma_V=DAVIS_DINO_DEFAULTS["sigma_V"],
        Psi_V=DAVIS_DINO_DEFAULTS["Psi_V"].copy(),
        nu_V=DAVIS_DINO_DEFAULTS["nu_V"],
        outlier_prob=DAVIS_DINO_DEFAULTS["outlier_prob"],
        gamma_shape=DAVIS_DINO_DEFAULTS["gamma_shape"],
        gamma_rate=DAVIS_DINO_DEFAULTS["gamma_rate"],
        translation_max_radius=DAVIS_DINO_DEFAULTS["translation_max_radius"],
        translation_gaussian_scale=DAVIS_DINO_DEFAULTS["translation_gaussian_scale"],
        group_frozen={k: 0 for k in GROUPS},
    )


def _make_yaml_cfg(em: EmState, infer: dict) -> dict:
    """Clone the default YAML, overlay all 18 EM hypers + the inference-strategy
    knobs (``infer``: use_sam_frame0 / num_blobs / num_hyperblobs /
    init_gibbs_sweeps). ONE global config — the same builder is used for EM,
    accept-test, the validate chosen pass, and the emitted YAML."""
    import genmatter_rt
    cfg = genmatter_rt.load_yaml_hypers(DEFAULT_YAML)
    cfg["tracking"]["use_sam_frame0"] = bool(infer["use_sam_frame0"])
    cfg["tracking"]["num_blobs"] = int(infer["num_blobs"])
    cfg["tracking"]["num_hyperblobs"] = int(infer["num_hyperblobs"])
    cfg["tracking"]["init_gibbs_sweeps"] = int(infer["init_gibbs_sweeps"])
    # Architectural-structure flags (the v2 cluster-freeze fix). These keys are
    # READ by genmatter_rt.run_tracker_from_cache (feature_aware_final_assignment /
    # final_assignment_outlier / freeze_hyperblob_assignment / pure_object_seed /
    # blob_means_updates_per_frame) but were NEVER set here — so the EM/accept/
    # validate path silently ran the vendored structure while the demo ran the
    # fixed one. Setting them makes what-we-calibrate == what-we-validate ==
    # what-we-ship. GLOBAL, never per-video. (.get fallbacks keep old cached
    # `infer` dicts working — they default to the shipped-structure constants.)
    cfg["tracking"]["feature_aware_final_assignment"] = bool(
        infer.get("feature_aware_final", INFER_FEATURE_AWARE_FINAL))
    cfg["tracking"]["final_assignment_outlier"] = bool(
        infer.get("final_outlier", INFER_FINAL_OUTLIER))
    cfg["tracking"]["freeze_hyperblob_assignment"] = bool(
        infer.get("freeze_hyperblob", INFER_FREEZE_HYPERBLOB))
    cfg["tracking"]["pure_object_seed"] = bool(
        infer.get("pure_object_seed", INFER_PURE_OBJECT_SEED))
    cfg["tracking"]["blob_means_updates_per_frame"] = int(
        infer.get("blob_means_updates", INFER_BLOB_MEANS_UPDATES))
    # Anti-drift feature update (Phase A): the DAMPED generalization of the freeze
    # fix. Written into EVERY _make_yaml_cfg output so the EM learns the hypers
    # UNDER the damped inference, selection sweeps it, and the emitted YAML ships
    # the LEARNED value. d=1.0 is bit-exact vendored; legacy freeze flag kept off.
    cfg["tracking"]["freeze_blob_features"] = bool(
        infer.get("freeze_blob_features", INFER_FREEZE_BLOB_FEATURES))
    cfg["tracking"]["feature_update_damping"] = float(
        infer.get("feature_update_damping", INFER_FEATURE_UPDATE_DAMPING))
    # INFERENCE feature-temperature (Phase 1): tau on the final assignment. Written
    # into EVERY _make_yaml_cfg output so the EM learns the hypers UNDER the chosen
    # tau, selection sweeps it, and the emitted YAML ships the SELECTED value. tau=1
    # is the re-impl of the vendored feature-aware-final (bit-exact). Only emitted
    # when != 1.0 (absent key => off => vendored path) keeps default configs clean.
    _tau = float(infer.get("final_feature_temp", INFER_FINAL_FEATURE_TEMP))
    if _tau != 1.0:
        cfg["tracking"]["final_feature_temp"] = _tau
        cfg["tracking"]["final_assignment_anchor"] = bool(
            infer.get("final_assignment_anchor", INFER_FINAL_ASSIGNMENT_ANCHOR))
    else:
        cfg["tracking"].pop("final_feature_temp", None)
        cfg["tracking"].pop("final_assignment_anchor", None)
    # TokenCut self-supervised seed augmentation (default-off). Carried on the infer
    # dict so the validate BASELINE (un-augmented shipping config) and CHOSEN
    # (augmented re-fit) differ by exactly this flag. Honored at the one chokepoint
    # _run_tracker_on_video; the live streaming_default.yaml never sets it.
    if infer.get("tokencut_seed_augment"):
        cfg["tracking"]["tokencut_seed_augment"] = True
        if infer.get("tokencut_knobs"):
            cfg["tracking"]["tokencut_knobs"] = dict(infer["tokencut_knobs"])
    else:
        cfg["tracking"].pop("tokencut_seed_augment", None)
        cfg["tracking"].pop("tokencut_knobs", None)
    cfg["tracking"]["calibrate_feature_sigmas"] = False
    cfg["tracking"]["use_calibrated_priors"] = bool(em.use_calibrated_priors)
    hp = cfg["tracking"]["hyperparams"]
    hp["sigma_F"] = float(em.sigma_F)
    hp["sigma_F_H"] = float(em.sigma_F_H)
    hp["sigma_V"] = float(em.sigma_V)
    hp["outlier_prob"] = float(em.outlier_prob)
    hp["outlier_velocity_gamma_shape"] = float(em.gamma_shape)
    hp["outlier_velocity_gamma_rate"] = float(em.gamma_rate)
    hp["alpha"] = float(em.alpha)
    hp["beta"] = float(em.beta)
    hp["sigma_H"] = float(em.sigma_H)
    hp["translation_max_radius"] = float(em.translation_max_radius)
    hp["translation_gaussian_scale"] = float(em.translation_gaussian_scale)
    # Calibrated-prior keys: only set when the flag is true so the YAML stays
    # diff-clean against streaming_default.yaml for the baseline path.
    if em.use_calibrated_priors:
        hp["Psi_B"] = np.asarray(em.Psi_B, dtype=np.float32).tolist()
        hp["Psi_H"] = np.asarray(em.Psi_H, dtype=np.float32).tolist()
        hp["Psi_V"] = np.asarray(em.Psi_V, dtype=np.float32).tolist()
        hp["nu_B"] = float(em.nu_B)
        hp["nu_H"] = float(em.nu_H)
        hp["nu_V"] = float(em.nu_V)
        hp["mu_H"] = np.asarray(em.mu_H, dtype=np.float32).tolist()
    return cfg


def _run_tracker_on_video(labels_TN: dict, yaml_cfg: dict, max_frames: int, *,
                           num_sweeps: int = 1,
                           capture_blob_weights: bool = False,
                           sam_grid: Optional[np.ndarray] = None,
                           capture_data_loglik: bool = False,
                           vid: Optional[str] = None) -> dict:
    """Run the Gibbs tracker over a video's CACHED perception outputs (the
    positions/velocities/features stored in its pseudo-label .npz) — no
    depth/flow/DINO recompute, since perception is invariant to the calibration
    hypers. ``num_sweeps`` is the Level-1 per-frame measurement-update depth.
    ``sam_grid`` (when given AND yaml_cfg enables use_sam_frame0) seeds the
    frame-0 semantic SAM init. When ``yaml_cfg['tracking']['tokencut_seed_augment']``
    is set (default-off) AND ``vid`` is known, the frame-0 seed is additively
    augmented with TokenCut+SAM2-discovered objects SAM missed (working videos are
    a no-op by construction). Returns stacked arrays (blob_a/hyperblob_a, plus
    blob_w/n_blobs/indices when requested) or {'error': ...}."""
    import genmatter_rt
    try:
        pos = labels_TN["positions"]
        vel = labels_TN["velocities"]
        feat = labels_TN["features"]
        idx = labels_TN["indices"]
        if max_frames is not None and max_frames > 0:
            pos, vel, feat = pos[:max_frames], vel[:max_frames], feat[:max_frames]
        nb = int(yaml_cfg["tracking"]["num_blobs"])
        nh = int(yaml_cfg["tracking"]["num_hyperblobs"])
        # Default-off TokenCut seed augmentation (self-supervised object discovery
        # for the broken videos SAM2 misses; additive + uncovered-only → working
        # videos untouched). Never breaks the tracker: any failure falls back to
        # the base SAM grid.
        if (sam_grid is not None and vid is not None
                and yaml_cfg.get("tracking", {}).get("tokencut_seed_augment")):
            try:
                import tokencut
                sam_grid, _ = tokencut.augment_seed_grid(
                    vid, labels_TN, sam_grid, yaml_cfg["tracking"].get("tokencut_knobs"))
            except Exception as _e:
                _log(f"tokencut augment failed for {vid}: {_e!r} (using base grid)")
        # Phase-3: the temporally-consistent SAM instances (Z_sam) drive the
        # instance-purity objective term (host-side; self-supervised, NOT GT).
        z_sam_TN = labels_TN.get("Z_sam") if capture_data_loglik else None
        if z_sam_TN is not None and max_frames is not None and max_frames > 0:
            z_sam_TN = z_sam_TN[:max_frames]
        return genmatter_rt.run_tracker_from_cache(
            pos, vel, feat, idx, yaml_cfg=yaml_cfg, num_blobs=nb, num_hyperblobs=nh,
            num_sweeps=num_sweeps, capture_blob_weights=capture_blob_weights,
            sam_segmentation=sam_grid, capture_data_loglik=capture_data_loglik,
            z_sam_TN=z_sam_TN)
    except Exception as e:
        return {"error": repr(e)}


def _score_video(vid: str, blob_a_TN: np.ndarray, hb_a_TN: np.ndarray,
                  labels_TN: dict) -> dict:
    """Slice the loaded labels to the tracker's T frames and score."""
    Tt = blob_a_TN.shape[0]
    Tl = labels_TN["Z_sam"].shape[0]
    T = min(Tt, Tl)
    return consistency_score(
        blob_a_TN[:T], hb_a_TN[:T],
        labels_TN["Z_dino"][:T], labels_TN["Z_motion"][:T],
        labels_TN["Z_sam"][:T],
    )


def _aggregate_S(per_vid_scores: Dict[str, dict]) -> float:
    """Median over videos of S (the (S_fine + S_coarse + S_semantic) / 3)."""
    vals = [v["S"] for v in per_vid_scores.values()
            if "S" in v and np.isfinite(v["S"])]
    return float(np.median(vals)) if vals else float("nan")


# ----------------------------------------------------------------------
# J-mean (matter-weighted Jaccard) scorer — the primary gate metric
# ----------------------------------------------------------------------

def _score_video_jmean(vid: str, blob_a_TN: np.ndarray, blob_w_TN: Optional[np.ndarray],
                       n_blobs: Optional[int], indices_N: np.ndarray,
                       annotations_path: str) -> dict:
    """Matter-weighted Jaccard (J-mean) vs per-frame masks at ``annotations_path``
    via the GenMatter++ evaluator (reused unchanged). The objective passes the
    SAM2-propagated masks (self-supervised); the validation report additionally
    passes the real DAVIS GT. Returns {'J_mean','J_recall','J_precision'}.

    Frame alignment: the streaming tracker skips video frame 0 (no flow yet), so
    its first output is video frame 1. ``evaluate_single_davis_video`` compares
    ``trial_list[k]`` to mask frame k, so we duplicate the first captured frame
    into the frame-0 slot — then ``trial_list[k]`` aligns with mask frame k for
    all k>=1, and only the frame-0 reference slot is a 1-frame proxy. Outliers
    (blob_a == -1) fold to ``n_blobs`` so the evaluator's ``!= n_blobs`` filter
    (evaluation.py:110/132) drops them.
    """
    from genmatter.evaluation import evaluate_single_davis_video

    T = int(blob_a_TN.shape[0])
    if T == 0 or blob_w_TN is None or n_blobs is None:
        return {"J_mean": float("nan"), "J_recall": float("nan"),
                "J_precision": float("nan")}

    def _frame(k: int) -> dict:
        a = blob_a_TN[k].astype(np.int64)
        a = np.where(a < 0, n_blobs, a)            # fold outliers -> n_blobs
        return {"n_blobs": int(n_blobs),
                "blob_assignments": a,
                "blob_weights": np.asarray(blob_w_TN[k], dtype=np.float64)}

    # [frame0 proxy] + frames 1..T → trial_list[k] aligns with mask frame k (k>=1).
    trial_list = [_frame(0)] + [_frame(k) for k in range(T)]
    try:
        res, _viz = evaluate_single_davis_video(
            vid, [trial_list], str(annotations_path),
            counting_threshold=0, img_dims=(GRID_H, GRID_W),
            render_results_video=False,
            subsampled_indices=np.asarray(indices_N),
        )
    except Exception as e:
        _log(f"  J-mean scorer failed on {vid}: {e}")
        return {"J_mean": float("nan"), "J_recall": float("nan"),
                "J_precision": float("nan")}
    return {
        "J_mean": float(res.get("avg_matter_weighted_jaccard_fixed", float("nan"))),
        "J_recall": float(res.get("avg_matter_weighted_recall_fixed", float("nan"))),
        "J_precision": float(res.get("avg_matter_weighted_precision_fixed", float("nan"))),
    }


def _aggregate_J(per_vid_scores: Dict[str, dict]) -> float:
    """Median matter-weighted J_mean across videos (logged side-metric)."""
    vals = [v["J_mean"] for v in per_vid_scores.values()
            if "J_mean" in v and np.isfinite(v["J_mean"])]
    return float(np.median(vals)) if vals else float("nan")


def _aggregate_region_J(per_vid_scores: Dict[str, dict]) -> float:
    """Median REAL region-J across videos (the primary self-supervised EM
    objective when reference='sam'; the gated benchmark when reference='gt')."""
    vals = [v["region_J"] for v in per_vid_scores.values()
            if "region_J" in v and np.isfinite(v["region_J"])]
    return float(np.median(vals)) if vals else float("nan")


# ----------------------------------------------------------------------
# REAL region-Jaccard (pixel/datapoint IoU of the CLUSTER foreground mask)
# — the metric the calibration actually optimizes (self-supervised) and is
# gated on (true DAVIS GT). Mirrors scripts/_real_jaccard_rank.real_region_J,
# but scores CLUSTER (hyperblob) labels and switches the foreground reference
# between cached SAM pseudo-labels (NO ground truth, NO file I/O) and the
# held-out DAVIS GT segmasks.
# ----------------------------------------------------------------------

def _gt_dp_at_datapoints(vid: str, frame_idx: int, indices_N: np.ndarray):
    """True DAVIS-GT foreground at the tracker's datapoints (ported from
    ``_real_jaccard_rank.gt_dp`` so this module never imports that script — it
    forces XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 on import). DAVIS GT is a 3-channel
    RGB mask (object in color, NOT channel 0), so foreground = ANY channel >= 1.
    Resize the full mask to the (GRID_H, GRID_W) stride-8 grid (nearest), then
    index the datapoints. Returns an (N,) bool array, or None if the mask is
    missing/unreadable."""
    _assert_gt_readable(vid)   # GT scope required; held-out GT gate-only
    import config as gm_config
    import cv2
    p = Path(gm_config.DAVIS_SEGMASKS_PATH) / vid / f"{int(frame_idx):05d}.png"
    if not p.is_file():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    fg = (m >= 1) if m.ndim == 2 else np.any(m >= 1, axis=2)
    g = cv2.resize(fg.astype(np.uint8), (GRID_W, GRID_H),
                   interpolation=cv2.INTER_NEAREST).reshape(-1) >= 1   # (GRID_H*GRID_W,)
    return g[np.asarray(indices_N)]                                    # (N,) at datapoints


def _score_video_region_J(vid: str, label_TN: np.ndarray, indices_N: np.ndarray,
                          labels: dict, *, reference: str = "sam") -> dict:
    """REAL region-IoU of the tracker's CLUSTER (hyperblob) foreground mask vs a
    per-frame foreground reference, averaged over frames (the standard
    'propagate the frame-0 object' definition, scored at the stride-8
    datapoints). Returns ``{"region_J": float}`` (NaN-robust like the other
    scorers).

    Foreground reference by ``reference``:
      * ``"sam"`` (SELF-SUPERVISED; learning objective + accept-test): per-frame
        foreground = ``labels["Z_sam"][t] > 0`` (cached per-datapoint SAM instance
        id, already aligned to ``indices``; 0 == background). NO file I/O, NO GT.
      * ``"gt"`` (EVALUATION; validate gate): per-frame foreground = the held-out
        true DAVIS GT segmasks via ``_gt_dp_at_datapoints`` — exactly
        ``_real_jaccard_rank.real_region_J``'s behaviour.

    The frame-0 reference cluster set = clusters majority-inside the frame-0
    foreground; tracked clusters' union is the foreground prediction per frame.
    Outliers (label == -1) are never in the reference set, so they never count as
    foreground.
    """
    label_TN = np.asarray(label_TN)
    T = int(label_TN.shape[0])
    if T == 0:
        return {"region_J": float("nan")}

    if reference == "sam":
        z_sam = np.asarray(labels["Z_sam"])
        Tz = int(z_sam.shape[0])
        Tc = min(T, Tz)
        fg_of = lambda t: (z_sam[t] > 0)                # (N,) bool, self-supervised
    elif reference == "gt":
        _assert_gt_readable(vid)   # GT scope required; held-out GT gate-only
        fi = np.asarray(labels["frame_idx"]).reshape(-1)
        Tc = min(T, int(fi.shape[0]))
        fg_of = lambda t: _gt_dp_at_datapoints(vid, int(fi[t]), indices_N)
    else:
        raise ValueError(f"reference must be 'sam' or 'gt', got {reference!r}")

    if Tc == 0:
        return {"region_J": float("nan")}

    g0 = fg_of(0)
    if g0 is None:
        return {"region_J": float("nan")}
    g0 = np.asarray(g0).reshape(-1)
    c0 = label_TN[0]
    ref = [int(c) for c in np.unique(c0)
           if c >= 0 and c0.shape[0] and g0[c0 == c].size and g0[c0 == c].mean() > 0.5]

    Js: List[float] = []
    for t in range(Tc):
        g = fg_of(t)
        if g is None:
            continue
        g = np.asarray(g).reshape(-1)
        pred = np.isin(label_TN[t], ref)
        inter = int(np.logical_and(pred, g).sum())
        union = int(np.logical_or(pred, g).sum())
        Js.append(inter / union if union > 0 else 0.0)
    return {"region_J": float(np.mean(Js)) if Js else float("nan")}


# ----------------------------------------------------------------------
# Richer COMPOSITE self-supervised objective (NO GT). A fixed-weight, a-priori
# combination of binary region-J (DOMINANT) + per-SAM-instance matched J +
# temporal cluster-ID continuity + motion-ARI + blob-level J. Used as the EM
# objective, the per-group accept-test score, and the phase_select score —
# ALWAYS self-supervised (reference = cached Z_sam, never GT). The held-out GATE
# in phase_validate stays PURE binary GT region-J; the composite's correlation
# with GT is measured ONCE there as a reported kill-switch.
# ----------------------------------------------------------------------

# Weights fixed A-PRIORI (region-J dominant; inst-matched + temporal as
# object-tracking regularizers; motion-ARI + blob-J as small structural
# cross-checks). NEVER tuned against the GT gate — they are design choices,
# validated once on held-out. Renormalized over finite components at score time.
COMPOSITE_WEIGHTS: Dict[str, float] = {
    "region_J":   0.60,   # binary CLUSTER foreground IoU vs Z_sam>0 (strongest GT proxy)
    "inst_J":     0.22,   # per-SAM-instance matched IoU (catches instance MERGING)
    "temporal":   0.10,   # cluster-ID continuity across frames (penalizes flicker/bleed)
    "motion_ari": 0.04,   # ARI(blob_a, Z_motion) (motion-coherence cross-check)
    "blob_J":     0.04,   # blob-level foreground IoU vs Z_sam>0 (finer granularity)
}

# The self-supervised objective the EM / accept-test / selection optimize:
# "composite" (default) | "data_loglik" (Phase-B probabilistic complete-data
# log-likelihood) | "region_J" (the binary kill-switch fallback). The held-out GT
# gate is ALWAYS binary region-J regardless of this. Set via main()'s --objective
# flag. data_loglik is the Phase-B re-optimization objective (replaces the
# anti-correlated region-J-vs-union signal with the model's OWN likelihood, which a
# bleeding tracker scores LOWER on); it is gated by _loglik_objective_proto.py
# before use, and the operator reverts to composite/region_J if that gate fails.
_OBJECTIVE_MODE = "composite"
# When _OBJECTIVE_MODE == "data_loglik", which term decomposition from
# _data_loglik.frame_data_loglik_terms drives selection (the Phase-B proto picks
# the best GT-correlating variant). The NAIVE live-feature variants ("full",
# "pos_feat") reward feature DRIFT (anti-correlate with GT); the ANCHOR-referenced
# variants ("feat_anchor", "anchor_pos_feat", "anchor_full") score features vs the
# FROZEN frame-0 appearance and TRACK GT. Default = anchor_pos_feat.
_DATA_LOGLIK_VARIANT = "anchor_pos_feat"


def _instance_matched_J(hb_a_TN: np.ndarray, z_sam_TN: np.ndarray, Tc: int,
                        *, min_inst_size: int = 5) -> float:
    """Per-SAM-instance matched region-J (mean over instances, "J&F"-style).

    For each frame-0 SAM instance k (>= min_inst_size datapoints), the reference
    cluster set = clusters majority-inside instance k at frame 0; per frame, IoU
    of those clusters' union vs (Z_sam==k). Instances captured by NO cluster score
    0 (penalizes missed/merged instances — exactly what binary region-J misses
    when two instances collapse into one cluster). Mean over instances+frames."""
    c0 = hb_a_TN[0]
    g0 = z_sam_TN[0]
    inst_ids = [int(k) for k in np.unique(g0)
                if k > 0 and int((g0 == k).sum()) >= min_inst_size]
    if not inst_ids:
        return float("nan")
    per_inst: List[float] = []
    for k in inst_ids:
        gk0 = (g0 == k)
        ref = [int(c) for c in np.unique(c0)
               if c >= 0 and gk0[c0 == c].size and gk0[c0 == c].mean() > 0.5]
        if not ref:
            per_inst.append(0.0)            # instance captured by no cluster → missed
            continue
        Jk: List[float] = []
        for t in range(Tc):
            gk = (z_sam_TN[t] == k)
            pred = np.isin(hb_a_TN[t], ref)
            inter = int(np.logical_and(pred, gk).sum())
            union = int(np.logical_or(pred, gk).sum())
            Jk.append(inter / union if union > 0 else 0.0)
        per_inst.append(float(np.mean(Jk)) if Jk else 0.0)
    return float(np.mean(per_inst)) if per_inst else float("nan")


def _temporal_id_continuity(hb_a_TN: np.ndarray, z_sam_TN: np.ndarray, Tc: int) -> float:
    """Fraction of datapoints whose tracker CLUSTER id is stable across
    consecutive frames, among datapoints whose Z_sam instance id is ALSO stable
    (and foreground). Penalizes cluster-identity flicker/bleed without GT. The
    streaming tracker's hyperblob ids are persistent slots (freeze_hyperblob keeps
    the frame-0 blob->hyperblob map), so frame-to-frame equality is meaningful."""
    conts: List[float] = []
    for t in range(1, Tc):
        stable_sam = (z_sam_TN[t] == z_sam_TN[t - 1]) & (z_sam_TN[t] > 0)
        denom = int(stable_sam.sum())
        if denom == 0:
            continue
        stable_clust = (hb_a_TN[t] == hb_a_TN[t - 1]) & stable_sam
        conts.append(int(stable_clust.sum()) / denom)
    return float(np.mean(conts)) if conts else float("nan")


def _score_video_composite(vid: str, hb_a_TN: np.ndarray, blob_a_TN: np.ndarray,
                           indices_N: np.ndarray, labels: dict) -> dict:
    """Self-supervised composite objective for one video (NO GT). Returns the
    component scores + the renormalized weighted ``composite``."""
    hb_a_TN = np.asarray(hb_a_TN)
    blob_a_TN = np.asarray(blob_a_TN)
    z_sam = np.asarray(labels["Z_sam"])
    Tc = min(int(hb_a_TN.shape[0]), int(z_sam.shape[0]))
    comps: dict = {
        "region_J": _score_video_region_J(vid, hb_a_TN, indices_N, labels,
                                          reference="sam")["region_J"],
        "blob_J":   _score_video_region_J(vid, blob_a_TN, indices_N, labels,
                                          reference="sam")["region_J"],
        "inst_J":   _instance_matched_J(hb_a_TN, z_sam, Tc) if Tc else float("nan"),
        "temporal": _temporal_id_continuity(hb_a_TN, z_sam, Tc) if Tc else float("nan"),
    }
    # motion-ARI via the existing JAX consistency scorer (S_coarse = ARI(blob_a, Z_motion)).
    try:
        sc = _score_video(vid, blob_a_TN, hb_a_TN, labels)
        comps["motion_ari"] = float(sc.get("S_coarse", float("nan")))
    except Exception:
        comps["motion_ari"] = float("nan")
    # Renormalized weighted mean over finite components (a NaN component drops out
    # rather than tanking the score; region_J dominates and is ~always finite).
    num = 0.0
    den = 0.0
    for k, w in COMPOSITE_WEIGHTS.items():
        v = comps.get(k)
        if v is not None and np.isfinite(v):
            num += w * float(v)
            den += w
    comps["composite"] = (num / den) if den > 0 else float("nan")
    return comps


def _pick_data_loglik(tr: dict) -> Optional[float]:
    """Select the configured data-loglik VARIANT (_DATA_LOGLIK_VARIANT) from a
    tracker result's captured term breakdown. Falls back to the headline scalar.
    Returns None when the tracker didn't capture loglik (objective != data_loglik)."""
    terms = tr.get("data_loglik_terms")
    if terms is not None:
        return terms.get(_DATA_LOGLIK_VARIANT, tr.get("data_loglik"))
    return tr.get("data_loglik")


def _self_sup_score(vid: str, hb_a_TN: np.ndarray, blob_a_TN: np.ndarray,
                    indices_N: np.ndarray, labels: dict,
                    data_loglik: Optional[float] = None) -> dict:
    """The self-supervised score the EM / accept-test / selection optimize.
    Returns the full composite dict plus ``score`` per _OBJECTIVE_MODE:
      composite   -> the fixed-weight region-J composite (default),
      data_loglik -> the model's complete-data log-likelihood (Phase B; passed in
                     from the tracker run since it needs the live state, NOT the
                     assignments), higher = better, comparable across videos,
      region_J    -> binary CLUSTER region-J vs Z_sam>0 (the kill-switch fallback).
    The composite components are ALWAYS computed (logging + kill-switch corr). GT is
    never touched here."""
    comp = _score_video_composite(vid, hb_a_TN, blob_a_TN, indices_N, labels)
    if _OBJECTIVE_MODE == "data_loglik":
        comp["data_loglik"] = (float(data_loglik) if data_loglik is not None
                               and np.isfinite(data_loglik) else float("nan"))
        comp["score"] = comp["data_loglik"]
    elif _OBJECTIVE_MODE == "composite":
        comp["score"] = comp["composite"]
    else:
        comp["score"] = comp["region_J"]
    return comp


def _defaults_for_priors() -> dict:
    """Prior-centering defaults consumed by `aggregate_posteriors` (sigma_F,
    sigma_F_H, Psi_B/H/V). The conjugate prior centers at these values so
    sparse-N regimes return the default; n ≫ n_0 → MAP ≈ MLE.
    """
    return {
        "sigma_F": DAVIS_DINO_DEFAULTS["sigma_F"],
        "sigma_F_H": DAVIS_DINO_DEFAULTS["sigma_F_H"],
        "sigma_V": DAVIS_DINO_DEFAULTS["sigma_V"],   # finite seed = M-step prior center
        "Psi_B": DAVIS_DINO_DEFAULTS["Psi_B"].copy(),
        "Psi_H": DAVIS_DINO_DEFAULTS["Psi_H"].copy(),
        "Psi_V": DAVIS_DINO_DEFAULTS["Psi_V"].copy(),
    }


def _apply_sanity_veto(raw_proposals: dict) -> Tuple[dict, dict]:
    """Plan §4: a RAW proposal outside its numerical-stability range gets
    NaN'd → the damper short-circuits → that field freezes for this iter.

    Returns (vetoed_proposals, veto_log). veto_log maps field name → reason.
    γ shape veto also cascades to γ rate (jointly parameterized).
    """
    out = dict(raw_proposals)
    log: dict = {}
    # Scalar sanity ranges
    for key, (lo, hi) in SANITY_FLOORS.items():
        if key not in out:
            continue
        val = out[key]
        if val is None or not np.isfinite(val):
            log[key] = f"non-finite raw proposal ({val!r})"
            out[key] = float("nan")
            continue
        if val < lo or val > hi:
            log[key] = f"raw proposal {val:.4g} outside [{lo}, {hi}]"
            out[key] = float("nan")
    # γ shape veto cascades to γ rate
    if "gamma_shape" in log and "gamma_rate" not in log:
        log["gamma_rate"] = "cascaded from gamma_shape veto"
        out["gamma_rate"] = float("nan")
    # Psi_* diagonals
    for key in ("Psi_B", "Psi_H", "Psi_V"):
        mat = out.get(key)
        if mat is None:
            log[key] = "no proposal produced"
            continue
        diag = np.asarray(mat).diagonal()
        lo, hi = PSI_DIAG_FLOOR
        if not (np.all(diag >= lo) and np.all(diag <= hi)):
            log[key] = f"diag {diag.tolist()} outside [{lo}, {hi}]"
            out[key] = None
    # mu_H per-axis
    mu_H = out.get("mu_H")
    if mu_H is not None:
        arr = np.asarray(mu_H)
        lo, hi = MU_H_RANGE
        if not (np.all(arr >= lo) and np.all(arr <= hi)):
            log["mu_H"] = f"{arr.tolist()} outside [{lo}, {hi}]"
            out["mu_H"] = None
    return out, log


def _damp_field(prop, cur, alpha: float, is_array: bool = False):
    """θ_new = α · prop + (1-α) · cur. NaN / None prop → no change."""
    if prop is None:
        return cur
    if is_array:
        prop_arr = np.asarray(prop, dtype=np.float32)
        if not np.all(np.isfinite(prop_arr)):
            return cur
        cur_arr = np.asarray(cur, dtype=np.float32)
        return (alpha * prop_arr + (1.0 - alpha) * cur_arr).astype(np.float32)
    if not np.isfinite(prop):
        return cur
    return float(alpha * float(prop) + (1.0 - alpha) * float(cur))


def _damp_em(em: EmState, vetoed: dict, alpha: float) -> EmState:
    """Apply damped update to ALL 18 fields. Vetoed (NaN/None) fields inherit.

    use_calibrated_priors becomes True the moment any Psi_*/nu_*/mu_H field
    actually changes (i.e. a non-NaN damped value lands), so the tracker
    starts using the YAML overrides.
    """
    new = _clone_em(em)
    new.sigma_F = _damp_field(vetoed.get("sigma_F"), em.sigma_F, alpha)
    new.sigma_F_H = _damp_field(vetoed.get("sigma_F_H"), em.sigma_F_H, alpha)
    new.outlier_prob = _damp_field(vetoed.get("outlier_prob"), em.outlier_prob, alpha)
    new.gamma_shape = _damp_field(vetoed.get("gamma_shape"), em.gamma_shape, alpha)
    new.gamma_rate = _damp_field(vetoed.get("gamma_rate"), em.gamma_rate, alpha)
    new.alpha = _damp_field(vetoed.get("alpha"), em.alpha, alpha)
    new.beta = _damp_field(vetoed.get("beta"), em.beta, alpha)
    new.sigma_H = _damp_field(vetoed.get("sigma_H"), em.sigma_H, alpha)
    # sigma_V is now a learnable motion-group scalar (closed-form M-step proxy in
    # aggregate_posteriors). Damp it like the other scalars; a vetoed/NaN proposal
    # leaves it at the current (finite-seeded) value.
    new.sigma_V = _damp_field(vetoed.get("sigma_V"), em.sigma_V, alpha)
    new.nu_B = _damp_field(vetoed.get("nu_B"), em.nu_B, alpha)
    new.nu_H = _damp_field(vetoed.get("nu_H"), em.nu_H, alpha)
    new.nu_V = _damp_field(vetoed.get("nu_V"), em.nu_V, alpha)
    new.translation_max_radius = _damp_field(
        vetoed.get("translation_max_radius"), em.translation_max_radius, alpha)
    new.translation_gaussian_scale = _damp_field(
        vetoed.get("translation_gaussian_scale"), em.translation_gaussian_scale, alpha)
    new.Psi_B = _damp_field(vetoed.get("Psi_B"), em.Psi_B, alpha, is_array=True)
    new.Psi_H = _damp_field(vetoed.get("Psi_H"), em.Psi_H, alpha, is_array=True)
    new.Psi_V = _damp_field(vetoed.get("Psi_V"), em.Psi_V, alpha, is_array=True)
    new.mu_H = _damp_field(vetoed.get("mu_H"), em.mu_H, alpha, is_array=True)
    # Any non-vetoed Psi_*/nu_*/mu_H means the candidate now carries
    # calibrated values worth swapping into the tracker.
    if any(vetoed.get(k) is not None and (
                np.all(np.isfinite(np.asarray(vetoed[k]))) if not np.isscalar(vetoed[k])
                else np.isfinite(vetoed[k]))
            for k in ("Psi_B", "Psi_H", "Psi_V", "nu_B", "nu_H", "nu_V", "mu_H")):
        new.use_calibrated_priors = True
    return new


def _overlay_group_fields(target: EmState, source: EmState, group: str) -> EmState:
    """Copy only the given group's fields from `source` onto `target`.

    Also forwards `use_calibrated_priors` from source so the trial config
    actually uses the overlaid Psi_*/nu_*/mu_H instead of silently falling
    back to the K-means seed path (without this, the iter-0 accept-test
    would compare baseline-vs-baseline).
    """
    out = _clone_em(target)
    for key in GROUPS[group]:
        val = getattr(source, key)
        if isinstance(val, np.ndarray):
            setattr(out, key, val.copy())
        else:
            setattr(out, key, val)
    out.use_calibrated_priors = bool(source.use_calibrated_priors or
                                       out.use_calibrated_priors)
    return out


def _json_safe_proposals(proposals: dict) -> dict:
    """Convert ndarray entries to lists for em_iter_*.json serialization."""
    out: dict = {}
    for k, v in proposals.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif v is None:
            out[k] = None
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
    return out


def phase_em(out_root: Path, *, force: bool = False,
              max_frames_per_video: int = -1,
              max_iters: int = EM_MAX_OUTER_ITERS,
              num_gibbs_sweeps_per_frame: int = DEFAULT_NUM_GIBBS_SWEEPS,
              em_plateau_tol: float = EM_PLATEAU_TOL,
              keep_ari_diagnostic: bool = True,
              compute_side_metrics: bool = True,
              train_videos: Optional[List[str]] = None,
              em_val_videos: Optional[List[str]] = None,
              sigma_v_seed: Optional[float] = None,
              use_sam_frame0: bool = INFER_USE_SAM_FRAME0,
              num_blobs: int = INFER_NUM_BLOBS,
              num_hyperblobs: int = INFER_NUM_HYPERBLOBS,
              init_gibbs_sweeps: int = INFER_INIT_GIBBS_SWEEPS,
              feature_aware_final: bool = INFER_FEATURE_AWARE_FINAL,
              final_outlier: bool = INFER_FINAL_OUTLIER,
              freeze_hyperblob: bool = INFER_FREEZE_HYPERBLOB,
              pure_object_seed: bool = INFER_PURE_OBJECT_SEED,
              blob_means_updates: int = INFER_BLOB_MEANS_UPDATES,
              freeze_blob_features: bool = INFER_FREEZE_BLOB_FEATURES,
              feature_update_damping: float = INFER_FEATURE_UPDATE_DAMPING,
              final_feature_temp: float = INFER_FINAL_FEATURE_TEMP,
              final_assignment_anchor: bool = INFER_FINAL_ASSIGNMENT_ANCHOR) -> dict:
    """Outer EM loop: run the deep-N-sweep tracker → MAP-posterior M-step → veto →
    damp → per-group composite-J accept-test → repeat.

    Primary (self-supervised) objective: the median COMPOSITE score (binary
    region-J dominant + per-instance matched J + temporal continuity + motion-ARI
    + blob-J vs each video's cached SAM2 pseudo-labels), NO ground truth. The
    inference strategy is the strong one: SAM2 frame-0 semantic init, num_blobs=128,
    deep per-frame Gibbs (num_gibbs_sweeps_per_frame) + a long warmup
    (init_gibbs_sweeps). The deeper E-step de-biases the conjugate M-step's
    sufficient stats (the MCEM/SAEM lever); the real DAVIS GT is held out for
    validation only.

    Generalization loop (Phase D): when ``em_val_videos`` is given, the M-step
    pools sufficient stats over TRAIN \\ em_val_videos and ``val_composite`` is
    scored on em_val_videos (a held-WITHIN-TRAIN probe). ``best_em`` is then the
    iterate with the best VAL composite (not train), with patience-based early
    stop — so the shipped hypers are the best-GENERALIZING iterate, all measured
    self-supervised. ``sigma_v_seed`` (when given) overrides the initial sigma_V
    (the phase_select motion-anchor fallback axis). See the two-level (filtering +
    SAEM) framing in ~/.claude/plans/velvet-knitting-nova.md.
    """
    info_path = out_root / "em.json"
    cached = _load_json(info_path)
    if cached and not force:
        _log("em: reusing cached")
        return cached

    _ensure_jax_setup()
    videos = discover_videos()
    infer = _infer_cfg(use_sam_frame0=use_sam_frame0, num_blobs=num_blobs,
                       num_hyperblobs=num_hyperblobs, init_gibbs_sweeps=init_gibbs_sweeps,
                       feature_aware_final=feature_aware_final, final_outlier=final_outlier,
                       freeze_hyperblob=freeze_hyperblob, pure_object_seed=pure_object_seed,
                       blob_means_updates=blob_means_updates,
                       freeze_blob_features=freeze_blob_features,
                       feature_update_damping=feature_update_damping,
                       final_feature_temp=final_feature_temp,
                       final_assignment_anchor=final_assignment_anchor)
    # TRAIN-split restriction: learning (main pass + M-step + accept-test) sees
    # ONLY train_videos; held-out + GT are reserved for phase_validate. None =
    # legacy all-cached behaviour.
    train_set = set(train_videos) if train_videos is not None else None
    info: dict = {"phase": "em", "iters": [], "n_videos": len(videos),
                  "num_gibbs_sweeps_per_frame": num_gibbs_sweeps_per_frame,
                  "em_plateau_tol": em_plateau_tol,
                  "objective_mode": _OBJECTIVE_MODE,
                  "composite_weights": dict(COMPOSITE_WEIGHTS),
                  "train_videos": sorted(train_set) if train_set is not None else None,
                  "objective": ("self-supervised median COMPOSITE on TRAIN "
                                f"(mode={_OBJECTIVE_MODE}; best iterate by val_composite "
                                "when em_val_videos set)"),
                  "infer": infer}
    _log(f"em: inference strategy — use_sam_frame0={infer['use_sam_frame0']} "
         f"num_blobs={infer['num_blobs']} init_gibbs_sweeps={infer['init_gibbs_sweeps']} "
         f"per_frame_sweeps={num_gibbs_sweeps_per_frame} objective={_OBJECTIVE_MODE}")

    em = _initial_em_state()
    if sigma_v_seed is not None:
        em.sigma_V = float(sigma_v_seed)   # phase_select motion-anchor fallback axis
        _log(f"em: sigma_V seeded at {em.sigma_V:.3e} (motion-anchor seed override)")
    prior_defaults = _defaults_for_priors()

    labels_cache: Dict[str, dict] = {}
    for vid in videos:
        if train_set is not None and vid not in train_set:
            continue   # held-out video — never enters learning
        path = LABELS_DIR / f"{vid}.npz"
        if not path.is_file():
            _log(f"em: skipping {vid} — labels not cached")
            continue
        try:
            labels_cache[vid] = _load_labels(vid)
        except Exception as e:
            _log(f"em: failed to load {vid} labels: {e}")
    info["n_labels_loaded"] = len(labels_cache)
    # Generalization split (Phase D): em_val_videos ⊆ labels_cache held out of the
    # M-step pool and scored as the val probe. pool_set = the M-step pooling set
    # (TRAIN \ em_val); val_set = the generalization probe. Empty val_set ==
    # legacy behaviour (pool == all, best-by-train).
    val_set = (set(em_val_videos) & set(labels_cache)) if em_val_videos else set()
    pool_set = set(labels_cache) - val_set
    info["val_videos"] = sorted(val_set) if val_set else None
    info["pool_videos"] = sorted(pool_set)
    if train_set is not None:
        _log(f"em: TRAIN split — learning on {len(labels_cache)}/{len(train_set)} "
             f"cached train videos (held-out reserved for validate)")
    if val_set:
        _log(f"em: generalization loop — M-step pools {len(pool_set)} videos, "
             f"val_composite scored on {len(val_set)} disjoint TRAIN videos "
             f"(best iterate by val, patience={EM_VAL_PATIENCE})")

    sam_grids = _build_sam_grids(videos, labels_cache, infer["use_sam_frame0"])

    best_obj = float("-inf")     # the selection objective (val_composite or train)
    best_em = em
    best_snapshot: dict = {}
    prev_J_for_plateau: List[float] = []
    iters_since_val_improve = 0

    for iter_t in range(max_iters):
        _log(f"em iter {iter_t}: running {num_gibbs_sweeps_per_frame}-sweep tracker "
             f"on {len(labels_cache)} videos (sigma_F={em.sigma_F:.4f} "
             f"use_calibrated_priors={em.use_calibrated_priors})")
        yaml_cfg = _make_yaml_cfg(em, infer)
        per_vid_score: Dict[str, dict] = {}      # PRIMARY self-supervised composite objective
        per_vid_jmean: Dict[str, dict] = {}      # logged side-metric (matter-J)
        per_vid_ari: Dict[str, dict] = {}
        per_vid_pseudo: Dict[str, dict] = {}     # M-step pool stats (TRAIN \ val)
        per_vid_tracker: Dict[str, dict] = {}    # M-step pool stats (TRAIN \ val)
        tracker_errors: Dict[str, str] = {}
        for j, vid in enumerate(labels_cache.keys(), 1):
            entry = videos[vid]
            t0 = time.monotonic()
            is_davis = (entry.kind == "davis")
            # All 38 videos: deep tracker (capture weights for SAM2-J), SAM init.
            tr = _run_tracker_on_video(labels_cache[vid], yaml_cfg, max_frames_per_video,
                                       num_sweeps=num_gibbs_sweeps_per_frame,
                                       capture_blob_weights=True,
                                       sam_grid=sam_grids.get(vid), vid=vid,
                                       capture_data_loglik=(_OBJECTIVE_MODE == "data_loglik"))
            if "error" in tr:
                tracker_errors[vid] = tr["error"]
                _log(f"em iter {iter_t} [{j}/{len(labels_cache)}] {vid}: tracker FAILED")
                continue
            # M-step sufficient stats from POOL videos only (TRAIN \ em_val). The
            # held-WITHIN-TRAIN val probe never enters the conjugate M-step pool —
            # this is what makes val_composite a genuine generalization signal.
            if vid in pool_set:
                try:
                    pseudo_stats, tracker_stats = per_video_sufficient_stats(
                        labels_cache[vid], tr["blob_a"], tr["hyperblob_a"])
                    per_vid_pseudo[vid] = pseudo_stats
                    per_vid_tracker[vid] = tracker_stats
                except Exception as e:
                    _log(f"em iter {iter_t} [{j}/{len(labels_cache)}] {vid}: mstep FAILED: {e}")
                    traceback.print_exc()
            # Self-supervised PRIMARY objective: the COMPOSITE score (binary
            # region-J dominant + per-instance matched J + temporal continuity +
            # motion-ARI + blob-J vs cached Z_sam), NO ground truth. region_J is
            # kept alongside (logging + the kill-switch fallback objective).
            cs = _self_sup_score(vid, tr["hyperblob_a"], tr["blob_a"], tr.get("indices"),
                                 labels_cache[vid],
                                 data_loglik=_pick_data_loglik(tr))
            per_vid_score[vid] = {**cs, "kind": entry.kind,
                                  "matter_fps": tr["matter_fps"],
                                  "outlier_frac_p95": tr["outlier_frac_p95"],
                                  "is_val": vid in val_set}
            # Logged side-metric (NOT the objective): matter-weighted Jaccard vs
            # the same SAM masks. SKIPPED when compute_side_metrics=False (model
            # selection): the GenMatter++ matter-J evaluator dominates per-video
            # wall time (~20s) and only the composite objective drives selection.
            jm_str = ""
            if compute_side_metrics:
                jm = _score_video_jmean(vid, tr["blob_a"], tr.get("blob_w"),
                                        tr.get("n_blobs"), tr.get("indices"),
                                        annotations_path=_sam_annotations_path(entry))
                per_vid_jmean[vid] = {**jm, "kind": entry.kind,
                                      "matter_fps": tr["matter_fps"],
                                      "outlier_frac_p95": tr["outlier_frac_p95"]}
                jm_str = f" (matter-J={jm['J_mean']:.4f})"
            # Optional cheap side-diagnostic: consistency-ARI on local videos.
            if (not is_davis) and keep_ari_diagnostic:
                sc = _score_video(vid, tr["blob_a"], tr["hyperblob_a"], labels_cache[vid])
                per_vid_ari[vid] = {**sc, "outlier_frac_p95": tr["outlier_frac_p95"]}
            tag = "val" if vid in val_set else "pool"
            _log(f"em iter {iter_t} [{j}/{len(labels_cache)}] {vid} ({entry.kind},{tag}): "
                 f"composite={cs['composite']:.4f} (region-J={cs['region_J']:.4f}, "
                 f"inst={cs['inst_J']:.3f}, temp={cs['temporal']:.3f}){jm_str} "
                 f"nblobs={tr.get('n_blobs')} wall={time.monotonic()-t0:.1f}s")

        # ---- Aggregate the self-supervised objective. train_composite = median
        # over the M-step POOL; val_composite = median over the held-within-TRAIN
        # probe (Phase D generalization signal). The selection objective J_obj is
        # val_composite when a val set exists, else train_composite. ----
        def _median_over(vids, key="score") -> float:
            vals = [per_vid_score[v][key] for v in vids
                    if v in per_vid_score and np.isfinite(per_vid_score[v].get(key, float("nan")))]
            return float(np.median(vals)) if vals else float("nan")

        train_composite = _median_over(pool_set, "score")
        val_composite = _median_over(val_set, "score") if val_set else float("nan")
        train_region_J = _median_over(pool_set, "region_J")
        J_obj = val_composite if val_set else train_composite
        matter_J_t = _aggregate_J(per_vid_jmean)     # logged side-metric (matter-J)
        davis_J = _median_over([v for v in pool_set
                                if per_vid_score.get(v, {}).get("kind") == "davis"], "region_J")
        local_J = _median_over([v for v in pool_set
                                if per_vid_score.get(v, {}).get("kind") == "local"], "region_J")
        local_S = _aggregate_S(per_vid_ari) if per_vid_ari else float("nan")
        comp_medians = {k: _median_over(pool_set, k)
                        for k in ("region_J", "inst_J", "temporal", "motion_ari", "blob_J")}
        iter_log: dict = {
            "iter": iter_t,
            "train_composite": train_composite,        # PRIMARY objective (pool)
            "val_composite": val_composite,            # generalization probe (Phase D)
            "objective": J_obj,                        # the iterate-selection objective
            "region_J": train_region_J,                # binary region-J on pool (kill-switch)
            "J_mean": train_composite,                 # back-compat alias (now composite)
            "matter_J": matter_J_t,                    # logged side-metric (matter-weighted)
            "davis_region_J": davis_J,
            "local_region_J": local_J,
            "composite_components_pool_median": comp_medians,
            "per_vid_composite": {v: float(s["composite"]) for v, s in per_vid_score.items()},
            "per_vid_region_J": {v: float(s["region_J"]) for v, s in per_vid_score.items()},
            "per_vid_matter_J": {v: float(s["J_mean"]) for v, s in per_vid_jmean.items()},
            "local_diagnostics": {
                "median_S_ari": local_S,
                "per_vid_S": {v: float(s["S"]) for v, s in per_vid_ari.items()},
            },
            "hypers_before": em.to_dict(),
            "tracker_errors": tracker_errors,
        }
        n_scored = len(per_vid_score)
        _log(f"em iter {iter_t}: train_composite={train_composite:.4f} "
             f"val_composite={val_composite:.4f} (objective={J_obj:.4f}; {n_scored} scored; "
             f"region-J pool={train_region_J:.4f} DAVIS {davis_J:.4f} / local {local_J:.4f}); "
             f"matter-J (side) = {matter_J_t:.4f}; local diag S_ari = {local_S:.4f}")

        # Best iterate by the generalization objective (val when present). Snapshot
        # the matching region_J so emit/report can show the binary metric too.
        if np.isfinite(J_obj) and J_obj > best_obj:
            best_obj = J_obj
            best_em = _clone_em(em)
            best_snapshot = {"iter": iter_t, "train_composite": train_composite,
                             "val_composite": val_composite, "region_J": train_region_J}
            iters_since_val_improve = 0
        else:
            iters_since_val_improve += 1

        # Conjugate-posterior MAP across the POOL videos (sufficient stats from both
        # partitions; deeper sweeps de-bias these — Level-1 of the framework).
        raw_proposals = aggregate_posteriors(
            per_vid_pseudo, per_vid_tracker, prior_defaults)
        iter_log["raw_proposals"] = _json_safe_proposals(raw_proposals)

        # Veto on RAW proposals (protects conjugate-update validity)
        vetoed, veto_log = _apply_sanity_veto(raw_proposals)
        iter_log["vetoed_layers"] = veto_log

        # Damped SAEM step (α=0.3), then per-group composite accept on the crossval
        # subset (Level-2 guard). Retry alphas (0.15, 0.075, 0.0375) re-damp per group.
        candidate = _damp_em(em, vetoed, EM_ALPHA)
        em_next, group_log = _accept_groups(em, candidate, vetoed,
                                             labels_cache, videos, sam_grids, infer,
                                             max_frames_per_video,
                                             num_gibbs_sweeps_per_frame,
                                             em_plateau_tol)
        any_accepted = any(g.get("accepted") for g in group_log.values())
        iter_log["group_accepts"] = group_log
        iter_log["accepted"] = any_accepted
        iter_log["hypers_after"] = em_next.to_dict()
        iter_log["iters_since_val_improve"] = iters_since_val_improve
        em = em_next

        info["iters"].append(iter_log)
        _save_json(out_root / f"em_iter_{iter_t}.json", iter_log)
        _save_json(info_path, info)

        # Early stop: patience on the val objective (overfit signal) when a val set
        # exists; else the legacy plateau (3 iters within tol AND no group accepted).
        if val_set:
            if iters_since_val_improve >= EM_VAL_PATIENCE:
                iter_log["stopped_reason"] = "val_patience"
                _save_json(out_root / f"em_iter_{iter_t}.json", iter_log)
                _log(f"em: val_composite has not improved for {EM_VAL_PATIENCE} iters "
                     f"(best={best_obj:.4f}); early-stopping (overfit guard)")
                break
        else:
            prev_J_for_plateau.append(J_obj)
            if len(prev_J_for_plateau) >= 3:
                recent = prev_J_for_plateau[-3:]
                if max(recent) - min(recent) < em_plateau_tol and not any_accepted:
                    iter_log["stopped_reason"] = "plateau"
                    _save_json(out_root / f"em_iter_{iter_t}.json", iter_log)
                    _log(f"em: plateau detected (last 3 objective within {em_plateau_tol}, "
                         f"no group accepted this iter); stopping")
                    break

    info["best_hypers"] = best_em.to_dict()
    info["best_snapshot"] = best_snapshot
    info["best_objective"] = best_obj
    # best_region_J / best_J_mean kept for downstream readers (emit header, select
    # rec). They now carry the region_J / composite of the best-GENERALIZING iterate.
    info["best_region_J"] = best_snapshot.get("region_J", best_obj)
    info["best_composite"] = best_snapshot.get("val_composite" if val_set else "train_composite", best_obj)
    info["best_J_mean"] = info["best_composite"]
    _save_json(info_path, info)
    _log(f"em done: best {'val' if val_set else 'train'}_composite={best_obj:.4f} "
         f"(region-J={info['best_region_J']:.4f}) at iter {best_snapshot.get('iter')}")
    return info


def _score_subset_J(em: EmState, labels_cache: Dict[str, dict],
                    videos: Dict[str, VideoEntry], sam_grids: Dict[str, object],
                    infer: dict, max_frames: int, vids: List[str], num_sweeps: int
                    ) -> Tuple[float, dict]:
    """Run the deep-N-sweep tracker (SAM init) on `vids` from cached perception,
    score each via the SELF-SUPERVISED COMPOSITE objective (CLUSTER region-J vs
    Z_sam>0 + per-instance matched J + temporal continuity + motion-ARI + blob-J;
    NO ground truth) — the Level-2 accept-test guard on the SAEM step. Returns
    (median_score, per_vid_score). capture_blob_weights stays on so the tracker
    path is bit-identical to the EM main pass."""
    yaml_cfg = _make_yaml_cfg(em, infer)
    per_vid_J: Dict[str, float] = {}
    for vid in vids:
        if vid not in labels_cache:
            continue
        tr = _run_tracker_on_video(labels_cache[vid], yaml_cfg, max_frames,
                                   num_sweeps=num_sweeps, capture_blob_weights=True,
                                   sam_grid=sam_grids.get(vid), vid=vid,
                                   capture_data_loglik=(_OBJECTIVE_MODE == "data_loglik"))
        if "error" in tr:
            continue
        sc = _self_sup_score(vid, tr["hyperblob_a"], tr["blob_a"], tr.get("indices"),
                             labels_cache[vid], data_loglik=_pick_data_loglik(tr))
        if np.isfinite(sc["score"]):
            per_vid_J[vid] = float(sc["score"])
    if not per_vid_J:
        return float("nan"), per_vid_J
    return float(np.median(list(per_vid_J.values()))), per_vid_J


def _accept_groups(current: EmState, candidate: EmState, vetoed: dict,
                    labels_cache: Dict[str, dict],
                    videos: Dict[str, VideoEntry], sam_grids: Dict[str, object],
                    infer: dict, max_frames: int, num_sweeps: int, tol: float
                    ) -> Tuple[EmState, dict]:
    """Per-group accept-test on the CROSSVAL subset (DAVIS + local), gated on the
    SELF-SUPERVISED REAL region-J (the Level-2 guard on the damped SAEM step).

    Iterates (cluster, particle, motion). For each:
      • Build trial = overlay group's candidate fields onto current accepted.
      • Score trial's median composite (reference='sam') on CROSSVAL; > anchor+tol → accept.
      • Else retry α=0.15, 0.075, 0.0375 (re-damp this group only from raw vetoed proposal).
      • If ALL α attempts regress → put the group on a short COOLDOWN (sit out
        EM_GROUP_COOLDOWN iters, then re-attempt). This replaces the old
        PERMANENT soft-freeze: the accept-test is itself the regression guard, so
        a group that merely fluctuated in the SAEM neighborhood should be retried
        with fresh M-step proposals — especially the motion group, now that
        sigma_V is finite and the velocity model is live.
    """
    log: dict = {}
    crossval = [v for v in CROSSVAL_VIDEOS if v in labels_cache]
    if not crossval:
        for g in GROUPS:
            log[g] = {"accepted": False, "reason": "no_crossval_videos"}
        return current, log

    # Anchor: current EM's SAM2-J on the crossval subset (computed once)
    J_anchor, _ = _score_subset_J(current, labels_cache, videos, sam_grids, infer,
                                  max_frames, crossval, num_sweeps)
    _log(f"  accept anchor: current J_sub={J_anchor:.4f}")

    accepted = _clone_em(current)
    for group_name in ("cluster", "particle", "motion"):
        cooldown = int(current.group_frozen.get(group_name, 0))
        if cooldown > 0:
            # Cooldown (NOT a permanent freeze): sit this iter out, decrement, and
            # re-attempt once it elapses with fresh M-step proposals.
            accepted.group_frozen[group_name] = cooldown - 1
            log[group_name] = {"accepted": False,
                               "reason": f"cooldown ({cooldown} iter(s) left)",
                               "cooldown_left": cooldown - 1}
            _log(f"  accept[{group_name}]: cooldown {cooldown}→{cooldown - 1} (re-attempt later)")
            continue
        attempts: List[dict] = []
        accepted_this_group = False
        alpha = EM_ALPHA
        for attempt_idx in range(EM_MAX_ALPHA_RETRIES + 1):
            # Re-damp the vetoed proposal at this α (only the group's keys
            # would change, but _damp_em is a no-op for vetoed/NaN fields)
            cand_at_alpha = _damp_em(current, vetoed, alpha)
            trial = _overlay_group_fields(accepted, cand_at_alpha, group_name)
            J_trial, _ = _score_subset_J(trial, labels_cache, videos, sam_grids,
                                          infer, max_frames, crossval, num_sweeps)
            attempts.append({"alpha": float(alpha), "J_trial": J_trial})
            _log(f"  accept[{group_name}] α={alpha:.3f}: J_trial={J_trial:.4f} "
                 f"vs J_accepted={J_anchor:.4f}")
            if (np.isfinite(J_trial) and np.isfinite(J_anchor) and
                    J_trial > J_anchor + tol):
                accepted = trial
                # Re-anchor: future groups compare against this improved baseline
                J_anchor = J_trial
                accepted_this_group = True
                break
            alpha /= EM_ALPHA_RETRY_DIVISOR
        # Cooldown only when ALL α attempts gave J strictly worse than anchor.
        # Marginal-positive-but-below-tol cases get retried next iter with fresh
        # M-step proposals (a group merely fluctuating in the SAEM neighborhood).
        all_negative = all(
            (not np.isfinite(a["J_trial"])) or a["J_trial"] < J_anchor
            for a in attempts
        )
        log[group_name] = {
            "accepted": accepted_this_group,
            "attempts": attempts,
            "accepted_alpha": float(attempts[-1]["alpha"]) if accepted_this_group else None,
        }
        if accepted_this_group:
            accepted.group_frozen[group_name] = 0                     # stays active
        elif all_negative:
            accepted.group_frozen[group_name] = EM_GROUP_COOLDOWN     # short cooldown
            log[group_name]["cooldown_set"] = EM_GROUP_COOLDOWN
        else:
            accepted.group_frozen[group_name] = 0                     # marginal: retry next iter
    return accepted, log


# ----------------------------------------------------------------------
# Phase 4: validate + YAML/report emission
# ----------------------------------------------------------------------

def _load_selected_infer(out_root: Path) -> Optional[dict]:
    """Return the phase_select-chosen global scalar tuple (num_blobs /
    init_gibbs_sweeps / num_gibbs_sweeps_per_frame) from selected_infer.json, or
    None if selection has not run. This tuple is AUTHORITATIVE for em/validate/
    emit so what-we-select == what-we-validate == what-we-ship."""
    sel = _load_json(out_root / "selected_infer.json")
    if not sel or "num_blobs" not in sel:
        return None
    return sel


def phase_select(out_root: Path, *, force: bool = False,
                 max_frames_per_video: int = -1,
                 grid: Optional[dict] = None, em_iters: int = 2,
                 base_infer_kwargs: Optional[dict] = None) -> dict:
    """Self-supervised model selection of the ONE global scalar tuple — the only
    NON-conjugate global knobs (num_blobs / init_gibbs_sweeps /
    num_gibbs_sweeps_per_frame). The conjugate hypers are learned by the M-step;
    these few discrete scalars are picked by grid search, scored ENTIRELY
    self-supervised on TRAIN (region-J of the CLUSTER foreground vs Z_sam>0). The
    held-out set and DAVIS GT are NEVER touched here.

    For each grid combo: run a SHORT EM (``em_iters``, default 2) on TRAIN to fit
    the conjugate hypers UNDER that inference combo, then score the learned hypers
    by median self-supervised region-J on SELECT_VAL_VIDEOS (a TRAIN subset
    disjoint from the SAEM accept-test's CROSSVAL videos). Pick the combo with the
    best median region-J; tie-break toward the SIMPLER/faster config (smaller
    num_blobs, then fewer per-frame sweeps, then fewer warmup sweeps), then lower
    median outlier-p95. Per-combo checkpointed in select_grid.json (resumable);
    the winner is written to selected_infer.json.
    """
    _assert_split_discipline()
    if grid is None:
        grid = {"num_blobs": [64, 128], "init_gibbs_sweeps": [1, 5, 15],
                "num_gibbs_sweeps_per_frame": [1, 4]}
    # New non-conjugate axes default to a SINGLE value (no grid explosion). The
    # sigma_V axis is the motion-anchor fallback: pass e.g. [1e14, 0.1] to let
    # selection choose self-supervised between "anchor off" (1e14) and "anchor on"
    # (the finite seed). blob_means_updates is the crude per-frame anti-bleed proxy.
    grid.setdefault("sigma_V", [_SIGMA_V_SEED])
    grid.setdefault("blob_means_updates", [INFER_BLOB_MEANS_UPDATES])
    # Phase-A anti-drift damping axis: default a SINGLE value (no explosion).
    # Pass e.g. [0.0, 0.25, 0.5, 1.0] to SELECT the damping self-supervised on
    # TRAIN (scored by the Phase-B data_loglik objective on SELECT_VAL).
    grid.setdefault("feature_update_damping", [INFER_FEATURE_UPDATE_DAMPING])
    # Phase-1 INFERENCE feature-temperature axis: default a SINGLE value (no
    # explosion). Pass e.g. [1.0, 2.0, 4.0, 8.0] to SELECT tau self-supervised on
    # TRAIN (scored by the data_loglik objective on SELECT_VAL). Capped (no
    # feat_only) keeps a position floor against the R-circ degeneracy.
    grid.setdefault("final_feature_temp", [INFER_FINAL_FEATURE_TEMP])
    if base_infer_kwargs is None:
        base_infer_kwargs = dict(use_sam_frame0=INFER_USE_SAM_FRAME0,
                                 num_hyperblobs=INFER_NUM_HYPERBLOBS,
                                 num_blobs=INFER_NUM_BLOBS,
                                 init_gibbs_sweeps=INFER_INIT_GIBBS_SWEEPS)
    grid_path = out_root / "select_grid.json"
    sel_path = out_root / "selected_infer.json"
    cached = _load_json(grid_path) if not force else None
    results: List[dict] = list(cached.get("combos", [])) if cached else []
    done = {tuple(c["combo"]) for c in results}

    _ensure_jax_setup()
    videos = discover_videos()
    # SELECT_VAL labels (the inner self-supervised scoring subset of TRAIN).
    labels_cache: Dict[str, dict] = {}
    for vid in SELECT_VAL_VIDEOS:
        if (LABELS_DIR / f"{vid}.npz").is_file():
            try:
                labels_cache[vid] = _load_labels(vid)
            except Exception as e:
                _log(f"select: failed to load {vid} labels: {e}")
    _log(f"select: scoring on {len(labels_cache)}/{len(SELECT_VAL_VIDEOS)} SELECT_VAL "
         f"videos (self-supervised COMPOSITE, reference=sam); short EM on TRAIN "
         f"({len(TRAIN_VIDEOS)} videos), {em_iters} iters/combo")

    # Combo = (num_blobs, init_gibbs_sweeps, per_frame_sweeps, sigma_V,
    #          blob_means_updates, feature_update_damping, final_feature_temp).
    combos = list(itertools.product(
        grid["num_blobs"], grid["init_gibbs_sweeps"], grid["num_gibbs_sweeps_per_frame"],
        grid["sigma_V"], grid["blob_means_updates"], grid["feature_update_damping"],
        grid["final_feature_temp"]))
    _log(f"select: {len(combos)} grid combos "
         f"(num_blobs={grid['num_blobs']} init_gibbs_sweeps={grid['init_gibbs_sweeps']} "
         f"per_frame_sweeps={grid['num_gibbs_sweeps_per_frame']} "
         f"sigma_V={grid['sigma_V']} blob_means_updates={grid['blob_means_updates']} "
         f"feature_update_damping={grid['feature_update_damping']} "
         f"final_feature_temp={grid['final_feature_temp']})")

    for ci, (nb, igs, nsw, sv, bmu, fud, tau) in enumerate(combos, 1):
        combo_key = (nb, igs, nsw, float(sv), bmu, float(fud), float(tau))
        sv_tag = f"{float(sv):g}".replace("+", "")
        fud_tag = f"{float(fud):g}"
        tau_tag = f"{float(tau):g}"
        label = f"nb{nb}/igs{igs}/sw{nsw}/sv{sv_tag}/bmu{bmu}/fud{fud_tag}/tau{tau_tag}"
        if combo_key in done:
            _log(f"select [{ci}/{len(combos)}] {label}: cached, skip")
            continue
        t0 = time.monotonic()
        combo_root = out_root / "select" / f"nb{nb}_igs{igs}_sw{nsw}_sv{sv_tag}_bmu{bmu}_fud{fud_tag}_tau{tau_tag}"
        combo_root.mkdir(parents=True, exist_ok=True)
        combo_infer_kwargs = {**base_infer_kwargs, "num_blobs": nb,
                              "init_gibbs_sweeps": igs, "blob_means_updates": bmu,
                              "feature_update_damping": float(fud),
                              "final_feature_temp": float(tau)}
        # Short EM on TRAIN under this inference combo (resumable: phase_em caches
        # combo_root/em.json and returns it on re-entry unless force). The short EM
        # does NOT use the Phase-D val loop (em_val_videos=None) so SELECT_VAL stays
        # purely a combo-selection probe (no circular iterate-selection on it).
        em_info = phase_em(combo_root, force=force,
                           max_frames_per_video=max_frames_per_video,
                           max_iters=em_iters, num_gibbs_sweeps_per_frame=nsw,
                           keep_ari_diagnostic=False, compute_side_metrics=False,
                           train_videos=list(TRAIN_VIDEOS), sigma_v_seed=float(sv),
                           **combo_infer_kwargs)
        em = _em_from_dict(em_info["best_hypers"])
        infer = _infer_cfg(**combo_infer_kwargs)
        yaml_cfg = _make_yaml_cfg(em, infer)
        sam_grids = _build_sam_grids(videos, labels_cache, infer["use_sam_frame0"])
        # Score the learned hypers self-supervised on SELECT_VAL (composite + p95).
        per_vid_score: Dict[str, float] = {}
        per_vid_regionJ: Dict[str, float] = {}
        p95s: List[float] = []
        for vid in labels_cache:
            tr = _run_tracker_on_video(labels_cache[vid], yaml_cfg, max_frames_per_video,
                                       num_sweeps=nsw, capture_blob_weights=True,
                                       sam_grid=sam_grids.get(vid), vid=vid,
                                       capture_data_loglik=(_OBJECTIVE_MODE == "data_loglik"))
            if "error" in tr:
                continue
            sc = _self_sup_score(vid, tr["hyperblob_a"], tr["blob_a"], tr.get("indices"),
                                 labels_cache[vid], data_loglik=_pick_data_loglik(tr))
            if np.isfinite(sc["score"]):
                per_vid_score[vid] = float(sc["score"])
            if np.isfinite(sc["region_J"]):
                per_vid_regionJ[vid] = float(sc["region_J"])
            p95 = tr.get("outlier_frac_p95", float("nan"))
            if np.isfinite(p95):
                p95s.append(float(p95))
        median_score = float(np.median(list(per_vid_score.values()))) if per_vid_score else float("nan")
        median_regionJ = float(np.median(list(per_vid_regionJ.values()))) if per_vid_regionJ else float("nan")
        median_p95 = float(np.median(p95s)) if p95s else float("nan")
        rec = {"combo": [nb, igs, nsw, float(sv), bmu, float(fud), float(tau)],
               "num_blobs": nb, "init_gibbs_sweeps": igs,
               "num_gibbs_sweeps_per_frame": nsw, "sigma_V": float(sv),
               "blob_means_updates": bmu, "feature_update_damping": float(fud),
               "final_feature_temp": float(tau),
               "select_val_score": median_score,         # objective (composite | data_loglik)
               "select_val_region_J": median_regionJ,    # binary region-J (display/diag)
               "select_val_median_p95": median_p95,
               "best_train_score": em_info.get("best_composite"),
               "per_vid_score": per_vid_score}
        results.append(rec)
        done.add(combo_key)
        _save_json(grid_path, {"phase": "select", "grid": grid, "em_iters": em_iters,
                               "objective_mode": _OBJECTIVE_MODE,
                               "select_val_videos": list(SELECT_VAL_VIDEOS),
                               "train_videos": list(TRAIN_VIDEOS), "combos": results})
        _log(f"select [{ci}/{len(combos)}] {label}: "
             f"SELECT_VAL composite={median_score:.4f} (region-J={median_regionJ:.4f}, "
             f"p95={median_p95:.4f}) wall={time.monotonic()-t0:.1f}s")

    finite = [c for c in results if np.isfinite(c.get("select_val_score", float("nan")))]
    if not finite:
        raise RuntimeError("phase_select: no grid combo produced a finite composite score")
    # Best = highest self-supervised composite; tie-break SIMPLER/faster (fewer
    # blobs, fewer per-frame sweeps, fewer warmup sweeps, LARGER sigma_V = less
    # anchoring, fewer blob_means_updates), then lower p95.
    def _combo_at(c, i, default):
        cc = c["combo"]
        return cc[i] if len(cc) > i else default
    best = sorted(finite, key=lambda c: (
        -round(float(c["select_val_score"]), 4),
        int(c["combo"][0]), int(c["combo"][2]), int(c["combo"][1]),
        -float(_combo_at(c, 3, _SIGMA_V_SEED)), int(_combo_at(c, 4, INFER_BLOB_MEANS_UPDATES)),
        -float(_combo_at(c, 5, INFER_FEATURE_UPDATE_DAMPING)),   # prefer LARGER damping (less intervention) on ties
        float(_combo_at(c, 6, INFER_FINAL_FEATURE_TEMP)),        # prefer SMALLER tau (position floor) on ties
        float(c.get("select_val_median_p95") if np.isfinite(
            c.get("select_val_median_p95", float("nan"))) else 1e9),
    ))[0]
    selected = {
        "num_blobs": int(best["combo"][0]),
        "init_gibbs_sweeps": int(best["combo"][1]),
        "num_gibbs_sweeps_per_frame": int(best["combo"][2]),
        "sigma_V_seed": float(_combo_at(best, 3, _SIGMA_V_SEED)),
        "blob_means_updates": int(_combo_at(best, 4, INFER_BLOB_MEANS_UPDATES)),
        "feature_update_damping": float(_combo_at(best, 5, INFER_FEATURE_UPDATE_DAMPING)),
        "final_feature_temp": float(_combo_at(best, 6, INFER_FINAL_FEATURE_TEMP)),
        "select_val_score": float(best["select_val_score"]),
        "select_val_region_J": float(best.get("select_val_region_J", float("nan"))),
        "select_val_median_p95": float(best.get("select_val_median_p95", float("nan"))),
        "objective_mode": _OBJECTIVE_MODE,
        "select_val_videos": list(SELECT_VAL_VIDEOS),
        "train_videos": list(TRAIN_VIDEOS),
        "reference": "sam (self-supervised composite; TRAIN learning + SELECT_VAL scoring; NO GT, NO held-out)",
        "n_combos_scored": len(finite),
    }
    _save_json(sel_path, selected)
    _log(f"select DONE: chose num_blobs={selected['num_blobs']} "
         f"init_gibbs_sweeps={selected['init_gibbs_sweeps']} "
         f"per_frame_sweeps={selected['num_gibbs_sweeps_per_frame']} "
         f"sigma_V_seed={selected['sigma_V_seed']:.3e} "
         f"blob_means_updates={selected['blob_means_updates']} "
         f"feature_update_damping={selected['feature_update_damping']:.3g} "
         f"final_feature_temp={selected['final_feature_temp']:.3g} "
         f"(SELECT_VAL composite={selected['select_val_score']:.4f}); wrote {sel_path}")
    return selected


def _em_from_yaml_cfg(cfg: dict) -> EmState:
    """Reconstruct an EmState from a streaming YAML cfg dict (inverse of the
    hyper-writing half of _make_yaml_cfg). Missing keys keep the DAVIS-DINO
    defaults; calibrated-prior matrices are read only when use_calibrated_priors."""
    trk = cfg.get("tracking", {})
    hp = trk.get("hyperparams", {})
    em = _initial_em_state()
    em.use_calibrated_priors = bool(trk.get("use_calibrated_priors", False))
    em.sigma_F = float(hp.get("sigma_F", em.sigma_F))
    em.sigma_F_H = float(hp.get("sigma_F_H", em.sigma_F_H))
    em.sigma_V = float(hp.get("sigma_V", em.sigma_V))
    em.outlier_prob = float(hp.get("outlier_prob", em.outlier_prob))
    em.gamma_shape = float(hp.get("outlier_velocity_gamma_shape", em.gamma_shape))
    em.gamma_rate = float(hp.get("outlier_velocity_gamma_rate", em.gamma_rate))
    em.alpha = float(hp.get("alpha", em.alpha))
    em.beta = float(hp.get("beta", em.beta))
    em.sigma_H = float(hp.get("sigma_H", em.sigma_H))
    em.translation_max_radius = float(hp.get("translation_max_radius", em.translation_max_radius))
    em.translation_gaussian_scale = float(hp.get("translation_gaussian_scale", em.translation_gaussian_scale))
    if em.use_calibrated_priors:
        if "Psi_B" in hp: em.Psi_B = np.asarray(hp["Psi_B"], dtype=np.float32)
        if "Psi_H" in hp: em.Psi_H = np.asarray(hp["Psi_H"], dtype=np.float32)
        if "Psi_V" in hp: em.Psi_V = np.asarray(hp["Psi_V"], dtype=np.float32)
        if "nu_B" in hp: em.nu_B = float(hp["nu_B"])
        if "nu_H" in hp: em.nu_H = float(hp["nu_H"])
        if "nu_V" in hp: em.nu_V = float(hp["nu_V"])
        if "mu_H" in hp: em.mu_H = np.asarray(hp["mu_H"], dtype=np.float32)
    return em


def _infer_from_yaml_cfg(cfg: dict) -> dict:
    """Reconstruct the inference/structure `infer` dict from a streaming YAML.
    Structural-flag defaults are the VENDORED-OFF values, so a config predating
    the cluster-freeze fix (no structural keys) reads back as the vendored
    structure — exactly how it deployed."""
    trk = cfg.get("tracking", {})
    return _infer_cfg(
        use_sam_frame0=bool(trk.get("use_sam_frame0", False)),
        num_blobs=int(trk.get("num_blobs", 64)),
        num_hyperblobs=int(trk.get("num_hyperblobs", 4)),
        init_gibbs_sweeps=int(trk.get("init_gibbs_sweeps", 15)),
        feature_aware_final=bool(trk.get("feature_aware_final_assignment", False)),
        final_outlier=bool(trk.get("final_assignment_outlier", True)),
        freeze_hyperblob=bool(trk.get("freeze_hyperblob_assignment", False)),
        pure_object_seed=bool(trk.get("pure_object_seed", False)),
        blob_means_updates=int(trk.get("blob_means_updates_per_frame", 15)),
        freeze_blob_features=bool(trk.get("freeze_blob_features", False)),
        feature_update_damping=float(trk.get("feature_update_damping",
                                             INFER_FEATURE_UPDATE_DAMPING)),
        final_feature_temp=float(trk.get("final_feature_temp",
                                         INFER_FINAL_FEATURE_TEMP)),
        final_assignment_anchor=bool(trk.get("final_assignment_anchor",
                                             INFER_FINAL_ASSIGNMENT_ANCHOR)),
        # Read the augmentation flag FROM the (baseline) YAML, NOT the global — so the
        # validate baseline loaded from the un-flagged shipping config stays un-augmented
        # even when the run global --tokencut-seed-augment is ON.
        tokencut_seed_augment=bool(trk.get("tokencut_seed_augment", False)),
        tokencut_knobs=trk.get("tokencut_knobs"),
    )


def _load_shipping_baseline() -> Tuple[EmState, dict, str]:
    """The validate baseline = the CURRENTLY-SHIPPING streaming_general.yaml,
    loaded exactly as it deploys (hypers + inference/structure). This is the
    honest 'before': beating it on held-out GT means the recalibrated, structured
    config genuinely improves on what ships. Falls back to the inline
    streaming_default shipping values (under the vendored structure) when the file
    is absent. Returns (EmState, infer, source_label)."""
    import genmatter_rt
    if GENERAL_YAML.is_file():
        try:
            cfg = genmatter_rt.load_yaml_hypers(GENERAL_YAML)
            return (_em_from_yaml_cfg(cfg), _infer_from_yaml_cfg(cfg),
                    f"streaming_general.yaml (as shipping at validate time)")
        except Exception as e:
            _log(f"validate: could not load baseline from {GENERAL_YAML}: {e}; "
                 f"using inline streaming_default fallback")
    em = _initial_em_state()
    em.use_calibrated_priors = False
    em.sigma_F = 2.0; em.sigma_F_H = 2.0; em.outlier_prob = 5.0
    em.gamma_shape = 5.0; em.gamma_rate = 1.0; em.alpha = 1.0; em.beta = 1.0
    em.sigma_H = 25.0; em.translation_max_radius = 0.35
    em.translation_gaussian_scale = 0.2; em.sigma_V = 1.0e14
    return em, SHIPPING_INFER, "inline streaming_default fallback"


def phase_validate(out_root: Path, *, force: bool = False,
                    max_frames_per_video: int = -1,
                    num_gibbs_sweeps_per_frame: int = DEFAULT_NUM_GIBBS_SWEEPS,
                    heldout_videos: Optional[List[str]] = None,
                    use_sam_frame0: bool = INFER_USE_SAM_FRAME0,
                    num_blobs: int = INFER_NUM_BLOBS,
                    num_hyperblobs: int = INFER_NUM_HYPERBLOBS,
                    init_gibbs_sweeps: int = INFER_INIT_GIBBS_SWEEPS,
                    feature_aware_final: bool = INFER_FEATURE_AWARE_FINAL,
                    final_outlier: bool = INFER_FINAL_OUTLIER,
                    freeze_hyperblob: bool = INFER_FREEZE_HYPERBLOB,
                    pure_object_seed: bool = INFER_PURE_OBJECT_SEED,
                    blob_means_updates: int = INFER_BLOB_MEANS_UPDATES,
                    freeze_blob_features: bool = INFER_FREEZE_BLOB_FEATURES,
                    feature_update_damping: float = INFER_FEATURE_UPDATE_DAMPING,
                    final_feature_temp: float = INFER_FINAL_FEATURE_TEMP,
                    final_assignment_anchor: bool = INFER_FINAL_ASSIGNMENT_ANCHOR) -> dict:
    """Generalization gate on the HELD-OUT split — the ONLY place true DAVIS GT
    is read. Learning + selection (phase_em / phase_select) never saw these
    videos; all 5 demo videos are here, so their quality measures generalization.

    Two passes over the held-out videos:
      • ``baseline`` — the CURRENTLY-SHIPPING streaming_general.yaml, loaded
        exactly as it deploys (hypers + inference/structure; structural flags are
        whatever that file sets — vendored-OFF for a config predating the
        cluster-freeze fix). The honest 'before'.
      • ``chosen``   — the recalibrated hypers (learned UNDER the shipped
        structure on TRAIN) + the phase_select-chosen global scalar tuple + the
        architectural-structure flags ON.

    Per held-out video we score the self-supervised region-J (CLUSTER IoU vs
    Z_sam>0); for the held-out DAVIS we ALSO score real-GT region-J (vs the true
    DAVIS GT) + a matter-weighted GT-J side-metric. The GATE is median Δ real-GT
    region-J over the held-out DAVIS > 0 (and held-out outlier-p95 not regressed).
    Local held-out videos (wine_swirl) get a consistency-ARI diagnostic only.
    """
    _assert_split_discipline()
    info_path = out_root / "validate.json"
    cached = _load_json(info_path)
    if cached and not force:
        _log("validate: reusing cached")
        return cached

    em_info = _load_json(out_root / "em.json")
    if not em_info or "best_hypers" not in em_info:
        raise RuntimeError("validate requires phase_em to have completed")
    chosen = _em_from_dict(em_info["best_hypers"])
    chosen_infer = _infer_cfg(use_sam_frame0=use_sam_frame0, num_blobs=num_blobs,
                              num_hyperblobs=num_hyperblobs,
                              init_gibbs_sweeps=init_gibbs_sweeps,
                              feature_aware_final=feature_aware_final,
                              final_outlier=final_outlier,
                              freeze_hyperblob=freeze_hyperblob,
                              pure_object_seed=pure_object_seed,
                              blob_means_updates=blob_means_updates,
                              freeze_blob_features=freeze_blob_features,
                              feature_update_damping=feature_update_damping,
                              final_feature_temp=final_feature_temp,
                              final_assignment_anchor=final_assignment_anchor)
    # Baseline = the currently-shipping streaming_general.yaml (or inline
    # streaming_default fallback). Captured BEFORE this run's emit overwrites it.
    baseline, baseline_infer, baseline_src = _load_shipping_baseline()
    _log(f"validate: baseline source = {baseline_src}")

    _ensure_jax_setup()
    videos = discover_videos()
    # HELD-OUT restriction: validate (and thus GT) sees ONLY held-out videos.
    held_set = set(heldout_videos) if heldout_videos is not None else None
    labels_cache: Dict[str, dict] = {}
    for vid in videos:
        if held_set is not None and vid not in held_set:
            continue
        path = LABELS_DIR / f"{vid}.npz"
        if path.is_file():
            try:
                labels_cache[vid] = _load_labels(vid)
            except Exception as e:
                _log(f"validate: failed to load {vid} labels: {e}")
    if held_set is not None:
        _log(f"validate: HELD-OUT split — gating on {len(labels_cache)}/{len(held_set)} "
             f"cached held-out videos (GT read here only)")
    # SAM grids only needed for the chosen (SAM-init) pass; harmless for baseline.
    sam_grids = _build_sam_grids(videos, labels_cache, chosen_infer["use_sam_frame0"])

    def _run_pass(label: str, em: EmState, infer: dict, num_sweeps: int) -> Dict[str, dict]:
        yaml_cfg = _make_yaml_cfg(em, infer)
        out: Dict[str, dict] = {}
        vids = list(labels_cache.keys())
        for j, vid in enumerate(vids, 1):
            entry = videos[vid]
            is_davis = (entry.kind == "davis")
            tr = _run_tracker_on_video(labels_cache[vid], yaml_cfg, max_frames_per_video,
                                       num_sweeps=num_sweeps, capture_blob_weights=True,
                                       sam_grid=sam_grids.get(vid), vid=vid,
                                       capture_data_loglik=(_OBJECTIVE_MODE == "data_loglik"))
            if "error" in tr:
                out[vid] = {"error": tr["error"], "kind": entry.kind}
                _log(f"validate[{label}][{j}/{len(vids)}] {vid}: FAILED")
                continue
            rec: dict = {"kind": entry.kind, "matter_fps": tr["matter_fps"],
                         "outlier_frac_p95": tr["outlier_frac_p95"]}
            # Phase-B probabilistic objective value (self-supervised) — captured
            # for the corr(data_loglik, GT) kill-switch below (held-out DAVIS only).
            if _OBJECTIVE_MODE == "data_loglik":
                rec["data_loglik"] = _pick_data_loglik(tr)
                rec["data_loglik_terms"] = tr.get("data_loglik_terms")
            # Self-supervised REAL region-J (CLUSTER IoU vs Z_sam>0) — the EM
            # objective metric, logged here for every video.
            sam_rj = _score_video_region_J(vid, tr["hyperblob_a"], tr.get("indices"),
                                           labels_cache[vid], reference="sam")
            rec["sam2_region_J"] = sam_rj["region_J"]
            # Self-supervised COMPOSITE (reference Z_sam, NO GT) — needed for the
            # composite-vs-GT correlation kill-switch below (held-out DAVIS only).
            comp = _score_video_composite(vid, tr["hyperblob_a"], tr["blob_a"],
                                          tr.get("indices"), labels_cache[vid])
            rec["composite_sam"] = comp["composite"]
            rec["composite_components"] = {k: comp.get(k) for k in
                                           ("region_J", "inst_J", "temporal", "motion_ari", "blob_J")}
            # matter-weighted SAM2-J side-metric (kept for back-compat / diagnostics).
            sam_jm = _score_video_jmean(vid, tr["blob_a"], tr.get("blob_w"),
                                        tr.get("n_blobs"), tr.get("indices"),
                                        annotations_path=_sam_annotations_path(entry))
            rec["sam2_J"] = sam_jm["J_mean"]
            rec["sam2_J_recall"] = sam_jm["J_recall"]
            rec["sam2_J_precision"] = sam_jm["J_precision"]
            if is_davis:
                # GATE METRIC: REAL region-J vs the held-out true DAVIS GT.
                gt_rj = _score_video_region_J(vid, tr["hyperblob_a"], tr.get("indices"),
                                              labels_cache[vid], reference="gt")
                rec["gt_J"] = gt_rj["region_J"]       # gated benchmark (region-IoU)
                # matter-weighted GT-J kept ALONGSIDE as a logged side-metric.
                gt_jm = _score_video_jmean(vid, tr["blob_a"], tr.get("blob_w"),
                                           tr.get("n_blobs"), tr.get("indices"),
                                           annotations_path=_davis_gt_path(vid))
                rec["gt_matter_J"] = gt_jm["J_mean"]
                rec["gt_matter_J_recall"] = gt_jm["J_recall"]
                rec["gt_matter_J_precision"] = gt_jm["J_precision"]
                # back-compat alias so any downstream reader of "J_mean" still works
                rec["J_mean"] = gt_rj["region_J"]
                _log(f"validate[{label}][{j}/{len(vids)}] {vid}: "
                     f"GT-region-J={gt_rj['region_J']:.4f} (GT-matter-J={gt_jm['J_mean']:.4f}) "
                     f"SAM-region-J={sam_rj['region_J']:.4f} "
                     f"nblobs={tr.get('n_blobs')} p95={tr['outlier_frac_p95']:.4f}")
            else:
                sc = _score_video(vid, tr["blob_a"], tr["hyperblob_a"], labels_cache[vid])
                rec["S_ari"] = float(sc["S"])
                _log(f"validate[{label}][{j}/{len(vids)}] {vid}: "
                     f"SAM-region-J={sam_rj['region_J']:.4f} (matter-J={sam_jm['J_mean']:.4f}) "
                     f"S_ari={sc['S']:.4f} (diag) p95={tr['outlier_frac_p95']:.4f}")
            out[vid] = rec
        return out

    # GT scoring is permitted ONLY within this scope (the held-out gate). Both
    # passes score reference="gt" on the held-out DAVIS; the guard makes any GT
    # access from learning/selection raise instead. `_heldout_gate_enabled()` is
    # the held-out-specific lock: this gate is the one place HELD-OUT GT may be read.
    with _gt_scoring_enabled(), _heldout_gate_enabled():
        _log(f"validate: baseline pass ({baseline_src}; 1 sweep/frame)")
        baseline_pv = _run_pass("baseline", baseline, baseline_infer, 1)
        _log("validate: chosen pass (recalibrated hypers + structure: "
             f"SAM init / {num_blobs} blobs / {num_gibbs_sweeps_per_frame}-sweep/frame)")
        chosen_pv = _run_pass("chosen", chosen, chosen_infer, num_gibbs_sweeps_per_frame)

    # ---- Primary gate: median Δ REAL region-J vs the held-out DAVIS GT over the
    # 30 DAVIS videos (the honest benchmark; gt_J now holds region-IoU). ----
    def _median_metric(pv: Dict[str, dict], key: str, kind: Optional[str]) -> float:
        vals = [v[key] for v in pv.values()
                if (kind is None or v.get("kind") == kind)
                and key in v and np.isfinite(v[key])]
        return float(np.median(vals)) if vals else float("nan")

    deltas: List[float] = []
    for vid, b in baseline_pv.items():
        if b.get("kind") != "davis":
            continue
        c = chosen_pv.get(vid)
        if c is None or "gt_J" not in b or "gt_J" not in c:
            continue
        if not (np.isfinite(b["gt_J"]) and np.isfinite(c["gt_J"])):
            continue
        deltas.append(c["gt_J"] - b["gt_J"])
    median_delta = float(np.median(deltas)) if deltas else float("nan")

    # SAM2-J deltas over all 38 (the self-supervised objective) — reported, not gated.
    sam_deltas: List[float] = []
    for vid, b in baseline_pv.items():
        c = chosen_pv.get(vid)
        if c is None or "sam2_J" not in b or "sam2_J" not in c:
            continue
        if np.isfinite(b["sam2_J"]) and np.isfinite(c["sam2_J"]):
            sam_deltas.append(c["sam2_J"] - b["sam2_J"])
    median_sam_delta = float(np.median(sam_deltas)) if sam_deltas else float("nan")

    p95s_chosen = [v["outlier_frac_p95"] for v in chosen_pv.values()
                    if "outlier_frac_p95" in v and np.isfinite(v["outlier_frac_p95"])]
    p95s_baseline = [v["outlier_frac_p95"] for v in baseline_pv.values()
                      if "outlier_frac_p95" in v and np.isfinite(v["outlier_frac_p95"])]
    max_p95 = float(max(p95s_chosen)) if p95s_chosen else float("nan")
    max_p95_baseline = float(max(p95s_baseline)) if p95s_baseline else float("nan")
    # Relative gate: chosen max p95 must not regress vs baseline max p95.
    p95_pass = (np.isfinite(max_p95) and np.isfinite(max_p95_baseline) and
                max_p95 <= max_p95_baseline + 0.05)
    delta_pass = (np.isfinite(median_delta) and median_delta > 0)
    # HARD FLOOR (no-rollback): the chosen AGGREGATE median GT-J must NOT regress
    # below the baseline's. The per-video median DELTA can be marginally positive
    # (a slim majority improve by a hair) while a few large regressions pull the
    # aggregate BELOW the baseline — median-of-deltas != delta-of-medians. Emitting
    # there would SHIP A REGRESSION, so we require BOTH the paired-delta AND the
    # aggregate floor. (Observed: tau0.75 gave median Δ +0.0007 yet chosen 0.6780 <
    # baseline 0.6812 — a regression that the delta-only gate let through.)
    _chosen_med_gt = _median_metric(chosen_pv, "gt_J", "davis")
    _baseline_med_gt = _median_metric(baseline_pv, "gt_J", "davis")
    floor_pass = (np.isfinite(_chosen_med_gt) and np.isfinite(_baseline_med_gt)
                  and _chosen_med_gt >= _baseline_med_gt - 1e-9)
    passed = bool(p95_pass and delta_pass and floor_pass)

    # ---- Composite-vs-GT correlation KILL-SWITCH (Phase C). Measured ONCE, here,
    # inside the GT scope: does the self-supervised composite proxy track held-out
    # GT region-J at least as well as binary region-J does? If the composite
    # ANTI-correlates (spearman < 0), the operator reverts to --objective region_J
    # and re-runs. The GATE metric itself is UNCHANGED (binary GT region-J). ----
    def _pearson(xs: List[float], ys: List[float]) -> float:
        if len(xs) < 3:
            return float("nan")
        x = np.asarray(xs, dtype=np.float64); y = np.asarray(ys, dtype=np.float64)
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def _spearman(xs: List[float], ys: List[float]) -> float:
        if len(xs) < 3:
            return float("nan")
        rx = np.argsort(np.argsort(np.asarray(xs, dtype=np.float64))).astype(np.float64)
        ry = np.argsort(np.argsort(np.asarray(ys, dtype=np.float64))).astype(np.float64)
        return _pearson(rx.tolist(), ry.tolist())

    comp_xs: List[float] = []
    regionj_xs: List[float] = []
    loglik_xs: List[float] = []
    gt_ys: List[float] = []
    loglik_ys: List[float] = []   # GT paired with the (possibly fewer) finite loglik points
    for vid, c in chosen_pv.items():
        if c.get("kind") != "davis":
            continue
        gj = c.get("gt_J"); cs = c.get("composite_sam"); rj = c.get("sam2_region_J")
        if all(v is not None and np.isfinite(v) for v in (gj, cs, rj)):
            gt_ys.append(float(gj)); comp_xs.append(float(cs)); regionj_xs.append(float(rj))
        ll = c.get("data_loglik")
        if gj is not None and np.isfinite(gj) and ll is not None and np.isfinite(ll):
            loglik_xs.append(float(ll)); loglik_ys.append(float(gj))
    composite_gt = {
        "pearson": _pearson(comp_xs, gt_ys), "spearman": _spearman(comp_xs, gt_ys),
        "n": len(gt_ys), "objective_mode": _OBJECTIVE_MODE,
    }
    region_gt = {
        "pearson": _pearson(regionj_xs, gt_ys), "spearman": _spearman(regionj_xs, gt_ys),
    }
    # Phase-B kill-switch: the PROBABILISTIC objective (data_loglik) must POSITIVELY
    # correlate with held-out GT (it replaced the anti-correlated region-J-vs-union
    # signal precisely to fix that). spearman >= 0 required when it is the objective.
    data_loglik_gt = {
        "pearson": _pearson(loglik_xs, loglik_ys), "spearman": _spearman(loglik_xs, loglik_ys),
        "n": len(loglik_ys),
    }
    # Kill-switch verdict: the ACTIVE objective must not anti-correlate with GT.
    composite_ok = bool(_OBJECTIVE_MODE != "composite" or
                        (np.isfinite(composite_gt["spearman"]) and composite_gt["spearman"] >= 0.0))
    loglik_ok = bool(_OBJECTIVE_MODE != "data_loglik" or
                     (np.isfinite(data_loglik_gt["spearman"]) and data_loglik_gt["spearman"] >= 0.0))

    info: dict = {
        "phase": "validate",
        "num_gibbs_sweeps_per_frame": num_gibbs_sweeps_per_frame,
        "objective_mode": _OBJECTIVE_MODE,
        "composite_weights": dict(COMPOSITE_WEIGHTS),
        "composite_gt_corr": composite_gt,
        "region_J_gt_corr": region_gt,
        "data_loglik_gt_corr": data_loglik_gt,
        "composite_kill_switch_ok": composite_ok,
        "loglik_kill_switch_ok": loglik_ok,
        "chosen_infer": chosen_infer,
        "baseline_infer": baseline_infer,
        "baseline_source": baseline_src,
        "heldout_videos": sorted(held_set) if held_set is not None else None,
        "baseline_per_video": baseline_pv,
        "chosen_per_video": chosen_pv,
        "chosen_hypers": chosen.to_dict(),
        "baseline_hypers": baseline.to_dict(),
        # Headline benchmark: real DAVIS GT.
        "median_delta_gt_J": median_delta,
        "median_delta_J_mean": median_delta,   # back-compat alias (== GT-J delta)
        "chosen_median_gt_J_davis": _median_metric(chosen_pv, "gt_J", "davis"),
        "baseline_median_gt_J_davis": _median_metric(baseline_pv, "gt_J", "davis"),
        # Self-supervised side-metric: SAM region-J over the held-out videos.
        "median_delta_sam2_J": median_sam_delta,
        "chosen_median_sam2_J_all": _median_metric(chosen_pv, "sam2_J", None),
        "baseline_median_sam2_J_all": _median_metric(baseline_pv, "sam2_J", None),
        "chosen_median_sam2_J_davis": _median_metric(chosen_pv, "sam2_J", "davis"),
        "chosen_median_sam2_J_local": _median_metric(chosen_pv, "sam2_J", "local"),
        "n_davis_scored": len(deltas),
        "chosen_max_outlier_p95": max_p95,
        "baseline_max_outlier_p95": max_p95_baseline,
        "delta_pass": delta_pass,
        "p95_pass": p95_pass,
        "passed": passed,
    }
    _save_json(info_path, info)
    _log(f"validate done: median Δ GT-J={median_delta:+.4f} over {len(deltas)} held-out DAVIS "
         f"(delta_pass={delta_pass}); chosen GT-J={info['chosen_median_gt_J_davis']:.4f} "
         f"vs baseline {info['baseline_median_gt_J_davis']:.4f} (floor_pass={floor_pass}). "
         f"Median Δ SAM-J (held-out)={median_sam_delta:+.4f}. "
         f"chosen max p95={max_p95:.4f} vs baseline {max_p95_baseline:.4f} "
         f"(pass={p95_pass}) — overall {'PASS' if passed else 'FAIL'}")
    _log(f"validate kill-switch: corr(composite, GT) spearman="
         f"{composite_gt['spearman']:.3f}/pearson={composite_gt['pearson']:.3f} "
         f"vs corr(binary region-J, GT) spearman={region_gt['spearman']:.3f} "
         f"(objective={_OBJECTIVE_MODE}; composite_ok={composite_ok}). "
         + ("" if composite_ok else
            "WARNING: composite ANTI-correlates with GT — re-run with --objective region_J."))
    if _OBJECTIVE_MODE == "data_loglik":
        _log(f"validate kill-switch: corr(data_loglik, GT) spearman="
             f"{data_loglik_gt['spearman']:.3f}/pearson={data_loglik_gt['pearson']:.3f} "
             f"(n={data_loglik_gt['n']}; loglik_ok={loglik_ok}). "
             + ("" if loglik_ok else
                "WARNING: data_loglik ANTI-correlates with GT — revert to --objective composite."))
    return info


def phase_validate_augmentation(out_root: Path, *, force: bool = False,
                                max_frames_per_video: int = -1,
                                num_gibbs_sweeps_per_frame: int = 1,
                                heldout_videos: Optional[List[str]] = None,
                                eps_regression: float = 0.02) -> dict:
    """The ONE sanctioned held-out validation of the TokenCut seed augmentation —
    the gate-routed replacement for the (deleted) ad-hoc `_tokencut_official_eval.py`.

    Both passes use the SAME shipping hypers + structure; they differ by EXACTLY
    the `tokencut_seed_augment` flag (NOT an EM re-fit — that confound already
    failed phase_validate's strict gate). So this isolates the augmentation:
      • baseline = shipping config, flag forced OFF (un-augmented).
      • chosen   = shipping config, flag forced ON  (+ self-supervised knobs).

    Held-out GT is read ONLY here, inside `_heldout_gate_enabled()` — the durable
    lock makes any held-out GT access elsewhere assert. The gate criterion is the
    one appropriate to a SPARSE additive augmentation (working/demo videos no-op
    by construction → their Δ is 0, so a median-of-Δ gate is ill-posed):
        floor (chosen median GT-J ≥ baseline)
        ∧ no held-out regression worse than `eps_regression`
        ∧ mean Δ GT-J > 0.
    Emits nothing (the shipping config already carries the flag); on PASS it
    confirms the number through the lock, on FAIL it WARNs. Writes
    `validate_augmentation.json` + `augmentation_report.md`."""
    _assert_split_discipline()
    info_path = out_root / "validate_augmentation.json"
    cached = _load_json(info_path)
    if cached and not force:
        _log("validate_augmentation: reusing cached")
        return cached

    # Shipping hypers + structure (the honest 'before'); we toggle ONLY the flag.
    em, infer, src = _load_shipping_baseline()
    knobs = infer.get("tokencut_knobs")
    if not knobs:
        kp = OUT_ROOT / "tokencut_knobs.json"
        if kp.is_file():
            knobs = json.loads(kp.read_text())
        else:
            import tokencut
            knobs = dict(tokencut.DEFAULT_KNOBS)
    base_infer = {**infer, "tokencut_seed_augment": False, "tokencut_knobs": None}
    chosen_infer = {**infer, "tokencut_seed_augment": True, "tokencut_knobs": knobs}
    _log(f"validate_augmentation: baseline={src} (flag OFF) vs SAME hypers (flag ON); "
         f"knobs={knobs}")

    _ensure_jax_setup()
    videos = discover_videos()
    held_set = set(heldout_videos) if heldout_videos is not None else set(HELDOUT_VIDEOS)
    labels_cache: Dict[str, dict] = {}
    for vid in videos:
        if vid not in held_set or videos[vid].kind != "davis":
            continue   # held-out DAVIS only (locals have no GT)
        if (LABELS_DIR / f"{vid}.npz").is_file():
            try:
                labels_cache[vid] = _load_labels(vid)
            except Exception as e:
                _log(f"validate_augmentation: failed to load {vid}: {e}")
    sam_grids = _build_sam_grids(videos, labels_cache, chosen_infer["use_sam_frame0"])
    _log(f"validate_augmentation: held-out DAVIS gate over {len(labels_cache)} videos")

    def _pass(label: str, infer_d: dict) -> Dict[str, dict]:
        yaml_cfg = _make_yaml_cfg(em, infer_d)
        out: Dict[str, dict] = {}
        for j, vid in enumerate(labels_cache, 1):
            tr = _run_tracker_on_video(labels_cache[vid], yaml_cfg, max_frames_per_video,
                                       num_sweeps=num_gibbs_sweeps_per_frame,
                                       capture_blob_weights=True,
                                       sam_grid=sam_grids.get(vid), vid=vid)
            if "error" in tr:
                out[vid] = {"error": tr["error"]}
                _log(f"validate_augmentation[{label}][{j}/{len(labels_cache)}] {vid}: FAILED")
                continue
            gt = _score_video_region_J(vid, tr["hyperblob_a"], tr.get("indices"),
                                       labels_cache[vid], reference="gt")
            out[vid] = {"gt_J": gt["region_J"], "outlier_frac_p95": tr["outlier_frac_p95"]}
            _log(f"validate_augmentation[{label}][{j}/{len(labels_cache)}] {vid}: "
                 f"GT-region-J={gt['region_J']:.4f}")
        return out

    # The ONE held-out GT read, behind both the GT scope and the held-out lock.
    with _gt_scoring_enabled(), _heldout_gate_enabled():
        base_pv = _pass("OFF", base_infer)
        chosen_pv = _pass("ON", chosen_infer)

    rows = []
    for vid in labels_cache:
        b, c = base_pv.get(vid, {}), chosen_pv.get(vid, {})
        if np.isfinite(b.get("gt_J", np.nan)) and np.isfinite(c.get("gt_J", np.nan)):
            rows.append((vid, float(b["gt_J"]), float(c["gt_J"])))
    if not rows:
        raise RuntimeError("validate_augmentation: no held-out DAVIS scored")
    off = np.array([r[1] for r in rows]); on = np.array([r[2] for r in rows])
    delta = on - off
    median_off, median_on = float(np.median(off)), float(np.median(on))
    mean_delta = float(np.mean(delta)); worst = float(np.min(delta))
    floor_pass = median_on >= median_off - 1e-9
    no_regression = worst >= -eps_regression
    mean_pass = mean_delta > 0
    passed = bool(floor_pass and no_regression and mean_pass)

    worst_vid = rows[int(np.argmin(delta))][0]
    best_i = int(np.argmax(delta))
    info = {
        "phase": "validate_augmentation",
        "baseline_source": src, "knobs": knobs,
        "num_gibbs_sweeps_per_frame": num_gibbs_sweeps_per_frame,
        "eps_regression": eps_regression,
        "n_davis": len(rows),
        "per_video": {v: {"off": o, "on": n, "delta": n - o} for v, o, n in rows},
        "median_off": median_off, "median_on": median_on,
        "mean_delta": mean_delta, "worst_delta": worst, "worst_video": worst_vid,
        "best_delta": float(delta[best_i]), "best_video": rows[best_i][0],
        "floor_pass": floor_pass, "no_regression_pass": no_regression,
        "mean_pass": mean_pass, "passed": passed,
        "n_up": int((delta > 1e-4).sum()), "n_down": int((delta < -1e-4).sum()),
        "n_flat": int((np.abs(delta) <= 1e-4).sum()),
    }
    _save_json(info_path, info)

    md = ["# TokenCut seed-augmentation — sanctioned held-out validation", "",
          "Augmentation-only A/B (same shipping hypers; differ ONLY by the "
          "`tokencut_seed_augment` flag). Held-out GT read once, inside the locked "
          "gate (`_heldout_gate_enabled`). Gate = floor ∧ no-regression(ε="
          f"{eps_regression}) ∧ mean Δ>0 (the criterion for a sparse additive "
          "augmentation; working/demo videos no-op → Δ=0).", "",
          f"- baseline source: `{src}`",
          f"- knobs (self-supervised, TRAIN-selected): `{knobs}`",
          f"- held-out DAVIS scored: {len(rows)}  "
          f"(up {info['n_up']} / down {info['n_down']} / bit-flat {info['n_flat']})",
          f"- median GT region-J: OFF {median_off:.4f} → ON {median_on:.4f}  "
          f"(floor_pass={floor_pass})",
          f"- mean Δ GT region-J: {mean_delta:+.4f}  (mean_pass={mean_pass})",
          f"- worst Δ: {worst:+.4f} ({worst_vid})  (no_regression_pass={no_regression})",
          f"- best Δ: {info['best_delta']:+.4f} ({info['best_video']})",
          f"- **gate: {'PASS' if passed else 'FAIL'}**", "",
          "| video | OFF | ON | Δ |", "|---|---|---|---|"]
    for v, o, n in sorted(rows, key=lambda r: (r[2] - r[1])):
        md.append(f"| {v} | {o:.4f} | {n:.4f} | {n - o:+.4f} |")
    (out_root / "augmentation_report.md").write_text("\n".join(md) + "\n")

    _log(f"validate_augmentation done: median OFF={median_off:.4f} → ON={median_on:.4f} "
         f"(floor={floor_pass}); mean Δ={mean_delta:+.4f} (mean_pass={mean_pass}); "
         f"worst Δ={worst:+.4f} @{worst_vid} (no_regression={no_regression}) — "
         f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        _log("validate_augmentation: gate FAILED — the shipping config's flag is NOT "
             "reconfirmed; investigate before relying on the augmentation.")
    return info


def _emit_yaml(out_path: Path, *, val_info: dict, em_info: dict) -> None:
    """Write configs/streaming_general.yaml with header + full 18-hyper chosen
    config (use_calibrated_priors flag flipped on)."""
    import yaml as _yaml
    chosen_dict = val_info["chosen_hypers"]
    chosen_em = _em_from_dict(chosen_dict)
    infer = val_info.get("chosen_infer", _infer_cfg(
        use_sam_frame0=INFER_USE_SAM_FRAME0, num_blobs=INFER_NUM_BLOBS,
        num_hyperblobs=INFER_NUM_HYPERBLOBS, init_gibbs_sweeps=INFER_INIT_GIBBS_SWEEPS))
    n_sweeps = val_info.get("num_gibbs_sweeps_per_frame", DEFAULT_NUM_GIBBS_SWEEPS)
    median_delta = val_info.get("median_delta_gt_J", val_info.get("median_delta_J_mean"))
    median_sam_delta = val_info.get("median_delta_sam2_J", float("nan"))
    chosen_gt = val_info.get("chosen_median_gt_J_davis", float("nan"))
    base_gt = val_info.get("baseline_median_gt_J_davis", float("nan"))
    chosen_sam_all = val_info.get("chosen_median_sam2_J_all", float("nan"))
    max_p95 = val_info["chosen_max_outlier_p95"]
    n_davis = val_info.get("n_davis_scored", 0)
    passed = val_info.get("passed", False)
    n_iters = len(em_info.get("iters", []))

    cfg = _make_yaml_cfg(chosen_em, infer)

    header_lines = [
        "# StreamingVision: self-supervised, structure-aware calibrated hypers.",
        "# Produced by scripts/calibrate_consistency.py. ONE GLOBAL config; NO",
        "# per-video knobs; NO test-set tuning.",
        "#",
        "# Methodology: hypers LEARNED + the global scalar tuple SELECTED entirely",
        "# self-supervised on a TRAIN split (region-J of the CLUSTER foreground vs",
        "# cached SAM Z_sam>0). Real DAVIS GT was read ONCE, on a disjoint HELD-OUT",
        "# split, as the generalization gate only. Two-level approximate inference:",
        f"# per-frame Bayesian filtering with {n_sweeps} Gibbs sweeps + a {infer['init_gibbs_sweeps']}-sweep",
        "# warmup (the deep E-step that de-biases the conjugate M-step), outer",
        "# SAEM (α=0.3 damping, numerical-stability veto, per-group accept-test on",
        "# an 8-video crossval subset OF TRAIN: cluster / particle / motion).",
        "#",
        "# Inference strategy (ONE global config — these ship alongside the hypers):",
        f"#   use_sam_frame0    = {infer['use_sam_frame0']}  (SAM2 frame-0 semantic init)",
        f"#   num_blobs         = {infer['num_blobs']}",
        f"#   num_hyperblobs    = {infer['num_hyperblobs']}  (k-means fallback; SAM init sets it dynamically)",
        f"#   init_gibbs_sweeps = {infer['init_gibbs_sweeps']}",
        f"#   per-frame sweeps  = {n_sweeps}  (calibration depth; live demo runs 1)",
        f"# EM iterations run: {n_iters}",
        f"# Best-iteration median self-supervised region-J (TRAIN) = "
        f"{em_info.get('best_region_J', em_info.get('best_J_mean', float('nan'))):.4f}",
        "#",
        "# Chosen scalar values:",
        f"#   sigma_F      = {chosen_em.sigma_F:.4f}",
        f"#   sigma_F_H    = {chosen_em.sigma_F_H:.4f}",
        f"#   sigma_V      = {chosen_em.sigma_V:.3e}",
        f"#   outlier_prob = {chosen_em.outlier_prob:.3f}",
        f"#   gamma_shape  = {chosen_em.gamma_shape:.3f}",
        f"#   gamma_rate   = {chosen_em.gamma_rate:.3f}",
        f"#   alpha        = {chosen_em.alpha:.3f}",
        f"#   beta         = {chosen_em.beta:.3f}",
        f"#   sigma_H      = {chosen_em.sigma_H:.3f}",
        f"#   nu_B / nu_H / nu_V = {chosen_em.nu_B:.1f} / {chosen_em.nu_H:.1f} / {chosen_em.nu_V:.1f}",
        f"#   translation_max_radius     = {chosen_em.translation_max_radius:.4f}",
        f"#   translation_gaussian_scale = {chosen_em.translation_gaussian_scale:.4f}",
        f"#   use_calibrated_priors      = {chosen_em.use_calibrated_priors}",
        f"#   feature_update_damping     = {infer.get('feature_update_damping', INFER_FEATURE_UPDATE_DAMPING):.4f}  "
        f"(Phase-A anti-drift; d=0 freeze .. d=1 vendored; SELECTED self-supervised)",
        f"#   final_feature_temp         = {infer.get('final_feature_temp', INFER_FINAL_FEATURE_TEMP):.4f}  "
        f"(Phase-1 inference; tau=1 vendored feat-aware .. tau>1 feature-driven; SELECTED self-supervised)",
        "#",
        "# HELD-OUT validation (chosen = this config; baseline = previously-shipping):",
        f"#   real DAVIS GT region-J: chosen {chosen_gt:.4f} vs baseline {base_gt:.4f} "
        f"(median Δ over {n_davis} held-out DAVIS = {median_delta:+.4f})  <- GENERALIZATION GATE",
        f"#   self-supervised SAM region-J (held-out): chosen {chosen_sam_all:.4f} "
        f"(median Δ matter-J = {median_sam_delta:+.4f})",
        f"#   chosen max outlier_frac p95 (held-out) = {max_p95:.4f}",
        f"#   gate passed (median Δ GT region-J > 0 AND max p95 ≤ baseline + 0.05): {passed}",
        f"#   self-supervised objective = {val_info.get('objective_mode', 'composite')}; "
        f"corr(composite, GT) spearman = "
        f"{val_info.get('composite_gt_corr', {}).get('spearman', float('nan')):.3f} "
        f"(kill-switch OK = {val_info.get('composite_kill_switch_ok', True)})",
        f"#   sigma_V (velocity anchor) = {chosen_em.sigma_V:.3e}  "
        f"(streaming_default.yaml keeps 1e14 → live path bit-exact)",
    ]
    if not passed:
        header_lines.insert(0,
            "# WARN: GT-J benchmark gate failed; review runs/calibrate_consistency/report.md.")
    header_lines.append("")
    body = _yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    text = "\n".join(header_lines) + body
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    _log(f"wrote {out_path}")


def _emit_demo_yaml(out_path: Path, *, val_info: dict, em_info: dict) -> None:
    """Emit the demo-facing config (streaming_render_v2.yaml): the SAME chosen
    calibrated hypers + GLOBAL architectural-structure flags as
    streaming_general.yaml, with the demo's richer k-means-fallback
    num_hyperblobs (the SAM/GT frame-0 seed overrides it dynamically). ONE global
    config — NO per-video keys; the render grid reads the single
    tracking.num_blobs / blob_means_updates_per_frame."""
    import yaml as _yaml
    chosen_em = _em_from_dict(val_info["chosen_hypers"])
    infer = dict(val_info.get("chosen_infer") or {})
    infer.setdefault("use_sam_frame0", INFER_USE_SAM_FRAME0)
    infer.setdefault("num_blobs", INFER_NUM_BLOBS)
    infer.setdefault("init_gibbs_sweeps", INFER_INIT_GIBBS_SWEEPS)
    infer["num_hyperblobs"] = RENDER_V2_NUM_HYPERBLOBS
    cfg = _make_yaml_cfg(chosen_em, infer)
    passed = val_info.get("passed", False)
    median_delta = val_info.get("median_delta_gt_J", float("nan"))
    n_sweeps = val_info.get("num_gibbs_sweeps_per_frame", DEFAULT_NUM_GIBBS_SWEEPS)
    header = [
        "# StreamingVision demo config (streaming_render_v2.yaml) — AUTO-EMITTED.",
        "# ONE GLOBAL config: the SAME calibrated hypers + global architectural-",
        "# structure flags as streaming_general.yaml (the v2 cluster-freeze fix:",
        "# feature_aware_final_assignment / final_assignment_outlier=false /",
        "# freeze_hyperblob_assignment / pure_object_seed), with the demo's richer",
        f"# k-means-fallback num_hyperblobs={RENDER_V2_NUM_HYPERBLOBS} (SAM/GT frame-0 seed sets it",
        "# dynamically). NO per-video knobs — the render grid reads the single",
        "# tracking.num_blobs / blob_means_updates_per_frame.",
        "#",
        f"# num_blobs={infer.get('num_blobs')}, init_gibbs_sweeps={infer.get('init_gibbs_sweeps')}, "
        f"calibration per-frame sweeps={n_sweeps} (live demo runs 1).",
        "# Scalar tuple selected SELF-SUPERVISED on the TRAIN split; hypers learned",
        "# on TRAIN under this structure; validated ONCE on the HELD-OUT split vs the",
        f"# shipping config (median Δ real-GT region-J = {median_delta:+.4f}; gate passed: {passed}).",
        "# Produced by scripts/calibrate_consistency.py. Do not hand-edit.",
    ]
    if not passed:
        header.insert(0,
            "# WARN: held-out GT gate FAILED — candidate only; review report.md.")
    header.append("")
    body = _yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header) + body)
    _log(f"wrote {out_path} (demo global config, num_hyperblobs={RENDER_V2_NUM_HYPERBLOBS}, "
         f"num_blobs={infer.get('num_blobs')})")


def _emit_configs(out_root: Path, *, val_info: dict, em_info: dict) -> None:
    """Emit the calibrated general + demo configs and the report. On a PASSING
    held-out gate, overwrite the shipping configs (streaming_general.yaml +
    streaming_render_v2.yaml). On FAIL, write *_CANDIDATE.yaml siblings and leave
    the known-good shipping configs untouched (per the plan: don't overwrite
    known-good on fail)."""
    passed = bool(val_info.get("passed", False))
    if passed:
        _emit_yaml(GENERAL_YAML, val_info=val_info, em_info=em_info)
        _emit_demo_yaml(RENDER_V2_YAML, val_info=val_info, em_info=em_info)
        _log("validate PASSED held-out gate — updated streaming_general.yaml + "
             "streaming_render_v2.yaml (one global config, no per-video).")
    else:
        cand_general = GENERAL_YAML.with_name("streaming_general_CANDIDATE.yaml")
        cand_demo = RENDER_V2_YAML.with_name("streaming_render_v2_CANDIDATE.yaml")
        _emit_yaml(cand_general, val_info=val_info, em_info=em_info)
        _emit_demo_yaml(cand_demo, val_info=val_info, em_info=em_info)
        _log(f"validate FAILED held-out gate — wrote {cand_general.name} / "
             f"{cand_demo.name}; known-good {GENERAL_YAML.name} / "
             f"{RENDER_V2_YAML.name} left intact.")
    _emit_report(out_root, val_info=val_info, em_info=em_info)


def _emit_report(out_root: Path, *, val_info: dict, em_info: dict) -> None:
    md = []
    md.append("# Self-supervised calibration report (TRAIN-learned, HELD-OUT-gated)")
    md.append("")
    # ---- Provenance: the methodology guarantee (no test-set tuning, no per-video) ----
    sel = _load_json(out_root / "selected_infer.json") or {}
    grid = _load_json(out_root / "select_grid.json") or {}
    md.append("## Provenance / methodology")
    md.append("- **Learning + model selection: SELF-SUPERVISED on the TRAIN split only** "
              "(region-J of the CLUSTER foreground vs cached `Z_sam>0`). No ground truth, "
              "no held-out data entered learning or selection.")
    md.append("- **Ground-truth DAVIS GT was read ONLY in phase_validate, on the HELD-OUT "
              "split** — the single generalization gate. (Enforced at runtime: GT access "
              "outside validate asserts.)")
    md.append("- **ONE global config; no per-video knobs anywhere.**")
    md.append(f"- TRAIN ({len(TRAIN_VIDEOS)}): {', '.join(TRAIN_VIDEOS)}")
    md.append(f"- HELD-OUT ({len(HELDOUT_VIDEOS)}, incl. all 5 demos): "
              f"{', '.join(HELDOUT_VIDEOS)}")
    md.append(f"- CROSSVAL (SAEM accept-test, ⊆ TRAIN): {', '.join(CROSSVAL_VIDEOS)}")
    md.append(f"- SELECT_VAL (selection scoring, ⊆ TRAIN, ∩CROSSVAL=∅): "
              f"{', '.join(SELECT_VAL_VIDEOS)}")
    md.append(f"- **Self-supervised objective**: {val_info.get('objective_mode', em_info.get('objective_mode', 'composite'))} "
              f"(composite weights {em_info.get('composite_weights', dict(COMPOSITE_WEIGHTS))}). "
              "The held-out GT gate is ALWAYS binary region-J regardless.")
    if sel:
        md.append(f"- **Selected global scalar tuple** (self-supervised on TRAIN): "
                  f"num_blobs={sel.get('num_blobs')}, "
                  f"init_gibbs_sweeps={sel.get('init_gibbs_sweeps')}, "
                  f"num_gibbs_sweeps_per_frame={sel.get('num_gibbs_sweeps_per_frame')}, "
                  f"sigma_V_seed={sel.get('sigma_V_seed', _SIGMA_V_SEED):.3e}, "
                  f"blob_means_updates={sel.get('blob_means_updates', INFER_BLOB_MEANS_UPDATES)} "
                  f"(SELECT_VAL composite={sel.get('select_val_score', sel.get('select_val_region_J', float('nan'))):.4f})")
    if grid.get("combos"):
        md.append("")
        md.append("### phase_select grid (self-supervised composite on SELECT_VAL)")
        md.append("| num_blobs | igs | sweeps | sigma_V | bmu | SELECT_VAL composite | region-J | median p95 |")
        md.append("|---|---|---|---|---|---|---|---|")

        def _cat(c, i, d):
            cc = c.get("combo", [])
            return cc[i] if len(cc) > i else d
        for c in sorted(grid["combos"],
                        key=lambda c: -float(c.get("select_val_score", c.get("select_val_region_J", float("-inf")))
                                             if np.isfinite(c.get("select_val_score", c.get("select_val_region_J", float("nan"))))
                                             else float("-inf"))):
            md.append(f"| {c['combo'][0]} | {c['combo'][1]} | {c['combo'][2]} | "
                      f"{float(_cat(c, 3, _SIGMA_V_SEED)):.2e} | {_cat(c, 4, INFER_BLOB_MEANS_UPDATES)} | "
                      f"{c.get('select_val_score', float('nan')):.4f} | "
                      f"{c.get('select_val_region_J', float('nan')):.4f} | "
                      f"{c.get('select_val_median_p95', float('nan')):.4f} |")
    ci = val_info.get("chosen_infer", {})
    md.append("")
    md.append("### Architectural-structure flags (GLOBAL, shipped)")
    md.append(f"- feature_aware_final={ci.get('feature_aware_final')}, "
              f"final_outlier={ci.get('final_outlier')}, "
              f"freeze_hyperblob={ci.get('freeze_hyperblob')}, "
              f"pure_object_seed={ci.get('pure_object_seed')}, "
              f"blob_means_updates={ci.get('blob_means_updates')}")
    md.append(f"- Baseline source (validate 'before'): {val_info.get('baseline_source', 'n/a')}")
    md.append("")
    md.append("## Chosen hyperparameters")
    chosen = val_info["chosen_hypers"]
    md.append(f"- sigma_F: {chosen['sigma_F']:.4f}")
    md.append(f"- sigma_F_H: {chosen['sigma_F_H']:.4f}")
    md.append(f"- sigma_V: {chosen['sigma_V']:.3e}")
    md.append(f"- outlier_prob: {chosen['outlier_prob']:.3f}")
    md.append(f"- gamma_shape: {chosen['gamma_shape']:.3f}")
    md.append(f"- gamma_rate: {chosen['gamma_rate']:.3f}")
    md.append(f"- alpha: {chosen['alpha']:.3f}")
    md.append(f"- beta: {chosen['beta']:.3f}")
    md.append(f"- sigma_H: {chosen['sigma_H']:.3f}")
    md.append(f"- nu_B / nu_H / nu_V: {chosen['nu_B']:.1f} / "
              f"{chosen['nu_H']:.1f} / {chosen['nu_V']:.1f}")
    md.append(f"- translation_max_radius: {chosen['translation_max_radius']:.4f}")
    md.append(f"- translation_gaussian_scale: {chosen['translation_gaussian_scale']:.4f}")
    md.append(f"- use_calibrated_priors: {chosen['use_calibrated_priors']}")
    md.append("")
    infer = val_info.get("chosen_infer", {})
    md.append("## Inference strategy (ONE global config)")
    md.append(f"- use_sam_frame0: {infer.get('use_sam_frame0')}  (SAM2 frame-0 semantic init)")
    md.append(f"- num_blobs: {infer.get('num_blobs')}, num_hyperblobs: {infer.get('num_hyperblobs')} "
              f"(k-means fallback; SAM init sets it dynamically)")
    md.append(f"- init_gibbs_sweeps (warmup): {infer.get('init_gibbs_sweeps')}")
    md.append(f"- per-frame Gibbs sweeps (calibration depth): "
              f"{val_info.get('num_gibbs_sweeps_per_frame')}  (live demo runs 1)")
    md.append("")
    md.append("## EM trajectory (objective: self-supervised COMPOSITE; train=M-step pool, "
              "val=held-within-TRAIN generalization probe; best iterate by val_composite)")
    md.append("| iter | train_comp | val_comp | region-J | DAVIS rJ | local rJ | sigma_F | sigma_V | "
              "cluster | particle | motion |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for it in em_info.get("iters", []):
        h = it.get("hypers_after", it.get("hypers_before", {}))
        ga = it.get("group_accepts", {})
        cluster_acc = ga.get("cluster", {}).get("accepted", "-")
        particle_acc = ga.get("particle", {}).get("accepted", "-")
        motion_acc = ga.get("motion", {}).get("accepted", "-")
        md.append(f"| {it['iter']} | "
                  f"{it.get('train_composite', it.get('J_mean', float('nan'))):.4f} | "
                  f"{it.get('val_composite', float('nan')):.4f} | "
                  f"{it.get('region_J', float('nan')):.4f} | "
                  f"{it.get('davis_region_J', float('nan')):.4f} | "
                  f"{it.get('local_region_J', float('nan')):.4f} | "
                  f"{h.get('sigma_F', float('nan')):.4f} | "
                  f"{h.get('sigma_V', float('nan')):.3e} | "
                  f"{cluster_acc} | {particle_acc} | {motion_acc} |")
    # Motion-group acceptance summary (the #1 lever): did the live sigma_V anchor stick?
    motion_accepts = sum(1 for it in em_info.get("iters", [])
                         if it.get("group_accepts", {}).get("motion", {}).get("accepted"))
    seed_sv = (em_info.get("iters", [{}])[0].get("hypers_before", {}).get("sigma_V")
               if em_info.get("iters") else None)
    final_sv = chosen.get("sigma_V")
    md.append("")
    md.append(f"- **Motion group accepted in {motion_accepts}/{len(em_info.get('iters', []))} iters**; "
              f"sigma_V seed={seed_sv if seed_sv is None else f'{seed_sv:.3e}'} → "
              f"chosen={final_sv if final_sv is None else f'{final_sv:.3e}'} "
              f"(moved off seed = {seed_sv is not None and final_sv is not None and abs(final_sv - seed_sv) > 1e-9}).")
    md.append("")
    md.append("## Vetoed layers per iter")
    md.append("| iter | vetoed |")
    md.append("|---|---|")
    for it in em_info.get("iters", []):
        v = it.get("vetoed_layers", {})
        if v:
            md.append(f"| {it['iter']} | {', '.join(v.keys())} |")
        else:
            md.append(f"| {it['iter']} | (none) |")
    md.append("")
    md.append("## Validation gate (chosen = calibrated+strong-inference; baseline = shipping)")
    md.append(f"- **BENCHMARK** median Δ real-GT-J over {val_info.get('n_davis_scored', 0)} "
              f"DAVIS videos: {val_info.get('median_delta_gt_J', float('nan')):+.4f} "
              f"(chosen {val_info.get('chosen_median_gt_J_davis', float('nan')):.4f} "
              f"vs baseline {val_info.get('baseline_median_gt_J_davis', float('nan')):.4f}) "
              f"(GATE pass: {val_info['delta_pass']})")
    md.append(f"- self-supervised median Δ SAM2-J over 38: "
              f"{val_info.get('median_delta_sam2_J', float('nan')):+.4f} "
              f"(chosen all {val_info.get('chosen_median_sam2_J_all', float('nan')):.4f}; "
              f"DAVIS {val_info.get('chosen_median_sam2_J_davis', float('nan')):.4f} / "
              f"local {val_info.get('chosen_median_sam2_J_local', float('nan')):.4f})")
    md.append(f"- chosen max outlier_frac p95: {val_info['chosen_max_outlier_p95']:.4f} "
              f"vs baseline {val_info['baseline_max_outlier_p95']:.4f} "
              f"(pass: {val_info['p95_pass']})")
    md.append(f"- overall: {'PASS' if val_info['passed'] else 'FAIL'}")
    # Composite/GT correlation KILL-SWITCH (Phase C): measured ONCE on held-out
    # DAVIS. composite must not anti-correlate with GT; whether it BEATS binary
    # region-J is reported for the operator (revert to --objective region_J if not).
    cgt = val_info.get("composite_gt_corr", {})
    rgt = val_info.get("region_J_gt_corr", {})
    md.append(f"- **composite↔GT correlation (kill-switch)**: spearman="
              f"{cgt.get('spearman', float('nan')):.3f}, pearson={cgt.get('pearson', float('nan')):.3f} "
              f"(n={cgt.get('n', 0)}) vs binary region-J↔GT spearman={rgt.get('spearman', float('nan')):.3f}. "
              f"objective={val_info.get('objective_mode', 'composite')}; "
              f"kill-switch OK={val_info.get('composite_kill_switch_ok', True)}"
              + ("" if val_info.get("composite_kill_switch_ok", True)
                 else " — **composite ANTI-correlates with GT; re-run with --objective region_J**"))
    md.append("")
    base = val_info["baseline_per_video"]
    cho = val_info["chosen_per_video"]
    md.append("## Per-video real-GT-J (DAVIS — chosen vs baseline) + SAM2-J")
    md.append("| video | GT_baseline | GT_chosen | Δ GT | SAM2_chosen | matter_fps | outlier_p95 |")
    md.append("|---|---|---|---|---|---|---|")
    for vid in sorted(cho.keys()):
        c = cho.get(vid, {})
        if c.get("kind") != "davis":
            continue
        b = base.get(vid, {})
        jb = b.get("gt_J", float("nan")); jc = c.get("gt_J", float("nan"))
        delta = jc - jb if (np.isfinite(jb) and np.isfinite(jc)) else float("nan")
        md.append(f"| {vid} | {jb:.4f} | {jc:.4f} | {delta:+.4f} | "
                  f"{c.get('sam2_J', float('nan')):.4f} | "
                  f"{c.get('matter_fps', float('nan')):.2f} | "
                  f"{c.get('outlier_frac_p95', float('nan')):.4f} |")
    md.append("")
    md.append("## Per-video SAM2-J (local — chosen vs baseline) + ARI diagnostic")
    md.append("| video | SAM2_baseline | SAM2_chosen | Δ SAM2 | S_ari_chosen | outlier_p95 |")
    md.append("|---|---|---|---|---|---|")
    for vid in sorted(cho.keys()):
        c = cho.get(vid, {})
        if c.get("kind") != "local":
            continue
        b = base.get(vid, {})
        sb = b.get("sam2_J", float("nan")); sc = c.get("sam2_J", float("nan"))
        delta = sc - sb if (np.isfinite(sb) and np.isfinite(sc)) else float("nan")
        md.append(f"| {vid} | {sb:.4f} | {sc:.4f} | {delta:+.4f} | "
                  f"{c.get('S_ari', float('nan')):.4f} | "
                  f"{c.get('outlier_frac_p95', float('nan')):.4f} |")
    (out_root / "report.md").write_text("\n".join(md) + "\n")
    _save_json(out_root / "report.json",
               {"em": em_info, "validate": val_info})


# ----------------------------------------------------------------------
# Plumbing smoke (Section 1 verification)
# ----------------------------------------------------------------------

def plumbing_smoke(*, max_warmup_sweeps: int = 1) -> dict:
    """Section 1 plumbing smoke: capture the K-means-seed Psi_*/nu_*/mu_H
    that the streaming tracker derives at init on test.mp4 frame 0, then
    write `configs/streaming_general_smoke.yaml` with those exact values and
    `use_calibrated_priors: true`.

    Once written, the user verifies the calibrated-priors path round-trips
    by running the live tracker against the smoke YAML — the output should
    match streaming_default.yaml (K-means path) bit-for-bit within JAX
    nondeterminism.
    """
    import jax
    import genmatter_rt

    test_npz = LABELS_DIR / "test.npz"
    if not test_npz.is_file():
        raise FileNotFoundError(
            f"Required pseudo-label cache missing: {test_npz}. "
            f"Run `--phase pseudo_labels` first."
        )
    with np.load(test_npz) as d:
        positions = np.asarray(d["positions"][0], dtype=np.float32)   # (N, 3)
        velocities = np.asarray(d["velocities"][0], dtype=np.float32)
        features = np.asarray(d["features"][0], dtype=np.float32)

    # Drive a single init pass with the default YAML to harvest empirical
    # Psi_*/nu_*/mu_H. `num_warmup_sweeps=1` saves time; we only need
    # debug_capture to be populated, which happens before any Gibbs sweeps.
    cfg = genmatter_rt.load_yaml_hypers(DEFAULT_YAML)
    cfg["tracking"]["use_sam_frame0"] = False
    cfg["tracking"]["num_blobs"] = 64
    cfg["tracking"]["num_hyperblobs"] = 4
    cfg["tracking"]["calibrate_feature_sigmas"] = False
    cfg["tracking"]["use_calibrated_priors"] = False

    debug: dict = {}
    key = jax.random.PRNGKey(0)
    _log("plumbing_smoke: running init_state on test.mp4 frame 0 to capture K-means seed")
    genmatter_rt.init_state(positions, velocities, features, key,
                             yaml_cfg=cfg, num_blobs=64, num_hyperblobs=4,
                             num_warmup_sweeps=max_warmup_sweeps,
                             verbose=False, debug_capture=debug)

    em = _initial_em_state()
    em.use_calibrated_priors = True
    em.Psi_B = np.asarray(debug["empirical_Psi_B"], dtype=np.float32)
    em.Psi_H = np.asarray(debug["empirical_Psi_H"], dtype=np.float32)
    em.Psi_V = np.asarray(debug["empirical_Psi_V"], dtype=np.float32)
    em.nu_B = float(debug["empirical_nu_B"])
    em.nu_H = float(debug["empirical_nu_H"])
    em.nu_V = float(debug["empirical_nu_V"])
    em.mu_H = np.asarray(debug["empirical_mu_H"], dtype=np.float32)
    # Scalar hypers match streaming_default.yaml so the smoke yaml differs from
    # default ONLY in the calibrated-prior matrix values + flag.
    em.sigma_F = 2.0
    em.sigma_F_H = 2.0
    em.outlier_prob = 5.0
    em.gamma_shape = 5.0
    em.gamma_rate = 1.0
    em.alpha = 1.0
    em.beta = 1.0
    em.sigma_H = 25.0
    em.sigma_V = 1.0e14
    em.translation_max_radius = 0.35
    em.translation_gaussian_scale = 0.2

    # Plumbing smoke uses the shipping inference (k-means init / 64 blobs) so the
    # only diff from streaming_default is the calibrated-prior matrices + flag.
    smoke_cfg = _make_yaml_cfg(em, SHIPPING_INFER)
    import yaml as _yaml
    header = (
        "# StreamingVision: Section 1 plumbing-smoke YAML.\n"
        "# Psi_*/nu_*/mu_H equal to the K-means seed values _build_hypers\n"
        "# would derive on test.mp4 frame 0. With use_calibrated_priors:true,\n"
        "# this config should produce identical tracker output to\n"
        "# streaming_default.yaml (which uses use_calibrated_priors:false and\n"
        "# runs K-means at init).\n"
        "#\n"
        "# Generated by scripts.calibrate_consistency.plumbing_smoke().\n"
        "#\n"
        "# Verify with:\n"
        "#   python scripts/run_streaming_live.py \\\n"
        "#     assets/custom_videos/test/source.mp4 \\\n"
        "#     --config configs/streaming_general_smoke.yaml --max-frames 20\n\n"
    )
    body = _yaml.safe_dump(smoke_cfg, sort_keys=False, default_flow_style=False)
    GENERAL_SMOKE_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(GENERAL_SMOKE_YAML, "w", encoding="utf-8") as f:
        f.write(header + body)
    _log(f"plumbing_smoke: wrote {GENERAL_SMOKE_YAML}")
    # JSON-friendly view of captured empirical values for inspection
    return {
        "empirical_Psi_B": em.Psi_B.tolist(),
        "empirical_Psi_H": em.Psi_H.tolist(),
        "empirical_Psi_V": em.Psi_V.tolist(),
        "empirical_nu_B": em.nu_B,
        "empirical_nu_H": em.nu_H,
        "empirical_nu_V": em.nu_V,
        "empirical_mu_H": em.mu_H.tolist(),
        "smoke_yaml": str(GENERAL_SMOKE_YAML),
    }


# ----------------------------------------------------------------------
# Driver / CLI
# ----------------------------------------------------------------------

def _run_with_retry(name: str, fn, *args, retries: int = 1, **kwargs):
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            _log(f"phase {name} attempt {attempt+1}/{retries+1} FAILED: {e}")
            traceback.print_exc()
            if attempt == retries:
                raise
            time.sleep(2.0)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    global _OBJECTIVE_MODE, _DATA_LOGLIK_VARIANT   # set from --objective / --data-loglik-variant below
    global INFER_TOKENCUT_SEED_AUGMENT, INFER_TOKENCUT_KNOBS  # set from --tokencut-seed-augment
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True,
                   choices=("preflight", "pseudo_labels", "select", "em", "validate",
                            "validate_augmentation", "render_local", "plumbing_smoke", "all"))
    p.add_argument("--force", action="store_true",
                   help="Recompute even if a cached phase JSON exists.")
    p.add_argument("--out", type=str, default=str(OUT_ROOT))
    p.add_argument("--max-frames-per-video", type=int, default=-1,
                   help="-1 = all frames; set to a positive int to cap.")
    p.add_argument("--max-iters", type=int, default=EM_MAX_OUTER_ITERS)
    p.add_argument("--k-motion", type=int, default=8)
    p.add_argument("--regen-z-sam-only", action="store_true",
                   help="Phase E cheap path: refresh Z_sam + Z_dino in every cached .npz "
                        "from regenerated SAM PNGs (run sam2_davis_propagate.py first); "
                        "NO depth/flow/DINO recompute. Use with --phase pseudo_labels.")
    p.add_argument("--num-gibbs-sweeps-per-frame", type=int, default=DEFAULT_NUM_GIBBS_SWEEPS,
                   help="Gibbs sweeps per frame during calibration tracker passes "
                        "(Level-1 measurement-update depth). Streaming demo always uses 1.")
    p.add_argument("--em-plateau-tol", type=float, default=EM_PLATEAU_TOL,
                   help="Δ SAM2-J below which the EM loop calls a plateau.")
    p.add_argument("--keep-ari-diagnostic", action="store_true", default=True,
                   help="Compute consistency-ARI on the 8 local videos as a "
                        "side-by-side diagnostic. Does not gate accept-test.")
    # ---- Inference-strategy knobs (ONE global config) ----
    p.add_argument("--num-blobs", type=int, default=INFER_NUM_BLOBS,
                   help="Blob budget (stronger inference). Must stay <= K_MAX-2 (254).")
    p.add_argument("--num-hyperblobs", type=int, default=INFER_NUM_HYPERBLOBS,
                   help="k-means hyperblob fallback; SAM init sets it dynamically.")
    p.add_argument("--init-gibbs-sweeps", type=int, default=INFER_INIT_GIBBS_SWEEPS,
                   help="Frame-0 warmup Gibbs sweeps (converge the SAM-seeded init).")
    p.add_argument("--use-sam-frame0", dest="use_sam_frame0", action="store_true",
                   default=INFER_USE_SAM_FRAME0,
                   help="Seed frame-0 clusters from the SAM2 mask (semantic init).")
    p.add_argument("--no-sam-frame0", dest="use_sam_frame0", action="store_false",
                   help="Disable SAM frame-0 init (flat k-means).")
    # ---- Architectural-structure flags (ONE global config; default ON = the v2
    # cluster-freeze fix the calibration must learn UNDER). All have a --no-*
    # counterpart so the orchestration can build the vendored-structure baseline. ----
    p.add_argument("--feature-aware-final", dest="feature_aware_final",
                   action="store_true", default=INFER_FEATURE_AWARE_FINAL,
                   help="Final blob assignment uses full pos+feature likelihood.")
    p.add_argument("--no-feature-aware-final", dest="feature_aware_final",
                   action="store_false")
    p.add_argument("--final-outlier", dest="final_outlier", action="store_true",
                   default=INFER_FINAL_OUTLIER,
                   help="Inject outliers on the final assignment step (vendored).")
    p.add_argument("--no-final-outlier", dest="final_outlier", action="store_false")
    p.add_argument("--freeze-hyperblob", dest="freeze_hyperblob", action="store_true",
                   default=INFER_FREEZE_HYPERBLOB,
                   help="Keep the frame-0 seeded blob->hyperblob map (the fix).")
    p.add_argument("--no-freeze-hyperblob", dest="freeze_hyperblob", action="store_false")
    p.add_argument("--pure-object-seed", dest="pure_object_seed", action="store_true",
                   default=INFER_PURE_OBJECT_SEED,
                   help="Rebuild the CHM so each object hyperblob is pure at frame 0.")
    p.add_argument("--no-pure-object-seed", dest="pure_object_seed", action="store_false")
    p.add_argument("--blob-means-updates", type=int, default=INFER_BLOB_MEANS_UPDATES,
                   help="Per-frame gibbs_blob_means refinement count (one global).")
    # ---- Model-selection grid (phase_select); comma-separated, all global ----
    p.add_argument("--select-num-blobs", type=str, default="64,128",
                   help="phase_select num_blobs grid (comma-separated).")
    p.add_argument("--select-init-sweeps", type=str, default="1,5,15",
                   help="phase_select init_gibbs_sweeps grid (comma-separated).")
    p.add_argument("--select-per-frame-sweeps", type=str, default="1,4",
                   help="phase_select num_gibbs_sweeps_per_frame grid (comma-separated).")
    p.add_argument("--select-sigma-v", type=str, default=f"{_SIGMA_V_SEED:g}",
                   help="phase_select sigma_V seed grid (comma-separated). Pass e.g. "
                        "'1e14,0.1' to let selection choose between the velocity anchor "
                        "OFF (1e14) and ON (the finite seed).")
    p.add_argument("--select-blob-means-updates", type=str,
                   default=f"{INFER_BLOB_MEANS_UPDATES}",
                   help="phase_select blob_means_updates grid (comma-separated).")
    p.add_argument("--select-feature-update-damping", type=str,
                   default=f"{INFER_FEATURE_UPDATE_DAMPING:g}",
                   help="phase_select feature_update_damping grid (comma-separated). "
                        "Pass e.g. '0.0,0.25,0.5,1.0' to SELECT the Phase-A anti-drift "
                        "damping self-supervised on TRAIN (d=0 freeze .. d=1 vendored).")
    p.add_argument("--select-final-feature-temp", type=str,
                   default=f"{INFER_FINAL_FEATURE_TEMP:g}",
                   help="phase_select final_feature_temp grid (comma-separated). Pass "
                        "e.g. '1,2,4,8' to SELECT the Phase-1 inference feature temperature "
                        "self-supervised on TRAIN (tau=1 vendored feat-aware .. tau>1 "
                        "feature-driven). Capped (no feat_only) = a position floor.")
    p.add_argument("--select-em-iters", type=int, default=2,
                   help="Short-EM iters per phase_select grid combo.")
    # ---- Self-supervised objective (the EM/accept/select score; the GT gate is
    # ALWAYS binary region-J regardless). 'composite' = the richer combination;
    # 'region_J' = the binary kill-switch fallback (revert here if validate reports
    # corr(composite, GT) <= 0). ----
    p.add_argument("--objective", choices=("composite", "data_loglik", "region_J"),
                   default=_OBJECTIVE_MODE,
                   help="Self-supervised EM/accept/select objective. 'composite' (default) | "
                        "'data_loglik' (Phase-B probabilistic complete-data log-likelihood; "
                        "gate with _loglik_objective_proto.py first) | 'region_J' (binary "
                        "kill-switch fallback). The held-out GT gate is always binary region-J.")
    p.add_argument("--data-loglik-variant", type=str, default=_DATA_LOGLIK_VARIANT,
                   help="When --objective data_loglik, which term decomposition drives "
                        "selection (full | pos_feat | feat_only | feat | pos | vel | mix). "
                        "The Phase-B proto picks the best GT-correlating variant.")
    p.add_argument("--feature-update-damping", type=float, default=None,
                   help="Override the em/validate Phase-A anti-drift damping (d=0 freeze .. "
                        "d=1 vendored). Default: the phase_select-chosen value, else 1.0.")
    p.add_argument("--final-feature-temp", type=float, default=None,
                   help="Override the em/validate Phase-1 final-assignment feature temperature "
                        "(tau=1 vendored feat-aware .. tau>1 feature-driven). Default: the "
                        "phase_select-chosen value, else 1.0.")
    p.add_argument("--final-assignment-anchor", action="store_true",
                   help="Score the re-impl final feature term vs the FROZEN frame-0 anchor "
                        "(no-op at damping d=0; pairs with Phase-2 drift).")
    p.add_argument("--sigma-v-seed", type=float, default=None,
                   help="Override the EM initial sigma_V seed (the motion velocity-prior "
                        "variance). Default: DAVIS_DINO_DEFAULTS / the phase_select-chosen seed.")
    p.add_argument("--tokencut-seed-augment", action="store_true",
                   help="Enable TokenCut self-supervised seed augmentation for the WHOLE "
                        "pipeline (learn-under + select-under + validate-chosen). Self-supervised "
                        "(no GT); knobs from runs/.../tokencut_knobs.json (or tokencut.DEFAULT_KNOBS). "
                        "Augmented grids are disk-cached so the EM stays fast.")
    args = p.parse_args(argv)

    # The self-supervised objective + loglik variant are process-wide modes
    # (mirror _GT_SCORING_ALLOWED).
    _OBJECTIVE_MODE = args.objective
    _DATA_LOGLIK_VARIANT = args.data_loglik_variant
    # TokenCut seed augmentation is a process-wide mode too (set the module globals
    # _infer_cfg reads at call time → EM / select / validate-chosen all build cfgs
    # under it; the shipping-baseline infer is built separately and stays off).
    if args.tokencut_seed_augment:
        INFER_TOKENCUT_SEED_AUGMENT = True
        _knobs_path = OUT_ROOT / "tokencut_knobs.json"
        if _knobs_path.is_file():
            INFER_TOKENCUT_KNOBS = json.loads(_knobs_path.read_text())
            _log(f"TokenCut seed augmentation ON; knobs from {_knobs_path}: {INFER_TOKENCUT_KNOBS}")
        else:
            _log(f"TokenCut seed augmentation ON; knobs file {_knobs_path} missing → "
                 f"tokencut.DEFAULT_KNOBS")

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    _assert_split_discipline()
    # Base inference config from CLI (architectural-structure flags default global-ON).
    infer_kwargs = dict(use_sam_frame0=args.use_sam_frame0,
                        num_blobs=args.num_blobs,
                        num_hyperblobs=args.num_hyperblobs,
                        init_gibbs_sweeps=args.init_gibbs_sweeps,
                        feature_aware_final=args.feature_aware_final,
                        final_outlier=args.final_outlier,
                        freeze_hyperblob=args.freeze_hyperblob,
                        pure_object_seed=args.pure_object_seed,
                        blob_means_updates=args.blob_means_updates)
    if args.feature_update_damping is not None:
        infer_kwargs["feature_update_damping"] = float(args.feature_update_damping)
    if args.final_feature_temp is not None:
        infer_kwargs["final_feature_temp"] = float(args.final_feature_temp)
    if args.final_assignment_anchor:
        infer_kwargs["final_assignment_anchor"] = True
    num_sweeps = args.num_gibbs_sweeps_per_frame
    sigma_v_seed = args.sigma_v_seed   # None unless overridden / selected below
    # phase_select (if it ran) is AUTHORITATIVE for the global scalar tuple
    # (num_blobs / init_gibbs_sweeps / per-frame sweeps / sigma_V seed /
    # blob_means_updates), selected self-supervised on TRAIN — overlay it so em +
    # validate + emit all use the SAME config.
    sel = _load_selected_infer(out_root)
    if sel is not None and args.phase in ("em", "validate", "render_local", "all"):
        infer_kwargs["num_blobs"] = int(sel["num_blobs"])
        infer_kwargs["init_gibbs_sweeps"] = int(sel["init_gibbs_sweeps"])
        infer_kwargs["blob_means_updates"] = int(sel.get("blob_means_updates",
                                                          infer_kwargs["blob_means_updates"]))
        if "feature_update_damping" in sel and args.feature_update_damping is None:
            infer_kwargs["feature_update_damping"] = float(sel["feature_update_damping"])
        if "final_feature_temp" in sel and args.final_feature_temp is None:
            infer_kwargs["final_feature_temp"] = float(sel["final_feature_temp"])
        num_sweeps = int(sel["num_gibbs_sweeps_per_frame"])
        if sigma_v_seed is None and "sigma_V_seed" in sel:
            sigma_v_seed = float(sel["sigma_V_seed"])
        _log(f"applying phase_select tuple: num_blobs={infer_kwargs['num_blobs']} "
             f"init_gibbs_sweeps={infer_kwargs['init_gibbs_sweeps']} "
             f"per_frame_sweeps={num_sweeps} "
             f"blob_means_updates={infer_kwargs['blob_means_updates']} "
             f"feature_update_damping={infer_kwargs.get('feature_update_damping', INFER_FEATURE_UPDATE_DAMPING)} "
             f"final_feature_temp={infer_kwargs.get('final_feature_temp', INFER_FINAL_FEATURE_TEMP)} "
             f"sigma_V_seed={sigma_v_seed} (self-supervised, TRAIN)")

    # render_all_local's signature takes only the original 4 inference knobs; the
    # architectural-structure flags reach it via the emitted YAML (+ its own
    # setdefaults). So pass the render-relevant subset only.
    render_infer = dict(use_sam_frame0=infer_kwargs["use_sam_frame0"],
                        num_blobs=infer_kwargs["num_blobs"],
                        num_hyperblobs=infer_kwargs["num_hyperblobs"],
                        init_gibbs_sweeps=infer_kwargs["init_gibbs_sweeps"])

    def _do_render():
        from render_tracking import render_all_local
        return render_all_local(
            out_root, config_path=GENERAL_YAML,
            max_frames=args.max_frames_per_video,
            num_sweeps=num_sweeps, **render_infer)

    select_grid = {
        "num_blobs": [int(x) for x in args.select_num_blobs.split(",") if x.strip()],
        "init_gibbs_sweeps": [int(x) for x in args.select_init_sweeps.split(",") if x.strip()],
        "num_gibbs_sweeps_per_frame": [int(x) for x in args.select_per_frame_sweeps.split(",") if x.strip()],
        "sigma_V": [float(x) for x in args.select_sigma_v.split(",") if x.strip()],
        "blob_means_updates": [int(x) for x in args.select_blob_means_updates.split(",") if x.strip()],
        "feature_update_damping": [float(x) for x in
                                   args.select_feature_update_damping.split(",") if x.strip()],
        "final_feature_temp": [float(x) for x in
                               args.select_final_feature_temp.split(",") if x.strip()],
    }

    if args.phase == "preflight":
        phase_preflight(out_root)
        return 0
    if args.phase == "plumbing_smoke":
        info = plumbing_smoke()
        _save_json(out_root / "plumbing_smoke.json", info)
        return 0
    if args.phase == "pseudo_labels":
        phase_pseudo_labels(out_root, force=args.force,
                              max_frames_per_video=args.max_frames_per_video,
                              k_motion=args.k_motion,
                              regen_z_sam_only=args.regen_z_sam_only)
        return 0
    if args.phase == "select":
        phase_select(out_root, force=args.force,
                     max_frames_per_video=args.max_frames_per_video,
                     grid=select_grid, em_iters=args.select_em_iters,
                     base_infer_kwargs=infer_kwargs)
        return 0
    if args.phase == "em":
        phase_em(out_root, force=args.force,
                  max_frames_per_video=args.max_frames_per_video,
                  max_iters=args.max_iters,
                  num_gibbs_sweeps_per_frame=num_sweeps,
                  em_plateau_tol=args.em_plateau_tol,
                  keep_ari_diagnostic=args.keep_ari_diagnostic,
                  train_videos=list(TRAIN_VIDEOS),
                  em_val_videos=list(SELECT_VAL_VIDEOS),   # Phase D generalization probe
                  sigma_v_seed=sigma_v_seed,
                  **infer_kwargs)
        return 0
    if args.phase == "validate":
        val_info = phase_validate(out_root, force=args.force,
                                    max_frames_per_video=args.max_frames_per_video,
                                    num_gibbs_sweeps_per_frame=num_sweeps,
                                    heldout_videos=list(HELDOUT_VIDEOS),
                                    **infer_kwargs)
        em_info = _load_json(out_root / "em.json") or {}
        _emit_configs(out_root, val_info=val_info, em_info=em_info)
        return 0 if val_info.get("passed", False) else 2
    if args.phase == "validate_augmentation":
        # The ONE sanctioned, gate-routed held-out A/B of the seed augmentation
        # (replaces the deleted ad-hoc _tokencut_official_eval.py). No emit — the
        # shipping config already carries the flag; this reconfirms it through the lock.
        # Runs at the SHIPPING / live sweep depth (1), NOT the 25-sweep calibration
        # depth, so the number matches what deploys (the streaming demo always uses 1).
        aug_info = phase_validate_augmentation(out_root, force=args.force,
                                               max_frames_per_video=args.max_frames_per_video,
                                               num_gibbs_sweeps_per_frame=1,
                                               heldout_videos=list(HELDOUT_VIDEOS))
        return 0 if aug_info.get("passed", False) else 2
    if args.phase == "render_local":
        _do_render()
        return 0

    # --phase all: preflight -> (pseudo_labels if missing) -> select -> em(TRAIN)
    # -> validate(HELD-OUT) -> emit -> render_local.
    _run_with_retry("preflight", phase_preflight, out_root)
    _run_with_retry("pseudo_labels", phase_pseudo_labels, out_root, force=args.force,
                    max_frames_per_video=args.max_frames_per_video,
                    k_motion=args.k_motion)
    _run_with_retry("select", phase_select, out_root, force=args.force,
                    max_frames_per_video=args.max_frames_per_video,
                    grid=select_grid, em_iters=args.select_em_iters,
                    base_infer_kwargs=infer_kwargs)
    # Re-load the just-selected tuple and overlay it for em + validate + emit.
    sel = _load_selected_infer(out_root)
    if sel is not None:
        infer_kwargs["num_blobs"] = int(sel["num_blobs"])
        infer_kwargs["init_gibbs_sweeps"] = int(sel["init_gibbs_sweeps"])
        infer_kwargs["blob_means_updates"] = int(sel.get("blob_means_updates",
                                                          infer_kwargs["blob_means_updates"]))
        if "feature_update_damping" in sel and args.feature_update_damping is None:
            infer_kwargs["feature_update_damping"] = float(sel["feature_update_damping"])
        if "final_feature_temp" in sel and args.final_feature_temp is None:
            infer_kwargs["final_feature_temp"] = float(sel["final_feature_temp"])
        num_sweeps = int(sel["num_gibbs_sweeps_per_frame"])
        if sigma_v_seed is None and "sigma_V_seed" in sel:
            sigma_v_seed = float(sel["sigma_V_seed"])
        render_infer["num_blobs"] = infer_kwargs["num_blobs"]
        render_infer["init_gibbs_sweeps"] = infer_kwargs["init_gibbs_sweeps"]
    _run_with_retry("em", phase_em, out_root, force=args.force,
                    max_frames_per_video=args.max_frames_per_video,
                    max_iters=args.max_iters,
                    num_gibbs_sweeps_per_frame=num_sweeps,
                    em_plateau_tol=args.em_plateau_tol,
                    keep_ari_diagnostic=args.keep_ari_diagnostic,
                    train_videos=list(TRAIN_VIDEOS),
                    em_val_videos=list(SELECT_VAL_VIDEOS),   # Phase D generalization probe
                    sigma_v_seed=sigma_v_seed,
                    **infer_kwargs)
    val_info = _run_with_retry("validate", phase_validate, out_root,
                                 force=args.force,
                                 max_frames_per_video=args.max_frames_per_video,
                                 num_gibbs_sweeps_per_frame=num_sweeps,
                                 heldout_videos=list(HELDOUT_VIDEOS),
                                 **infer_kwargs)
    em_info = _load_json(out_root / "em.json") or {}
    _emit_configs(out_root, val_info=val_info, em_info=em_info)
    # Render the local tracking videos with the freshly-emitted config.
    _run_with_retry("render_local", _do_render, retries=0)
    return 0 if val_info.get("passed", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())

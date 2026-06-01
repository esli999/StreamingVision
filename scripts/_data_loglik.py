#!/usr/bin/env python
"""Probabilistic self-supervised objective: the model's COMPLETE-DATA
log-likelihood of a tracked frame/video (Phase B of the re-optimization plan).

Motivation: the shipped self-supervised objective (region-J of the cluster vs the
polluted SAM union ``Z_sam>0``) is ANTI-correlated with GT (corr -0.29/-0.40) — a
bleeding tracker matches the polluted union, so it cannot select an anti-bleed
setting self-supervised. This module replaces it with the model's OWN data
log-likelihood, which a bleeding tracker should score LOWER on (background
datapoints absorbed into an object blob have a large DINO-feature residual ->
lower feature log-likelihood).

Faithful to the vendored per-datapoint assignment density
(``gibbs_blob_assignments_dino``, dino.py:700-721): for each datapoint n assigned
to blob z_n, the complete-data contribution is

    log p(z_n) + log p(pos_n, vel_n, feat_n | blob_{z_n})
  = log_mixture_weight[z_n]
    + logN(pos_n | blob_means, blob_covs)            (position term)
    + logN(vel_n | blob_vel_means, blob_vel_covs)    (velocity term)
    + Sum_d logN(feat_n,d | blob_features_d, sqrt(sigma_F))   (feature term)

and for an OUTLIER datapoint (z_n == n_blobs) the inlier likelihood is replaced by
the Gamma-over-speed outlier density (dino.py:713-718) plus the outlier mixture
weight. We GATHER the assigned blob (one density per datapoint) rather than
vmapping all L blobs (dino.py vmaps all L for the categorical) — L-times cheaper,
since here we only need the likelihood AT the committed assignment.

The terms are kept SEPARATE so the Phase-B proto can test which combination best
correlates with GT (full vs feature-up-weighted vs feature-only), per decision #2
("try hard to make the probabilistic objective work" before any region_J
fallback). Unweighted (pos_w=vel_w=feat_w=mix_w=1, include_outlier=True) reproduces
the vendored complete-data density exactly (verified in the proto).
"""
from __future__ import annotations

import numpy as _np
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as _mvn
from jax.scipy.stats import norm as _norm


def instance_purity_meanll(blob_a, z_sam, eps: float = 1e-6) -> float:
    """Mean per-datapoint instance-PURITY log-likelihood (HOST-SIDE numpy).

    Phase-3 self-supervised objective term, from the temporally-consistent SAM
    instances (``Z_sam``) — NOT the polluted union, NOT GT.  For each blob b,
    ``theta_b[k]`` is the empirical fraction of b's datapoints whose SAM instance is
    k (the MLE instance distribution of blob b; background ``z_sam==0`` is its own
    bucket).  The per-datapoint term is ``log theta_{b_n, z_sam_n}`` and we return
    its mean over inlier datapoints.

    It is MAXIMISED (-> 0) when every blob maps to exactly ONE instance, and a blob
    that BLEEDS across instances scores ``log(fraction) < 0`` — a probabilistic,
    differentiable analog of the discrete ``_instance_matched_J`` (calibrate_
    consistency.py).  Unlike the anchor feature term (which is freeze-conservative),
    purity credits any partition that keeps instances separate REGARDLESS of
    appearance drift, so it can reward a beneficial slow drift the anchor term cannot.

    ``blob_a``: (N,) int blob id (>=0; -1 = outlier, excluded).  ``z_sam``: (N,) int
    SAM instance id (0 = background)."""
    z = _np.asarray(blob_a).reshape(-1)
    zs = _np.asarray(z_sam).reshape(-1)
    m = (z >= 0)
    if not m.any():
        return float("nan")
    z = z[m]
    zs = zs[m]
    # Relabel instance ids to a dense [0, K) range (K static per frame, host-side).
    _uniq, zs_idx = _np.unique(zs, return_inverse=True)
    K = int(_uniq.size)
    L = int(z.max()) + 1
    flat = z * K + zs_idx                                  # (M,) joint (blob, instance)
    counts = _np.bincount(flat, minlength=L * K).reshape(L, K).astype(_np.float64)
    totals = counts.sum(axis=1, keepdims=True)             # (L,1) per-blob datapoint count
    theta = counts / _np.maximum(totals, 1.0)              # (L,K) MLE instance dist
    ll = _np.log(theta[z, zs_idx] + eps)                   # (M,) log theta at each datapoint
    return float(_np.mean(ll))


def per_datapoint_terms(state, anchor_blob_feat=None):
    """Return the per-datapoint term decomposition for the live tracker state.

    Returns a dict of (N,) arrays: pos_ll, vel_ll, feat_ll, feat_anchor_ll,
    mix_ll, outlier_ll, and the bool mask is_outlier. All gathered at each
    datapoint's COMMITTED blob assignment (state.datapoints_state.blob_assignments).

    ``anchor_blob_feat`` (L, D) — the FROZEN frame-0 blob appearance. When given,
    ``feat_anchor_ll`` scores each datapoint's feature against the anchor of its
    assigned blob (NOT the live, drifted blob_features). This is the key fix for
    the circularity of the naive complete-data loglik: evaluating features against
    the same drifted means that were FIT to the data rewards drift (d=1 always fits
    best). The anchor term instead measures "do the assigned datapoints still match
    the frame-0 object appearance" — the self-supervised analog of the frozen-
    centroid DINO classifier (0.71) that beats the drifting tracker; it PENALISES
    bleed, so it should track GT."""
    ds = state.datapoints_state
    bs = state.blobs_state
    h = state.hypers
    pos = ds.datapoint_positions          # (N, 3)
    vel = ds.datapoint_vels               # (N, 3)
    feat = ds.datapoint_features          # (N, D)
    z = ds.blob_assignments               # (N,) in [0, L]; ==L means outlier
    L = h.n_blobs
    sigma_F = h.sigma_F
    z_safe = jnp.minimum(z, L - 1)        # clamp outliers for the gather

    bm = bs.blob_means[z_safe]            # (N, 3)
    bc = bs.blob_covs[z_safe]             # (N, 3, 3)
    bvm = bs.blob_vel_means[z_safe]       # (N, 3)
    bvc = bs.blob_vel_covs[z_safe]        # (N, 3, 3)
    bf = bs.blob_features[z_safe]         # (N, D)

    pos_ll = jax.vmap(lambda x, m, c: _mvn.logpdf(x, m, c))(pos, bm, bc)     # (N,)
    vel_ll = jax.vmap(lambda x, m, c: _mvn.logpdf(x, m, c))(vel, bvm, bvc)   # (N,)
    feat_ll = jnp.sum(_norm.logpdf(feat, bf, jnp.sqrt(sigma_F)), axis=-1)    # (N,)
    # Anchor-referenced feature term (the anti-drift objective). Falls back to the
    # live feature term when no anchor is supplied (so callers without an anchor
    # keep the old behaviour).
    if anchor_blob_feat is not None:
        af = anchor_blob_feat[z_safe]                                        # (N, D)
        feat_anchor_ll = jnp.sum(_norm.logpdf(feat, af, jnp.sqrt(sigma_F)), axis=-1)
    else:
        feat_anchor_ll = feat_ll

    # Mixture (assignment-prior) weight at the committed assignment.
    bw = bs.blob_weights                                                      # (L,)
    ext = jnp.concatenate([bw, jnp.array([h.outlier_prob])])
    log_mix = jnp.log(ext / jnp.sum(ext))                                     # (L+1,)
    mix_ll = log_mix[jnp.minimum(z, L)]                                       # (N,)

    # Outlier component: Gamma density over speed (dino.py:713-718).
    speed = jnp.linalg.norm(vel, axis=-1)                                     # (N,)
    a = h.outlier_velocity_gamma_shape
    b = h.outlier_velocity_gamma_rate
    outlier_ll = ((a - 1) * jnp.log(speed + 1e-8) - b * speed
                  - a * jnp.log(1.0 / b) - jax.lax.lgamma(a))                 # (N,)

    is_outlier = (z >= L)
    return {
        "pos_ll": pos_ll, "vel_ll": vel_ll, "feat_ll": feat_ll,
        "feat_anchor_ll": feat_anchor_ll,
        "mix_ll": mix_ll, "outlier_ll": outlier_ll, "is_outlier": is_outlier,
    }


def frame_data_loglik(state, *, pos_w=1.0, vel_w=1.0, feat_w=1.0, mix_w=1.0,
                      include_outlier=True, reduce="mean", anchor_blob_feat=None,
                      use_anchor_feat=False):
    """Complete-data log-likelihood of one tracked frame.

    Weights default to 1 (faithful to the vendored density). reduce="mean"
    returns the per-datapoint mean (comparable across videos / frame counts).
    use_anchor_feat=True (with anchor_blob_feat) swaps the live feature term for
    the anchor-referenced one (the anti-drift objective)."""
    t = per_datapoint_terms(state, anchor_blob_feat=anchor_blob_feat)
    featterm = t["feat_anchor_ll"] if use_anchor_feat else t["feat_ll"]
    inlier = pos_w * t["pos_ll"] + vel_w * t["vel_ll"] + feat_w * featterm
    per = jnp.where(t["is_outlier"], t["outlier_ll"], inlier) + mix_w * t["mix_ll"]
    if not include_outlier:
        keep = ~t["is_outlier"]
        tot = jnp.sum(jnp.where(keep, per, 0.0))
        n = jnp.maximum(jnp.sum(keep), 1.0)
        return tot / n if reduce == "mean" else tot
    if reduce == "mean":
        return jnp.mean(per)
    return jnp.sum(per)


def frame_data_loglik_terms(state, anchor_blob_feat=None):
    """Per-frame MEAN of each term + each candidate OBJECTIVE variant, computed
    from a SINGLE per_datapoint_terms pass. The NAIVE (live-feature) variants
    (full/feat/pos_feat) reward drift and ANTI-correlate with GT; the ANCHOR
    variants (anchor_full/anchor_pos_feat/feat_anchor) score features against the
    FROZEN frame-0 appearance and should TRACK GT. The Phase-B proto picks the
    best GT-correlating one; that key becomes _DATA_LOGLIK_VARIANT for the EM."""
    t = per_datapoint_terms(state, anchor_blob_feat=anchor_blob_feat)
    pos, vel = t["pos_ll"], t["vel_ll"]
    feat, fa, mix = t["feat_ll"], t["feat_anchor_ll"], t["mix_ll"]
    is_out, out = t["is_outlier"], t["outlier_ll"]

    def _full(featterm):
        per = jnp.where(is_out, out, pos + vel + featterm) + mix
        return jnp.mean(per)

    return {
        # per-term means
        "pos": jnp.mean(pos), "vel": jnp.mean(vel),
        "feat": jnp.mean(feat), "feat_anchor": jnp.mean(fa), "mix": jnp.mean(mix),
        "outlier_frac": jnp.mean(is_out.astype(jnp.float32)),
        # NAIVE live-feature objectives (anti-drift-rewarding — kept for the record)
        "full": _full(feat),
        "pos_feat": jnp.mean(pos + feat),
        # ANCHOR-referenced objectives (the fix)
        "anchor_full": _full(fa),
        "anchor_pos_feat": jnp.mean(pos + fa),
    }

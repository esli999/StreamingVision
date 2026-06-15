#!/usr/bin/env python
"""JAX TokenCut — probabilistic self-supervised object discovery.

Finds the globally-coherent object/background partition on a DINO-feature affinity
graph via a normalized cut — a partition that greedy k-means / motion-outlier
heuristics tend to miss. Framed probabilistically: a degree-weighted 2-component
1-D GMM posterior on the Fiedler vector gives a soft foreground probability
``q∈[0,1]`` per token — the natural Bayesian relaxation of the hard ``sign(phi)``
cut, a graded low-variance signal the hard locators lack. Controllable knobs are
selected self-supervised by SAM-agreement — no GT, no text prompts, no per-video
tuning.

The spectral core ``discover()`` is pure JAX, end-to-end jit-compilable:
  - static shapes (N tokens fixed per resolution),
  - continuous knobs (T, sigma_s, tau_edge) TRACED — sweepable WITHOUT recompiling,
  - structural knobs (edge_mode, use_spatial, posterior, n_cuts, em_iters) STATIC.
Follows the native-JAX discipline: no python loops over data, no host transfers in
the per-call path, one compile per structural combo.

Seed augmentation (Part C) is additive + uncovered-only + gated behind one
default-off flag, so applying it to a video SAM already covers is a no-op by
construction (the discovery agrees with SAM → nothing painted).
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "scripts"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GH, GW = 45, 80                 # the cached pca32 datapoint lattice (== cc.GRID_H/W)
_EPS = 1e-8

# Augmented grids are deterministic per (vid, knobs, base_grid) — cache them to disk
# + in-process so callers that re-run the tracker many times per video never
# recompute the DINO+SAM2 frame-0 discovery. The hash includes the base grid bytes
# so a changed SAM seed invalidates the entry.
_AUG_CACHE_DIR = _REPO / "runs" / "calibrate_consistency" / "tokencut_grids"
_AUG_MEM = {}


# ----------------------------------------------------------------------
# Part A — the pure-JAX probabilistic spectral core
# ----------------------------------------------------------------------
@partial(jax.jit, static_argnames=("edge_mode", "use_spatial", "posterior",
                                   "n_cuts", "em_iters"))
def discover(features, gridpos, T=0.20, sigma_s=0.30, tau_edge=0.20,
             edge_mode=False, use_spatial=True, posterior="gmm",
             n_cuts=1, em_iters=25):
    """Soft foreground posterior from a normalized cut on the feature graph.

    features (N,D) DINO/PCA tokens; gridpos (N,2) lattice coords in [0,1].
    Returns ``(q (N,), eigengap_first_cut)``. Continuous knobs traced; structural
    knobs static. Pure / functional / no host transfers."""
    N = features.shape[0]
    F = features / (jnp.linalg.norm(features, axis=1, keepdims=True) + _EPS)
    s = F @ F.T                                          # cosine affinity (N,N)
    if edge_mode:                                        # original TokenCut: binary
        W_feat = jnp.where(s >= tau_edge, 1.0, 1e-5)
    else:                                                # soft exponential affinity
        W_feat = jnp.exp((s - 1.0) / T)
    if use_spatial:                                      # mild locality prior
        sq = (gridpos ** 2).sum(1)
        d2 = jnp.maximum(sq[:, None] + sq[None, :] - 2.0 * (gridpos @ gridpos.T), 0.0)
        W = W_feat * jnp.exp(-d2 / (2.0 * sigma_s ** 2 + _EPS))
    else:
        W = W_feat
    W = W * (1.0 - jnp.eye(N))                            # zero self-affinity

    # GT-free, text-free foreground orientation priors: objects sit away from the
    # frame border and are spatially compact, so a thin frame-connected structure
    # is demoted in favour of a compact blob.
    border = ((gridpos[:, 0] < 0.06) | (gridpos[:, 0] > 0.94) |
              (gridpos[:, 1] < 0.06) | (gridpos[:, 1] > 0.94)).astype(features.dtype)

    def _posterior(phi, deg):
        if posterior == "logistic":
            phi0 = jnp.median(phi)
            beta = jnp.std(phi) + _EPS
            return jax.nn.sigmoid((phi - phi0) / beta)
        # degree-weighted 2-component 1-D GMM via fixed-iteration EM (jit-safe).
        w = deg / (deg.sum() + _EPS)
        m = (w * phi).sum()
        sd = jnp.sqrt((w * (phi - m) ** 2).sum()) + _EPS
        mu = jnp.array([m - sd, m + sd])
        var = jnp.array([sd ** 2, sd ** 2])
        pi = jnp.array([0.5, 0.5])

        def em_step(carry, _):
            mu, var, pi = carry
            ll0 = -0.5 * (phi - mu[0]) ** 2 / var[0] - 0.5 * jnp.log(var[0]) + jnp.log(pi[0])
            ll1 = -0.5 * (phi - mu[1]) ** 2 / var[1] - 0.5 * jnp.log(var[1]) + jnp.log(pi[1])
            mx = jnp.maximum(ll0, ll1)
            e0 = jnp.exp(ll0 - mx); e1 = jnp.exp(ll1 - mx)
            r1 = e1 / (e0 + e1 + _EPS); r0 = 1.0 - r1
            w0 = r0 * deg; w1 = r1 * deg
            n0 = w0.sum() + _EPS; n1 = w1.sum() + _EPS
            mu0 = (w0 * phi).sum() / n0; mu1 = (w1 * phi).sum() / n1
            v0 = (w0 * (phi - mu0) ** 2).sum() / n0 + 1e-6
            v1 = (w1 * (phi - mu1) ** 2).sum() / n1 + 1e-6
            tot = n0 + n1
            return (jnp.array([mu0, mu1]), jnp.array([v0, v1]),
                    jnp.array([n0 / tot, n1 / tot])), None

        (mu, var, pi), _ = jax.lax.scan(em_step, (mu, var, pi), None, length=em_iters)
        ll0 = -0.5 * (phi - mu[0]) ** 2 / var[0] - 0.5 * jnp.log(var[0]) + jnp.log(pi[0])
        ll1 = -0.5 * (phi - mu[1]) ** 2 / var[1] - 0.5 * jnp.log(var[1]) + jnp.log(pi[1])
        mx = jnp.maximum(ll0, ll1)
        e0 = jnp.exp(ll0 - mx); e1 = jnp.exp(ll1 - mx)
        return e1 / (e0 + e1 + _EPS)                      # P(high-phi component)

    def _objectness(mask_soft):
        # higher = more object-like: away from the border + spatially compact.
        n = mask_soft.sum() + _EPS
        bf = (mask_soft * border).sum() / n
        mp = (mask_soft[:, None] * gridpos).sum(0) / n
        var = (mask_soft[:, None] * (gridpos - mp) ** 2).sum() / n
        return -bf - 0.5 * var

    def one_cut(Wc):
        deg = Wc.sum(1)
        Dm12 = 1.0 / jnp.sqrt(deg + _EPS)
        # symmetric normalized Laplacian (avoids the generalized eigenproblem JAX
        # lacks); Fiedler vector phi = D^{-1/2} y, y = 2nd-smallest eigenvector.
        L = jnp.eye(N) - Dm12[:, None] * Wc * Dm12[None, :]
        L = 0.5 * (L + L.T)                               # symmetrize (numerical)
        evals, evecs = jnp.linalg.eigh(L)
        phi = Dm12 * evecs[:, 1]
        gap = evals[2] - evals[1]                         # spectral-stability prior
        r = _posterior(phi, deg)
        # orient sign-arbitrary Fiedler so foreground = the more object-like side.
        q = jnp.where(_objectness(r) >= _objectness(1.0 - r), r, 1.0 - r)
        return q, gap

    def cut_step(carry, _):
        Wc, q_acc = carry
        q, gap = one_cut(Wc)
        q_acc = jnp.maximum(q_acc, q)
        Wn = Wc * (1.0 - q)[:, None] * (1.0 - q)[None, :]  # MaskCut down-weight
        return (Wn, q_acc), (q, gap)

    (_, q_total), (q_each, gaps) = jax.lax.scan(
        cut_step, (W, jnp.zeros(N)), None, length=n_cuts)
    # q_each (n_cuts, N): per-cut foregrounds — MaskCut isolates DISTINCT objects
    # across cuts, so the object may be a LATER cut, not the union.
    return q_total, gaps[0], q_each


# ----------------------------------------------------------------------
# Feature sources (perception, upstream of the JAX graph — same boundary as
# SAM2/DINO already are)
# ----------------------------------------------------------------------
def _labels_path(vid):
    return _REPO / "runs" / "calibrate_consistency" / "labels" / f"{vid}.npz"


def frame0_features(vid, source="pca32", labels=None):
    """Return ``(features (N,D) float32, gridpos (N,2) in [0,1])`` for the seed
    frame. ``pca32`` = the cached DINO PCA-32 tokens at the 2925 datapoints
    (instant); ``raw384`` = the full DINOv2 ViT-S lattice (cleaner, heavier)."""
    if source == "pca32":
        if labels is None:
            with np.load(_labels_path(vid)) as d:
                feat = np.asarray(d["features"])[0]
                idx = np.asarray(d["indices"]).reshape(-1)
        else:
            feat = np.asarray(labels["features"])[0]
            idx = np.asarray(labels["indices"]).reshape(-1)
        rows = (idx // GW).astype(np.float32) / float(GH)
        cols = (idx % GW).astype(np.float32) / float(GW)
        gridpos = np.stack([rows, cols], 1).astype(np.float32)
        return feat.astype(np.float32), gridpos
    if source == "raw384":
        return _raw384_features(vid)
    raise ValueError(f"unknown feature_source {source!r}")


_DINO_MODEL = None


def _raw384_features(vid):
    """Full DINOv2 ViT-S 384-dim patch lattice (26×46=1196 patches) for the seed
    frame — the clean, faithful TokenCut input (the cached pca32 is more
    compressed). Perception (the DINO model only — no depth/flow) sits upstream of
    the JAX graph, the same boundary SAM2/DINO already are."""
    global _DINO_MODEL
    import cv2
    import streaming_dino
    import config as gm_config                       # cc set GENMATTER_DAVIS_DIR
    with np.load(_labels_path(vid)) as d:
        fi = int(np.asarray(d["frame_idx"]).reshape(-1)[0])
    rgb_dir = Path(gm_config.DAVIS_RGB_PATH) / vid
    f0 = rgb_dir / f"{fi:05d}.jpg"
    if not f0.is_file():
        f0 = rgb_dir / f"{fi:05d}.png"
    bgr = cv2.imread(str(f0))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if _DINO_MODEL is None:
        _DINO_MODEL = streaming_dino.load_dino()
    patches, (gh, gw) = streaming_dino.dino_patches(_DINO_MODEL, rgb)
    ar = np.arange(gh * gw)
    gridpos = np.stack([(ar // gw) / float(gh), (ar % gw) / float(gw)], 1).astype(np.float32)
    return patches.astype(np.float32), gridpos


# ----------------------------------------------------------------------
# Part C — gated, additive, SAM2-refined seed augmentation (reuses the
# _sam2_reprompt_proto library)
# ----------------------------------------------------------------------
# Faithful TokenCut recipe: binary graph + NO spatial kernel + MaskCut isolates
# distinct objects, where the soft+spatial variant tends to capture only the
# dominant scene-texture partition. raw384 = the un-PCA'd DINO lattice.
DEFAULT_KNOBS = {
    "feature_source": "raw384", "T": 0.20, "sigma_s": 0.30, "tau_edge": 0.20,
    "edge_mode": True, "use_spatial": False, "posterior": "gmm", "n_cuts": 3,
    "q_thresh": 0.5, "uncovered_thresh": 0.5, "size_lo": 0.03, "size_hi": 0.30,
    "n_points": 8,
}


def _q_to_grid45(q, source, idx):
    """Rasterize the native-lattice posterior q onto the 45×80 datapoint grid so
    every downstream consumer (gate, q-peak, SAM-agreement) is lattice-agnostic.
    pca32: sparse scatter at ``idx``; raw384: resize the 26×46 lattice."""
    import cv2
    if source == "raw384":
        from streaming_dino import DINO_GH, DINO_GW
        return cv2.resize(np.asarray(q, np.float32).reshape(DINO_GH, DINO_GW),
                          (GW, GH), interpolation=cv2.INTER_LINEAR)
    g = np.zeros(GH * GW, np.float32)
    g[idx] = np.asarray(q, np.float32)
    return g.reshape(GH, GW)


def discover_q(vid, labels=None, knobs=None):
    """Host-side wrapper: run ``discover`` for one video. Returns ``(q_grid45
    (45,80), eigengap, q_each_grid45 (n_cuts,45,80), gridpos)`` — the 45×80 rasters
    are lattice-agnostic; ``q_each_grid45`` exposes the per-cut MaskCut foregrounds
    so the object (often a LATER cut, not the union) can be selected downstream."""
    k = {**DEFAULT_KNOBS, **(knobs or {})}
    feat, gridpos = frame0_features(vid, k["feature_source"], labels)
    q_total, gap, q_each = discover(jnp.asarray(feat), jnp.asarray(gridpos),
                                    T=k["T"], sigma_s=k["sigma_s"], tau_edge=k["tau_edge"],
                                    edge_mode=k["edge_mode"], use_spatial=k["use_spatial"],
                                    posterior=k["posterior"], n_cuts=int(k["n_cuts"]))
    q_total = np.asarray(q_total); q_each = np.asarray(q_each)
    idx = None if labels is None else np.asarray(labels["indices"]).reshape(-1)
    if idx is None and k["feature_source"] == "pca32":
        with np.load(_labels_path(vid)) as d:
            idx = np.asarray(d["indices"]).reshape(-1)
    q_grid = _q_to_grid45(q_total, k["feature_source"], idx)
    q_each_grid = np.stack([_q_to_grid45(q_each[i], k["feature_source"], idx)
                            for i in range(q_each.shape[0])], 0)
    return q_grid, float(gap), q_each_grid, gridpos


def _q_peak_rc(q_grid):
    """Smoothed-q argmax → (row,col) in the 45×80 grid (robust prompt landing)."""
    import cv2
    Mb = cv2.blur(np.asarray(q_grid, np.float32), (3, 3))
    r, c = np.unravel_index(int(np.argmax(Mb)), Mb.shape)
    return (float(r), float(c))


def _topk_uncovered_points(q_grid, white_grid, k, q_thresh):
    """Top-k high-q grid cells in the SAM-uncovered region — a MULTI-POINT SAM2
    prompt (thin/elongated objects that a single point under-segments). Returns
    [(row,col), ...]."""
    qm = (np.asarray(q_grid, np.float32) * white_grid).reshape(-1)
    order = np.argsort(qm)[::-1][:k]
    return [(int(i // GW), int(i % GW)) for i in order if qm[i] > q_thresh]


def _knobs_hash(knobs, base_grid):
    import hashlib, json
    k = {**DEFAULT_KNOBS, **(knobs or {})}
    h = hashlib.md5(json.dumps(k, sort_keys=True, default=str).encode())
    h.update(np.ascontiguousarray(base_grid).tobytes())
    return h.hexdigest()[:16]


def augment_seed_grid(vid, labels, base_grid, knobs=None, use_cache=True):
    """Cached wrapper around the (deterministic) TokenCut+SAM2 seed augmentation.
    First call per (vid, knobs, base_grid) computes + persists; later calls load
    instantly. Returns ``(grid, info)``."""
    import json
    if base_grid is None:
        return base_grid, {"added": False, "reason": "no base grid"}
    kh = _knobs_hash(knobs, base_grid)
    key = (vid, kh)
    if use_cache and key in _AUG_MEM:
        return _AUG_MEM[key]
    cf = _AUG_CACHE_DIR / f"{vid}.{kh}.npz"
    if use_cache and cf.is_file():
        with np.load(cf, allow_pickle=True) as d:
            res = (np.asarray(d["grid"]), json.loads(str(d["info"])))
        _AUG_MEM[key] = res
        return res
    try:
        res = _augment_seed_grid_compute(vid, labels, base_grid, knobs)
    except Exception as e:
        # e.g. a local (non-DAVIS) video whose frame-0 RGB isn't on the DAVIS path
        # → can't re-extract raw384 / prompt SAM2 → gracefully no-op (use base seed,
        # the current behavior). Deterministic per video, so caching it is correct.
        res = (base_grid, {"added": False, "reason": f"compute skipped ({type(e).__name__})"})
    if use_cache:
        _AUG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cf, grid=res[0], info=json.dumps(res[1], default=float))
        _AUG_MEM[key] = res
    return res


def _fire_candidate(vid, q_total_grid, q_each_grid, idx, white, white_grid, k):
    """The SAM2 reprompt candidate selection used by ``_augment_seed_grid_compute``.

    Each MaskCut foreground (and the union) proposes a MULTI-POINT prompt (the top-k
    high-q cells SAM left uncovered); SAM2 refines it into a crisp mask. A candidate
    counts only if its SAM2 mask is object-sized AND mostly NEW (in ``Z_sam==0``); the
    best (max new_frac) is returned. When SAM already covers the object the re-prompt
    returns that same object → new_frac low → no fire; when SAM missed an object the
    re-prompt returns the missed object → fire. Returns ``(cand_or_None, proposals)``
    where cand = ``(new_frac, msize, peak, mask_grid, ci)``."""
    import _sam2_reprompt_proto as rp
    cand = None
    proposals = [q_total_grid] + [q_each_grid[i] for i in range(q_each_grid.shape[0])]
    for ci, qg in enumerate(proposals):
        peak = _q_peak_rc(qg)
        pr, pc = int(round(peak[0])), int(round(peak[1]))
        if pr < 2 or pr > GH - 3 or pc < 2 or pc > GW - 3:   # reject frame-border peaks
            continue
        pts = _topk_uncovered_points(qg, white_grid, int(k["n_points"]), k["q_thresh"])
        if len(pts) < 2:                                 # need a confident multi-point prompt
            continue
        mask_grid = rp._sam2_mask_grid_multi(vid, pts)
        if mask_grid is None:
            continue
        mflat = mask_grid.reshape(-1)[idx]
        msize = float(mflat.mean())
        if not (k["size_lo"] <= msize <= k["size_hi"]):  # object-sized SAM2 mask
            continue
        new_frac = float((mflat & white).sum() / max(int(mflat.sum()), 1))
        if new_frac < k["uncovered_thresh"]:             # mask mostly already SAM-covered
            continue
        if cand is None or new_frac > cand[0]:
            cand = (new_frac, msize, peak, mask_grid, ci)
    return cand, proposals


def _augment_seed_grid_compute(vid, labels, base_grid, knobs=None):
    """Gated, additive TokenCut+SAM2 seed augmentation. Returns ``(grid, info)``.

    Fires ONLY when TokenCut finds a confident-compact foreground that SAM does
    NOT cover (object-sized, mostly in ``Z_sam==0``); otherwise returns
    ``base_grid`` unchanged → working videos are untouched by construction. On
    fire, the smoothed-q peak prompts SAM2 for a CRISP mask, painted additively
    only into still-uncovered cells (the validated _sam2_reprompt path)."""
    k = {**DEFAULT_KNOBS, **(knobs or {})}
    q_total_grid, gap, q_each_grid, _ = discover_q(vid, labels, k)
    idx = np.asarray(labels["indices"]).reshape(-1)
    z = np.asarray(labels["Z_sam"])[0]
    white = (z == 0)                                     # SAM-uncovered datapoints
    white_grid = np.zeros(GH * GW, np.float32)
    white_grid[idx[white]] = 1.0
    white_grid = white_grid.reshape(GH, GW)

    cand, proposals = _fire_candidate(vid, q_total_grid, q_each_grid, idx, white, white_grid, k)
    if cand is None:
        return base_grid, {"added": False, "reason": "no SAM-uncovered object", "eigengap": gap}
    import _sam2_reprompt_proto as rp
    new_frac, msize, peak, mask_grid, ci = cand
    aug, nadd = rp._augment_grid(base_grid, mask_grid, idx)
    if nadd < 5:
        return base_grid, {"added": False, "reason": f"nadd={nadd}", "eigengap": gap}
    # Self-supervised reprompt-quality signal (no GT): how well the crisp SAM2 mask
    # agrees with the soft TokenCut posterior of the fired proposal (the located
    # object). Higher = the reprompt captured the object without over-/under-
    # segmenting (n_points too low under-covers thin objects; too high over-extends
    # compact ones). Info-only; the returned `aug` grid is unchanged.
    q_fired_dp = np.asarray(proposals[ci]).reshape(-1)[idx]
    fg_agreement = float(soft_iou(q_fired_dp, mask_grid.reshape(-1)[idx]))
    return aug, {"added": True, "nadd": int(nadd), "new_frac": new_frac, "msize": msize,
                 "peak": peak, "cut": int(ci), "eigengap": gap,
                 "fg_agreement": fg_agreement}


# ----------------------------------------------------------------------
# Soft-IoU agreement — used by the SAM2 re-prompt fire test (_fire_candidate)
# and the discover smoke test below.
# ----------------------------------------------------------------------
def soft_iou(q, ref_bool):
    """Soft-IoU between a soft posterior q∈[0,1] and a boolean reference."""
    r = ref_bool.astype(np.float32)
    inter = float((q * r).sum())
    union = float(q.sum() + r.sum() - (q * r).sum())
    return inter / (union + _EPS)


if __name__ == "__main__":
    # Smoke: a planted 2-blob synthetic input — the cut must recover the object.
    rng = np.random.default_rng(0)
    N = 400
    pos = rng.uniform(0, 1, (N, 2)).astype(np.float32)
    obj = ((pos[:, 0] - 0.5) ** 2 + (pos[:, 1] - 0.5) ** 2) < 0.04   # compact center blob
    feat = rng.normal(0, 0.05, (N, 16)).astype(np.float32)
    feat[obj] += np.array([2.0] + [0.0] * 15, np.float32)            # distinct object feature
    feat[~obj] += np.array([0.0, 2.0] + [0.0] * 14, np.float32)
    q, gap, _ = discover(jnp.asarray(feat), jnp.asarray(pos), T=0.2, sigma_s=0.3)
    q = np.asarray(q)
    iou = soft_iou(q, obj)
    print(f"synthetic: q in [{q.min():.2f},{q.max():.2f}] finite={np.isfinite(q).all()} "
          f"recovered-obj soft-IoU={iou:.3f} eigengap={float(gap):.4f}")
    assert np.isfinite(q).all() and 0.0 <= q.min() and q.max() <= 1.0
    assert iou > 0.5, f"cut failed to recover the planted object (IoU={iou:.3f})"
    print("OK")

#!/usr/bin/env python
"""Phase-B GATE: is the model's COMPLETE-DATA log-likelihood a self-supervised
objective that TRACKS held-out GT (so it can SELECT the anti-drift damping without
test-set tuning)? The shipped region-J-vs-union objective is ANTI-correlated
(-0.29/-0.40); this proto asks whether `video_data_loglik` is POSITIVELY correlated.

Design (differs from _clean_objective_proto.py on purpose): we induce bleed
variation by sweeping the Phase-A DAMPING (0=freeze .. 1=vendored), NOT sigma_F.
Scaling sigma_F would change the likelihood MODEL itself, making data_loglik
incomparable across grid points; damping varies the inference trajectory (hence
bleed) under a FIXED model, so data_loglik is comparable — and damping is exactly
the axis Phase-D selects on. We then report corr(data_loglik variant, GT region-J)
pooled and within-video, plus the decision-relevant check: does the damping that
MAXIMIZES median data_loglik match the damping that maximizes median GT?

GATE (decision #2 — try hard before any region_J fallback): REQUIRE the best
variant's spearman(data_loglik, GT) > 0 (target >= +0.3) AND argmax-damping
agreement with GT. GT is read ONCE here as a gate only (cc._GT_SCORING_ALLOWED).
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
import argparse
import copy
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "genmatterpp"))
import calibrate_consistency as cc   # noqa: E402
import genmatter_rt                   # noqa: E402


def _region_j(hb_TN, fg_of, Tc):
    g0 = fg_of(0)
    if g0 is None:
        return float("nan")
    g0 = np.asarray(g0).reshape(-1); c0 = hb_TN[0]
    ref = [int(c) for c in np.unique(c0)
           if c >= 0 and g0[c0 == c].size and g0[c0 == c].mean() > 0.5]
    Js = []
    for t in range(Tc):
        g = fg_of(t)
        if g is None:
            continue
        g = np.asarray(g).reshape(-1)
        pred = np.isin(hb_TN[t], ref)
        u = int(np.logical_or(pred, g).sum())
        Js.append(int(np.logical_and(pred, g).sum()) / u if u > 0 else 0.0)
    return float(np.mean(Js)) if Js else float("nan")


def _verify_math() -> bool:
    """genjax mv_normal / normal logpdf == the jax.scipy logpdf used in
    _data_loglik (guards against a distribution-convention mismatch)."""
    import jax, jax.numpy as jnp, genjax
    from jax.scipy.stats import multivariate_normal as mvn
    from jax.scipy.stats import norm
    k = jax.random.PRNGKey(0)
    x = jax.random.normal(k, (3,)); m = jax.random.normal(jax.random.PRNGKey(1), (3,))
    A = jax.random.normal(jax.random.PRNGKey(2), (3, 3)); C = A @ A.T + 3 * jnp.eye(3)
    gj_mvn = float(genjax.mv_normal.assess(genjax.ChoiceMapBuilder.v(x), (m, C))[0])
    sp_mvn = float(mvn.logpdf(x, m, C))
    xf = 0.7; mf = -0.2; sd = 0.5
    gj_n = float(genjax.normal.assess(genjax.ChoiceMapBuilder.v(xf), (mf, sd))[0])
    sp_n = float(norm.logpdf(xf, mf, sd))
    ok = abs(gj_mvn - sp_mvn) < 1e-3 and abs(gj_n - sp_n) < 1e-4
    print(f"[verify] mv_normal genjax={gj_mvn:+.5f} scipy={sp_mvn:+.5f} | "
          f"normal genjax={gj_n:+.5f} scipy={sp_n:+.5f} -> {'OK' if ok else 'MISMATCH'}",
          flush=True)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/streaming_general_premotion.yaml")
    ap.add_argument("--set", choices=["heldout", "dev", "demos"], default="heldout")
    ap.add_argument("--videos", nargs="+", default=None)
    ap.add_argument("--sweep", choices=["damping", "tau"], default="damping",
                    help="Which inference axis to sweep to induce GT variation. "
                         "'damping' (Phase-A feature_update_damping) | 'tau' (Phase-1 "
                         "final_feature_temp). For 'tau' pass --config the CURRENT SHIP.")
    ap.add_argument("--dampings", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--taus", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0],
                    help="final_feature_temp grid when --sweep tau (capped = position floor).")
    ap.add_argument("--drift-schedule", action="store_true", default=True,
                    help="apply init_gibbs_sweeps=1 + blob_means_updates=1 (the shipped anti-drift schedule)")
    args = ap.parse_args(argv)
    # The swept axis -> (cfg tracking key, value list). tau layers on the ship (which
    # already carries the drift schedule), so drift_schedule is only applied for damping.
    if args.sweep == "tau":
        sweep_key = "final_feature_temp"; sweep_vals = list(args.taus)
    else:
        sweep_key = "feature_update_damping"; sweep_vals = list(args.dampings)

    cc._ensure_jax_setup(); cc._GT_SCORING_ALLOWED = True
    if not _verify_math():
        print("[gate] FAIL: distribution-convention mismatch — fix _data_loglik first")
        return 2
    disc = cc.discover_videos()
    if args.videos:
        vids = [v for v in args.videos if v in disc]
    elif args.set == "heldout":
        vids = [v for v in cc.HELDOUT_VIDEOS if disc.get(v) and disc[v].kind == "davis"]
    elif args.set == "dev":
        vids = [v for v in cc.TRAIN_VIDEOS if disc.get(v) and disc[v].kind == "davis"]
    else:
        vids = [v for v in cc.DEMO_VIDEOS if disc.get(v) and disc[v].kind == "davis"]
    base = genmatter_rt.load_yaml_hypers(Path(args.config))

    # NAIVE live-feature variants (expected to ANTI-correlate — reward drift) +
    # ANCHOR-referenced variants (the fix — expected to TRACK GT).
    VARIANTS = ["full", "pos_feat", "feat", "pos", "vel", "mix",
                "feat_anchor", "anchor_pos_feat", "anchor_full",
                # Phase-3 instance-purity (present only when the tracker captured Z_sam):
                "inst_purity", "anchor_pos_feat_purity"]
    # grid rows: each (vid, damping) -> {variant: loglik, "gt": region_J}
    rows = []
    per_vid = {}   # vid -> {sweep_val: {variant:.., gt:..}}
    print(f"[loglik_proto] set={args.set} ({len(vids)} vids) sweep={args.sweep} "
          f"vals={sweep_vals} drift_schedule={args.drift_schedule and args.sweep=='damping'}", flush=True)
    hdr = f"  {'video':16s} {args.sweep[:5]:>5s}  " + " ".join(f"{v[:8]:>8s}" for v in VARIANTS) + f" {'GT':>6s}"
    print(hdr, flush=True)
    for vid in vids:
        npz = cc.LABELS_DIR / f"{vid}.npz"
        if not npz.is_file():
            continue
        with np.load(npz) as d:
            labels = {k: np.asarray(d[k]) for k in d.files}
        fi = np.asarray(labels["frame_idx"]).reshape(-1); idx = labels["indices"]
        z = labels["Z_sam"]; Tz = z.shape[0]
        gt_dp = [cc._gt_dp_at_datapoints(vid, int(fi[t]), idx) for t in range(Tz)]
        sam_grid = cc._frame0_sam_grid(disc[vid], labels)
        per_vid[vid] = {}
        for sval in sweep_vals:
            cfg = copy.deepcopy(base)
            cfg["tracking"][sweep_key] = float(sval)
            if args.drift_schedule and args.sweep == "damping":
                cfg["tracking"]["init_gibbs_sweeps"] = 1
                cfg["tracking"]["blob_means_updates_per_frame"] = 1
            tr = cc._run_tracker_on_video(labels, cfg, -1, num_sweeps=1,
                                          capture_blob_weights=False, sam_grid=sam_grid,
                                          capture_data_loglik=True)
            if "error" in tr:
                print(f"  {vid:16s} {sval:5.2f}  ERR {tr['error'][:40]}"); continue
            hb = tr["hyperblob_a"]; Tc = min(hb.shape[0], Tz)
            gt_j = _region_j(hb, lambda t: gt_dp[t], Tc)
            terms = tr.get("data_loglik_terms", {})
            row = {v: float(terms.get(v, float("nan"))) for v in VARIANTS}
            row["gt"] = gt_j
            rows.append(row); per_vid[vid][sval] = row
            cells = " ".join(f"{row[v]:8.2f}" for v in VARIANTS)
            print(f"  {vid:16s} {sval:5.2f}  {cells} {gt_j:6.3f}", flush=True)

    from scipy.stats import pearsonr, spearmanr
    g = np.asarray([r["gt"] for r in rows])
    okg = np.isfinite(g)
    print("\n=== POOLED corr(variant, GT) over all (video x damping) ===", flush=True)
    pooled = {}
    for v in VARIANTS:
        x = np.asarray([r[v] for r in rows]); m = okg & np.isfinite(x)
        if m.sum() >= 3:
            pr = pearsonr(x[m], g[m])[0]; sr = spearmanr(x[m], g[m])[0]
            pooled[v] = sr
            print(f"  {v:10s} pearson={pr:+.3f} spearman={sr:+.3f}", flush=True)

    print("\n=== WITHIN-VIDEO mean spearman(variant, GT) across damping ===", flush=True)
    within = {}
    for v in VARIANTS:
        srs = []
        for vid, dd in per_vid.items():
            xs = np.asarray([dd[k][v] for k in sorted(dd)])
            gs = np.asarray([dd[k]["gt"] for k in sorted(dd)])
            m = np.isfinite(xs) & np.isfinite(gs)
            if m.sum() >= 3 and np.unique(gs[m]).size > 1 and np.unique(xs[m]).size > 1:
                srs.append(spearmanr(xs[m], gs[m])[0])
        if srs:
            within[v] = float(np.nanmean(srs))
            print(f"  {v:10s} mean_within_spearman={np.nanmean(srs):+.3f} (n={len(srs)})", flush=True)

    print(f"\n=== SELECTION SIM: argmax-{args.sweep} by median(variant) vs median(GT) ===", flush=True)
    damps = sorted({d for dd in per_vid.values() for d in dd})
    med_gt = {d: np.nanmedian([per_vid[v][d]["gt"] for v in per_vid if d in per_vid[v]]) for d in damps}
    gt_best = max(med_gt, key=med_gt.get)
    print(f"  median GT per {args.sweep}: " + " ".join(f"{d:.2f}:{med_gt[d]:.3f}" for d in damps)
          + f"   -> GT-best {args.sweep} = {gt_best:.2f}", flush=True)
    agree = {}
    for v in VARIANTS:
        med_v = {d: np.nanmedian([per_vid[vid][d][v] for vid in per_vid if d in per_vid[vid]]) for d in damps}
        v_best = max(med_v, key=med_v.get)
        agree[v] = (v_best == gt_best)
        print(f"  {v:10s} per-damping median: " + " ".join(f"{d:.2f}:{med_v[d]:7.2f}" for d in damps)
              + f"  -> argmax={v_best:.2f} {'== GT' if v_best==gt_best else '!= GT('+format(gt_best,'.2f')+')'}",
              flush=True)

    # gate verdict
    best_var = max(within, key=within.get) if within else None
    print("\n=== GATE VERDICT ===", flush=True)
    if best_var is None:
        print("  FAIL: no usable variant"); return 1
    ok = (within[best_var] > 0) and (pooled.get(best_var, -1) > 0)
    strong = within[best_var] >= 0.3 and agree.get(best_var, False)
    print(f"  best variant by within-video spearman = '{best_var}' "
          f"(within={within[best_var]:+.3f}, pooled={pooled.get(best_var, float('nan')):+.3f}, "
          f"argmax{'==' if agree.get(best_var) else '!='}GT)", flush=True)
    print(f"  PASS(>0)={ok}  STRONG(>=0.3 & argmax==GT)={strong}", flush=True)
    print(f"  -> recommended objective variant: {best_var}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

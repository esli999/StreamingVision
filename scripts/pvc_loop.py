#!/usr/bin/env python
"""Propose -> Validate -> Check loop for the StreamingVision tracker.

Motivation (user, 2026-05-30): naive DINO-feature clustering sometimes BEATS the
hierarchical Bayesian tracker -> something simple is wrong, not a fundamental
ceiling. This harness lets us (a) measure naive feature baselines as a CEILING the
tracker should match, (b) sweep tracker structural variants ("proposals"), (c)
validate each on a DEV split and confirm winners ONCE on held-out — all from the
cached perception .npz (fast, no depth/flow/DINO recompute).

Discipline: DEV = TRAIN DAVIS, HELDOUT = held-out DAVIS (disjoint, from
calibrate_consistency's splits). GT is used to SCORE (region-J) — this is debugging
to find a STRUCTURAL bug, not continuous-knob tuning. A fix only ships if it is a
principled structural correction that ALSO passes on held-out. Every run is logged
to runs/calibrate_consistency/pvc_ledger.json (the propose->validate->check trail).

Candidates:
  baseline:dino_nc[:seed]   per-frame nearest-class-centroid on DINO features
                            (seed=gt|sam; centroids frozen at frame 0). The CEILING.
  baseline:dino_nc_ema      same but centroids EMA-updated from the prediction.
  baseline:dino_kmeans      per-frame 2-means on features, pick cluster ~ frame-0 seed.
  tracker:<name>            the Gibbs tracker with a named config-override (see PROPOSALS).

Usage:
  python scripts/pvc_loop.py --set dev                         # all candidates on DEV
  python scripts/pvc_loop.py --set dev --candidates baseline:dino_nc:gt tracker:premotion
  python scripts/pvc_loop.py --set heldout --candidates tracker:premotion tracker:<winner>
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "genmatterpp"))
import calibrate_consistency as cc  # noqa: E402
import genmatter_rt  # noqa: E402

LEDGER = _REPO / "runs/calibrate_consistency/pvc_ledger.json"
BASE_CFG_PATH = _REPO / "configs/streaming_general_premotion.yaml"


# ---- proposals: named structural/config overrides applied on top of premotion ----
# Each maps tracking.* keys. These are STRUCTURAL hypotheses about the bug, not
# continuous knob fits. Keep names descriptive; add freely as the loop iterates.
def _proposals():
    return {
        "premotion": {},  # the shipped baseline
        # --- warmup / refinement depth (seed-purity hypotheses) ---
        "warmup1": {"init_gibbs_sweeps": 1},
        "blobupd1": {"blob_means_updates_per_frame": 1},
        "blobupd5": {"blob_means_updates_per_frame": 5},
        "warmup1_blobupd1": {"init_gibbs_sweeps": 1, "blob_means_updates_per_frame": 1},
        # --- hyperblob aggregation (does the cluster layer discard feature info?) ---
        "hb8": {"num_hyperblobs": 8},
        "hb16": {"num_hyperblobs": 16},
        "nofreeze_hb": {"freeze_hyperblob_assignment": False},
        # --- final assignment ---
        "no_feat_final": {"feature_aware_final_assignment": False},
        "final_outlier": {"final_assignment_outlier": True},
        # --- ANTI-DRIFT: freeze the frame-0 appearance (the bug fix) ---
        "freeze_feat": {"freeze_blob_features": True},
        "freeze_feat_warmup1": {"freeze_blob_features": True, "init_gibbs_sweeps": 1},
        "freeze_feat_blobupd1": {"freeze_blob_features": True, "blob_means_updates_per_frame": 1},
        "freeze_all_drift": {"freeze_blob_features": True, "init_gibbs_sweeps": 1,
                             "blob_means_updates_per_frame": 1},
        # --- DAMPED anti-drift (the generalization being calibrated): blend each
        # per-frame Gibbs feature update toward the frame-0 anchor by (1 - d).
        # d=0 == freeze (via blend), d=1 == vendored. The "_drift" combos add the
        # warmup1 + blob_means_updates=1 schedule that freeze_all_drift ships. ---
        "damp0.0": {"feature_update_damping": 0.0},
        "damp0.25": {"feature_update_damping": 0.25},
        "damp0.5": {"feature_update_damping": 0.5},
        "damp0.75": {"feature_update_damping": 0.75},
        "damp1.0": {"feature_update_damping": 1.0},
        "damp0.0_drift": {"feature_update_damping": 0.0, "init_gibbs_sweeps": 1,
                          "blob_means_updates_per_frame": 1},
        "damp0.25_drift": {"feature_update_damping": 0.25, "init_gibbs_sweeps": 1,
                           "blob_means_updates_per_frame": 1},
        "damp0.5_drift": {"feature_update_damping": 0.5, "init_gibbs_sweeps": 1,
                          "blob_means_updates_per_frame": 1},
        "damp0.75_drift": {"feature_update_damping": 0.75, "init_gibbs_sweeps": 1,
                           "blob_means_updates_per_frame": 1},
        # --- INFERENCE: feature-TEMPERATURE on the FINAL assignment (Phase 1). tau>1
        # up-weights the DINO features at the step that WRITES the labels the scorer
        # reads, pushing the model tracker toward the frozen-centroid DINO classifier
        # (~0.71 region-J) that beats it from the same seed. tau=1 reproduces the
        # vendored feature-aware-final bit-for-bit (re-impl verification). Layer these
        # on the CURRENT SHIP via `--base configs/streaming_general.yaml`. ---
        # tau<1 DOWN-weights features (~ a larger effective feature sigma -> more
        # appearance TOLERANCE), the proxy for "deforming objects need to tolerate
        # within-object appearance variance" (rigid-vs-deforming diagnosis).
        "tau0.25": {"final_feature_temp": 0.25},
        "tau0.5": {"final_feature_temp": 0.5},
        "tau0.75": {"final_feature_temp": 0.75},
        "tau1": {"final_feature_temp": 1.0},
        "tau2": {"final_feature_temp": 2.0},
        "tau4": {"final_feature_temp": 4.0},
        "tau8": {"final_feature_temp": 8.0},
        "tau16": {"final_feature_temp": 16.0},
        "feat_only": {"final_feature_temp": 1000.0},
    }


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


def _seg_iou(pred, g):
    u = int(np.logical_or(pred, g).sum())
    return int(np.logical_and(pred, g).sum()) / u if u > 0 else 0.0


def _load(vid):
    npz = cc.LABELS_DIR / f"{vid}.npz"
    with np.load(npz) as d:
        labels = {k: np.asarray(d[k]) for k in d.files}
    fi = np.asarray(labels["frame_idx"]).reshape(-1)
    idx = labels["indices"]
    gt = [cc._gt_dp_at_datapoints(vid, int(fi[t]), idx) for t in range(fi.shape[0])]
    return labels, gt


def _seed0(labels, gt, seed):
    """Frame-0 foreground at datapoints from the chosen seed source."""
    if seed == "gt":
        return None if gt[0] is None else np.asarray(gt[0]).reshape(-1)
    return (labels["Z_sam"][0] > 0)  # sam


def score_baseline(vid, kind="dino_nc", seed="gt"):
    labels, gt = _load(vid)
    feat = labels["features"]  # (T, N, D)
    T = feat.shape[0]
    s0 = _seed0(labels, gt, seed)
    if s0 is None or s0.sum() < 3 or (~s0).sum() < 3:
        return {"region_J": float("nan")}
    Tc = min(T, len(gt))
    if kind in ("dino_nc", "dino_nc_ema"):
        o = feat[0][s0].mean(0); b = feat[0][~s0].mean(0)
        Js = []
        for t in range(Tc):
            if gt[t] is None:
                continue
            do = ((feat[t] - o) ** 2).sum(1); db = ((feat[t] - b) ** 2).sum(1)
            pred = do < db
            Js.append(_seg_iou(pred, np.asarray(gt[t]).reshape(-1)))
            if kind == "dino_nc_ema" and pred.sum() > 2 and (~pred).sum() > 2:
                o = 0.7 * o + 0.3 * feat[t][pred].mean(0)
                b = 0.7 * b + 0.3 * feat[t][~pred].mean(0)
        return {"region_J": float(np.mean(Js)) if Js else float("nan")}
    if kind == "dino_kmeans":
        from sklearn.cluster import KMeans
        Js = []
        for t in range(Tc):
            if gt[t] is None:
                continue
            lab = KMeans(2, n_init=3, random_state=0).fit_predict(feat[t])
            # pick the cluster whose frame-0-seed overlap is larger
            c_for = 0 if (lab[s0] == 0).mean() >= (lab[s0] == 1).mean() else 1
            pred = (lab == c_for)
            Js.append(_seg_iou(pred, np.asarray(gt[t]).reshape(-1)))
        return {"region_J": float(np.mean(Js)) if Js else float("nan")}
    return {"region_J": float("nan")}


def _frame0_gt_grid(vid, labels):
    """GT frame-0 seed grid (DIAGNOSTIC: isolates seed-quality from inference).
    Same recipe as render_live_grid._seed_grid(is_gt=True): black bg -> white
    sentinel, object colors kept as instances, resized to (GRID_H, GRID_W)."""
    import cv2
    import config as gm_config
    frame0 = int(np.asarray(labels["frame_idx"]).reshape(-1)[0])
    p = Path(gm_config.DAVIS_SEGMASKS_PATH) / vid / f"{frame0:05d}.png"
    if not p.is_file():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if m is None:
        return None
    bg = np.all(m == 0, axis=2); m = m.copy(); m[bg] = (255, 255, 255)
    return cv2.resize(m, (cc.GRID_W, cc.GRID_H), interpolation=cv2.INTER_NEAREST)


def score_tracker(vid, overrides, reference="gt", seed="sam"):
    labels, gt = _load(vid)
    cfg = genmatter_rt.load_yaml_hypers(BASE_CFG_PATH)
    for k, v in overrides.items():
        cfg["tracking"][k] = v
    if seed == "gt":
        sam_grid = _frame0_gt_grid(vid, labels)
    else:
        sam_grid = cc._frame0_sam_grid(cc.discover_videos()[vid], labels)
    nsw = 1
    tr = cc._run_tracker_on_video(labels, cfg, -1, num_sweeps=nsw,
                                  capture_blob_weights=False, sam_grid=sam_grid, vid=vid)
    if "error" in tr:
        return {"region_J": float("nan"), "error": tr["error"][:60]}
    hb = tr["hyperblob_a"]; idx = labels["indices"]; z = labels["Z_sam"]
    Tc = min(hb.shape[0], z.shape[0])
    if reference == "gt":
        fg = lambda t: (None if gt[t] is None else np.asarray(gt[t]).reshape(-1))
    else:
        fg = lambda t: (z[t] > 0)
    return {"region_J": _region_j(hb, fg, Tc)}


def _resolve(cand):
    """('baseline'|'tracker', spec) -> callable(vid)->dict."""
    if cand.startswith("baseline:"):
        parts = cand.split(":")
        kind = parts[1]; seed = parts[2] if len(parts) > 2 else "gt"
        return lambda vid: score_baseline(vid, kind=kind, seed=seed)
    if cand.startswith("tracker:"):
        parts = cand.split(":")
        name = parts[1]; seed = parts[2] if len(parts) > 2 else "sam"
        ov = _proposals().get(name)
        if ov is None:
            raise SystemExit(f"unknown tracker proposal '{name}'; have {list(_proposals())}")
        return lambda vid: score_tracker(vid, ov, seed=seed)
    raise SystemExit(f"bad candidate '{cand}' (use baseline:... or tracker:...)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=["dev", "heldout", "demos"], default="dev")
    ap.add_argument("--candidates", nargs="+", default=None,
                    help="default: dino_nc baselines + all tracker proposals")
    ap.add_argument("--videos", nargs="+", default=None, help="explicit video override")
    ap.add_argument("--base", default=None,
                    help="base config the proposals layer on (default: premotion). "
                         "Use configs/streaming_general.yaml to gate vs the CURRENT SHIP.")
    args = ap.parse_args(argv)

    if args.base:
        global BASE_CFG_PATH
        BASE_CFG_PATH = Path(args.base)
        print(f"[pvc] base config := {BASE_CFG_PATH}", flush=True)

    cc._ensure_jax_setup(); cc._GT_SCORING_ALLOWED = True
    disc = cc.discover_videos()
    if args.videos:
        vids = [v for v in args.videos if v in disc]
    elif args.set == "dev":
        vids = [v for v in cc.TRAIN_VIDEOS if disc.get(v) and disc[v].kind == "davis"]
    elif args.set == "heldout":
        vids = [v for v in cc.HELDOUT_VIDEOS if disc.get(v) and disc[v].kind == "davis"]
    else:
        vids = [v for v in cc.DEMO_VIDEOS if disc.get(v) and disc[v].kind == "davis"]

    cands = args.candidates or (
        ["baseline:dino_nc:gt", "baseline:dino_nc:sam", "baseline:dino_nc_ema:gt"]
        + [f"tracker:{n}" for n in _proposals()])
    print(f"[pvc] set={args.set} ({len(vids)} videos): {', '.join(vids)}", flush=True)
    print(f"[pvc] candidates: {', '.join(cands)}", flush=True)

    results = {}  # cand -> {vid: region_J}
    for cand in cands:
        fn = _resolve(cand); per = {}
        t0 = time.monotonic()
        for vid in vids:
            try:
                per[vid] = float(fn(vid).get("region_J", float("nan")))
            except Exception as e:
                per[vid] = float("nan"); print(f"   {cand} {vid}: ERR {e}", flush=True)
        results[cand] = per
        vals = [x for x in per.values() if np.isfinite(x)]
        med = float(np.median(vals)) if vals else float("nan")
        mean = float(np.mean(vals)) if vals else float("nan")
        print(f"  {cand:28s} median={med:.3f} mean={mean:.3f}  ({time.monotonic()-t0:.0f}s)", flush=True)

    # ranked summary
    print("\n=== RANKED (median GT region-J, {} set) ===".format(args.set), flush=True)
    ranked = sorted(results.items(),
                    key=lambda kv: -np.nanmedian([v for v in kv[1].values()] or [np.nan]))
    for cand, per in ranked:
        vals = [x for x in per.values() if np.isfinite(x)]
        print(f"  {cand:28s} {np.median(vals) if vals else float('nan'):.3f}", flush=True)

    # per-video table (tracker vs the gt baseline ceiling)
    base = results.get("baseline:dino_nc:gt")
    if base:
        print("\n=== per-video (vs dino_nc:gt ceiling) ===", flush=True)
        hdr = f"  {'video':16s} {'dino_nc:gt':>10s}"
        trk = [c for c in results if c.startswith("tracker:")]
        for c in trk:
            hdr += f" {c.split(':',1)[1][:9]:>10s}"
        print(hdr, flush=True)
        for vid in vids:
            row = f"  {vid:16s} {base.get(vid, float('nan')):10.3f}"
            for c in trk:
                row += f" {results[c].get(vid, float('nan')):10.3f}"
            print(row, flush=True)

    # append to ledger
    try:
        led = json.loads(LEDGER.read_text()) if LEDGER.is_file() else []
    except Exception:
        led = []
    led.append({"set": args.set, "videos": vids,
                "medians": {c: float(np.nanmedian([v for v in per.values()] or [np.nan]))
                            for c, per in results.items()},
                "per_video": results})
    LEDGER.write_text(json.dumps(led, indent=2))
    print(f"\n[pvc] appended to {LEDGER}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

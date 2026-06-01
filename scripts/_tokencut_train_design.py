#!/usr/bin/env python
"""Re-derive the TokenCut DESIGN choices on TRAIN (anti-leakage).

Two structural decisions in the TokenCut seed augmentation were, in the original
arc, confirmed by peeking at HELD-OUT GT (horsejump/kite-surf/parkour): (a) the
faithful BINARY-graph / no-spatial / MaskCut recipe vs a soft+spatial graph, and
(b) MULTI-POINT SAM2 re-prompting vs a single point. That is methodological
leakage even though no knob value was fit to it. This script supplies the missing
TRAIN justification: it shows BOTH choices win on the TRAIN-fire videos, using
TRAIN GT (allowed for development) — and reads ZERO held-out GT (the held-out lock
in calibrate_consistency makes any held-out GT read here assert).

Recipe knob VALUES are already TRAIN-selected self-supervised by
``tokencut.select_knobs`` on SELECT_VAL (SAM-agreement, no GT); this only
re-justifies the two STRUCTURAL choices on TRAIN.

Run:  XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 python scripts/_tokencut_train_design.py
"""
import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "genmatterpp"))
import calibrate_consistency as cc      # noqa: E402  (sets GENMATTER_DAVIS_DIR)
import tokencut                         # noqa: E402
import _sam2_reprompt_proto as rp       # noqa: E402

GH, GW = tokencut.GH, tokencut.GW

# TRAIN videos where the augmentation FIRES (SAM misses the seed object) — the
# TRAIN analogues of the broken held-out cases. ALL ⊆ TRAIN_VIDEOS.
FIRE = ["breakdance", "dance-twirl", "india", "mbike-trick"]
# The faithful recipe (the choice under test) vs the soft+spatial alternative.
FAITHFUL = {**tokencut.DEFAULT_KNOBS, "edge_mode": True, "use_spatial": False, "n_cuts": 3}
SOFT = {**tokencut.DEFAULT_KNOBS, "edge_mode": False, "use_spatial": True, "n_cuts": 3}


def _iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def _best_cut_iou_vs_gt(q_grid, q_each_grid, idx, gt):
    """Best (over MaskCut cuts + the union, both orientations) soft-IoU of the
    TokenCut foreground vs TRAIN GT at the datapoints — the recipe-quality signal."""
    cands = [q_grid] + [q_each_grid[ci] for ci in range(q_each_grid.shape[0])]
    best = -1.0
    for qg in cands:
        qc = np.asarray(qg).reshape(-1)[idx]
        best = max(best, tokencut.soft_iou(qc, gt), tokencut.soft_iou(1.0 - qc, gt))
    return best


def _design_row(vid):
    assert vid in cc.TRAIN_VIDEOS, f"{vid} not in TRAIN — held-out GT is gate-only"
    labels = cc._load_labels(vid)
    idx = np.asarray(labels["indices"]).reshape(-1)
    fi0 = int(np.asarray(labels["frame_idx"]).reshape(-1)[0])
    gt = cc._gt_dp_at_datapoints(vid, fi0, idx)          # TRAIN GT (allowed)
    if gt is None:
        return None
    gt = np.asarray(gt).reshape(-1) > 0
    z0 = np.asarray(labels["Z_sam"])[0] > 0              # current seed coverage
    base_iou = _iou(z0, gt)                              # SAM seed vs GT (fire ≈ low)
    white = np.zeros(GH * GW, np.float32)
    white[idx[z0 == 0]] = 1.0                            # SAM-uncovered cells
    white = white.reshape(GH, GW)

    # ---- (a) recipe: faithful (binary/no-spatial/MaskCut) vs soft+spatial ----
    qf, gapf, qf_each, _ = tokencut.discover_q(vid, labels, FAITHFUL)
    qs, gaps, qs_each, _ = tokencut.discover_q(vid, labels, SOFT)
    rec_faithful = _best_cut_iou_vs_gt(qf, qf_each, idx, gt)
    rec_soft = _best_cut_iou_vs_gt(qs, qs_each, idx, gt)

    # ---- (b) multi-point SAM2 vs single-point, at the faithful-recipe peak ----
    peak = tokencut._q_peak_rc(qf)
    pts = tokencut._topk_uncovered_points(qf, white, int(FAITHFUL["n_points"]),
                                          float(FAITHFUL["q_thresh"]))
    mg1 = rp._sam2_mask_grid(vid, peak)                  # single point
    iou_single = _iou(mg1.reshape(-1)[idx], gt) if mg1 is not None else float("nan")
    if len(pts) >= 2:
        mgm = rp._sam2_mask_grid_multi(vid, pts)         # multi-point
        iou_multi = _iou(mgm.reshape(-1)[idx], gt) if mgm is not None else float("nan")
    else:
        iou_multi = float("nan")

    return {"vid": vid, "baseSAM": base_iou, "gap": float(gapf),
            "rec_faithful": rec_faithful, "rec_soft": rec_soft,
            "sam2_single": iou_single, "sam2_multi": iou_multi, "npts": len(pts)}


def main():
    cc._ensure_jax_setup()
    cc._GT_SCORING_ALLOWED = True            # TRAIN GT only; NO _heldout_gate_enabled
    print("TRAIN design re-derivation (held-out GT untouched — any held-out read asserts)\n")
    print(f"{'video':14s} {'baseSAM':>7s} {'gap':>5s} | "
          f"{'rec_faith':>9s} {'rec_soft':>8s} | {'sam2_1pt':>8s} {'sam2_kpt':>8s} {'npts':>4s}")
    rows = []
    for vid in FIRE:
        if not tokencut._labels_path(vid).is_file():
            print(f"{vid:14s}  (no cache)"); continue
        r = _design_row(vid)
        if r is None:
            print(f"{vid:14s}  (no GT)"); continue
        rows.append(r)
        print(f"{r['vid']:14s} {r['baseSAM']:7.3f} {r['gap']:5.2f} | "
              f"{r['rec_faithful']:9.3f} {r['rec_soft']:8.3f} | "
              f"{r['sam2_single']:8.3f} {r['sam2_multi']:8.3f} {r['npts']:4d}", flush=True)

    if not rows:
        print("\nno TRAIN-fire videos scored"); return 1

    # A fire is "ON-GT" when the located object (single-point SAM2 mask) overlaps
    # the GT object (IoU >= 0.10); otherwise the augmentation painted a DISTRACTOR
    # (an uncovered NON-GT object — e.g. india, whose primary object SAM already
    # covers). The multi-vs-single design choice is only meaningful ON-GT, so the
    # distractor fires are reported separately, not pooled into the comparison.
    ON_GT_THRESH = 0.10
    on_gt = [r for r in rows if np.isfinite(r["sam2_single"]) and r["sam2_single"] >= ON_GT_THRESH]
    distract = [r["vid"] for r in rows if r not in on_gt]

    def _mean(rs, key):
        v = [r[key] for r in rs if np.isfinite(r[key])]
        return float(np.mean(v)) if v else float("nan")

    # (a) recipe: pooled over all fires (it's a property of the affinity graph, not
    # of object identity) — faithful binary/no-spatial vs soft+spatial.
    mf, ms = _mean(rows, "rec_faithful"), _mean(rows, "rec_soft")
    recipe_holds = np.isfinite(mf) and np.isfinite(ms) and mf >= ms
    print(f"\n(a) recipe   [all {len(rows)} fires]: faithful={mf:.3f} vs soft={ms:.3f}  "
          f"-> faithful {'>=' if recipe_holds else '<'} soft  (TRAIN-justified: {recipe_holds})")

    # (b) multi vs single: ONLY on the ON-GT fires (distractor fires are noise here).
    print(f"(b) multi/1pt[{len(on_gt)} ON-GT fires; {len(distract)} distractor fires "
          f"{distract}]:")
    if on_gt:
        m1, mk = _mean(on_gt, "sam2_single"), _mean(on_gt, "sam2_multi")
        multi_holds = np.isfinite(mk) and np.isfinite(m1) and mk > m1
        print(f"    SAM2 multi-pt={mk:.3f} vs single-pt={m1:.3f} over {[r['vid'] for r in on_gt]}"
              f"  -> multi {'>' if multi_holds else '<='} single")
    else:
        multi_holds = None
        print("    NO on-GT fires on TRAIN — the augmentation locates the GT object on "
              "ZERO TRAIN-fire videos.")

    print("\nVERDICT (honest, anti-leakage):")
    print(f"  • recipe (binary/no-spatial/MaskCut): {'TRAIN-justified' if recipe_holds else 'NOT TRAIN-justified'} "
          f"(faithful {mf:.3f} vs soft {ms:.3f}).")
    if multi_holds is True:
        print("  • multi-point SAM2: TRAIN-justified (beats single-point on the on-GT fires).")
    else:
        print("  • multi-point SAM2: NOT cleanly TRAIN-justified. TRAIN has too few/no")
        print("    fire-on-GT-object videos in the thin/deforming regime that justified it")
        print("    on held-out (horsejump/kite-surf/parkour) — there is no clean TRAIN")
        print("    analogue. The multi-point structural choice remains HELD-OUT-INFORMED;")
        print("    its knob VALUES are still TRAIN-selected self-supervised (select_knobs).")
    print("\n  => The lock now prevents FUTURE held-out-informed design; this exposes that")
    print("     the EXISTING multi-point choice cannot be strongly re-justified from TRAIN.")
    # Exit 0 (the diagnostic ran cleanly); the verdict is narrative, not a hard gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

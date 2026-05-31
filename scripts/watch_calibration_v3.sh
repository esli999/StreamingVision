#!/usr/bin/env bash
# Status snapshot for the run_calibration_v3.sh pipeline. Parses the resumable
# checkpoints (select_grid.json / selected_infer.json / em.json + em_iter_*.json /
# validate.json) and tails the run log. Run repeatedly, or `watch_calibration_v3.sh loop`
# to refresh every 30s.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN_DIR="runs/calibrate_consistency"
LOG="$RUN_DIR/run_v3.log"

snapshot() {
  echo "===== calibration-v3 status @ $(date +%H:%M:%S) ====="
  if pgrep -f "calibrate_consistency.py" >/dev/null 2>&1; then
    echo "process: RUNNING ($(pgrep -f calibrate_consistency.py | tr '\n' ' '))"
  else
    echo "process: not running"
  fi
  python - "$RUN_DIR" <<'PY'
import json, sys, glob, os
rd = sys.argv[1]
def load(p):
    try:
        with open(os.path.join(rd, p)) as f: return json.load(f)
    except Exception: return None
pf = load("preflight.json")
if pf:
    print(f"[preflight] cache TRAIN {pf.get('train_npz_present')}/{pf.get('train_npz_present',0)+len(pf.get('train_npz_missing',[]))} "
          f"HELD-OUT {pf.get('heldout_npz_present')}/{pf.get('heldout_npz_present',0)+len(pf.get('heldout_npz_missing',[]))} "
          f"| GPU free {pf.get('cuda_mem_free_gb',float('nan')):.1f}GB")
    if pf.get("train_npz_missing") or pf.get("heldout_npz_missing"):
        print(f"           MISSING train={pf.get('train_npz_missing')} held={pf.get('heldout_npz_missing')}")
def _g(c, key, fallback=None):
    v = c.get(key)
    return v if v is not None else c.get(fallback, float('nan'))
def _ci(c, i, d):
    cc = c.get("combo", [])
    return cc[i] if len(cc) > i else d
sg = load("select_grid.json")
if sg:
    combos = sg.get("combos", [])
    grid = sg.get("grid", {})
    total = 1
    for ax in ("num_blobs","init_gibbs_sweeps","num_gibbs_sweeps_per_frame","sigma_V","blob_means_updates"):
        total *= max(len(grid.get(ax,[])), 1)
    print(f"[select] {len(combos)}/{total or '?'} combos scored (objective={sg.get('objective_mode','composite')})")
    for c in sorted(combos, key=lambda c: -(_g(c,'select_val_score','select_val_region_J') or -9)):
        print(f"         nb{c['combo'][0]:>3} igs{c['combo'][1]:>2} sw{c['combo'][2]} "
              f"sv{float(_ci(c,3,0.1)):.0e} bmu{_ci(c,4,15)} -> "
              f"comp {_g(c,'select_val_score','select_val_region_J'):.4f} "
              f"(rJ {c.get('select_val_region_J',float('nan')):.4f} p95 {c.get('select_val_median_p95',float('nan')):.4f})")
sel = load("selected_infer.json")
if sel:
    print(f"[selected] num_blobs={sel.get('num_blobs')} init_gibbs_sweeps={sel.get('init_gibbs_sweeps')} "
          f"per_frame_sweeps={sel.get('num_gibbs_sweeps_per_frame')} "
          f"sigma_V_seed={sel.get('sigma_V_seed','?')} bmu={sel.get('blob_means_updates','?')} "
          f"(SELECT_VAL composite {_g(sel,'select_val_score','select_val_region_J'):.4f})")
em = load("em.json")
iters = sorted(glob.glob(os.path.join(rd, "em_iter_*.json")))
if em or iters:
    bc = (em or {}).get("best_composite"); bj = (em or {}).get("best_region_J")
    snap = (em or {}).get("best_snapshot", {})
    print(f"[em] {len(iters)} iter checkpoints; best composite="
          f"{bc if bc is None else round(bc,4)} (region-J {bj if bj is None else round(bj,4)}; "
          f"val={snap.get('val_composite')}) at iter {snap.get('iter')}; done={'best_hypers' in (em or {})}")
val = load("validate.json")
if val:
    cgt = val.get("composite_gt_corr", {})
    print(f"[validate] median Δ GT region-J = {val.get('median_delta_gt_J',float('nan')):+.4f} "
          f"(chosen {val.get('chosen_median_gt_J_davis',float('nan')):.4f} vs baseline "
          f"{val.get('baseline_median_gt_J_davis',float('nan')):.4f}) | GATE "
          f"{'PASS' if val.get('passed') else 'FAIL'} | corr(comp,GT) spearman="
          f"{cgt.get('spearman',float('nan')):.2f} (ok={val.get('composite_kill_switch_ok',True)}) "
          f"| baseline={val.get('baseline_source','?')}")
PY
  if [ -f "$LOG" ]; then
    echo "----- tail run_v3.log -----"
    tail -n 8 "$LOG"
  fi
  echo "==========================================="
}

if [ "${1:-}" = "loop" ]; then
  while true; do clear; snapshot; sleep 30; done
else
  snapshot
fi

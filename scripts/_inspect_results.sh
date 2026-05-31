#!/usr/bin/env bash
# Post-calibration inspection: print the held-out GT gain + the composite↔GT
# kill-switch, then crop the cluster/particle panels from the new demos for a
# visual before/after. Run after run_motion_calibration.sh completes.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN_DIR="runs/calibrate_consistency"
echo "=================== HELD-OUT GT GATE (vs the shipped 0.6514 baseline) ==================="
python - <<'PY'
import json
v=json.load(open("runs/calibrate_consistency/validate.json"))
print(f"  chosen held-out GT region-J = {v['chosen_median_gt_J_davis']:.4f}")
print(f"  baseline held-out GT region-J = {v['baseline_median_gt_J_davis']:.4f}")
print(f"  median Δ (gate metric)        = {v['median_delta_gt_J']:+.4f}  -> GATE {'PASS' if v['passed'] else 'FAIL'}")
print(f"  chosen max outlier p95 = {v['chosen_max_outlier_p95']:.4f} vs baseline {v['baseline_max_outlier_p95']:.4f} (p95_pass={v['p95_pass']})")
cg=v.get('composite_gt_corr',{}); rg=v.get('region_J_gt_corr',{})
print(f"  KILL-SWITCH corr(composite,GT) spearman={cg.get('spearman',float('nan')):.3f} pearson={cg.get('pearson',float('nan')):.3f} "
      f"vs binary region-J spearman={rg.get('spearman',float('nan')):.3f} (ok={v.get('composite_kill_switch_ok')})")
# per-video biggest movers
rows=[]
for vid,c in v["chosen_per_video"].items():
    if c.get("kind")!="davis": continue
    b=v["baseline_per_video"].get(vid,{})
    if "gt_J" in c and "gt_J" in b: rows.append((c["gt_J"]-b["gt_J"], vid, b["gt_J"], c["gt_J"]))
rows.sort(reverse=True)
print("  per-video held-out GT region-J (Δ, baseline -> chosen):")
for d,vid,bj,cj in rows: print(f"    {vid:18s} {d:+.3f}   {bj:.3f} -> {cj:.3f}")
PY
echo
echo "=================== EM provenance (motion group + sigma_V learned) ==================="
python - <<'PY'
import json
e=json.load(open("runs/calibrate_consistency/em.json"))
bh=e.get("best_hypers",{}); snap=e.get("best_snapshot",{})
print(f"  best iterate = {snap.get('iter')}  val_composite={snap.get('val_composite'):.4f} train={snap.get('train_composite'):.4f} region-J={snap.get('region_J'):.4f}")
print(f"  learned sigma_V = {bh.get('sigma_V'):.3e}  (seed 0.03; streaming_default keeps 1e14)")
macc=sum(1 for it in e.get("iters",[]) if it.get("group_accepts",{}).get("motion",{}).get("accepted"))
print(f"  motion group accepted in {macc}/{len(e.get('iters',[]))} iters")
PY

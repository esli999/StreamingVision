"""Tiny real-tracker smoke of the new phase_em integration (needs GPU + cache).

Runs 1 EM iter on a 4-video TRAIN subset (2 pooled incl. CROSSVAL for the
accept-test, 2 held as the val probe), few frames, to exercise: composite scoring
on REAL tracker output, train/val split, M-step pooling, the motion-group
accept-test, best-by-val, and the JSON logging — end to end.
"""
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "scripts", _REPO / "genmatterpp"):
    sys.path.insert(0, str(p))

import calibrate_consistency as cc  # noqa: E402

# pooled (incl. CROSSVAL members so the accept-test actually runs) + val probe
TRAIN = ["camel", "drift-straight", "bmx-trees", "goat"]
VAL = ["bmx-trees", "goat"]

out = Path(tempfile.mkdtemp(prefix="smoke_em_"))
print(f"[smoke] out={out}  TRAIN={TRAIN}  VAL={VAL}")
info = cc.phase_em(
    out, force=True, max_frames_per_video=6, max_iters=1,
    num_gibbs_sweeps_per_frame=2, compute_side_metrics=False,
    keep_ari_diagnostic=False, train_videos=TRAIN, em_val_videos=VAL,
    sigma_v_seed=0.1, use_sam_frame0=True, num_blobs=64, num_hyperblobs=4,
    init_gibbs_sweeps=1,
)
it0 = info["iters"][0]
print("\n[smoke] iter0 keys:", sorted(k for k in it0 if not k.startswith("per_vid")))
print(f"[smoke] train_composite = {it0.get('train_composite')}")
print(f"[smoke] val_composite   = {it0.get('val_composite')}")
print(f"[smoke] region_J (pool) = {it0.get('region_J')}")
print(f"[smoke] composite components (pool median) = {it0.get('composite_components_pool_median')}")
print(f"[smoke] group_accepts motion = {it0.get('group_accepts', {}).get('motion')}")
print(f"[smoke] pool_videos = {info.get('pool_videos')}")
print(f"[smoke] val_videos  = {info.get('val_videos')}")
print(f"[smoke] best_objective = {info.get('best_objective')}  best_composite = {info.get('best_composite')}")
print(f"[smoke] best_snapshot = {info.get('best_snapshot')}")

ok = (
    "best_hypers" in info
    and info.get("val_videos") == sorted(VAL)
    and set(info.get("pool_videos", [])) == {"camel", "drift-straight"}
    and it0.get("train_composite") is not None
    and "motion" in it0.get("group_accepts", {})
    and "sigma_V" in info["best_hypers"]
)
print("\n[smoke] RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

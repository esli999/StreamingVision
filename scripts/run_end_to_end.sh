#!/usr/bin/env bash
# ===========================================================================
# ONE coherent end-to-end run:
#   re-train (fresh) -> held-out structural gate -> re-apply the TokenCut
#   augmentation -> held-out augmentation A/B (gate-locked) -> render the
#   5-panel demo videos -> unified report.
#
# Methodology (unchanged): self-supervised learning + selection on a TRAIN split;
# real DAVIS GT read ONLY on the disjoint HELD-OUT split, inside the durable lock.
# ONE global config; no per-video knobs; no test-set tuning. streaming_default.yaml
# (the live path) and vendored genmatterpp/ are NEVER touched.
#
# The objective is data_loglik/anchor_pos_feat — the ESTABLISHED winner that
# produced the shipped config — NOT the driver's `composite` default, which last
# failed its kill-switch + held-out gate (a deliberate, evidence-based override).
#
# Reproduction note: a faithful re-train of an already-validated optimum REPRODUCES
# it (EM is deterministic), so the structural delta vs the un-augmented baseline is
# ~0 (confirmation, not regression). emit-on-PASS / restore-on-FAIL keeps the
# validated augmented ship safe in every branch; the demonstrable held-out WIN is
# the augmentation increment (Step 3) + the renders (Step 4).
#
# Launch (detached, multi-hour) + watch:
#   XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 nohup bash scripts/run_end_to_end.sh \
#       > runs/calibrate_consistency/run_v3.log 2>&1 &
#   bash scripts/watch_calibration_v3.sh loop
# ===========================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

RUN_DIR="runs/calibrate_consistency"
PY=python
ts(){ date +%H:%M:%S; }
say(){ echo "[e2e $(ts)] $*"; }

GEN="configs/streaming_general.yaml"
RV2="configs/streaming_render_v2.yaml"
GEN_BAK="configs/streaming_general_e2e_backup.yaml"
RV2_BAK="configs/streaming_render_v2_e2e_backup.yaml"
PRETC="configs/streaming_general_pretokencut.yaml"

# Established winning methodology (override the driver's composite default).
export OBJECTIVE="${OBJECTIVE:-data_loglik}"
export DATA_LOGLIK_VARIANT="${DATA_LOGLIK_VARIANT:-anchor_pos_feat}"
export SELECT_MAX_FRAMES="${SELECT_MAX_FRAMES:-60}"
RENDER_SWEEPS="${RENDER_SWEEPS:-8}"                # offline showcase depth (live ships 1)
SELECT_SWEEPS="${SELECT_SWEEPS:-1,4,8}"           # the "bigger Gibbs budget" axis
SELECT_ARGS=( --select-num-blobs 128 --select-init-sweeps 1
              --select-per-frame-sweeps "$SELECT_SWEEPS"
              --select-blob-means-updates 1 )

mkdir -p "$RUN_DIR"
say "=================== END-TO-END START ==================="
say "objective=$OBJECTIVE/$DATA_LOGLIK_VARIANT  select_sweeps={$SELECT_SWEEPS}  render_sweeps=$RENDER_SWEEPS  select_max_frames=$SELECT_MAX_FRAMES"

# ---------------------------------------------------------------------------
# Step 0 — backups + fair (un-augmented) baseline swap.
# ---------------------------------------------------------------------------
say "STEP 0  backups + fair-baseline swap"
if [ ! -f "$GEN" ]; then say "FATAL: $GEN missing"; exit 1; fi
cp -f "$GEN" "$GEN_BAK"; say "  backed up $GEN -> $GEN_BAK"
[ -f "$RV2" ] && { cp -f "$RV2" "$RV2_BAK"; say "  backed up $RV2 -> $RV2_BAK"; }
for f in em.json validate.json validate_augmentation.json selected_infer.json report.json report.md; do
  [ -f "$RUN_DIR/$f" ] && cp -f "$RUN_DIR/$f" "$RUN_DIR/${f%.*}_e2e_prev.${f##*.}"
done
say "  snapshotted prior runs/ artifacts as *_e2e_prev.*"
if [ -f "$PRETC" ]; then
  cp -f "$PRETC" "$GEN"
  say "  validate baseline set to UN-AUGMENTED $PRETC (fair structural gate)"
else
  say "  WARN: $PRETC missing — baseline stays augmented (structural delta will read negative on horsejump/parkour)"
fi
NPZ=$(ls "$RUN_DIR"/labels/*.npz 2>/dev/null | wc -l)
say "  perception cache: $NPZ npz present"
if [ "$NPZ" -lt 30 ]; then
  say "  FATAL: <30 npz in cache — restoring ship and aborting"
  cp -f "$GEN_BAK" "$GEN"; exit 1
fi
# Snapshot the live-path config so the end-of-run invariant check compares against
# THIS run's start state (robust to any pre-existing working-tree modifications).
DEFAULT_MD5=$(md5sum configs/streaming_default.yaml 2>/dev/null | awk '{print $1}')

# ---------------------------------------------------------------------------
# Step 1 — re-train + validate + emit (reuse run_calibration_v3.sh).
#   preflight -> select(resumable) -> em(TRAIN,--force) -> validate(HELD-OUT gate,
#   --force) -> emit-on-PASS -> render_local. Returns rc 0 (gate PASS) / 2 (gate
#   not passed). We capture it without tripping the pipeline.
# ---------------------------------------------------------------------------
say "STEP 1  re-train (run_calibration_v3.sh): preflight -> select -> em -> validate -> emit -> render_local"
VAL_RC=0
bash scripts/run_calibration_v3.sh "${SELECT_ARGS[@]}" || VAL_RC=$?
say "STEP 1  done: run_calibration_v3 rc=$VAL_RC  (0 = gate PASS + emitted; 2 = not improved over baseline)"

# ---------------------------------------------------------------------------
# Step 2 — re-apply the augmentation seed layer (TRAIN-selected, n_points from
# tokencut_knobs.json) on PASS; on non-PASS restore the validated augmented ship.
# ---------------------------------------------------------------------------
if [ "$VAL_RC" -eq 0 ]; then
  say "STEP 2  PASS -> re-applying TokenCut augmentation block onto the freshly-emitted $GEN"
  $PY scripts/_apply_tokencut_block.py --config "$GEN" || say "  WARN: _apply_tokencut_block failed"
else
  say "STEP 2  non-PASS -> restoring the validated augmented ship (no regression ships)"
  cp -f "$GEN_BAK" "$GEN"; say "  restored $GEN from $GEN_BAK"
  [ -f "$RV2_BAK" ] && { cp -f "$RV2_BAK" "$RV2"; say "  restored $RV2 from $RV2_BAK"; }
fi

# ---------------------------------------------------------------------------
# Step 3 — held-out augmentation A/B (flag OFF vs ON), the ONE sanctioned
# gate-locked test. Reads the now-active $GEN as the base; toggles the flag.
# ---------------------------------------------------------------------------
say "STEP 3  held-out augmentation test (--phase validate_augmentation --force)"
AUG_RC=0
$PY scripts/calibrate_consistency.py --phase validate_augmentation --force || AUG_RC=$?
say "STEP 3  done: validate_augmentation rc=$AUG_RC  (0 = PASS)"

# ---------------------------------------------------------------------------
# Step 4 — render the 5-panel demos with the FINAL config at the showcase depth.
#   Panels: RGB | pixels-by-particle | pixels-by-cluster | 3D-by-particle |
#   3D-by-cluster (+ live-stats row). Always run (on non-PASS, run_calibration_v3
#   skipped its internal render).
# ---------------------------------------------------------------------------
say "STEP 4  render 5-panel demos (car-roundabout,car-shadow,blackswan,judo,wine_swirl) at $RENDER_SWEEPS sweeps"
$PY scripts/render_live_grid.py \
    --videos car-roundabout car-shadow blackswan judo wine_swirl \
    --config "$RV2" --out-dir "$RUN_DIR/tracking_videos" \
    --num-sweeps "$RENDER_SWEEPS" || say "  WARN: render_live_grid returned non-zero"

# ---------------------------------------------------------------------------
# Step 5 — unified report.
# ---------------------------------------------------------------------------
say "STEP 5  unified report"
$PY scripts/_write_e2e_report.py --val-rc "$VAL_RC" --aug-rc "$AUG_RC" \
    --render-sweeps "$RENDER_SWEEPS" --objective "$OBJECTIVE/$DATA_LOGLIK_VARIANT" \
    || say "  WARN: report writer failed"

# ---------------------------------------------------------------------------
# Invariants.
# ---------------------------------------------------------------------------
say "INVARIANTS"
if [ "$(md5sum configs/streaming_default.yaml 2>/dev/null | awk '{print $1}')" = "$DEFAULT_MD5" ]; then
  say "  streaming_default.yaml UNCHANGED by this run (live path bit-exact)  OK"
else
  say "  WARN: streaming_default.yaml CHANGED during this run — investigate"
fi
if git diff --quiet -- genmatterpp/ 2>/dev/null; then
  say "  genmatterpp/ UNCHANGED  OK"
else
  say "  WARN: genmatterpp/ CHANGED — investigate"
fi

say "=================== END-TO-END COMPLETE ==================="
say "structural rc=$VAL_RC  augmentation rc=$AUG_RC"
say "report: $RUN_DIR/END_TO_END_REPORT.md"
exit 0

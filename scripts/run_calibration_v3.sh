#!/usr/bin/env bash
# Self-supervised, structure-aware calibration run (the methodology-correct v3,
# now with the LEARNABLE MOTION MODEL + COMPOSITE objective + generalization loop):
#
#   preflight  (env/GPU/disk + TRAIN∪HELD-OUT perception-cache inventory)
#   -> [regen] (OPTIONAL, REGEN_PSEUDO_LABELS=1: higher-quality SAM2 + cheap Z_sam
#               refresh; the prior masks + .npz are backed up first — reversible)
#   -> select  (grid the GLOBAL scalar tuple num_blobs / init_gibbs_sweeps /
#               per-frame sweeps / sigma_V seed / blob_means_updates; short EM on
#               TRAIN; score median self-supervised COMPOSITE on SELECT_VAL ⊆ TRAIN.
#               NO ground truth, NO held-out.)
#   -> em      (full EM on TRAIN with the selected tuple, structural flags ON,
#               sigma_V finite = LIVE velocity anchor; learns the conjugate hypers
#               incl. the motion group; best iterate by val_composite on SELECT_VAL)
#   -> validate(the ONE held-out generalization gate vs the shipping config; GT
#               read ONLY here; reports corr(composite, GT) kill-switch; emits
#               streaming_general.yaml + streaming_render_v2.yaml + report.md on a
#               passing gate, else *_CANDIDATE.yaml + WARN)
#   -> render  (local tracking mp4s + the 5 demo grids, if the gate passed)
#
# ONE global config, no per-video knobs, no test-set tuning. By default reuses the
# EXISTING perception cache (runs/.../labels/*.npz; hyper-invariant). `select` is
# per-combo checkpointed and RESUMABLE; `em`/`validate` are --force'd.
#
# Env knobs:
#   REGEN_PSEUDO_LABELS=1   regenerate higher-quality SAM2 masks + refresh Z_sam
#                           (the user-confirmed scope; ~50min SAM + a few min; the
#                           prior masks AND labels/ are backed up first so it is
#                           fully reversible). Default 0 (reuse the known-good cache).
#   SELECT_SIGMA_V=1e14,0.1 add the velocity-anchor-OFF (1e14) fallback to the
#                           select grid (doubles select cost). Default = the finite
#                           seed only (anchor on; the motion-group M-step still
#                           learns it, the accept-test guards it).
#   OBJECTIVE=region_J      kill-switch: revert the EM objective to binary region-J
#                           (do this if validate reports corr(composite, GT) <= 0).
#   SELECT_MAX_FRAMES=60    frame cap for the select grid (compute budget, not tuning).
#
# Args after the script name pass through to every phase (e.g. --select-em-iters 2,
# --max-frames-per-video 60). Foreground; the caller backgrounds + tees the log
# (see the launch one-liner at the bottom of this file).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

RUN_DIR="runs/calibrate_consistency"
PY=python
ts() { date +%H:%M:%S; }

# Thread the new global knobs into every phase (argparse ignores irrelevant ones).
SELECT_SIGMA_V="${SELECT_SIGMA_V:-}"
OBJECTIVE="${OBJECTIVE:-composite}"
EXTRA_ARGS=( --objective "$OBJECTIVE" )
[ -n "$SELECT_SIGMA_V" ] && EXTRA_ARGS+=( --select-sigma-v "$SELECT_SIGMA_V" )

mkdir -p "$RUN_DIR"

# --- Preserve prior top-level artifacts (idempotent; cp -n never clobbers a
#     backup). The stale em.json/validate.json are moved aside so the fresh
#     structural run recomputes them; the per-combo select/ caches are kept for
#     resumability. ---
echo "[run_v3 $(ts)] backing up prior artifacts"
for f in em.json validate.json report.md report.json; do
  [ -f "$RUN_DIR/$f" ] || continue
  cp -n "$RUN_DIR/$f" "$RUN_DIR/${f%.*}_jmean_v1.${f##*.}" 2>/dev/null || true
  cp -f "$RUN_DIR/$f" "$RUN_DIR/${f%.*}_structv2_prev.${f##*.}" 2>/dev/null || true
done

echo "[run_v3 $(ts)] PHASE preflight"
$PY scripts/calibrate_consistency.py --phase preflight "${EXTRA_ARGS[@]}" "$@" || {
  echo "[run_v3 $(ts)] preflight FAILED — aborting"; exit 1; }

# --- OPTIONAL Phase E: regenerate higher-quality SAM2 pseudo-labels, then the
#     cheap Z_sam/Z_dino refresh (no perception recompute). Gated + reversible:
#     the prior PNG masks (--backup) AND the labels/ caches are checkpointed first. ---
if [ "${REGEN_PSEUDO_LABELS:-0}" = "1" ]; then
  echo "[run_v3 $(ts)] PHASE regen-pseudo-labels (higher-quality SAM2 + Z_sam refresh)"
  if [ -d "$RUN_DIR/labels" ] && [ ! -d "$RUN_DIR/labels_prev_backup" ]; then
    cp -r "$RUN_DIR/labels" "$RUN_DIR/labels_prev_backup"
    echo "[run_v3 $(ts)] backed up labels/ → labels_prev_backup/ (revert point)"
  fi
  $PY scripts/sam2_davis_propagate.py --backup --force \
      ${SAM2_REGEN_ARGS:-} || {
    echo "[run_v3 $(ts)] sam2 propagate FAILED — aborting (cache + masks intact via backup)"; exit 1; }
  $PY scripts/calibrate_consistency.py --phase pseudo_labels --regen-z-sam-only || {
    echo "[run_v3 $(ts)] Z_sam refresh FAILED — restore from labels_prev_backup/ if needed"; exit 1; }
  echo "[run_v3 $(ts)] regen done; compare per-video select region-J vs the backup before committing"
else
  echo "[run_v3 $(ts)] skipping pseudo-label regen (REGEN_PSEUDO_LABELS!=1; reusing known-good cache)"
fi

echo "[run_v3 $(ts)] PHASE select  (self-supervised composite grid on TRAIN; resumable; args: $*)"
# Selection only needs to RANK combos, so cap frames here (≈60) to bound the
# grid × 2-iter cost — a compute budget, NOT test-set tuning or per-video.
# The final em + validate below run FULL-frame for the real numbers. A user
# --max-frames-per-video in "$@" overrides this (argparse takes the last value).
$PY scripts/calibrate_consistency.py --phase select --max-frames-per-video "${SELECT_MAX_FRAMES:-60}" \
    "${EXTRA_ARGS[@]}" "$@" || {
  echo "[run_v3 $(ts)] select FAILED — aborting before em"; exit 1; }
if [ -f "$RUN_DIR/selected_infer.json" ]; then
  echo "[run_v3 $(ts)] selected: $(cat "$RUN_DIR/selected_infer.json")"
fi

echo "[run_v3 $(ts)] PHASE em  (full EM on TRAIN: live motion model, composite obj, val loop, --force)"
$PY scripts/calibrate_consistency.py --phase em --force "${EXTRA_ARGS[@]}" "$@" || {
  echo "[run_v3 $(ts)] em FAILED — aborting before validate"; exit 1; }

echo "[run_v3 $(ts)] PHASE validate  (HELD-OUT GT gate + composite↔GT kill-switch, --force)"
# validate returns rc=2 when the held-out gate FAILS (not an error); capture it
# without tripping the pipeline so renders still run and we report the outcome.
VAL_RC=0
$PY scripts/calibrate_consistency.py --phase validate --force "${EXTRA_ARGS[@]}" "$@" || VAL_RC=$?
echo "[run_v3 $(ts)] validate rc=$VAL_RC (0=gate passed, 2=gate failed)"

echo "[run_v3 $(ts)] PHASE render_local"
$PY scripts/calibrate_consistency.py --phase render_local "$@" || true

# Demo grids (the 5 held-out demo videos) — only when the gate passed AND the
# emitted demo config exists. Non-fatal; this is the "much better visuals" output.
# Render with the SELECTED per-frame sweep count so the demo matches the
# validated config (the live real-time demo may still run 1; this is offline).
if [ "$VAL_RC" -eq 0 ] && [ -f "configs/streaming_render_v2.yaml" ]; then
  SEL_SWEEPS=$($PY -c "import json;print(json.load(open('$RUN_DIR/selected_infer.json')).get('num_gibbs_sweeps_per_frame',1))" 2>/dev/null || echo 1)
  echo "[run_v3 $(ts)] PHASE render_demos (car-roundabout,car-shadow,blackswan,judo,wine_swirl; sweeps=$SEL_SWEEPS)"
  $PY scripts/render_live_grid.py \
      --videos car-roundabout car-shadow blackswan judo wine_swirl \
      --config configs/streaming_render_v2.yaml \
      --out-dir "$RUN_DIR/tracking_videos" --num-sweeps "$SEL_SWEEPS" || true
else
  echo "[run_v3 $(ts)] skipping demo grids (gate not passed or demo config missing)"
fi

echo "[run_v3 $(ts)] DONE (validate rc=$VAL_RC). Report: $RUN_DIR/report.md"
exit "$VAL_RC"

# ---------------------------------------------------------------------------
# Background launch + tee (run from repo root):
#   nohup bash scripts/run_calibration_v3.sh > runs/calibrate_consistency/run_v3.log 2>&1 &
# Watch progress:
#   bash scripts/watch_calibration_v3.sh
# ---------------------------------------------------------------------------

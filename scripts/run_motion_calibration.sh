#!/usr/bin/env bash
# Lean overnight run of the NEW calibration levers (learnable motion model +
# composite objective + generalization val loop) on the ALREADY-SELECTED global
# structure (runs/.../selected_infer.json: 128 blobs / igs5 / sw1). Skips the
# expensive select grid re-search — the structure is fixed and good; this run
# RE-LEARNS the conjugate hypers (incl. the motion group / sigma_V) under the
# composite objective, then gates ONCE on held-out GT vs the currently-shipping
# config (whose sigma_V=1e14 IS the "velocity anchor OFF" baseline), emits the
# improved streaming_general.yaml + streaming_render_v2.yaml on a pass, and
# re-renders the 5 demo grids (12 s slow-mo loops).
#
# Env knobs: OBJECTIVE=region_J (kill-switch), MAX_ITERS (default 8),
#            SIGMA_V_SEED (override the EM sigma_V seed), TARGET_DUR (render loop s).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
RUN_DIR="runs/calibrate_consistency"; PY=python
ts() { date +%H:%M:%S; }
mkdir -p "$RUN_DIR"

EXTRA=( --objective "${OBJECTIVE:-composite}" )
[ -n "${SIGMA_V_SEED:-}" ] && EXTRA+=( --sigma-v-seed "$SIGMA_V_SEED" )

# Checkpoint the current shipped configs (revert point) + prior em/validate.
for f in configs/streaming_general.yaml configs/streaming_render_v2.yaml; do
  cp -n "$f" "${f%.yaml}_premotion.yaml" 2>/dev/null || true
done
for f in em.json validate.json report.md; do
  [ -f "$RUN_DIR/$f" ] && cp -f "$RUN_DIR/$f" "$RUN_DIR/${f%.*}_premotion.${f##*.}" 2>/dev/null || true
done

echo "[motion $(ts)] PHASE em (live motion model + composite + val loop, --force, max-iters=${MAX_ITERS:-8})"
$PY scripts/calibrate_consistency.py --phase em --force --max-iters "${MAX_ITERS:-8}" \
    "${EXTRA[@]}" || { echo "[motion $(ts)] em FAILED"; exit 1; }

echo "[motion $(ts)] PHASE validate (held-out GT gate + composite↔GT kill-switch, --force)"
VAL_RC=0
$PY scripts/calibrate_consistency.py --phase validate --force "${EXTRA[@]}" || VAL_RC=$?
echo "[motion $(ts)] validate rc=$VAL_RC (0=gate passed, 2=gate failed)"

if [ "$VAL_RC" -eq 0 ] && [ -f configs/streaming_render_v2.yaml ]; then
  SEL_SWEEPS=$($PY -c "import json;print(json.load(open('$RUN_DIR/selected_infer.json')).get('num_gibbs_sweeps_per_frame',1))" 2>/dev/null || echo 1)
  echo "[motion $(ts)] PHASE render_demos (5 demos, ${TARGET_DUR:-12}s loops, sweeps=$SEL_SWEEPS)"
  $PY scripts/render_live_grid.py \
      --videos car-roundabout car-shadow blackswan judo wine_swirl \
      --config configs/streaming_render_v2.yaml --out-dir "$RUN_DIR/tracking_videos" \
      --num-sweeps "$SEL_SWEEPS" --target-duration "${TARGET_DUR:-12}" || true
else
  echo "[motion $(ts)] gate not passed — known-good configs kept; review $RUN_DIR/report.md"
fi
echo "[motion $(ts)] DONE (validate rc=$VAL_RC). Report: $RUN_DIR/report.md"
exit "$VAL_RC"

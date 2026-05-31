#!/usr/bin/env bash
# Phase E launcher: the FULL self-supervised re-optimization under the Phase-A
# damped inference + the Phase-B anchor-referenced data_loglik objective.
#
# Pins the validated anti-drift STRUCTURE (num_blobs=128, init_gibbs_sweeps=1,
# per-frame sweeps=1, blob_means_updates=1 — the "drift schedule" the pvc gate
# used) and GRIDS ONLY the anti-drift damping, SELECTED self-supervised on TRAIN
# by the anchor-referenced complete-data log-likelihood (anchor_pos_feat — the
# variant the Phase-B proto validated as GT-tracking). The full EM then re-learns
# ALL conjugate hypers UNDER that inference + objective; validate gates ONCE on
# held-out GT and emits the configs on PASS.
#
#   --force          : fresh select (clears the stale 5-tuple grid cache) + em + validate
#   OBJECTIVE        : data_loglik (the corrected probabilistic objective)
#   DATA_LOGLIK_VARIANT=anchor_pos_feat : score features vs the FROZEN frame-0 anchor
#   SELECT_FEATURE_UPDATE_DAMPING : the damping grid to select over
#
# Held-out gate (once): median Δ GT region-J > 0 vs the shipping (freeze-fix)
# baseline AND chosen median >= max(0.664, 0.6514) AND p95 not regressed AND
# corr(data_loglik, GT) spearman > 0 (loglik kill-switch).
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OBJECTIVE=data_loglik \
DATA_LOGLIK_VARIANT=anchor_pos_feat \
SELECT_FEATURE_UPDATE_DAMPING="${SELECT_FEATURE_UPDATE_DAMPING:-0.0,0.25,0.5}" \
SELECT_MAX_FRAMES="${SELECT_MAX_FRAMES:-60}" \
bash scripts/run_calibration_v3.sh \
  --force \
  --select-num-blobs 128 \
  --select-init-sweeps 1 \
  --select-per-frame-sweeps 1 \
  --select-blob-means-updates 1 \
  --num-blobs 128 \
  --init-gibbs-sweeps 1 \
  --blob-means-updates 1 \
  --max-iters "${MAX_ITERS:-4}" \
  "$@"

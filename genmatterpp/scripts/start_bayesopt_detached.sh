#!/usr/bin/env bash
# Long-running BayesOpt (resume latest run). Safe to disconnect SSH / close laptop.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
VIDEO_ID="${1:-original_244622072067}"
RUN_ID="${2:-20260520_013644}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/bayesopt_${VIDEO_ID}.log"
PIDFILE="$LOG_DIR/bayesopt_${VIDEO_ID}.pid"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Already running pid $(cat "$PIDFILE"). Log: $LOG"
  exit 0
fi

uv sync --extra bayesopt
nohup uv run --extra bayesopt genmatter bayesopt \
  --video-id "$VIDEO_ID" \
  --config configs/custom_default.yaml \
  --bayesopt-config configs/custom_bayesopt.yaml \
  --run-id "$RUN_ID" \
  >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "BayesOpt pid $(cat "$PIDFILE") run_id=$RUN_ID"
echo "Log: $LOG"
echo "Trials: assets/custom_videos/$VIDEO_ID/bayesopt/$RUN_ID/trials.jsonl"

#!/usr/bin/env bash
# Run BayesOpt in background (plain log lines when stdout is not a TTY).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
VIDEO_ID="${1:-original_244622072067}"
RUN_ID="${2:-}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/bayesopt_${VIDEO_ID}.log"
PIDFILE="$LOG_DIR/bayesopt_${VIDEO_ID}.pid"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "bayesopt already running (pid $(cat "$PIDFILE")). Log: $LOG"
  exit 0
fi

EXTRA=()
if [[ -n "$RUN_ID" ]]; then
  EXTRA+=(--run-id "$RUN_ID")
fi

uv sync --extra bayesopt
nohup uv run --extra bayesopt genmatter bayesopt \
  --video-id "$VIDEO_ID" \
  --config configs/custom_default.yaml \
  --bayesopt-config configs/custom_bayesopt.yaml \
  "${EXTRA[@]}" \
  >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started bayesopt pid $(cat "$PIDFILE"). Log: $LOG"

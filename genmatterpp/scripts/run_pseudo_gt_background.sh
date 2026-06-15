#!/usr/bin/env bash
# Run pseudo-GT in background (survives SSH/laptop disconnect on VM).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
VIDEO_ID="${1:-original_244622072067}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pseudo_gt_${VIDEO_ID}.log"
PIDFILE="$LOG_DIR/pseudo_gt_${VIDEO_ID}.pid"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "pseudo-gt already running (pid $(cat "$PIDFILE")). Log: $LOG"
  exit 0
fi

nohup uv run genmatter pseudo-gt --video-id "$VIDEO_ID" --force \
  >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started pseudo-gt pid $(cat "$PIDFILE"). Log: $LOG"

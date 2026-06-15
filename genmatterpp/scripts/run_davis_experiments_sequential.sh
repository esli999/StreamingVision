#!/usr/bin/env bash
# Run DAVIS-related GenMatter experiments one after another.
# Continues to the next step even if a step fails (non-zero exit).
#
# Usage:
#   chmod +x scripts/run_davis_experiments_sequential.sh
#
#   # Foreground — all output to a log file:
#   ./scripts/run_davis_experiments_sequential.sh /path/to/your.log
#
#   # Background + survives closing the terminal (recommended):
#   cd /path/to/GenMatter
#   nohup ./scripts/run_davis_experiments_sequential.sh ./results/logs/davis_batch.log </dev/null >/dev/null 2>&1 &
#   echo "PID: $!  Log: ./results/logs/davis_batch.log"
#   tail -f ./results/logs/davis_batch.log
#
# Default log if no argument: ./results/logs/davis_batch_YYYYMMDD_HHMMSS.log

set +e
set -o pipefail 2>/dev/null || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "Cannot cd to $REPO_ROOT"; exit 1; }

LOG_FILE="${1:-$REPO_ROOT/results/logs/davis_batch_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

echo "================================================================================"
echo "DAVIS experiment batch — $(date -Is)"
echo "Repository: $REPO_ROOT"
echo "Log file:   $LOG_FILE"
echo "================================================================================"

run_step() {
  local title="$1"
  shift
  echo ""
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  echo "START: $title — $(date -Is)"
  echo "Command: $*"
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  "$@"
  local ec=$?
  echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  echo "END:   $title — exit code $ec — $(date -Is)"
  echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  return 0
}

# Exact order requested:
run_step "1 davis-ablation-sam" \
  uv run python run_experiments.py davis-ablation-sam

run_step "2 davis-ablation-gt-init" \
  uv run python run_experiments.py davis-ablation-gt-init

run_step "3 postprocess-davis" \
  uv run python run_experiments.py postprocess-davis

echo ""
echo "================================================================================"
echo "Batch finished — $(date -Is)"
echo "Full log: $LOG_FILE"
echo "================================================================================"

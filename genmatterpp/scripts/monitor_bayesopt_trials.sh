#!/usr/bin/env bash
# Wait until N trials appear in trials.jsonl and print a summary.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VIDEO_ID="${1:-original_244622072067}"
WANT="${2:-4}"
RUN_ROOT="${3:-}"

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$(ls -td "$REPO_ROOT/assets/custom_videos/$VIDEO_ID/bayesopt"/*/ 2>/dev/null | head -1)"
fi
TRIALS="${RUN_ROOT%/}/trials.jsonl"
echo "Watching $TRIALS for $WANT trials..."

for _ in $(seq 1 720); do
  if [[ -f "$TRIALS" ]]; then
    n=$(wc -l <"$TRIALS" | tr -d ' ')
    if [[ "$n" -ge "$WANT" ]]; then
      echo "=== First $WANT trials ==="
      head -n "$WANT" "$TRIALS"
      echo "=== Summary ==="
      python3 - <<PY
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("$TRIALS").read_text().splitlines() if l.strip()][:int("$WANT")]
for r in rows:
    print(
        f"trial {r.get('trial_index')}: {r.get('status')} "
        f"jaccard={r.get('objective', 0):.4f} "
        f"outlier%={r.get('mean_outlier_pct', 'n/a')} "
        f"elapsed={r.get('elapsed_seconds', 0):.1f}s"
    )
PY
      exit 0
    fi
  fi
  sleep 30
done
echo "Timeout waiting for $WANT trials"
exit 1

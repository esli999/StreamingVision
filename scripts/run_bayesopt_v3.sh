#!/usr/bin/env bash
# Run the multi-objective bayesopt unattended for ~3 hours, then extract
# the tuned config and render the comparison demo.  One invocation, one
# terminal session, walk-away, return to assets/matter_demo_v2.mp4.
#
# Usage:
#   ./scripts/run_bayesopt_v3.sh                # full run: preflight + 3h + demo
#   ./scripts/run_bayesopt_v3.sh --skip-preflight   # resume an aborted run
#   ./scripts/run_bayesopt_v3.sh --duration 600 # short test (10 min instead of 3h)

set -euo pipefail
cd "$(dirname "$0")/.."

# Self-sufficient env: don't rely on the parent shell having the
# streamingvision conda env activated.  Pinning PATH to the env's bin/
# guarantees `python`, `pip`, etc., resolve correctly even when the
# wrapper is launched from a clean session (setsid, systemd, cron, etc.).
CONDA_ENV_BIN=/home/esli/miniconda3/envs/streamingvision/bin
if [[ -x "$CONDA_ENV_BIN/python" ]]; then
    export PATH="$CONDA_ENV_BIN:$PATH"
else
    echo "ERROR: $CONDA_ENV_BIN/python not found — is the streamingvision conda env installed?" >&2
    exit 1
fi

LOG=/tmp/bayesopt_v3.log
RUN_ID=${RUN_ID:-v3_moo_3hr}
VIDEO_LIST=gray_jacket,jello_trim,new_eagle_trim,purple_jacket,snake_trim,wine_swirl,two_blocks_centered
PRIMARY_VIDEO=gray_jacket
RUN_DIR=assets/custom_videos/$PRIMARY_VIDEO/bayesopt/$RUN_ID
TRIALS_FILE=$RUN_DIR/trials.jsonl
DURATION=${DURATION:-10800}       # 3 hours
STALL_S=900                       # 15 min without a new trial row → restart
MAX_RESTARTS=3
SKIP_PREFLIGHT=0
SKIP_DEMO=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --skip-demo)      SKIP_DEMO=1; shift ;;
        --duration)       DURATION=$2; shift 2 ;;
        --run-id)         RUN_ID=$2; RUN_DIR=assets/custom_videos/$PRIMARY_VIDEO/bayesopt/$RUN_ID; TRIALS_FILE=$RUN_DIR/trials.jsonl; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

log() {
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    # If stdout is a terminal, write to both terminal and file.  When the
    # wrapper is launched with stdout already redirected to "$LOG" (the
    # autonomous case), `tee -a "$LOG"` would duplicate every line; just
    # append to the file instead.
    if [[ -t 1 ]]; then
        printf '%s\n' "$msg" | tee -a "$LOG"
    else
        printf '%s\n' "$msg" >> "$LOG"
    fi
}
fail() { log "HALT: $*"; exit 2; }

# Truncate the log only on a brand-new run.
if [[ $SKIP_PREFLIGHT -eq 0 && ! -f "$TRIALS_FILE" ]]; then
    : > "$LOG"
fi
log "wrapper start; run_id=$RUN_ID duration=${DURATION}s"
log "log file: $LOG"

# ---- 1. Pre-flight ---------------------------------------------------------

if [[ $SKIP_PREFLIGHT -eq 0 ]]; then
    log "preflight (5-15 min)..."
    if ! python scripts/bayesopt_v3_preflight.py >> "$LOG" 2>&1; then
        fail "preflight failed; aborting before launching long run"
    fi
    log "preflight OK"
else
    log "preflight skipped (--skip-preflight)"
fi

# ---- 2. Launch bayesopt in background --------------------------------------

# Set the LAUNCH_TS marker so the watchdog can compute time-since-launch.
launch_bayesopt() {
    local resume_flag="$1"   # "--no-resume" on first launch, "" on resume
    log "launching bayesopt ($resume_flag) → $LOG"
    nohup env \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        GENMATTER_DEFAULT_MAX_FRAMES=128 \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
        python -m genmatterpp.genmatter.custom.cli bayesopt \
            --video-id "$VIDEO_LIST" \
            --config configs/streaming_default.yaml \
            --bayesopt-config configs/streaming_bayesopt.yaml \
            $resume_flag --run-id "$RUN_ID" \
        >> "$LOG" 2>&1 &
    echo $! > /tmp/bayesopt_v3.pid
    BAYES_PID=$(cat /tmp/bayesopt_v3.pid)
    log "bayesopt pid=$BAYES_PID"
}

# On a brand-new run, --no-resume; on a resumed wrapper, let the runner
# pick up the existing run_config.yaml + trials.jsonl.
if [[ -f "$TRIALS_FILE" ]]; then
    log "found existing trials.jsonl ($(wc -l < "$TRIALS_FILE") rows) — resuming"
    launch_bayesopt ""
else
    launch_bayesopt "--no-resume"
fi

LAUNCH_TS=$(date +%s)
RESTARTS=0

# ---- 3. Watchdog + 3-hour timeout ------------------------------------------

while kill -0 "$BAYES_PID" 2>/dev/null; do
    now=$(date +%s)
    elapsed=$((now - LAUNCH_TS))
    if (( elapsed >= DURATION )); then
        log "duration ${DURATION}s reached — SIGINT bayesopt"
        kill -INT "$BAYES_PID" 2>/dev/null || true
        # Give it 60s to finalize trial + write state.
        for _ in {1..60}; do
            kill -0 "$BAYES_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$BAYES_PID" 2>/dev/null || true
        break
    fi

    # Stall detect: trial count hasn't grown in $STALL_S.
    if [[ -f "$TRIALS_FILE" ]]; then
        cur_count=$(wc -l < "$TRIALS_FILE" 2>/dev/null || echo 0)
        if [[ -z "${prev_count:-}" ]]; then
            prev_count=$cur_count
            prev_count_ts=$now
        elif (( cur_count > prev_count )); then
            prev_count=$cur_count
            prev_count_ts=$now
        elif (( now - prev_count_ts >= STALL_S )); then
            if (( RESTARTS >= MAX_RESTARTS )); then
                fail "stall detected after $RESTARTS restarts in 30 min — giving up"
            fi
            RESTARTS=$((RESTARTS + 1))
            log "STALL DETECTED: $((now - prev_count_ts))s without a new trial (restart $RESTARTS/$MAX_RESTARTS)"
            kill -INT "$BAYES_PID" 2>/dev/null || true
            for _ in {1..30}; do
                kill -0 "$BAYES_PID" 2>/dev/null || break
                sleep 1
            done
            kill -KILL "$BAYES_PID" 2>/dev/null || true
            launch_bayesopt ""    # resume from existing trials.jsonl
            prev_count_ts=$(date +%s)
        fi
    fi

    sleep 60
done

log "bayesopt process ended"

# ---- 4. Extract winning trial → streaming_tuned.yaml -----------------------

if [[ ! -f "$TRIALS_FILE" ]]; then
    fail "no trials.jsonl at $TRIALS_FILE — nothing to extract"
fi
COMPLETED=$(grep -c '"status": "completed"' "$TRIALS_FILE" || true)
log "completed trials: $COMPLETED"
if [[ "$COMPLETED" -lt 1 ]]; then
    fail "no completed trials in $TRIALS_FILE"
fi

log "extracting best trial (--pick-by min)"
python scripts/extract_best_tuned_yaml.py \
    --video-id "$PRIMARY_VIDEO" \
    --run-id "$RUN_ID" \
    --pick-by min >> "$LOG" 2>&1

# Report the per-video floor; if it's below the accuracy gate we still
# render the demo but the wrapper exits non-zero.
PV_MIN=$(python -c "
import yaml
m = yaml.safe_load(open('configs/streaming_tuned.yaml')).get('_meta', {})
print(m.get('per_video_min_persistent_iou', -1))
")
log "per-video min persistent IoU: $PV_MIN"

# ---- 5. Render the v2 comparison demo --------------------------------------

if [[ $SKIP_DEMO -eq 0 ]]; then
    log "rendering v2 demo → assets/matter_demo_v2.mp4"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    GENMATTER_DEFAULT_MAX_FRAMES=128 \
    python render_demo.py \
        --duration 20 --fps 30 \
        --out assets/matter_demo_v2.mp4 \
        --config configs/streaming_tuned.yaml >> "$LOG" 2>&1
else
    log "render skipped (--skip-demo)"
fi

# ---- 6. Final acceptance ---------------------------------------------------

if python -c "import sys; sys.exit(0 if float('$PV_MIN') >= 0.55 else 1)"; then
    log "DONE: accuracy gate (per-video min ≥ 0.55) PASSED at $PV_MIN"
    log "demo: assets/matter_demo_v2.mp4"
    exit 0
else
    log "DONE BUT BELOW GATE: per-video min $PV_MIN < 0.55"
    log "demo (for inspection): assets/matter_demo_v2.mp4"
    exit 1
fi

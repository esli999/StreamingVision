#!/usr/bin/env bash
# Helper: run all calibrate_consistency phases (preflight → pseudo_labels → em →
# validate → emit) in one process so the JAX disk cache is hit across phases
# (compile-once discipline). Foreground; the caller backgrounds + tees the log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
exec python scripts/calibrate_consistency.py --phase all "$@"

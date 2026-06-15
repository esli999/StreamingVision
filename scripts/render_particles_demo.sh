#!/usr/bin/env bash
# Render the particle-visualization demo (scripts/render_gaussian_demo.py).
# RENDER_MODE=focused = the ONE unified pipeline figure per clip (<vid>_unified.mp4);
# unset = the legacy 2x3.
#
# The recommended entry point for re-rendering the particle demo videos. Sets the JAX
# GPU memory fraction (easy to forget, and XLA starves without it) and forwards every
# argument straight through to render_gaussian_demo.py.
#
# Usage:
#   RENDER_MODE=focused scripts/render_particles_demo.sh --source assets/test.mp4 --target-duration 0
#                                                          # the bundled video (no dataset needed)
#   RENDER_MODE=focused DEMO_NUM_BLOBS=256 STICKINESS=12 ROBUST_DELTA=6 \
#     scripts/render_particles_demo.sh --target-duration 0 --out-fps 12   # featured subjects
#   scripts/render_particles_demo.sh --videos blackswan    # one video (legacy 2x3)
#   scripts/render_particles_demo.sh --videos wine_swirl two_blocks_centered --out-dir runs/mine
#   RENDER_MODE=focused scripts/render_particles_demo.sh --videos all-davis   # all 30 DAVIS clips
#   RENDER_MODE=focused scripts/render_particles_demo.sh --videos all-custom  # all custom clips
#   RENDER_MODE=focused scripts/render_particles_demo.sh --videos all         # every test video
#   MEM_FRACTION=0.8 scripts/render_particles_demo.sh      # override the JAX mem fraction
#
# See docs/RENDER_PARTICLES.md for the full layout, env-knob tuning, and how to add a
# new video. `python scripts/render_gaussian_demo.py --help` lists all flags + knobs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEM_FRACTION="${MEM_FRACTION:-0.6}"

# Default to a 6-second render of the featured focus videos if no args are given.
if [ "$#" -eq 0 ]; then
  set -- --target-duration 6
fi

echo "[render_particles_demo] XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION}  args: $*"
XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}" \
  python "${REPO}/scripts/render_gaussian_demo.py" "$@"

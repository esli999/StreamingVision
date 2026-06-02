#!/usr/bin/env bash
# Render the 2x3 particle-visualization demo (scripts/render_gaussian_demo.py).
#
# The recommended entry point for re-rendering the particle demo videos. Sets the JAX
# GPU memory fraction (easy to forget, and XLA starves without it) and forwards every
# argument straight through to render_gaussian_demo.py.
#
# Usage:
#   scripts/render_particles_demo.sh                       # the 5 focus videos
#   scripts/render_particles_demo.sh --videos blackswan    # one video
#   scripts/render_particles_demo.sh --videos wine_swirl car-roundabout --out-dir runs/mine
#   MEM_FRACTION=0.8 scripts/render_particles_demo.sh      # override the JAX mem fraction
#
# See docs/RENDER_PARTICLES.md for the full layout, env-knob tuning, and how to add a
# new video. `python scripts/render_gaussian_demo.py --help` lists all flags + knobs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEM_FRACTION="${MEM_FRACTION:-0.6}"

# Default to a 6-second render of the 5 focus videos if no args are given.
if [ "$#" -eq 0 ]; then
  set -- --target-duration 6
fi

echo "[render_particles_demo] XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION}  args: $*"
XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}" \
  python "${REPO}/scripts/render_gaussian_demo.py" "$@"

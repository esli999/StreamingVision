#!/usr/bin/env python3
"""Diagnose JAX recompilation across DAVIS / DINO tracking runs.

Run:
  uv run python scripts/diagnose_jax_recompile.py
  uv run python scripts/diagnose_jax_recompile.py --profile blackswan bike-packing
  JAX_LOG_COMPILES=1 uv run python scripts/diagnose_jax_recompile.py --profile blackswan 2>&1 | rg 'Compiling '
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from collections import Counter
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402


def survey_davis_shapes() -> Counter:
    import numpy as np

    pct = 0.78125
    shapes: Counter = Counter()
    for vid in config.TAPVID_DAVIS_VIDEO_NAMES:
        found = False
        for name in (f"{vid}_3d_motion.npz", f"{vid}_3d_data.npz"):
            path = config.DAVIS_3D_MOTION_PATH / name
            if not path.is_file():
                continue
            with np.load(path) as z:
                t, h, w, _ = z["points_3d"].shape
            n_sub = max(1, int(h * w * pct / 100.0))
            n_dense_batches = (h * w) // 975
            n_sub_batches = n_sub // 975
            shapes[(t, h, w, n_sub, n_sub_batches, n_dense_batches)] += 1
            found = True
            break
        if not found:
            shapes[("MISSING", vid)] += 1
    return shapes


def benchmark_nested_vs_module_jit() -> None:
    """Minimal repro: nested @jax.jit creates a new cache entry every call."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    def make_nested_step():
        @jax.jit
        def step(carry, _):
            x, = carry
            return (x + 1.0,), x

        return step

    @jax.jit
    def module_step(carry, _):
        x, = carry
        return (x + 1.0,), x

    def bench(step_fn, n: int, label: str) -> float:
        t0 = time.perf_counter()
        out = lax.scan(step_fn, (0.0,), None, length=n)
        out[0][0].block_until_ready()
        elapsed = time.perf_counter() - t0
        print(f"  {label:22s} scan_length={n:3d}  {elapsed:.4f}s")
        return elapsed

    print("\n=== Nested @jax.jit inside a factory (like genmatter_tracking_gibbs_dino) ===")
    for n in (48, 67, 48):
        bench(make_nested_step(), n, "nested (new fn each time)")

    print("\n=== Module-level @jax.jit (like genmatter/inference.py f_tracking_sweep) ===")
    for n in (48, 67, 48):
        bench(module_step, n, "module-level")


def _tracking_params():
    from genmatter.tracking.dino import (
        DinoTrackingHyperparams,
        DinoTrackingInputs,
        DinoTrackingParams,
        configure_jax_cache,
        run_dino_tracking,
    )

    configure_jax_cache()
    return (
        DinoTrackingInputs,
        DinoTrackingParams(
            num_blobs=500,
            num_hyperblobs=9,
            datapoint_retain_pct=0.78125,
            random_seed=42,
            focal_length=520.0,
            use_sam_frame0=True,
            init_gibbs_sweeps=15,
            tracking_outlier_prob=1e-28,
            measure_fps=True,
            hyperparams=DinoTrackingHyperparams(),
        ),
        run_dino_tracking,
    )


def _inputs(video_id: str):
    from genmatter.tracking.dino import DinoTrackingInputs

    motion = config.DAVIS_3D_MOTION_PATH / f"{video_id}_3d_motion.npz"
    if not motion.is_file():
        motion = config.DAVIS_3D_MOTION_PATH / f"{video_id}_3d_data.npz"
    return DinoTrackingInputs(
        video_id=video_id,
        motion_npz=motion,
        dino_npz=config.DAVIS_DINO_PATH / f"{video_id}_dino_pca_per_pixel.npz",
        sam_frame0_png=config.DAVIS_SAM_FRAME0_PATH / f"{video_id}_SAM_frame0.png",
    )


def profile_videos(
    video_ids: list[str],
    *,
    clear_caches_between: bool,
) -> None:
    import jax
    from genmatter.tracking.dino import configure_jax_cache, run_dino_tracking

    _, params, _ = _tracking_params()
    configure_jax_cache()

    print(
        f"\n=== run_dino_tracking JIT warmup timings "
        f"(clear_caches_between={clear_caches_between}) ==="
    )
    for vid in video_ids:
        if clear_caches_between and hasattr(jax, "clear_caches"):
            jax.clear_caches()
            gc.collect()

        buf = StringIO()
        # JAX compile logs go through logging; capture is best-effort.
        t0 = time.perf_counter()
        result = run_dino_tracking(_inputs(vid), params)
        elapsed = time.perf_counter() - t0
        t = result.timings
        print(
            f"  {vid:16s} T={t.num_frames:3d}  "
            f"jit_init={t.jit_init_gibbs_seconds:6.2f}s  "
            f"jit_track={t.jit_tracking_seconds:6.2f}s  "
            f"jit_dense={t.jit_dense_seconds:5.2f}s  "
            f"total={elapsed:6.1f}s"
        )
        _ = buf


def print_findings() -> None:
    print(
        """
=== Likely recompilation causes (davis-tracking-sam / run_dino_tracking) ===

1. Nested @jax.jit in genmatter_tracking_gibbs_dino (genmatter/tracking/dino.py)
   - f_tracking_sweep is defined INSIDE the function and decorated there.
   - Every call builds a new Python callable → JAX treats it as a new jit target.
   - Minimal repro in benchmark_nested_vs_module_jit(): repeating the same scan
     length still pays compile cost with nested jit, but not module-level jit.

2. Variable number of frames T across DAVIS videos (~20 unique T values)
   - timestep_indices = jnp.arange(1, len(tracked_points)) fixes scan length to T-1 at trace time.
   - Even with hoisted module-level jit, you need one XLA program per (T-1) unless you pad/mask.

3. jax.clear_caches() in experiments/davis/run_davis_tracking.py cleanup_between_runs()
   - Called after every video; drops in-memory compilation cache.
   - Persistent cache under .jax_cache helps reload, but warmup paths still run and
     nested jit + new T can miss until disk cache is warm.

4. init_gibbs_sweep_dino is NOT module-level @jax.jit
   - 15 Gibbs sweeps traced each run; contributes to jit_init_gibbs_seconds.

5. Shapes that are STABLE across standard TAP-Vid DAVIS (520x960):
   - Subsampled datapoints: 3900 (0.78125% of 499200)
   - Dense grid: 499200 pixels → dense_eval_blob_assignments should reuse one compile
   - num_blobs=500, num_hyperblobs=9 fixed

=== Ideas to reduce recompilation ===

A. Hoist jit to module scope (highest leverage, matches inference.py)
   - @jax.jit def f_tracking_sweep(...)
   - @jax.jit def init_gibbs_sweep_dino(...) or jit the gibbs_iteration body
   - @jax.jit gibbs_blob_assignments_dino / blob_tracking_gibbs_dino if not inlined

B. Remove or gate jax.clear_caches() in the per-video loop
   - Prefer gc.collect() only; clear only on OOM or --reset-jax-cache flag

C. Pad sequences to max_T (or bucket by T)
   - static_argnums=('num_steps',) with a small set of T buckets
   - Or lax.scan with length=max_T and a carry mask (more work, one compile)

D. Rely on persistent cache (already enabled via configure_jax_cache)
   - Process videos grouped by T to maximize cache hits in one session
   - Do not redefine jitted functions inside run_dino_tracking

E. Split compile-once / run-many API
   - compile_tracking_program(T, num_datapoints) once per shape bucket
   - execute with different keys and arrays only

F. dense_eval is already module-level jit — keep it; avoid re-wrapping per frame
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        nargs="*",
        metavar="VIDEO",
        help="Run run_dino_tracking JIT timings for these video ids",
    )
    parser.add_argument(
        "--with-clear-caches",
        action="store_true",
        help="Call jax.clear_caches() between profiled videos (matches experiment loop)",
    )
    parser.add_argument("--skip-shape-survey", action="store_true")
    args = parser.parse_args()

    if not args.skip_shape_survey:
        shapes = survey_davis_shapes()
        print("=== TAP-Vid DAVIS shape survey (T, H, W, subsampled_N, sub_batches, dense_batches) ===")
        print(f"  unique shape keys: {len([k for k in shapes if k[0] != 'MISSING'])}")
        for key, count in shapes.most_common(12):
            print(f"  {count:2d}x  {key}")

    benchmark_nested_vs_module_jit()
    print_findings()

    if args.profile:
        profile_videos(args.profile, clear_caches_between=args.with_clear_caches)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

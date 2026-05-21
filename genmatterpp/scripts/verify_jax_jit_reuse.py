#!/usr/bin/env python3
"""Verify padded JIT tracking: parity vs pre-refactor + no recompile across videos."""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from genmatter.tracking.dino import (  # noqa: E402
    DinoTrackingHyperparams,
    DinoTrackingInputs,
    DinoTrackingParams,
    compile_dino_tracking_program,
    configure_jax_cache,
    run_dino_tracking,
)


def _params(*, measure_fps: bool = True) -> DinoTrackingParams:
    return DinoTrackingParams(
        num_blobs=500,
        num_hyperblobs=9,
        datapoint_retain_pct=0.78125,
        random_seed=42,
        focal_length=520.0,
        use_sam_frame0=True,
        init_gibbs_sweeps=15,
        tracking_outlier_prob=1e-28,
        measure_fps=measure_fps,
        hyperparams=DinoTrackingHyperparams(),
    )


def _inputs(video_id: str) -> DinoTrackingInputs:
    motion = config.DAVIS_3D_MOTION_PATH / f"{video_id}_3d_motion.npz"
    if not motion.is_file():
        motion = config.DAVIS_3D_MOTION_PATH / f"{video_id}_3d_data.npz"
    return DinoTrackingInputs(
        video_id=video_id,
        motion_npz=motion,
        dino_npz=config.DAVIS_DINO_PATH / f"{video_id}_dino_pca_per_pixel.npz",
        sam_frame0_png=config.DAVIS_SAM_FRAME0_PATH / f"{video_id}_SAM_frame0.png",
    )


def _load_pre_refactor_dino():
    path = REPO_ROOT / "scripts" / "_dino_pre_jit_reference.py"
    if not path.is_file():
        import subprocess

        subprocess.run(
            ["git", "show", "HEAD:genmatter/tracking/dino.py"],
            check=True,
            stdout=open(path, "w"),
        )
    spec = importlib.util.spec_from_file_location("dino_pre_jit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dino_pre_jit"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parity_vs_git_head() -> None:
    old = _load_pre_refactor_dino()
    import genmatter.tracking.dino as new

    inp = _inputs("blackswan")
    p = _params(measure_fps=False)
    a_old = np.stack(
        [f["blob_assignments"] for f in old.run_dino_tracking(inp, p).tracking_data]
    )
    prog = new.compile_dino_tracking_program()
    a_new = np.stack(
        [
            f["blob_assignments"]
            for f in new.run_dino_tracking(inp, p, compiled=prog).tracking_data
        ]
    )
    if not np.array_equal(a_old, a_new):
        raise AssertionError(
            f"blob_assignments differ: {(a_old != a_new).sum()} / {a_old.size} pixels"
        )
    print("[ok] parity vs git HEAD dino.py (blackswan, all frames)")


def test_multi_video(prog, vids: list[str]) -> None:
    params = _params(measure_fps=True)
    rows = []
    compiles: list[str] = []
    log = logging.getLogger("jax")

    class H(logging.Handler):
        def emit(self, r):
            m = re.search(r"Compiling (\S+) with", r.getMessage())
            if m:
                compiles.append(m.group(1))

    h = H()
    log.addHandler(h)
    log.setLevel(logging.WARNING)
    try:
        for vid in vids:
            t0 = time.perf_counter()
            r = run_dino_tracking(_inputs(vid), params, compiled=prog)
            rows.append((vid, r.timings.num_frames, r.timings.jit_tracking_seconds, time.perf_counter() - t0))
    finally:
        log.removeHandler(h)

    print("\n=== Multi-video ===")
    for vid, t, jit_t, total in rows:
        print(f"  {vid:16s} T={t:3d}  jit_track={jit_t:6.2f}s  total={total:6.1f}s")

    assert rows[0][2] > 0
    assert rows[1][2] == 0.0 and rows[2][2] == 0.0
    track_hits = [c for c in compiles if "f_tracking_sweep_dino" in c]
    assert len(track_hits) <= 1, track_hits
    print(f"  f_tracking_sweep_dino compiles: {len(track_hits)}")
    print("[ok] videos 2+ skip JIT timing; tracking kernel compiles once")


def main() -> int:
    configure_jax_cache(str(REPO_ROOT / ".jax_cache"))
    print(f"max_frames={config.tapvid_davis_max_frames()}")
    test_parity_vs_git_head()
    prog = compile_dino_tracking_program()
    test_multi_video(prog, ["blackswan", "bike-packing", "dog"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

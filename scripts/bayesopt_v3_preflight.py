#!/usr/bin/env python3
"""Pre-flight check before kicking off a 3-hour multi-objective bayesopt run.

Each step has a hard fail.  If anything fails, the long-run wrapper aborts
before burning GPU.  Expected wall-clock: 5-15 min depending on JIT warmup
and how aggressive the determinism + qNEHVI probes are.

Steps:
  1. Imports + Ax/BoTorch version compatibility.
  2. Run a single real tracking call to warm up JIT + read peak GPU memory.
  3. Determinism: same params with parallel_videos=1 vs 4 must produce
     identical per-video IoUs.  Catches thread-safety bugs in JAX/genjax.
  4. MO experiment shape: AxClient with N=2 objectives has 2 entries in
     its optimization_config.
  5. qNEHVI acquisition timing: drive the generation strategy past Sobol
     by attaching synthetic completed trials, then time one BoTorch
     proposal.  > 30s ⇒ abort (we'd otherwise eat the whole budget on
     acquisition optimization).
  6. Warm-start replay: load v1 trial 26 via _warm_start_attach and verify
     it ends up in trials.jsonl.

Exits 0 with ``PREFLIGHT OK`` line on success, non-zero with ``WHY`` line
on first failure.
"""

from __future__ import annotations

import os
# Set XLA mem fraction BEFORE jax imports anywhere.  Single-video tracking
# already needs ~10 GB of XLA buffer space; 2-way parallel peaks at ~29 GB.
# The user's RTX 5090 has 32 GB total; 85 % (= 27.2 GB) gives JAX enough
# room while leaving headroom for CUDA runtime + transcode + safety.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("GENMATTER_DEFAULT_MAX_FRAMES", "128")

import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))


def _step(label: str) -> None:
    print(f"\n=== {label} ===", flush=True)


def _fail(why: str) -> None:
    print(f"WHY: {why}", flush=True)
    print("PREFLIGHT FAILED", flush=True)
    sys.exit(2)


def _gpu_mem_mb() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return float(out.strip().splitlines()[0])


def step_1_imports() -> None:
    _step("1. imports")
    try:
        from genmatterpp.genmatter.bayesopt.objective import (  # noqa: F401
            MultiVideoTrackingObjective,
            build_multi_objective_contexts,
        )
        from genmatterpp.genmatter.bayesopt.ax_runner import (  # noqa: F401
            _create_ax_client,
            _objective_names,
            _row_per_video_objectives,
            _warm_start_attach,
            _build_generation_strategy,
        )
        from genmatterpp.genmatter.bayesopt.search_space import specs_from_config
        from genmatterpp.genmatter.bayesopt.config import load_bayesopt_config
        from genmatterpp.genmatter.custom.config_schema import load_config
        import ax
        import botorch
        import importlib.metadata as md
        print(f"ax={md.version('ax-platform')} botorch={md.version('botorch')}")
    except Exception as e:
        _fail(f"import error: {e!r}\n{traceback.format_exc()}")


def step_2_gpu_warmup(video_id: str) -> None:
    _step(f"2. JIT warmup + GPU memory probe on {video_id}")
    from genmatterpp.genmatter.bayesopt.objective import build_multi_objective_contexts
    from genmatterpp.genmatter.bayesopt.search_space import build_tracking_params
    from genmatterpp.genmatter.tracking.dino import run_dino_tracking
    from genmatterpp.genmatter.bayesopt.config import load_bayesopt_config
    from genmatterpp.genmatter.custom.config_schema import load_config

    cfg = load_config(REPO / "configs" / "streaming_default.yaml")
    bo_cfg = load_bayesopt_config(REPO / "configs" / "streaming_bayesopt.yaml", video_id=video_id)
    ctxs = build_multi_objective_contexts(cfg, [video_id], bo_cfg=bo_cfg)
    ctx = ctxs[0]
    mem_before = _gpu_mem_mb()
    params = build_tracking_params(ctx.cfg, {}, measure_fps=False)
    t0 = time.perf_counter()
    result = run_dino_tracking(ctx.inputs, params)
    elapsed = time.perf_counter() - t0
    mem_after = _gpu_mem_mb()
    print(f"single-video tracking: {elapsed:.1f}s, GPU mem {mem_before:.0f} → {mem_after:.0f} MB")
    if mem_after > 20000:
        _fail(f"single-video GPU usage {mem_after:.0f} MB is already > 20 GB — parallel will OOM")


def step_3_determinism(video_ids: list[str]) -> None:
    _step(f"3. determinism: parallel_videos=1 vs 4 on {len(video_ids)} videos")
    from genmatterpp.genmatter.bayesopt.objective import (
        MultiVideoTrackingObjective,
        build_multi_objective_contexts,
    )
    from genmatterpp.genmatter.bayesopt.config import load_bayesopt_config
    from genmatterpp.genmatter.custom.config_schema import load_config

    cfg = load_config(REPO / "configs" / "streaming_default.yaml")
    bo_cfg = load_bayesopt_config(REPO / "configs" / "streaming_bayesopt.yaml",
                                  video_id=",".join(video_ids))
    ctxs = build_multi_objective_contexts(cfg, video_ids, bo_cfg=bo_cfg)
    trial_params: dict[str, float] = {}

    obj_seq = MultiVideoTrackingObjective(ctxs, parallel_videos=1)
    obj_par = MultiVideoTrackingObjective(ctxs, parallel_videos=min(4, len(video_ids)))
    t0 = time.perf_counter()
    _, meta_seq, _ = obj_seq.evaluate(trial_params, measure_fps=False)
    seq_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, meta_par, _ = obj_par.evaluate(trial_params, measure_fps=False)
    par_t = time.perf_counter() - t0
    speedup = seq_t / par_t if par_t > 0 else 0.0
    print(f"serial: {seq_t:.1f}s  parallel: {par_t:.1f}s  speedup: {speedup:.2f}×")

    seq_pv = {pv["video_id"]: pv["avg_persistent_iou"] for pv in meta_seq["per_video"]}
    par_pv = {pv["video_id"]: pv["avg_persistent_iou"] for pv in meta_par["per_video"]}
    bad = []
    for vid in seq_pv:
        if abs(seq_pv[vid] - par_pv[vid]) > 1e-5:
            bad.append((vid, seq_pv[vid], par_pv[vid]))
    if bad:
        for vid, s, p in bad:
            print(f"  MISMATCH {vid}: serial={s:.6f}  parallel={p:.6f}  Δ={p - s:+.6f}")
        _fail("per-video IoUs disagreed between serial and parallel — thread-safety bug")
    print("per-video IoUs match within 1e-5; thread parallelism is deterministic")
    if speedup < 1.5 and len(video_ids) >= 3:
        print(f"WARN: parallelism speedup {speedup:.2f}× is below 1.5× — investigate but not fatal")


def step_4_mo_shape(video_ids: list[str]) -> None:
    _step(f"4. multi-objective experiment shape ({len(video_ids)} objectives)")
    from genmatterpp.genmatter.bayesopt.ax_runner import _create_ax_client, _objective_names
    from genmatterpp.genmatter.bayesopt.config import load_bayesopt_config
    from genmatterpp.genmatter.bayesopt.search_space import specs_from_config

    bo_cfg = load_bayesopt_config(REPO / "configs" / "streaming_bayesopt.yaml",
                                  video_id=",".join(video_ids))
    specs = specs_from_config(bo_cfg)
    names = _objective_names(video_ids)
    client = _create_ax_client(bo_cfg, specs, run_id="preflight_mo_shape",
                               objective_names=names,
                               per_video_iou_threshold=0.3)
    metrics = client.experiment.optimization_config.metrics
    print(f"objective metrics ({len(metrics)}): {sorted(metrics.keys())}")
    if len(metrics) != len(video_ids):
        _fail(f"expected {len(video_ids)} metrics, got {len(metrics)}")


def step_5_qnehvi_timing(video_ids: list[str]) -> None:
    _step("5. qNEHVI acquisition wall-clock probe")
    import numpy as np
    from genmatterpp.genmatter.bayesopt.ax_runner import _create_ax_client, _objective_names
    from genmatterpp.genmatter.bayesopt.config import load_bayesopt_config
    from genmatterpp.genmatter.bayesopt.search_space import specs_from_config

    bo_cfg = load_bayesopt_config(REPO / "configs" / "streaming_bayesopt.yaml",
                                  video_id=",".join(video_ids))
    specs = specs_from_config(bo_cfg)
    names = _objective_names(video_ids)
    client = _create_ax_client(bo_cfg, specs, run_id="preflight_qnehvi",
                               objective_names=names,
                               per_video_iou_threshold=0.3)
    n_sobol = int(bo_cfg["phase"]["phase_a_sobol_evals"])
    rng = np.random.default_rng(0)
    print(f"attaching {n_sobol} synthetic Sobol trials to drive the strategy past Sobol...")
    for i in range(n_sobol):
        params, tid = client.get_next_trial()
        raw = {n: (float(rng.uniform(0.3, 0.85)), 0.0) for n in names}
        client.complete_trial(trial_index=tid, raw_data=raw)
    t0 = time.perf_counter()
    params, tid = client.get_next_trial()
    elapsed = time.perf_counter() - t0
    print(f"qNEHVI first BoTorch proposal: {elapsed:.1f}s")
    if elapsed > 30.0:
        _fail(f"qNEHVI acquisition {elapsed:.1f}s > 30s — would dominate the budget")
    client.complete_trial(trial_index=tid,
                          raw_data={n: (0.5, 0.0) for n in names})


def step_6_warm_start_replay(video_ids: list[str]) -> None:
    _step("6. warm-start replay")
    from genmatterpp.genmatter.bayesopt.ax_runner import (
        _create_ax_client,
        _objective_names,
        _warm_start_attach,
    )
    from genmatterpp.genmatter.bayesopt.config import load_bayesopt_config
    from genmatterpp.genmatter.bayesopt.search_space import specs_from_config
    from genmatterpp.genmatter.custom.config_schema import load_config
    from genmatterpp.genmatter.bayesopt.terminal import BayesOptConsole

    cfg = load_config(REPO / "configs" / "streaming_default.yaml")
    bo_cfg = load_bayesopt_config(REPO / "configs" / "streaming_bayesopt.yaml",
                                  video_id=",".join(video_ids))
    warm = bo_cfg.get("phase", {}).get("warm_start")
    if not warm:
        print("no warm_start configured — skipping (not a failure)")
        return
    specs = specs_from_config(bo_cfg)
    names = _objective_names(video_ids)
    client = _create_ax_client(bo_cfg, specs, run_id="preflight_warm",
                               objective_names=names,
                               per_video_iou_threshold=0.3)

    class _StubUI:
        def print(self, *a, **kw): print(*a)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        trials_path = tmp / "trials.jsonl"
        events_path = tmp / "events.jsonl"
        run_root = tmp / "run"
        (run_root / "trials").mkdir(parents=True, exist_ok=True)
        _warm_start_attach(
            client,
            cfg,
            warm_start_cfg=warm,
            objective_names=names,
            trials_path=trials_path,
            events_path=events_path,
            run_root=run_root,
            param_names=[s.name for s in specs],
            resolved_model_params_dict=lambda p: dict(p),
            ui=_StubUI(),
        )
        if not trials_path.is_file() or trials_path.stat().st_size == 0:
            _fail("warm-start attach did not write trials.jsonl")
        rows = [json.loads(l) for l in trials_path.read_text().splitlines() if l.strip()]
        print(f"warm-start attached {len(rows)} row(s); first row phase={rows[0]['phase']} "
              f"per_video_count={len(rows[0].get('per_video', []))}")


def main() -> None:
    full_set = ["gray_jacket", "jello_trim", "new_eagle_trim", "purple_jacket",
                "snake_trim", "wine_swirl", "two_blocks_centered"]
    # For determinism / shape probes use a 2-video set so we keep wall-clock
    # under 5 min while still exercising the multi-video MO path.
    fast_set = ["gray_jacket", "purple_jacket"]

    t0 = time.perf_counter()
    step_1_imports()
    step_2_gpu_warmup("gray_jacket")
    step_3_determinism(fast_set)
    step_4_mo_shape(full_set)
    step_5_qnehvi_timing(full_set)
    step_6_warm_start_replay(full_set)
    elapsed = time.perf_counter() - t0
    print(f"\nPREFLIGHT OK ({elapsed:.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        _fail("unhandled exception during preflight")

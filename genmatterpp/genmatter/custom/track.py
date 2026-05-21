"""Orchestrate GenMatter DINO tracking on preprocessed custom videos."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from genmatter.custom.config_schema import CustomConfig
from genmatter.custom.console import PreprocessConsole
from genmatter.custom.paths import VideoPaths, resolve_video_paths
from genmatter.tracking.dino import (
    DinoTrackingInputs,
    dino_params_from_config,
    run_dino_tracking,
    save_dense_tracking_npz,
)


@dataclass
class TrackResult:
    video_id: str
    paths: VideoPaths
    manifest_path: Path
    success: bool


_REQUIRED_ARTIFACTS = ("npz_path", "dino_path", "sam_path")


def _load_preprocess_manifest(paths: VideoPaths) -> dict:
    if not paths.manifest_path.is_file():
        raise FileNotFoundError(
            f"Preprocess manifest not found: {paths.manifest_path}\n"
            "Run: uv run genmatter preprocess <mp4> first."
        )
    with open(paths.manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _validate_preprocess(paths: VideoPaths, preprocess_manifest: dict) -> list[str]:
    errors: list[str] = []
    for attr in _REQUIRED_ARTIFACTS:
        p = getattr(paths, attr)
        if not p.is_file():
            errors.append(f"Missing {attr}: {p}")
    if not paths.rgb_frames_dir.is_dir():
        errors.append(f"Missing rgb_frames_dir: {paths.rgb_frames_dir}")
    stages = preprocess_manifest.get("stages", {})
    for key in ("frames", "motion_3d", "dino", "sam_frame0"):
        st = stages.get(key, {})
        if st.get("status") not in ("success", "skipped"):
            errors.append(f"Preprocess stage '{key}' not successful")
    return errors


def _should_skip_tracking(paths: VideoPaths, cfg: CustomConfig) -> bool:
    if cfg.tracking.pipeline.force:
        return False
    if not cfg.tracking.pipeline.skip_existing:
        return False
    return paths.tracking_npz.is_file()


def run_track(
    video_id: str,
    cfg: CustomConfig,
    ui: PreprocessConsole,
) -> TrackResult:
    paths = resolve_video_paths(cfg, video_id)
    preprocess_manifest = _load_preprocess_manifest(paths)

    ui.print_header(
        mp4=Path(preprocess_manifest.get("mp4", paths.video_root / f"{video_id}.mp4")),
        video_id=video_id,
        config_path=cfg.config_path or Path("configs/custom_default.yaml"),
        output_root=paths.video_root,
        cuda_label="JAX (GPU if available)",
        est_frames=preprocess_manifest.get("stages", {})
        .get("frames", {})
        .get("num_frames"),
    )

    manifest: dict = {
        "video_id": video_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": cfg.config_fingerprint(),
        "preprocess_config_fingerprint": preprocess_manifest.get("config_fingerprint"),
        "artifacts": {
            "rgb_frames": str(paths.rgb_frames_dir),
            "motion_npz": str(paths.npz_path),
            "dino_npz": str(paths.dino_path),
            "sam_frame0": str(paths.sam_path),
            "tracking_dense": str(paths.tracking_npz),
        },
        "stages": {},
    }

    # ── [1/6] Validate ─────────────────────────────────────────────────────
    rec_validate = ui.numbered_track_stage(1, "Validate inputs")
    ui.begin_stage(rec_validate, "preprocess artifacts")
    errors = _validate_preprocess(paths, preprocess_manifest)
    if errors:
        for err in errors:
            ui.substep(err)
        raise FileNotFoundError("Preprocess incomplete:\n  " + "\n  ".join(errors))
    ui.end_stage_success(rec_validate, duration_s=0.0, detail="all artifacts present")
    manifest["stages"]["validate"] = {"status": "success"}

    if _should_skip_tracking(paths, cfg):
        rec_skip = ui.numbered_track_stage(2, "Tracking")
        ui.begin_stage(rec_skip)
        ui.end_stage_skipped(rec_skip, paths.tracking_npz, "tracking_dense.npz exists")
        manifest["stages"]["tracking"] = {"status": "skipped", "output": str(paths.tracking_npz)}
        paths.track_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(paths.track_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        ui.print_manifest_footer(paths.track_manifest_path, title="Track complete (skipped)")
        return TrackResult(video_id=video_id, paths=paths, manifest_path=paths.track_manifest_path, success=True)

    params = dino_params_from_config(cfg.tracking)
    inputs = DinoTrackingInputs(
        video_id=video_id,
        motion_npz=paths.npz_path,
        dino_npz=paths.dino_path,
        sam_frame0_png=paths.sam_path if cfg.tracking.use_sam_frame0 else None,
    )

    rec_load = ui.numbered_track_stage(2, "Load motion + DINO")
    rec_init = ui.numbered_track_stage(3, "Initialize model")
    rec_init_gibbs = ui.numbered_track_stage(4, "Frame-0 Gibbs burn-in")
    rec_tracking = ui.numbered_track_stage(5, "Temporal tracking")
    rec_dense = ui.numbered_track_stage(6, "Dense eval + save")

    stage_map = {
        "load": rec_load,
        "init": rec_init,
        "jit_init_gibbs": rec_init_gibbs,
        "init_gibbs": rec_init_gibbs,
        "jit_tracking": rec_tracking,
        "tracking": rec_tracking,
        "jit_dense": rec_dense,
        "dense": rec_dense,
        "save": rec_dense,
    }
    active_rec: dict[str, object] = {}

    def on_step_start(step: str) -> None:
        rec = stage_map.get(step)
        if rec is None:
            return
        if step.startswith("jit_"):
            ui.begin_stage(rec, f"JIT compile ({step[4:]})")
        elif step == "load":
            ui.begin_stage(rec, "NPZ + feature tensors")
        elif step == "init":
            ui.begin_stage(rec, "K-means + SAM segmentation")
        elif step == "init_gibbs":
            ui.begin_stage(rec, f"{params.init_gibbs_sweeps} Gibbs sweeps")
        elif step == "tracking":
            ui.begin_stage(rec, "Gibbs scan over time")
        elif step == "dense":
            ui.begin_stage(rec, "full-grid blob assignments")
        elif step == "save":
            ui.begin_stage(rec, "write tracking_dense.npz")
        active_rec[step] = rec

    dense_started = [False]

    def on_dense_progress(cur: int, tot: int) -> None:
        if not dense_started[0]:
            dense_started[0] = True
            ui.progress_start("Dense evaluation", total=tot)
        ui.progress_update(cur, tot)

    t_pipeline = time.perf_counter()
    try:
        result = run_dino_tracking(
            inputs,
            params,
            on_step_start=on_step_start,
            on_dense_progress=on_dense_progress,
        )
    finally:
        ui.progress_stop()

    tm = result.timings

    ui.timing_result("Load tensors", tm.load_seconds, frames=tm.num_frames)
    ui.end_stage_success(rec_load, duration_s=tm.load_seconds, fps=None)
    manifest["stages"]["load"] = {
        "status": "success",
        "timings": {"load_seconds": tm.load_seconds, "num_frames": tm.num_frames},
    }

    init_total = tm.kmeans_init_seconds + tm.importance_seconds
    ui.timing_result("K-means + SAM init", tm.kmeans_init_seconds)
    ui.timing_result("Importance sample", tm.importance_seconds)
    ui.end_stage_success(rec_init, duration_s=init_total, fps=None)
    manifest["stages"]["init"] = {
        "status": "success",
        "timings": {
            "kmeans_init_seconds": tm.kmeans_init_seconds,
            "importance_seconds": tm.importance_seconds,
        },
    }

    ui.timing_result("JIT compile (init Gibbs)", tm.jit_init_gibbs_seconds)
    ui.timing_result(
        "Init Gibbs sweeps",
        tm.init_gibbs_seconds,
        frames=params.init_gibbs_sweeps,
        unit="sweeps",
    )
    ui.end_stage_success(
        rec_init_gibbs,
        duration_s=tm.jit_init_gibbs_seconds + tm.init_gibbs_seconds,
        fps=tm.init_gibbs_fps if tm.init_gibbs_fps > 0 else None,
    )
    manifest["stages"]["init_gibbs"] = {
        "status": "success",
        "timings": {
            "jit_compile_seconds": tm.jit_init_gibbs_seconds,
            "init_gibbs_seconds": tm.init_gibbs_seconds,
            "init_gibbs_fps": tm.init_gibbs_fps,
        },
    }

    ui.timing_result("JIT compile (tracking)", tm.jit_tracking_seconds)
    ui.timing_result(
        "Gibbs temporal scan",
        tm.tracking_seconds,
        frames=tm.num_tracking_steps,
        unit="steps",
    )
    ui.end_stage_success(
        rec_tracking,
        duration_s=tm.jit_tracking_seconds + tm.tracking_seconds,
        fps=tm.tracking_fps if tm.tracking_fps > 0 else None,
    )
    manifest["stages"]["tracking"] = {
        "status": "success",
        "timings": {
            "jit_compile_seconds": tm.jit_tracking_seconds,
            "tracking_seconds": tm.tracking_seconds,
            "tracking_fps": tm.tracking_fps,
            "num_tracking_steps": tm.num_tracking_steps,
        },
    }

    on_step_start("save")
    t0 = time.perf_counter()
    ui.timing_result("JIT compile (dense)", tm.jit_dense_seconds)
    ui.timing_result(
        "Dense evaluation",
        tm.dense_seconds,
        frames=tm.num_frames,
    )
    file_size = save_dense_tracking_npz(
        paths.tracking_npz,
        result,
        rgb_frames_dir=paths.rgb_frames_dir,
        motion_npz=paths.npz_path,
        dino_npz=paths.dino_path,
        sam_frame0_png=paths.sam_path,
    )
    save_seconds = time.perf_counter() - t0
    ui.timing_result("Save NPZ", save_seconds)
    ui.end_stage_success(
        rec_dense,
        duration_s=tm.jit_dense_seconds + tm.dense_seconds + save_seconds,
        fps=tm.dense_fps if tm.dense_fps > 0 else None,
        detail=f"{file_size / 1e6:.1f} MB",
        output_path=paths.tracking_npz,
    )
    manifest["stages"]["dense"] = {
        "status": "success",
        "timings": {
            "jit_compile_seconds": tm.jit_dense_seconds,
            "dense_seconds": tm.dense_seconds,
            "dense_fps": tm.dense_fps,
            "save_seconds": save_seconds,
            "file_size_bytes": file_size,
        },
    }
    manifest["timings"] = asdict(tm)
    manifest["model"] = {
        "n_blobs": int(result.tracking_data[0]["n_blobs"]) if result.tracking_data else 0,
        "n_hyperblobs": int(result.tracking_data[0]["n_hyperblobs"]) if result.tracking_data else 0,
        "img_dims": list(result.img_dims),
        "focal_length": result.focal_length,
    }
    manifest["elapsed_seconds"] = time.perf_counter() - t_pipeline

    paths.track_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.track_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    ui.print_summary()
    ui.print_manifest_footer(paths.track_manifest_path, title="Track complete")

    return TrackResult(
        video_id=video_id,
        paths=paths,
        manifest_path=paths.track_manifest_path,
        success=True,
    )

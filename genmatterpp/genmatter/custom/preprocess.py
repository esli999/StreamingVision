"""Orchestrate custom MP4 preprocessing (frames → depth → motion → DINO → SAM)."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Repo root on path for ``preprocessing.motion_extraction_3d``
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from genmatter.custom.config_schema import CustomConfig
from genmatter.custom.console import PreprocessConsole
from genmatter.custom.paths import VideoPaths, ensure_video_dirs, resolve_video_paths
from genmatter.preprocessing.dino import DinoParams, load_dino_model, run_dino_extraction
from genmatter.preprocessing.frames import extract_frames_from_mp4, probe_mp4_frame_count
from genmatter.preprocessing.gpu import gpu_memory_snapshot, release_gpu
from genmatter.preprocessing.motion_3d import Motion3DParams, run_motion_3d
from genmatter.preprocessing.sam_frame0 import SamFrame0Params, run_sam_frame0


@dataclass
class PreprocessResult:
    video_id: str
    paths: VideoPaths
    manifest_path: Path
    success: bool


def _should_skip(path: Path, cfg: CustomConfig) -> bool:
    if cfg.preprocess.pipeline.force:
        return False
    if not cfg.preprocess.pipeline.skip_existing:
        return False
    return path.is_file()


def run_preprocess(
    mp4: Path,
    cfg: CustomConfig,
    video_id: str,
    ui: PreprocessConsole,
) -> PreprocessResult:
    mp4 = mp4.resolve()
    if not mp4.is_file():
        raise FileNotFoundError(f"MP4 not found: {mp4}")

    paths = resolve_video_paths(cfg, video_id)
    ensure_video_dirs(paths)

    import torch

    cuda_label = (
        f"CUDA — {torch.cuda.get_device_name(0)}"
        if torch.cuda.is_available()
        else "CPU (CUDA unavailable)"
    )
    try:
        est = probe_mp4_frame_count(mp4)
    except Exception:
        est = None

    ui.print_header(
        mp4=mp4,
        video_id=video_id,
        config_path=cfg.config_path or Path("configs/custom_default.yaml"),
        output_root=paths.video_root,
        cuda_label=cuda_label,
        est_frames=est,
    )

    manifest: dict = {
        "video_id": video_id,
        "mp4": str(mp4),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": cfg.config_fingerprint(),
        "stages": {},
    }

    # ── [1/5] Extract RGB frames ──────────────────────────────────────────
    rec_frames = ui.numbered_stage(1, "Extract RGB frames")
    ui.begin_stage(rec_frames, "decode MP4 → JPEG sequence")
    frame0 = paths.rgb_frames_dir / cfg.preprocess.sam_frame0.frame_name

    if _should_skip(frame0, cfg) and any(paths.rgb_frames_dir.glob("*.jpg")):
        ui.end_stage_skipped(rec_frames, paths.rgb_frames_dir, "frame folder exists")
        manifest["stages"]["frames"] = {"status": "skipped"}
    else:
        t0 = time.perf_counter()

        def on_prog(n: int, _total: int) -> None:
            ui.progress_update(n)

        ui.progress_start("Writing frames", total=None)
        try:
            fr = extract_frames_from_mp4(
                mp4,
                paths.rgb_frames_dir,
                max_len=cfg.preprocess.frames.max_len,
                target_fps=cfg.preprocess.frames.target_fps,
                skip_frames=cfg.preprocess.frames.skip_frames,
                max_res=cfg.preprocess.frames.max_res,
                jpg_quality=cfg.preprocess.frames.jpg_quality,
                frame_name_pattern=cfg.preprocess.frames.frame_name_pattern,
                on_progress=on_prog,
            )
        finally:
            ui.progress_stop()
        frame_elapsed = time.perf_counter() - t0
        ui.timing_result("Frame export", frame_elapsed, frames=fr.num_frames)
        ui.end_stage_success(
            rec_frames,
            duration_s=frame_elapsed,
            fps=None,
            detail="",
            output_path=paths.rgb_frames_dir,
        )
        manifest["stages"]["frames"] = {
            "status": "success",
            "num_frames": fr.num_frames,
            "fps": fr.fps,
            "elapsed_seconds": fr.elapsed_seconds,
        }

    # ── [2/5] Depth · [3/5] RAFT motion (single VDA+RAFT run, split UI) ───
    rec_depth = ui.numbered_stage(2, "Depth estimation")
    rec_flow = ui.numbered_stage(3, "RAFT 3D motion")

    if _should_skip(paths.npz_path, cfg):
        ui.begin_stage(rec_depth, "Video-Depth-Anything + metric depth")
        ui.end_stage_skipped(rec_depth, paths.npz_path, "included in existing NPZ")
        ui.begin_stage(rec_flow, "optical flow + NPZ")
        ui.end_stage_skipped(rec_flow, paths.npz_path, "NPZ already exists")
        manifest["stages"]["motion_3d"] = {"status": "skipped"}
    else:
        mparams = Motion3DParams(
            encoder=cfg.preprocess.motion_3d.encoder,
            input_size=cfg.preprocess.motion_3d.input_size,
            max_res=cfg.preprocess.motion_3d.max_res,
            max_len=cfg.preprocess.motion_3d.max_len,
            target_fps=cfg.preprocess.motion_3d.target_fps,
            fp32=cfg.preprocess.motion_3d.fp32,
            device=cfg.preprocess.motion_3d.device,
            skip_frames=cfg.preprocess.motion_3d.skip_frames,
            subsample=cfg.preprocess.motion_3d.subsample,
        )

        depth_closed = False

        def on_motion_step(step_name: str) -> None:
            nonlocal depth_closed
            if step_name == "depth_vda":
                ui.begin_stage(rec_depth, "Video-Depth-Anything inference")
                ui.status("Estimating per-frame depth maps")
                ui.progress_start("Depth inference", total=None)
            elif step_name == "depth_post":
                ui.progress_stop()
            elif step_name == "load_rgb" and not depth_closed:
                depth_closed = True
                ui.begin_stage(rec_flow, "optical flow → 3D points → NPZ")
            elif step_name == "optical_flow":
                pass

        flow_started = [False]

        def on_flow_progress(cur: int, total: int) -> None:
            if not flow_started[0]:
                flow_started[0] = True
                ui.progress_start("RAFT optical flow", total=total)
            ui.progress_update(cur, total)

        try:
            motion = run_motion_3d(
                mp4,
                paths.npz_path,
                mparams,
                on_step_start=on_motion_step,
                on_frame_progress=on_flow_progress,
                verbose=False,
                quiet=True,
            )
        finally:
            ui.progress_stop()

        tm = motion.timings
        depth_duration = tm.depth_seconds + tm.depth_post_seconds

        ui.timing_result(
            "Depth (VDA)",
            tm.depth_seconds,
            frames=tm.num_frames,
        )
        ui.timing_result("Depth post-process", tm.depth_post_seconds)
        ui.end_stage_success(
            rec_depth,
            duration_s=depth_duration,
            fps=None,
            detail="",
        )

        flow_duration = (
            tm.load_rgb_seconds + tm.flow_seconds + tm.save_seconds
        )
        ui.timing_result("Load RGB (flow align)", tm.load_rgb_seconds)
        ui.timing_result(
            "Optical flow (RAFT)",
            tm.flow_seconds,
            frames=tm.num_frame_pairs,
            unit="pairs",
        )
        ui.timing_result("Save 3D motion NPZ", tm.save_seconds)
        ui.end_stage_success(
            rec_flow,
            duration_s=flow_duration,
            fps=None,
            detail="",
            output_path=motion.output_path,
        )
        manifest["stages"]["depth"] = {
            "status": "success",
            "timings": {
                "depth_seconds": tm.depth_seconds,
                "depth_post_seconds": tm.depth_post_seconds,
                "depth_fps": tm.depth_fps,
                "num_frames": tm.num_frames,
            },
        }
        manifest["stages"]["motion_3d"] = {
            "status": "success",
            "timings": asdict(tm),
            "file_size_bytes": motion.file_size_bytes,
            "output": str(motion.output_path),
        }
        release_gpu()
        ui.vram_note()

    # ── [4/5] DINO features ───────────────────────────────────────────────
    rec_dino = ui.numbered_stage(4, "DINO features")
    ui.begin_stage(rec_dino, "DINOv2 + PCA per pixel")

    if _should_skip(paths.dino_path, cfg):
        ui.end_stage_skipped(rec_dino, paths.dino_path, "DINO NPZ already exists")
        manifest["stages"]["dino"] = {"status": "skipped"}
    else:
        dparams = DinoParams(
            target_h=cfg.preprocess.dino.target_h,
            target_w=cfg.preprocess.dino.target_w,
            patch_size=cfg.preprocess.dino.patch_size,
            n_components=cfg.preprocess.dino.n_components,
            device=cfg.preprocess.dino.device,
        )
        ui.status("Loading DINOv2 weights")
        model, device = load_dino_model(dparams.device)

        def on_ext(cur: int, tot: int) -> None:
            if cur == 1:
                ui.progress_start("Extracting patch features", total=tot)
            ui.progress_update(cur, tot)

        def on_save(cur: int, tot: int) -> None:
            if cur == 1:
                ui.progress_start("PCA resize to pixel grid", total=tot)
            ui.progress_update(cur, tot)

        t0 = time.perf_counter()
        try:
            dino = run_dino_extraction(
                video_id,
                paths.rgb_frames_dir,
                paths.dino_path.parent,
                model,
                device,
                dparams,
                on_extract_frame=on_ext,
                on_save_frame=on_save,
            )
        finally:
            ui.progress_stop()
            del model
            release_gpu()
            ui.vram_note()

        dt = dino.timings
        ui.timing_result("Load frames", dt.load_frames_seconds, frames=dt.num_frames)
        ui.timing_result(
            "Feature extraction (DINO)",
            dt.extract_seconds,
            frames=dt.num_frames,
        )
        ui.timing_result("PCA fit", dt.pca_seconds)
        ui.timing_result(
            "Resize PCA → pixel grid",
            dt.resize_save_seconds,
            frames=dt.num_frames,
        )
        ui.end_stage_success(
            rec_dino,
            duration_s=dino.elapsed_seconds,
            fps=None,
            detail=f"{dino.file_size_bytes / 1e6:.1f} MB NPZ",
            output_path=dino.output_path,
        )
        manifest["stages"]["dino"] = {
            "status": "success",
            "num_frames": dino.num_frames,
            "fps": dino.fps,
            "elapsed_seconds": dino.elapsed_seconds,
            "file_size_bytes": dino.file_size_bytes,
            "output": str(dino.output_path),
            "timings": asdict(dt),
        }

    # ── [5/5] SAM frame 0 ─────────────────────────────────────────────────
    rec_sam = ui.numbered_stage(5, "SAM frame 0")
    ui.begin_stage(rec_sam, "Ultralytics SAM2 segmentation mask")

    frame_path = paths.rgb_frames_dir / cfg.preprocess.sam_frame0.frame_name
    if not frame_path.is_file():
        raise FileNotFoundError(f"Frame 0 not found for SAM: {frame_path}")

    if _should_skip(paths.sam_path, cfg):
        ui.end_stage_skipped(rec_sam, paths.sam_path, "SAM PNG already exists")
        manifest["stages"]["sam_frame0"] = {"status": "skipped"}
    else:
        sparams = SamFrame0Params(
            model=cfg.preprocess.sam_frame0.model,
            min_threshold=cfg.preprocess.sam_frame0.min_threshold,
            seed=cfg.preprocess.sam_frame0.seed,
            frame_name=cfg.preprocess.sam_frame0.frame_name,
        )
        ui.status("Loading SAM2 weights")
        from ultralytics import SAM

        weights_dir = cfg.resolved_weights_dir()
        from genmatter.preprocessing.sam_frame0 import resolve_model_weights

        sam_model = SAM(resolve_model_weights(sparams.model, weights_dir))
        t0 = time.perf_counter()
        sam = run_sam_frame0(
            video_id,
            frame_path,
            paths.sam_path.parent,
            weights_dir,
            sparams,
            model=sam_model,
        )
        del sam_model
        release_gpu()
        ui.vram_note()
        ui.timing_result("SAM inference (frame 0)", sam.elapsed_seconds, frames=1, unit="frame")
        ui.end_stage_success(
            rec_sam,
            duration_s=sam.elapsed_seconds,
            fps=None,
            detail=f"{sam.file_size_bytes / 1024:.1f} KB PNG",
            output_path=sam.output_path,
        )
        manifest["stages"]["sam_frame0"] = {
            "status": "success",
            "elapsed_seconds": sam.elapsed_seconds,
            "file_size_bytes": sam.file_size_bytes,
            "output": str(sam.output_path),
        }

    snap = gpu_memory_snapshot()
    if snap:
        manifest["final_gpu_gib_used"] = snap[0]
        manifest["final_gpu_gib_total"] = snap[1]

    paths.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ui.print_manifest_footer(paths.manifest_path)
    ui.print_summary()

    failed = any(s.status == "failed" for s in ui.stages)
    return PreprocessResult(
        video_id=video_id,
        paths=paths,
        manifest_path=paths.manifest_path,
        success=not failed,
    )

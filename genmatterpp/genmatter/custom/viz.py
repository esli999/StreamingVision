"""Orchestrate export of tracking results to Rerun .rrd."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from genmatter.custom.config_schema import CustomConfig
from genmatter.custom.console import PreprocessConsole
from genmatter.custom.paths import VideoPaths, resolve_video_paths
from genmatter.viz.artifacts import load_viz_artifacts
from genmatter.viz.rerun_export import export_to_rrd


@dataclass
class VizResult:
    video_id: str
    paths: VideoPaths
    rrd_path: Path
    success: bool
    file_size_bytes: int = 0


def _should_skip(paths: VideoPaths, cfg: CustomConfig) -> bool:
    if cfg.viz.pipeline.force:
        return False
    if not cfg.viz.pipeline.skip_existing:
        return False
    return paths.rrd_path.is_file()


def _validate_tracking(paths: VideoPaths) -> None:
    if not paths.tracking_npz.is_file():
        raise FileNotFoundError(
            f"Missing {paths.tracking_npz}. "
            f"Run: uv run genmatter track --video-id {paths.video_id}"
        )
    if paths.track_manifest_path.is_file():
        manifest = json.loads(paths.track_manifest_path.read_text(encoding="utf-8"))
        dense = manifest.get("stages", {}).get("dense", {})
        if dense.get("status") not in ("success", "skipped"):
            raise RuntimeError(
                "track_manifest.json indicates tracking did not complete successfully"
            )


def run_viz(
    video_id: str,
    cfg: CustomConfig,
    ui: PreprocessConsole,
    *,
    output_path: Path | None = None,
) -> VizResult:
    paths = resolve_video_paths(cfg, video_id)
    rrd_path = (output_path or paths.rrd_path).resolve()

    ui.print_header(
        mp4=paths.video_root / f"{video_id}.mp4",
        video_id=video_id,
        config_path=cfg.config_path or Path("configs/custom_default.yaml"),
        output_root=paths.video_root,
        cuda_label="CPU (Rerun export)",
        est_frames=None,
    )

    rec_validate = ui.numbered_viz_stage(1, "Validate artifacts")
    ui.begin_stage(rec_validate)
    _validate_tracking(paths)
    ui.end_stage_success(rec_validate, duration_s=0.0, detail="tracking_dense.npz present")

    if _should_skip(paths, cfg):
        rec_skip = ui.numbered_viz_stage(2, "Export to Rerun")
        ui.begin_stage(rec_skip)
        ui.end_stage_skipped(rec_skip, rrd_path, ".rrd already exists")
        ui.print_manifest_footer(rrd_path, title="Viz complete (skipped)")
        return VizResult(
            video_id=video_id,
            paths=paths,
            rrd_path=rrd_path,
            success=True,
            file_size_bytes=rrd_path.stat().st_size,
        )

    rec_load = ui.numbered_viz_stage(2, "Load artifacts")
    ui.begin_stage(rec_load, "NPZ + RGB frames")
    t0 = time.perf_counter()
    artifacts = load_viz_artifacts(paths)
    load_s = time.perf_counter() - t0
    ui.timing_result(
        "Load tensors",
        load_s,
        frames=artifacts.num_frames,
    )
    ui.end_stage_success(rec_load, duration_s=load_s, detail=f"{artifacts.height}×{artifacts.width}")

    rec_export = ui.numbered_viz_stage(3, "Export frames to Rerun")
    ui.begin_stage(rec_export, "dense logging per frame")

    def on_prog(cur: int, tot: int) -> None:
        if cur == 1:
            ui.progress_start("Logging frames", total=tot)
        ui.progress_update(cur, tot)

    t0 = time.perf_counter()
    try:
        size = export_to_rrd(
            artifacts,
            cfg.viz,
            rrd_path,
            on_progress=on_prog,
        )
    finally:
        ui.progress_stop()
    export_s = time.perf_counter() - t0
    ui.timing_result(
        "Export",
        export_s,
        frames=artifacts.num_frames,
    )
    ui.end_stage_success(
        rec_export,
        duration_s=export_s,
        fps=artifacts.num_frames / export_s if export_s > 0 else None,
        detail=f"{size / 1e6:.1f} MB",
        output_path=rrd_path,
    )

    rec_done = ui.numbered_viz_stage(4, "Save recording")
    ui.begin_stage(rec_done)
    manifest = {
        "video_id": video_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": cfg.config_fingerprint(),
        "rrd_path": str(rrd_path),
        "file_size_bytes": size,
        "num_frames": artifacts.num_frames,
        "img_dims": [artifacts.height, artifacts.width],
    }
    manifest_path = paths.tracking_dir / "viz_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ui.end_stage_success(rec_done, duration_s=0.0, output_path=rrd_path)

    ui.print_summary()
    ui.print_manifest_footer(rrd_path, title="Viz complete — open with: rerun " + str(rrd_path))

    return VizResult(
        video_id=video_id,
        paths=paths,
        rrd_path=rrd_path,
        success=True,
        file_size_bytes=size,
    )

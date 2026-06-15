"""CLI orchestration for pseudo-GT SAM build."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from genmatter.custom.config_schema import CustomConfig
from genmatter.custom.console import PreprocessConsole
from genmatter.pseudo_gt.build import (
    build_pseudo_gt,
    colorize_pseudo_gt_segmasks,
    pseudo_gt_config_from_yaml,
)


@dataclass
class PseudoGtCmdResult:
    video_id: str
    manifest_path: Path
    success: bool


def run_pseudo_gt(
    video_id: str,
    cfg: CustomConfig,
    ui: PreprocessConsole,
    *,
    force: bool = False,
    colorize_only: bool = False,
) -> PseudoGtCmdResult:
    ui.print_header(
        mp4=Path(video_id),
        video_id=video_id,
        config_path=cfg.config_path or Path("configs/custom_default.yaml"),
        output_root=cfg.resolved_custom_videos_root() / video_id,
        cuda_label="SAM2",
        est_frames=None,
    )
    if colorize_only:
        try:
            color_dir = colorize_pseudo_gt_segmasks(video_id, cfg)
        except Exception:
            ui.console.print_exception()
            return PseudoGtCmdResult(video_id=video_id, manifest_path=Path(), success=False)
        ui.print_manifest_footer(color_dir.parent.parent / "manifest.json", title="Pseudo-GT colorized")
        ui.console.print(f"[green]Wrote[/] {color_dir} (consistent colors per track id)")
        return PseudoGtCmdResult(
            video_id=video_id,
            manifest_path=color_dir.parent.parent / "manifest.json",
            success=True,
        )

    pg_settings = cfg.pseudo_gt
    rec = ui.numbered_track_stage(1, "Build pseudo-GT (SAM2)")
    ui.begin_stage(
        rec,
        f"SAM2 segment all frames ({pg_settings.model}; ~3–5 s/frame on GPU)",
    )
    progress_started = False

    def on_segment_progress(cur: int, tot: int) -> None:
        nonlocal progress_started
        if not progress_started:
            ui.progress_start("SAM2 segmentation", total=tot)
            progress_started = True
        ui.progress_update(cur, tot)

    try:
        pg = pseudo_gt_config_from_yaml(cfg)
        pg.force = force
        result = build_pseudo_gt(
            video_id,
            cfg,
            pg,
            on_segment_progress=on_segment_progress,
        )
    except Exception:
        ui.progress_stop()
        ui.console.print_exception()
        return PseudoGtCmdResult(video_id=video_id, manifest_path=Path(), success=False)
    finally:
        ui.progress_stop()

    ui.end_stage_success(
        rec,
        duration_s=0.0,
        detail=f"{result.num_frames} frames, method={result.method}",
    )
    ui.print_manifest_footer(result.manifest_path, title="Pseudo-GT complete")
    return PseudoGtCmdResult(
        video_id=video_id,
        manifest_path=result.manifest_path,
        success=result.success,
    )

"""Rich console UI for custom preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

StageStatus = Literal["pending", "running", "success", "skipped", "failed"]

PREPROCESS_STAGE_COUNT = 5
TRACK_STAGE_COUNT = 6
VIZ_STAGE_COUNT = 4


@dataclass
class StageRecord:
    name: str
    status: StageStatus = "pending"
    duration_s: float = 0.0
    fps: float | None = None
    detail: str = ""
    output_path: str = ""


@dataclass
class PreprocessConsole:
    console: Console = field(default_factory=Console)
    stages: list[StageRecord] = field(default_factory=list)
    _progress: Progress | None = None
    _task: TaskID | None = None

    def print_header(
        self,
        mp4: Path,
        video_id: str,
        config_path: Path,
        output_root: Path,
        cuda_label: str,
        est_frames: int | None,
    ) -> None:
        lines = [
            f"[bold]Input[/]     {mp4}",
            f"[bold]Video ID[/]  [cyan]{video_id}[/]",
            f"[bold]Config[/]   {config_path}",
            f"[bold]Output[/]   {output_root}",
            f"[bold]Device[/]   {cuda_label}",
        ]
        if est_frames is not None and est_frames >= 0:
            lines.append(f"[bold]Frames[/]   ~{est_frames} in source MP4")
        self.console.print(
            Panel(
                "\n".join(lines),
                title="GenMatter Preprocess",
                border_style="blue",
                box=box.ROUNDED,
            )
        )

    def stage(self, name: str) -> StageRecord:
        rec = StageRecord(name=name)
        self.stages.append(rec)
        return rec

    def numbered_stage(
        self, index: int, name: str, *, total: int | None = None
    ) -> StageRecord:
        """Register a pipeline stage as ``[i/N] name``."""
        n = total if total is not None else PREPROCESS_STAGE_COUNT
        return self.stage(f"[{index}/{n}] {name}")

    def numbered_track_stage(self, index: int, name: str) -> StageRecord:
        """Register a tracking stage as ``[i/6] name``."""
        return self.numbered_stage(index, name, total=TRACK_STAGE_COUNT)

    def numbered_viz_stage(self, index: int, name: str) -> StageRecord:
        """Register a viz stage as ``[i/4] name``."""
        return self.numbered_stage(index, name, total=VIZ_STAGE_COUNT)

    def begin_stage(self, rec: StageRecord, subtitle: str = "") -> None:
        rec.status = "running"
        self.progress_stop()
        title = rec.name
        if subtitle:
            title = f"{title} — {subtitle}"
        self.console.print()
        self.console.rule(f"[bold cyan]▶ {title}[/]", style="cyan")

    def substep(self, message: str) -> None:
        """Indented line inside the current stage."""
        self.console.print(f"    [dim]›[/] {message}")

    def timing_result(
        self,
        label: str,
        seconds: float,
        *,
        frames: int | None = None,
        unit: str = "frames",
    ) -> None:
        """Print one substeps timing line; FPS when *frames* is set."""
        if seconds < 0:
            seconds = 0.0
        if frames is not None and frames > 0 and seconds > 0:
            fps = frames / seconds
            self.console.print(
                f"    [dim]›[/] {label:<24} [white]{seconds:>7.2f}s[/]  "
                f"[green]{fps:>7.2f} FPS[/]  [dim]({frames} {unit})[/]"
            )
        else:
            self.console.print(
                f"    [dim]›[/] {label:<24} [white]{seconds:>7.2f}s[/]"
            )

    def status(self, message: str) -> None:
        """Short status line under the stage header."""
        self.console.print(f"  [dim]{message}[/]")

    def end_stage_success(
        self,
        rec: StageRecord,
        *,
        duration_s: float,
        fps: float | None = None,
        detail: str = "",
        output_path: Path | str = "",
    ) -> None:
        rec.status = "success"
        rec.duration_s = duration_s
        rec.fps = fps
        rec.detail = detail
        rec.output_path = str(output_path)
        self.progress_stop()
        if fps is not None:
            self.console.print(
                f"  [green]✓[/]  [dim]{duration_s:.2f}s total[/]  "
                f"[green]{fps:.2f} FPS[/]  {detail}"
            )
        else:
            tail = f"  {detail}" if detail else ""
            self.console.print(f"  [green]✓[/]  [dim]{duration_s:.2f}s total[/]{tail}")
        if output_path:
            self.console.print(f"    [dim]→[/] [cyan]{output_path}[/]")

    def end_stage_skipped(self, rec: StageRecord, output_path: Path | str, reason: str) -> None:
        rec.status = "skipped"
        rec.output_path = str(output_path)
        rec.detail = reason
        self.progress_stop()
        self.console.print(f"  [yellow]⊘ skipped[/] — {reason}")
        if output_path:
            self.console.print(f"    [dim]→[/] [cyan]{output_path}[/]")

    def fail_stage(self, rec: StageRecord, error: Exception) -> None:
        rec.status = "failed"
        rec.detail = str(error)
        self.progress_stop()
        self.console.print(f"  [red]✗ failed[/]: {error}")

    def vram_note(self, label: str = "GPU memory after release") -> None:
        from genmatter.preprocessing.gpu import gpu_memory_snapshot

        snap = gpu_memory_snapshot()
        if snap is None:
            return
        used, total = snap
        self.console.print(f"  [dim]{label}:[/] {used:.2f} / {total:.2f} GiB used")

    def progress_start(self, description: str, total: int | None = None) -> None:
        self.progress_stop()
        cols = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ]
        self._progress = Progress(*cols, console=self.console, transient=False)
        self._progress.start()
        self._task = self._progress.add_task(description, total=total)

    def progress_update(self, completed: int, total: int | None = None) -> None:
        if self._progress is None or self._task is None:
            return
        kwargs: dict = {"completed": completed}
        if total is not None:
            kwargs["total"] = total
        self._progress.update(self._task, **kwargs)

    def progress_stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task = None

    def print_manifest_footer(
        self, manifest_path: Path, *, title: str = "Preprocess complete"
    ) -> None:
        self.console.print()
        self.console.print(
            Panel(
                f"[cyan]{manifest_path}[/]",
                title=f"[green]{title}[/]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

    def print_summary(self) -> None:
        table = Table(
            title="Preprocess summary",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Stage", style="bold")
        table.add_column("Status")
        table.add_column("Time (s)", justify="right")
        table.add_column("FPS", justify="right")
        table.add_column("Output / detail")

        status_style = {
            "success": "green",
            "skipped": "yellow",
            "failed": "red",
            "running": "cyan",
            "pending": "dim",
        }
        for rec in self.stages:
            st = Text(rec.status, style=status_style.get(rec.status, "white"))
            fps = f"{rec.fps:.2f}" if rec.fps is not None else "—"
            out = rec.output_path if rec.output_path else rec.detail
            table.add_row(rec.name, st, f"{rec.duration_s:.2f}", fps, out)

        self.console.print()
        self.console.print(table)

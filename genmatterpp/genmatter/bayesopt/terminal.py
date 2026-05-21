"""Rich terminal UI for custom-video BayesOpt."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class BayesOptConsole:
    def __init__(self, *, plain: bool | None = None) -> None:
        self.plain = not sys.stdout.isatty() if plain is None else plain
        self.console = Console()
        self._live: Live | None = None

    def start(self) -> None:
        if self.plain:
            return
        self._live = Live(self._render({}), console=self.console, refresh_per_second=4)
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def update(self, status: dict[str, Any]) -> None:
        if self.plain:
            return
        if self._live is not None:
            self._live.update(self._render(status))

    def log_trial(self, status: dict[str, Any]) -> None:
        """One-line status for background / nohup logs."""
        if not self.plain:
            return
        self.console.print(
            f"[trial {status.get('trial_index', '?')}] "
            f"phase={status.get('phase', '?')} "
            f"status={status.get('last_status', '?')} "
            f"mean_gt_iou={float(status.get('last_objective', 0)):.4f} "
            f"outlier%={status.get('last_outlier_pct', float('nan'))} "
            f"elapsed={float(status.get('last_elapsed_s', 0)):.1f}s "
            f"best={float(status.get('best_objective', float('nan'))):.4f}"
        )

    def print(self, message: str) -> None:
        self.console.print(message)

    def _render(self, status: dict[str, Any]) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("Phase", str(status.get("phase", "—")))
        table.add_row("Trial", str(status.get("trial_index", "—")))
        table.add_row("Last status", str(status.get("last_status", "—")))
        table.add_row("Last mean GT IoU", f"{status.get('last_objective', float('nan')):.4f}")
        out_pct = status.get("last_outlier_pct")
        if out_pct is not None:
            table.add_row(
                "Last outlier %",
                f"{float(out_pct):.2f} (max {float(status.get('max_outlier_pct', 10)):.1f})",
            )
        table.add_row("Invalid trials", str(status.get("invalid_count", 0)))
        table.add_row("Best mean GT IoU", f"{status.get('best_objective', float('nan')):.4f}")
        table.add_row("Elapsed (last)", f"{status.get('last_elapsed_s', 0):.1f}s")
        params = status.get("current_params") or status.get("best_params") or {}
        if params:
            pstr = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())
            table.add_row("Params", pstr[:120])
        return Panel(table, title="GenMatter BayesOpt", border_style="cyan")

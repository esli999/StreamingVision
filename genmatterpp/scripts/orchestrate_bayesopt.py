#!/usr/bin/env python3
"""Start BayesOpt, monitor first N trials, then detach for long run.

Writes status to logs/orchestrate_status.json for agents without shell output.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VIDEO_ID = os.environ.get("GENMATTER_VIDEO_ID", "original_244622072067")
SMOKE_TRIALS = int(os.environ.get("GENMATTER_SMOKE_TRIALS", "4"))
LOG_DIR = REPO / "logs"
STATUS_PATH = LOG_DIR / "orchestrate_status.json"
BAYESOPT_LOG = LOG_DIR / f"bayesopt_{VIDEO_ID}.log"
BAYESOPT_PID = LOG_DIR / f"bayesopt_{VIDEO_ID}.pid"


def _status(**kwargs: object) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), **kwargs}
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _count_segmasks() -> int:
    root = (
        REPO
        / "assets/custom_videos"
        / VIDEO_ID
        / "pseudo_gt_sam/segmasks"
        / VIDEO_ID
    )
    return len(list(root.glob("*.png"))) if root.is_dir() else 0


def _latest_run_dir() -> Path | None:
    base = REPO / "assets/custom_videos" / VIDEO_ID / "bayesopt"
    if not base.is_dir():
        return None
    runs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _trial_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "trials.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _validate_trial(row: dict) -> list[str]:
    """Return issues only for true failures (not outlier-gated invalid trials)."""
    issues: list[str] = []
    st = row.get("status")
    if st == "failed":
        issues.append(f"trial {row.get('trial_index')} failed: {row.get('error', row)}")
    elif st == "completed":
        if "elapsed_seconds" not in row:
            issues.append(f"trial {row.get('trial_index')} missing elapsed_seconds")
    return issues


def _start_bayesopt(*, resume: bool, run_id: str | None = None) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if BAYESOPT_PID.is_file():
        old = int(BAYESOPT_PID.read_text(encoding="utf-8").strip())
        try:
            os.kill(old, 0)
            raise RuntimeError(f"bayesopt already running (pid {old})")
        except OSError:
            BAYESOPT_PID.unlink(missing_ok=True)
    subprocess.run(
        ["uv", "sync", "--extra", "bayesopt"],
        cwd=str(REPO),
        check=True,
        capture_output=True,
        text=True,
    )
    cmd = [
        "uv",
        "run",
        "--extra",
        "bayesopt",
        "genmatter",
        "bayesopt",
        "--video-id",
        VIDEO_ID,
        "--config",
        str(REPO / "configs/custom_default.yaml"),
        "--bayesopt-config",
        str(REPO / "configs/custom_bayesopt.yaml"),
    ]
    if run_id:
        cmd.extend(["--run-id", run_id])
    if not resume:
        cmd.append("--no-resume")
    log_f = open(BAYESOPT_LOG, "a", encoding="utf-8")
    log_f.write(f"\n--- started {datetime.now(timezone.utc).isoformat()} ---\n")
    log_f.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    BAYESOPT_PID.write_text(str(proc.pid), encoding="utf-8")
    return proc


def main() -> int:
    n_masks = _count_segmasks()
    manifest = (
        REPO
        / "assets/custom_videos"
        / VIDEO_ID
        / "pseudo_gt_sam/manifest.json"
    )
    if not manifest.is_file() or n_masks < 1:
        _status(
            phase="error",
            error=f"pseudo-GT incomplete: manifest={manifest.is_file()} segmasks={n_masks}",
        )
        return 1

    _status(phase="starting", pseudo_gt_frames=n_masks, smoke_trials=SMOKE_TRIALS)
    proc = _start_bayesopt(resume=True)
    run_dir: Path | None = None

    for poll in range(360):  # up to 6h for first smoke trials
        time.sleep(30)
        if proc.poll() is not None:
            tail = BAYESOPT_LOG.read_text(encoding="utf-8")[-4000:] if BAYESOPT_LOG.is_file() else ""
            _status(phase="crashed", exit_code=proc.returncode, log_tail=tail)
            return 2

        run_dir = _latest_run_dir()
        if run_dir is None:
            _status(phase="waiting_for_run_dir", poll=poll, pid=proc.pid)
            continue

        rows = _trial_rows(run_dir)
        _status(
            phase="smoke_monitor",
            poll=poll,
            pid=proc.pid,
            run_dir=str(run_dir),
            trials_completed=len(rows),
            last_trial=rows[-1] if rows else None,
        )

        if len(rows) >= SMOKE_TRIALS:
            smoke = rows[:SMOKE_TRIALS]
            issues: list[str] = []
            for row in smoke:
                issues.extend(_validate_trial(row))
            completed = [r for r in smoke if r.get("status") == "completed"]
            invalid = [r for r in smoke if r.get("status") == "invalid"]
            if issues:
                _status(phase="smoke_failed", issues=issues, trials=smoke)
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                return 3
            if not completed:
                _status(
                    phase="smoke_failed",
                    issues=[f"no completed trials in first {SMOKE_TRIALS}"],
                    trials=smoke,
                )
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                return 3
            _status(
                phase="smoke_ok",
                message=(
                    f"First {SMOKE_TRIALS} trials OK "
                    f"({len(completed)} completed, {len(invalid)} invalid); "
                    f"leaving pid {proc.pid} running"
                ),
                run_dir=str(run_dir),
                trials=smoke,
            )
            return 0

    _status(phase="smoke_timeout", pid=proc.pid, run_dir=str(run_dir))
    return 4


if __name__ == "__main__":
    sys.exit(main())

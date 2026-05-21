"""Extract numbered RGB frames from an MP4 for DINO / SAM preprocessing."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2


@dataclass
class ExtractFramesResult:
    output_dir: Path
    num_frames: int
    elapsed_seconds: float
    fps: float
    frame_paths: list[Path]


def probe_mp4_frame_count(mp4_path: Path) -> int:
    """Return approximate frame count without decoding all frames."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {mp4_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 0)


def extract_frames_from_mp4(
    mp4_path: Path,
    output_dir: Path,
    *,
    max_len: int = -1,
    target_fps: int = -1,
    skip_frames: int = 1,
    max_res: int = 1280,
    jpg_quality: int = 95,
    frame_name_pattern: str = "{:05d}.jpg",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> ExtractFramesResult:
    """
    Decode MP4 to numbered JPEGs under ``output_dir``.

    Frame indices follow the same skipping rules as VDA preprocessing.
    """
    mp4_path = Path(mp4_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {mp4_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1
    if target_fps > 0 and src_fps > 0:
        frame_interval = max(1, int(round(src_fps / target_fps)))

    t0 = time.perf_counter()
    frame_paths: list[Path] = []
    read_idx = 0
    kept_after_fps = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if read_idx % frame_interval != 0:
            read_idx += 1
            continue
        if skip_frames > 1 and (kept_after_fps % skip_frames) != 0:
            read_idx += 1
            kept_after_fps += 1
            continue

        if max_len > 0 and len(frame_paths) >= max_len:
            break

        h, w = frame_bgr.shape[:2]
        if max_res > 0 and max(h, w) > max_res:
            scale = max_res / max(h, w)
            frame_bgr = cv2.resize(
                frame_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        name = frame_name_pattern.format(len(frame_paths))
        out_path = output_dir / name
        cv2.imwrite(
            str(out_path),
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)],
        )
        frame_paths.append(out_path)

        if on_progress is not None:
            on_progress(len(frame_paths), -1)

        read_idx += 1
        kept_after_fps += 1

    cap.release()
    elapsed = time.perf_counter() - t0
    n = len(frame_paths)
    fps = n / elapsed if elapsed > 0 else 0.0
    return ExtractFramesResult(
        output_dir=output_dir,
        num_frames=n,
        elapsed_seconds=elapsed,
        fps=fps,
        frame_paths=frame_paths,
    )


def list_frame_files(
    frames_dir: Path,
    frame_name_pattern: str = "{:05d}.jpg",
) -> list[Path]:
    """Sorted frame paths in a directory (numeric order)."""

    def _key(p: Path) -> int:
        nums = re.findall(r"\d+", p.stem)
        return int(nums[0]) if nums else 0

    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in frames_dir.iterdir() if p.suffix.lower() in exts]
    return sorted(paths, key=_key)

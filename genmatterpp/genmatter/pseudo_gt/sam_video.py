"""SAM2 segmentation on RGB frame sequences."""

from __future__ import annotations

import gc
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from genmatter.preprocessing.sam_frame0 import resolve_model_weights
from genmatter.pseudo_gt.correspondence import (
    align_chunk_track_ids,
    merge_tracked_with_new_objects,
)

logger = logging.getLogger(__name__)


def read_frame_hw(frame_path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` for an RGB frame on disk."""
    bgr = cv2.imread(str(frame_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read frame: {frame_path}")
    h, w = bgr.shape[:2]
    return int(h), int(w)


def list_rgb_frames(rgb_dir: Path) -> list[Path]:
    paths = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No RGB frames in {rgb_dir}")
    return paths


def _cuda_release() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def resize_mask_to_hw(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a boolean mask to native frame resolution (same as per-frame SAM output)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == (height, width):
        return mask
    resized = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def _masks_from_ultralytics_result(
    result: Any,
    min_threshold: float,
    *,
    native_hw: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    if result.masks is None or len(result.masks.data) == 0:
        return []
    out: list[np.ndarray] = []
    for i in range(int(result.masks.data.shape[0])):
        m = result.masks.data[i].detach().cpu().numpy()
        mask = m > min_threshold
        if native_hw is not None:
            mask = resize_mask_to_hw(mask, native_hw[0], native_hw[1])
        out.append(mask)
    return out


def tracked_masks_from_result(
    result: Any,
    min_threshold: float,
    *,
    native_hw: tuple[int, int] | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract boolean masks and SAM2 video object IDs from one frame result."""
    if result.masks is None or result.boxes is None or len(result.masks.data) == 0:
        return [], []
    data = result.masks.data
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    masks: list[np.ndarray] = []
    track_ids: list[int] = []
    for i in range(int(data.shape[0])):
        m = data[i].detach().cpu().numpy()
        mask = m if m.dtype == bool else m > min_threshold
        if not np.asarray(mask, dtype=bool).any():
            continue
        if native_hw is not None:
            mask = resize_mask_to_hw(mask, native_hw[0], native_hw[1])
        masks.append(np.asarray(mask, dtype=bool))
        track_ids.append(int(cls[i]))
    order = np.argsort(track_ids)
    return [masks[i] for i in order], [track_ids[i] for i in order]


def select_bboxes_by_area(bboxes: np.ndarray, max_objects: int | None) -> np.ndarray:
    """Keep the largest ``max_objects`` boxes by area (reduces video-tracker VRAM)."""
    if max_objects is None or len(bboxes) <= max_objects:
        return bboxes
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    keep = np.argsort(areas)[-max_objects:]
    return bboxes[keep]


def write_frames_mp4(frame_paths: list[Path], mp4_path: Path, *, fps: int = 30) -> None:
    """Encode RGB frames to an MP4 for ``SAM2VideoPredictor``."""
    bgr0 = cv2.imread(str(frame_paths[0]))
    if bgr0 is None:
        raise FileNotFoundError(f"Could not read frame: {frame_paths[0]}")
    h, w = bgr0.shape[:2]
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {mp4_path}")
    try:
        for fp in frame_paths:
            bgr = cv2.imread(str(fp))
            if bgr is None:
                raise FileNotFoundError(f"Could not read frame: {fp}")
            if bgr.shape[0] != h or bgr.shape[1] != w:
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
            writer.write(bgr)
    finally:
        writer.release()


def _track_video_chunk(
    frame_paths: list[Path],
    bboxes: np.ndarray,
    *,
    weights_path: Path,
    imgsz: int,
    min_threshold: float,
    id_offset: int,
    native_hw: tuple[int, int],
) -> tuple[list[list[np.ndarray]], list[list[int]], np.ndarray]:
    """Run ``SAM2VideoPredictor`` on a short clip; return masks, IDs, and last-frame boxes."""
    from ultralytics.models.sam import SAM2VideoPredictor

    with tempfile.TemporaryDirectory(prefix="genmatter_sam2vid_") as tmp:
        mp4_path = Path(tmp) / "rgb.mp4"
        write_frames_mp4(frame_paths, mp4_path)
        predictor = SAM2VideoPredictor(
            overrides=dict(
                model=str(weights_path),
                task="segment",
                mode="predict",
                imgsz=imgsz,
            ),
        )
        try:
            stream = predictor(source=str(mp4_path), bboxes=bboxes, stream=True)
            results = list(stream)
        finally:
            del predictor
            _cuda_release()

    if len(results) != len(frame_paths):
        raise RuntimeError(
            f"SAM2 video chunk: expected {len(frame_paths)} frames, got {len(results)}"
        )

    per_frame_masks: list[list[np.ndarray]] = []
    per_frame_track_ids: list[list[int]] = []
    for r in results:
        masks, tids = tracked_masks_from_result(
            r, min_threshold, native_hw=native_hw
        )
        per_frame_masks.append(masks)
        per_frame_track_ids.append([tid + id_offset for tid in tids])

    last = results[-1]
    if last.boxes is None or len(last.boxes) == 0:
        raise RuntimeError("SAM2 video chunk ended with no boxes on last frame")
    last_bboxes = last.boxes.xyxy.detach().cpu().numpy()
    return per_frame_masks, per_frame_track_ids, last_bboxes


def segment_sam2_video_tracked(
    frame_paths: list[Path],
    *,
    model: str,
    weights_dir: Path,
    min_threshold: float,
    imgsz: int = 1024,
    max_objects: int | None = 12,
    chunk_frames: int = 16,
    iou_threshold: float = 0.3,
    on_progress: Callable[[int, int], None] | None = None,
    show_tqdm: bool = True,
) -> tuple[list[list[np.ndarray]], list[list[int]], str] | None:
    """
    Segment with ``SAM2VideoPredictor`` using frame-0 auto masks as bbox prompts.

    ``imgsz`` is the model inference size only; exported masks are always resized to the
    native RGB frame resolution (same as per-frame SAM). Long videos are processed in
    short chunks to cap GPU memory. Chunk boundaries are linked via Hungarian IoU matching.
    """
    from ultralytics import SAM

    weights_path = resolve_model_weights(model, weights_dir)
    native_hw = read_frame_hw(frame_paths[0])
    n = len(frame_paths)
    chunk_frames = max(2, int(chunk_frames))

    sam_init = SAM(weights_path)
    try:
        init = sam_init.predict(str(frame_paths[0]), imgsz=imgsz, verbose=False)[0]
    finally:
        del sam_init
        _cuda_release()

    if init.boxes is None or len(init.boxes) == 0:
        logger.warning("SAM2 video: no objects on frame 0 for %s", frame_paths[0])
        return None

    bboxes = select_bboxes_by_area(
        init.boxes.xyxy.detach().cpu().numpy(),
        max_objects,
    )
    if len(bboxes) < len(init.boxes):
        logger.info(
            "SAM2 video: tracking %d / %d frame-0 objects (max_objects=%s)",
            len(bboxes),
            len(init.boxes),
            max_objects,
        )

    all_masks: list[list[np.ndarray]] = []
    all_track_ids: list[list[int]] = []
    prev_masks: list[np.ndarray] | None = None
    prev_track_ids: list[int] | None = None
    frames_done = 0

    chunk_starts = range(0, n, chunk_frames)
    chunk_iter: Any = chunk_starts
    if on_progress is None and show_tqdm and n > 1:
        chunk_iter = tqdm(
            list(chunk_starts),
            desc="SAM2 video tracked",
            unit="chunk",
        )

    for start in chunk_iter:
        end = min(start + chunk_frames, n)
        chunk_paths = frame_paths[start:end]
        id_offset = (
            max(tid for frame in all_track_ids for tid in frame) + 1
            if all_track_ids
            else 0
        )

        chunk_masks, chunk_tids, bboxes = _track_video_chunk(
            chunk_paths,
            bboxes,
            weights_path=weights_path,
            imgsz=imgsz,
            min_threshold=min_threshold,
            id_offset=id_offset,
            native_hw=native_hw,
        )

        if prev_masks is not None and prev_track_ids is not None:
            chunk_tids = align_chunk_track_ids(
                prev_masks,
                prev_track_ids,
                chunk_tids,
                chunk_masks,
                iou_threshold=iou_threshold,
            )

        all_masks.extend(chunk_masks)
        all_track_ids.extend(chunk_tids)
        prev_masks = chunk_masks[-1]
        prev_track_ids = chunk_tids[-1]
        frames_done = end

        if on_progress is not None:
            on_progress(frames_done, n)

    if not any(all_masks):
        logger.warning("SAM2 video: empty masks for all frames")
        return None

    return all_masks, all_track_ids, "sam2_video_tracked"


def segment_frames_per_frame(
    frame_paths: list[Path],
    *,
    model: str,
    weights_dir: Path,
    min_threshold: float,
    sam_model: Any | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    show_tqdm: bool = True,
) -> tuple[list[list[np.ndarray]], str]:
    """Run SAM on each frame independently. Returns masks and method tag."""
    from ultralytics import SAM

    if sam_model is None:
        weights_path = resolve_model_weights(model, weights_dir)
        sam_model = SAM(weights_path)

    n = len(frame_paths)
    per_frame: list[list[np.ndarray]] = []
    iterator: Any = frame_paths
    if on_progress is None and show_tqdm and n > 1:
        iterator = tqdm(frame_paths, desc="SAM2 per-frame", unit="frame", total=n)
    native_hw = read_frame_hw(frame_paths[0])
    for i, fp in enumerate(iterator):
        r = sam_model(str(fp), verbose=False)[0]
        per_frame.append(
            _masks_from_ultralytics_result(r, min_threshold, native_hw=native_hw)
        )
        if on_progress is not None:
            on_progress(i + 1, n)
    return per_frame, "per_frame_sam"


def segment_video_stream(
    frame_paths: list[Path],
    *,
    model: str,
    weights_dir: Path,
    min_threshold: float,
    sam_model: Any | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    show_tqdm: bool = True,
) -> tuple[list[list[np.ndarray]], str] | None:
    """
    Try SAM on the frame directory as a video source.

    Returns None if the API does not yield per-frame mask lists.
    """
    from ultralytics import SAM

    if sam_model is None:
        weights_path = resolve_model_weights(model, weights_dir)
        sam_model = SAM(weights_path)

    source = str(frame_paths[0].parent)
    n = len(frame_paths)
    try:
        stream = sam_model.predict(source=source, stream=True, verbose=False)
        if on_progress is None and show_tqdm and n > 1:
            stream = tqdm(stream, desc="SAM2 video stream", unit="frame", total=n)
        results = []
        for i, r in enumerate(stream):
            results.append(r)
            if on_progress is not None:
                on_progress(i + 1, n)
    except Exception:
        return None

    if len(results) != len(frame_paths):
        return None

    per_frame: list[list[np.ndarray]] = []
    for r in results:
        per_frame.append(_masks_from_ultralytics_result(r, min_threshold))
    if not any(per_frame):
        return None
    return per_frame, "sam_video_stream"


def segment_sequence(
    rgb_dir: Path,
    *,
    model: str,
    weights_dir: Path,
    min_threshold: float,
    prefer_video: bool = True,
    video_imgsz: int = 1024,
    max_objects: int | None = 12,
    video_chunk_frames: int = 16,
    iou_threshold: float = 0.3,
    detect_new_objects: bool = True,
    new_object_iou_threshold: float | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    show_tqdm: bool = True,
) -> tuple[list[list[np.ndarray]], str, tuple[int, int], list[list[int]] | None]:
    """
    Segment all frames.

    When ``prefer_video`` is true, tries ``SAM2VideoPredictor`` first (temporal tracking).
    With ``detect_new_objects``, runs per-frame SAM and merges masks that do not overlap
    video tracks (objects entering after frame 0). Returns per-frame track IDs when
    video tracking succeeds, else ``None``.
    """
    frame_paths = list_rgb_frames(rgb_dir)
    h, w = read_frame_hw(frame_paths[0])
    n = len(frame_paths)
    birth_iou = (
        iou_threshold if new_object_iou_threshold is None else new_object_iou_threshold
    )

    if prefer_video:
        try:
            tracked = segment_sam2_video_tracked(
                frame_paths,
                model=model,
                weights_dir=weights_dir,
                min_threshold=min_threshold,
                imgsz=video_imgsz,
                max_objects=max_objects,
                chunk_frames=video_chunk_frames,
                iou_threshold=iou_threshold,
                on_progress=on_progress if not detect_new_objects else None,
                show_tqdm=show_tqdm and not detect_new_objects,
            )
        except Exception:
            logger.exception("SAM2 video tracking failed for %s", rgb_dir)
            tracked = None
        if tracked is not None:
            per_frame, track_ids, method = tracked
            if detect_new_objects:
                from ultralytics import SAM

                weights_path = resolve_model_weights(model, weights_dir)
                sam_model = SAM(weights_path)
                try:
                    sam_per_frame, _ = segment_frames_per_frame(
                        frame_paths,
                        model=model,
                        weights_dir=weights_dir,
                        min_threshold=min_threshold,
                        sam_model=sam_model,
                        on_progress=on_progress,
                        show_tqdm=show_tqdm,
                    )
                finally:
                    del sam_model
                    _cuda_release()

                per_frame, track_ids = merge_tracked_with_new_objects(
                    per_frame,
                    track_ids,
                    sam_per_frame,
                    new_object_iou_threshold=birth_iou,
                    link_iou_threshold=iou_threshold,
                )
                method = "sam2_video_tracked_hybrid"
                logger.info(
                    "SAM2 hybrid: merged video tracks with per-frame SAM new-object pass (%d frames)",
                    n,
                )
            return per_frame, method, (h, w), track_ids

    from ultralytics import SAM

    weights_path = resolve_model_weights(model, weights_dir)
    sam_model = SAM(weights_path)
    per_frame, method = segment_frames_per_frame(
        frame_paths,
        model=model,
        weights_dir=weights_dir,
        min_threshold=min_threshold,
        sam_model=sam_model,
        on_progress=on_progress,
        show_tqdm=show_tqdm,
    )
    return per_frame, method, (h, w), None

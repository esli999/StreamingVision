"""Mask IoU, Hungarian matching, and track-ID propagation for pseudo-GT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boolean masks."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def mask_iou_matrix(masks_a: list[np.ndarray], masks_b: list[np.ndarray]) -> np.ndarray:
    """Return ``(len(masks_a), len(masks_b))`` IoU matrix."""
    na, nb = len(masks_a), len(masks_b)
    out = np.zeros((na, nb), dtype=np.float64)
    for i in range(na):
        for j in range(nb):
            out[i, j] = mask_iou(masks_a[i], masks_b[j])
    return out


@dataclass
class FramePairCorrespondence:
    t: int
    matches: list[list[float]]  # [local_i, local_j, iou]
    births: list[int]  # local indices in frame t+1 unmatched
    deaths: list[int]  # local indices in frame t unmatched


def hungarian_link(
    iou_matrix: np.ndarray,
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """
    Match masks between consecutive frames.

    Returns (matches as (i, j, iou), unmatched_i, unmatched_j).
    """
    iou = np.asarray(iou_matrix, dtype=np.float64)
    n, m = iou.shape
    if n == 0 and m == 0:
        return [], [], []
    if n == 0:
        return [], [], list(range(m))
    if m == 0:
        return [], list(range(n)), []

    cost = 1.0 - iou
    row_ind, col_ind = linear_sum_assignment(cost)
    matches: list[tuple[int, int, float]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(row_ind, col_ind, strict=True):
        iou_val = float(iou[r, c])
        if iou_val >= iou_threshold:
            matches.append((int(r), int(c), iou_val))
            matched_rows.add(int(r))
            matched_cols.add(int(c))
    deaths = [i for i in range(n) if i not in matched_rows]
    births = [j for j in range(m) if j not in matched_cols]
    return matches, deaths, births


def label_map_from_masks(
    masks: list[np.ndarray],
    track_ids: list[int],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize instance masks to uint16 label map (0=background, id+1=label)."""
    out = np.zeros((height, width), dtype=np.uint16)
    for mask, tid in zip(masks, track_ids, strict=True):
        m = np.asarray(mask, dtype=bool)
        out[m] = np.uint16(tid + 1)
    return out


def propagate_track_ids(
    per_frame_masks: list[list[np.ndarray]],
    iou_threshold: float = 0.3,
) -> tuple[list[np.ndarray], list[FramePairCorrespondence], list[dict[str, Any]]]:
    """
    Assign global track IDs across frames via Hungarian IoU linking.

    Returns (label_maps per frame, frame_pair records, track metadata list).
    """
    if not per_frame_masks:
        return [], [], []

    h, w = per_frame_masks[0][0].shape if per_frame_masks[0] else (0, 0)
    if h == 0 or w == 0:
        raise ValueError("Empty mask list in propagate_track_ids")

    # frame 0: new track per mask
    current_track_ids: list[int] = []
    next_tid = 0
    for _ in per_frame_masks[0]:
        current_track_ids.append(next_tid)
        next_tid += 1

    label_maps: list[np.ndarray] = [
        label_map_from_masks(per_frame_masks[0], current_track_ids, h, w)
    ]
    frame_pairs: list[FramePairCorrespondence] = []
    tracks: dict[int, dict[str, Any]] = {
        tid: {"track_id": tid, "birth_frame": 0} for tid in current_track_ids
    }

    for t in range(len(per_frame_masks) - 1):
        masks_t = per_frame_masks[t]
        masks_t1 = per_frame_masks[t + 1]
        iou_mat = mask_iou_matrix(masks_t, masks_t1)
        matches, deaths, births = hungarian_link(iou_mat, iou_threshold)

        pair = FramePairCorrespondence(
            t=t,
            matches=[[float(i), float(j), iou] for i, j, iou in matches],
            births=[int(b) for b in births],
            deaths=[int(d) for d in deaths],
        )
        frame_pairs.append(pair)

        next_track_ids = [-1] * len(masks_t1)
        for i, j, _ in matches:
            next_track_ids[j] = current_track_ids[i]
            tracks[current_track_ids[i]]["last_frame"] = t + 1

        for j in births:
            next_track_ids[j] = next_tid
            tracks[next_tid] = {"track_id": next_tid, "birth_frame": t + 1, "last_frame": t + 1}
            next_tid += 1

        for d in deaths:
            tid = current_track_ids[d]
            tracks[tid]["death_frame"] = t + 1

        if any(tid < 0 for tid in next_track_ids):
            raise RuntimeError(f"Unassigned masks at frame pair {t}->{t+1}")
        current_track_ids = [int(tid) for tid in next_track_ids]
        label_maps.append(
            label_map_from_masks(masks_t1, current_track_ids, h, w)
        )

    track_list = [tracks[k] for k in sorted(tracks)]
    return label_maps, frame_pairs, track_list


def propagate_track_ids_lists(
    per_frame_masks: list[list[np.ndarray]],
    iou_threshold: float = 0.3,
    *,
    id_offset: int = 0,
) -> list[list[int]]:
    """Assign global track IDs across frames; returns per-frame ID lists only."""
    if not per_frame_masks:
        return []

    current_track_ids: list[int] = []
    next_tid = id_offset
    for _ in per_frame_masks[0]:
        current_track_ids.append(next_tid)
        next_tid += 1

    out: list[list[int]] = [list(current_track_ids)]
    for t in range(len(per_frame_masks) - 1):
        masks_t = per_frame_masks[t]
        masks_t1 = per_frame_masks[t + 1]
        iou_mat = mask_iou_matrix(masks_t, masks_t1)
        matches, _, births = hungarian_link(iou_mat, iou_threshold)

        next_track_ids = [-1] * len(masks_t1)
        for i, j, _ in matches:
            next_track_ids[j] = current_track_ids[i]

        for j in births:
            next_track_ids[j] = next_tid
            next_tid += 1

        if any(tid < 0 for tid in next_track_ids):
            raise RuntimeError(f"Unassigned masks at frame pair {t}->{t+1}")
        current_track_ids = [int(tid) for tid in next_track_ids]
        out.append(current_track_ids)

    return out


def extract_unmatched_sam_masks(
    tracked_masks: list[np.ndarray],
    sam_masks: list[np.ndarray],
    *,
    iou_threshold: float,
) -> list[np.ndarray]:
    """Per-frame SAM masks that do not overlap any video-tracked mask."""
    unmatched: list[np.ndarray] = []
    for sam_mask in sam_masks:
        if not tracked_masks:
            unmatched.append(sam_mask)
            continue
        max_iou = max(mask_iou(sam_mask, tm) for tm in tracked_masks)
        if max_iou < iou_threshold:
            unmatched.append(sam_mask)
    return unmatched


def merge_tracked_with_new_objects(
    video_per_frame_masks: list[list[np.ndarray]],
    video_per_frame_track_ids: list[list[int]],
    sam_per_frame_masks: list[list[np.ndarray]],
    *,
    new_object_iou_threshold: float,
    link_iou_threshold: float,
) -> tuple[list[list[np.ndarray]], list[list[int]]]:
    """
    Append per-frame SAM instances that are not explained by video tracking.

    New instances receive track IDs via Hungarian linking across frames, offset
    above the highest video-track ID.
    """
    n = len(video_per_frame_masks)
    if len(sam_per_frame_masks) != n or len(video_per_frame_track_ids) != n:
        raise ValueError("video and SAM per-frame lists must have equal length")

    unmatched_per_frame = [
        extract_unmatched_sam_masks(
            video_per_frame_masks[t],
            sam_per_frame_masks[t],
            iou_threshold=new_object_iou_threshold,
        )
        for t in range(n)
    ]

    max_video_tid = max(
        (tid for frame in video_per_frame_track_ids for tid in frame),
        default=-1,
    )
    id_offset = max_video_tid + 1

    if any(unmatched_per_frame):
        new_tids_per_frame = propagate_track_ids_lists(
            unmatched_per_frame,
            link_iou_threshold,
            id_offset=id_offset,
        )
    else:
        new_tids_per_frame = [[] for _ in range(n)]

    merged_masks: list[list[np.ndarray]] = []
    merged_tids: list[list[int]] = []
    for t in range(n):
        merged_masks.append(video_per_frame_masks[t] + unmatched_per_frame[t])
        merged_tids.append(video_per_frame_track_ids[t] + new_tids_per_frame[t])
    return merged_masks, merged_tids


def build_from_sam2_track_ids(
    per_frame_masks: list[list[np.ndarray]],
    per_frame_track_ids: list[list[int]],
    height: int,
    width: int,
    iou_threshold: float = 0.3,
) -> tuple[list[np.ndarray], list[FramePairCorrespondence], list[dict[str, Any]]]:
    """
    Build label maps and correspondence using SAM2 video object IDs.

    Masks are linked across frames by stable ``track_id``, not Hungarian matching.
    """
    if len(per_frame_masks) != len(per_frame_track_ids):
        raise ValueError("per_frame_masks and per_frame_track_ids length mismatch")
    if not per_frame_masks:
        return [], [], []

    label_maps = [
        label_map_from_masks(masks, tids, height, width)
        for masks, tids in zip(per_frame_masks, per_frame_track_ids, strict=True)
    ]

    tracks: dict[int, dict[str, Any]] = {}
    for t, tids in enumerate(per_frame_track_ids):
        for tid in tids:
            if tid not in tracks:
                tracks[tid] = {"track_id": tid, "birth_frame": t}
            tracks[tid]["last_frame"] = t

    frame_pairs: list[FramePairCorrespondence] = []
    for t in range(len(per_frame_masks) - 1):
        masks_t = per_frame_masks[t]
        masks_t1 = per_frame_masks[t + 1]
        tids_t = per_frame_track_ids[t]
        tids_t1 = per_frame_track_ids[t + 1]
        idx_t = {tid: i for i, tid in enumerate(tids_t)}
        idx_t1 = {tid: j for j, tid in enumerate(tids_t1)}

        matches: list[list[float]] = []
        for tid in sorted(set(idx_t) & set(idx_t1)):
            i, j = idx_t[tid], idx_t1[tid]
            iou = mask_iou(masks_t[i], masks_t1[j])
            matches.append([float(i), float(j), iou])

        deaths = [int(idx_t[tid]) for tid in sorted(set(idx_t) - set(idx_t1))]
        births = [int(idx_t1[tid]) for tid in sorted(set(idx_t1) - set(idx_t))]
        frame_pairs.append(
            FramePairCorrespondence(t=t, matches=matches, births=births, deaths=deaths)
        )

        for tid in set(idx_t) - set(idx_t1):
            tracks[tid]["death_frame"] = t + 1

    # unused iou_threshold kept for API parity with propagate_track_ids callers
    _ = iou_threshold

    track_list = [tracks[k] for k in sorted(tracks)]
    return label_maps, frame_pairs, track_list


def align_chunk_track_ids(
    prev_masks: list[np.ndarray],
    prev_track_ids: list[int],
    chunk_track_ids: list[list[int]],
    chunk_masks: list[list[np.ndarray]],
    *,
    iou_threshold: float,
) -> list[list[int]]:
    """
    Remap SAM2 object IDs in a new chunk to continue global IDs from the prior chunk.

    Uses Hungarian matching between the last frame of the previous chunk and frame 0 of
    this chunk; unmatched objects in the new chunk receive fresh global IDs.
    """
    if not chunk_track_ids:
        return chunk_track_ids
    if not prev_masks or not prev_track_ids:
        return chunk_track_ids

    iou_mat = mask_iou_matrix(prev_masks, chunk_masks[0])
    matches, _, births = hungarian_link(iou_mat, iou_threshold)
    local_to_global: dict[int, int] = {}
    for i, j, _ in matches:
        local_to_global[int(chunk_track_ids[0][j])] = int(prev_track_ids[i])

    next_tid = max(prev_track_ids) + 1 if prev_track_ids else 0
    for j in births:
        local_tid = int(chunk_track_ids[0][j])
        if local_tid not in local_to_global:
            local_to_global[local_tid] = next_tid
            next_tid += 1

    return [
        [local_to_global.get(int(tid), int(tid)) for tid in frame_tids]
        for frame_tids in chunk_track_ids
    ]

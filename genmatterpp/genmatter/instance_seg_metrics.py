"""Multi-instance segmentation metrics (Jaccard / IoU over all objects)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def load_gt_label_map(
    video_id: str,
    frame_idx: int,
    annotations_path: str | Path,
    img_dims: tuple[int, int],
) -> np.ndarray:
    """Load uint16/float label map (0=background, 1..K=instances)."""
    root = Path(annotations_path) / video_id
    path = root / f"{frame_idx:05d}.png"
    lab = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(path)
    h, w = img_dims
    if lab.shape[:2] != (h, w):
        lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.asarray(lab, dtype=np.int32)


def rasterize_blob_label_map(
    blob_assignments: np.ndarray,
    n_blobs: int,
    img_dims: tuple[int, int],
) -> np.ndarray:
    """
    Dense per-pixel instance labels from blob assignments.

    Returns int32 map: 0=background/outlier, 1..=blob_id+1.
    """
    h, w = img_dims
    assign = np.asarray(blob_assignments).reshape(-1)
    expected = h * w
    if assign.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} assignments for {img_dims}, got {assign.shape[0]}"
        )
    out = np.zeros(expected, dtype=np.int32)
    valid = (assign >= 0) & (assign < n_blobs)
    out[valid] = assign[valid].astype(np.int32) + 1
    return out.reshape(h, w)


def rasterize_hyperblob_label_map(
    blob_assignments: np.ndarray,
    hyperblob_assignments: np.ndarray,
    n_blobs: int,
    img_dims: tuple[int, int],
) -> np.ndarray:
    """
    Dense per-pixel HYPERBLOB labels from blob assignments + blob→hyperblob map.

    The model carries persistent identity at the *hyperblob* level — each
    hyperblob is the union of multiple blobs and is meant to be an
    object-level cluster.  This gives larger, more comparable-to-GT regions
    than the raw 64-blob map.

    Returns int32 map: 0=background/outlier, 1..=hyperblob_id+1.
    """
    h, w = img_dims
    assign = np.asarray(blob_assignments).reshape(-1)
    hb = np.asarray(hyperblob_assignments).reshape(-1)
    expected = h * w
    if assign.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} assignments for {img_dims}, got {assign.shape[0]}"
        )
    out = np.zeros(expected, dtype=np.int32)
    valid = (assign >= 0) & (assign < n_blobs)
    out[valid] = hb[assign[valid].astype(np.int32)].astype(np.int32) + 1
    return out.reshape(h, w)


def _instance_ids(label_map: np.ndarray) -> np.ndarray:
    ids = np.unique(label_map)
    return ids[ids > 0]


def _mask_for_id(label_map: np.ndarray, inst_id: int) -> np.ndarray:
    return label_map == inst_id


def instance_iou_matrix(gt_map: np.ndarray, pred_map: np.ndarray) -> np.ndarray:
    """IoU between every GT instance (rows) and pred instance (cols)."""
    gt_ids = _instance_ids(gt_map)
    pred_ids = _instance_ids(pred_map)
    if gt_ids.size == 0 or pred_ids.size == 0:
        return np.zeros((gt_ids.size, pred_ids.size), dtype=np.float64)

    iou = np.zeros((gt_ids.size, pred_ids.size), dtype=np.float64)
    for i, gid in enumerate(gt_ids):
        g = _mask_for_id(gt_map, int(gid))
        for j, pid in enumerate(pred_ids):
            p = _mask_for_id(pred_map, int(pid))
            inter = np.logical_and(g, p).sum()
            if inter == 0:
                continue
            union = np.logical_or(g, p).sum()
            iou[i, j] = float(inter) / float(union)
    return iou


def hungarian_match_instances(
    iou_matrix: np.ndarray,
    *,
    iou_threshold: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Match GT rows to pred cols; return (gt_row, pred_col, iou)."""
    iou = np.asarray(iou_matrix, dtype=np.float64)
    n_gt, n_pred = iou.shape
    if n_gt == 0 or n_pred == 0:
        return []

    cost = 1.0 - iou
    row_ind, col_ind = linear_sum_assignment(cost)
    matches: list[tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind, strict=True):
        v = float(iou[r, c])
        if v >= iou_threshold:
            matches.append((int(r), int(c), v))
    return matches


def frame_instance_jaccard_metrics(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    *,
    match_iou_threshold: float = 0.0,
    score_iou_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Per-frame multi-object Jaccard summary.

    - ``mean_matched_iou``: mean IoU over Hungarian-matched pairs.
    - ``mean_gt_iou``: mean IoU per GT instance (unmatched GT → 0).
    - ``gt_recall_at_thresh``: fraction of GT with IoU ≥ thresh.
    - ``pred_precision_at_thresh``: fraction of pred instances matched at ≥ thresh.
    - ``pixel_jaccard``: standardized binary foreground/background Jaccard
      (instance-agnostic union-of-all-objects IoU vs union-of-all-GT).
    """
    gt_ids = _instance_ids(gt_map)
    pred_ids = _instance_ids(pred_map)
    n_gt = int(gt_ids.size)
    n_pred = int(pred_ids.size)

    iou_mat = instance_iou_matrix(gt_map, pred_map)
    matches = hungarian_match_instances(iou_mat, iou_threshold=match_iou_threshold)

    if matches:
        mean_matched_iou = float(np.mean([m[2] for m in matches]))
    else:
        mean_matched_iou = 0.0

    gt_ious = np.zeros(n_gt, dtype=np.float64)
    for r, _, v in matches:
        gt_ious[r] = v
    if n_gt > 0 and not matches:
        for r in range(n_gt):
            gt_ious[r] = float(iou_mat[r].max()) if n_pred > 0 else 0.0
    mean_gt_iou = float(gt_ious.mean()) if n_gt > 0 else 0.0

    gt_recall = float(np.mean(gt_ious >= score_iou_threshold)) if n_gt > 0 else 0.0
    matched_pred_cols = {c for _, c, v in matches if v >= score_iou_threshold}
    pred_precision = float(len(matched_pred_cols) / n_pred) if n_pred > 0 else 0.0

    # Standardized binary Jaccard: collapse all instances to a foreground mask,
    # then compute pixel-wise IoU.  Instance-agnostic; rewards correct coverage
    # of the foreground regardless of how it's segmented internally.
    gt_fg = gt_map > 0
    pred_fg = pred_map > 0
    inter_fg = float(np.logical_and(gt_fg, pred_fg).sum())
    union_fg = float(np.logical_or(gt_fg, pred_fg).sum())
    pixel_jaccard = inter_fg / union_fg if union_fg > 0 else 0.0

    return {
        "mean_matched_iou": mean_matched_iou,
        "mean_gt_iou": mean_gt_iou,
        "gt_recall_at_thresh": gt_recall,
        "pred_precision_at_thresh": pred_precision,
        "pixel_jaccard": pixel_jaccard,
        "num_gt_instances": float(n_gt),
        "num_pred_instances": float(n_pred),
        "num_matched": float(len(matches)),
    }


def evaluate_instance_segmentation_tracking(
    video_id: str,
    tracking_data: list[dict],
    *,
    annotations_path: str | Path,
    img_dims: tuple[int, int],
    match_iou_threshold: float = 0.0,
    score_iou_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Video-level multi-instance Jaccard vs pseudo-GT label maps (all objects).

    Primary score: ``avg_mean_gt_iou`` — mean per-GT-instance IoU, averaged over
    frames (unmatched objects count as 0).
    """
    per_frame: list[dict[str, float]] = []
    for frame_idx, fd in enumerate(tracking_data):
        gt_map = load_gt_label_map(video_id, frame_idx, annotations_path, img_dims)
        pred_map = rasterize_blob_label_map(
            fd["blob_assignments"],
            int(fd["n_blobs"]),
            img_dims,
        )
        per_frame.append(
            frame_instance_jaccard_metrics(
                gt_map,
                pred_map,
                match_iou_threshold=match_iou_threshold,
                score_iou_threshold=score_iou_threshold,
            )
        )

    if not per_frame:
        return {
            "avg_mean_matched_iou": 0.0,
            "avg_mean_gt_iou": 0.0,
            "avg_gt_recall_at_thresh": 0.0,
            "avg_pred_precision_at_thresh": 0.0,
            "avg_persistent_iou": 0.0,
            "num_frames": 0.0,
        }

    def _avg(key: str) -> float:
        return float(np.mean([f[key] for f in per_frame]))

    # Persistent-identity IoU: rerun Hungarian matching ONCE on frame 0
    # (between blob IDs and GT instance IDs) and reuse that mapping for every
    # subsequent frame.  This exercises the model's claim that blob IDs are
    # latent identities preserved across frames — if a blob silently swaps
    # objects mid-video, the fixed mapping makes the post-swap IoU drop.
    persistent = _persistent_identity_iou(
        video_id=video_id,
        tracking_data=tracking_data,
        annotations_path=annotations_path,
        img_dims=img_dims,
        match_iou_threshold=match_iou_threshold,
    )

    return {
        "avg_mean_matched_iou": _avg("mean_matched_iou"),
        "avg_mean_gt_iou": _avg("mean_gt_iou"),
        "avg_gt_recall_at_thresh": _avg("gt_recall_at_thresh"),
        "avg_pred_precision_at_thresh": _avg("pred_precision_at_thresh"),
        "avg_pixel_jaccard": _avg("pixel_jaccard"),
        "avg_persistent_iou": persistent["avg_persistent_iou"],
        "num_persistent_matched": persistent["num_matched"],
        "num_frames": float(len(per_frame)),
        "score_iou_threshold": float(score_iou_threshold),
    }


def _persistent_identity_iou(
    *,
    video_id: str,
    tracking_data: list[dict],
    annotations_path: str | Path,
    img_dims: tuple[int, int],
    match_iou_threshold: float = 0.0,
) -> dict[str, float]:
    """Match HYPERBLOB IDs → GT instance IDs once on frame 0, then average
    IoU across frames using that fixed mapping.

    Uses hyperblob granularity (not individual blobs) because each hyperblob
    is the persistent object-level cluster the model maintains — the union
    of blobs that collectively cover one foreground region.  Individual
    blobs are too small to ever match a GT instance well; hyperblobs are
    sized to align with object-level GT.

    If a hyperblob's mapped GT instance is missing in a later frame, IoU=0
    for that frame.  This rewards identity persistence and penalizes silent
    ID swaps (a hyperblob silently jumping to a different physical object
    mid-video makes the post-swap IoU collapse).
    """
    if len(tracking_data) == 0:
        return {"avg_persistent_iou": 0.0, "num_matched": 0.0}

    fd0 = tracking_data[0]
    gt0 = load_gt_label_map(video_id, 0, annotations_path, img_dims)
    pred0 = rasterize_hyperblob_label_map(
        fd0["blob_assignments"], fd0["hyperblob_assignments"],
        int(fd0["n_blobs"]), img_dims,
    )
    gt_ids0 = _instance_ids(gt0)
    pred_ids0 = _instance_ids(pred0)
    if gt_ids0.size == 0 or pred_ids0.size == 0:
        return {"avg_persistent_iou": 0.0, "num_matched": 0.0}

    iou0 = instance_iou_matrix(gt0, pred0)
    matches = hungarian_match_instances(iou0, iou_threshold=match_iou_threshold)
    if not matches:
        return {"avg_persistent_iou": 0.0, "num_matched": 0.0}

    pairs = [(int(gt_ids0[r]), int(pred_ids0[c])) for r, c, _ in matches]

    per_frame_iou_sum = 0.0
    per_frame_count = 0
    for frame_idx, fd in enumerate(tracking_data):
        gt_map = load_gt_label_map(video_id, frame_idx, annotations_path, img_dims)
        pred_map = rasterize_hyperblob_label_map(
            fd["blob_assignments"], fd["hyperblob_assignments"],
            int(fd["n_blobs"]), img_dims,
        )
        frame_iou = 0.0
        for gt_id, pred_id in pairs:
            g = gt_map == gt_id
            p = pred_map == pred_id
            inter = float(np.logical_and(g, p).sum())
            union = float(np.logical_or(g, p).sum())
            frame_iou += inter / union if union > 0 else 0.0
        per_frame_iou_sum += frame_iou / len(pairs)
        per_frame_count += 1

    avg = per_frame_iou_sum / per_frame_count if per_frame_count > 0 else 0.0
    return {"avg_persistent_iou": float(avg), "num_matched": float(len(pairs))}

"""Sequential single-process live runner over one video.

The live demo (`render_demo.py`) drives depth + flow + DINO + GenMatter++ from
four threads with monotonic frame ids and an EMA depth filter. The calibrator
(`scripts/calibrate_general.py`) needs the **same** perception regime — but
without the Recorder/sidebar/mp4 machinery, and with the per-frame
``(positions, velocities, features, blob_a, hyperblob_a, outlier_frac)``
returned so feature/velocity statistics can be computed offline.

This module re-uses:
- ``render_demo._load_depth_model`` / ``_load_flow_model`` (factored out so we
  don't need to import render_demo's thread + faulthandler setup).
- ``streaming_dino.load_dino`` / ``dino_patches``.
- ``genmatter_rt.{subsample_indices, unproject, dino_features_to_datapoints,
  init_state, step, extract_assignments}``.

The runner pins the same static shapes the live demo uses
(``STRIDE = 8``, ``N_KEEP = 2925``, ``num_blobs / num_hyperblobs`` from the YAML,
DINO grid ``(DINO_GH, DINO_GW)``) so the JIT compile happens once across all
videos when the disk cache is warm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp

import streaming_dino  # noqa: E402  (kept here because render_demo path tweak comes from REPO_ROOT)
import genmatter_rt    # noqa: E402

# render_demo loads the SEA-RAFT + HF depth + DINO models at module import via
# `_load_depth_model` / `_load_flow_model` / `streaming_dino.load_dino`. Importing
# render_demo therefore triggers the same one-time model load the live demo pays
# for. We re-export those globals here so the calibrator gets the *same*
# (proc, depth_model, flow_model, dino_model) instances.
import render_demo  # noqa: E402  -- side-effect import: loads depth/flow/dino models
from render_demo import depth_proc, depth_model, flow_model, dino_model, DEVICE  # noqa: E402


WORK_HW: Tuple[int, int] = (360, 640)   # (H, W) — matches render_demo.RESIZE (W, H)


# ----------------------------------------------------------------------
# Frame iteration
# ----------------------------------------------------------------------

def _iter_frames_mp4(path: Path, max_frames: int) -> Iterator[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open {path!r}")
    try:
        i = 0
        while max_frames < 0 or i < max_frames:
            ok, frame = cap.read()
            if not ok:
                return
            frame = cv2.resize(frame, (WORK_HW[1], WORK_HW[0]), interpolation=cv2.INTER_AREA)
            yield i, frame
            i += 1
    finally:
        cap.release()


def _iter_frames_dir(dir_path: Path, max_frames: int) -> Iterator[Tuple[int, np.ndarray]]:
    jpgs = sorted(p for p in dir_path.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for i, jpg in enumerate(jpgs):
        if max_frames >= 0 and i >= max_frames:
            return
        bgr = cv2.imread(str(jpg), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        bgr = cv2.resize(bgr, (WORK_HW[1], WORK_HW[0]), interpolation=cv2.INTER_AREA)
        yield i, bgr


def iter_frames(source: Path, max_frames: int = -1) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(index, bgr_640x360)`` frames from an mp4/mov OR an RGB-frames dir."""
    source = Path(source)
    if source.is_dir():
        return _iter_frames_dir(source, max_frames)
    if source.is_file():
        return _iter_frames_mp4(source, max_frames)
    raise FileNotFoundError(source)


# ----------------------------------------------------------------------
# Per-frame perception kernels
# ----------------------------------------------------------------------

def _depth_forward(bgr: np.ndarray) -> np.ndarray:
    """Run DepthAnythingV2-Small on one frame. Returns float32 (H, W)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        d_in = depth_proc(images=rgb, return_tensors="pt").to(DEVICE, dtype=torch.float16)
        d = depth_model(**d_in).predicted_depth
        d = F.interpolate(d[:, None], size=rgb.shape[:2],
                          mode="bilinear", align_corners=False)[0, 0].float()
        return d.cpu().numpy()


def _bgr_to_raft_tensor(bgr: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """RAFT expects (1, 3, H', W') with H', W' multiples of 8. Returns padded tensor + (H, W)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE).float()
    _, _, h, w = t.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    return F.pad(t, (0, pw, 0, ph)), (h, w)


def _flow_forward(prev_tensor: torch.Tensor, cur_tensor: torch.Tensor,
                  hw: Tuple[int, int], iters: int = 4) -> np.ndarray:
    """Run SEA-RAFT(prev → cur). Returns float32 (2, H, W) flow."""
    with torch.inference_mode():
        out = flow_model(prev_tensor, cur_tensor, iters=iters, test_mode=True)
        flow = out["flow"][-1][..., :hw[0], :hw[1]]
        return flow[0].cpu().numpy().astype(np.float32)


def _features_forward(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Run DINOv2-S dense patches. Returns (features (P, D), (gh, gw))."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return streaming_dino.dino_patches(dino_model, rgb, DEVICE)


# ----------------------------------------------------------------------
# Top-level entry points used by the calibrator
# ----------------------------------------------------------------------

@dataclass
class LiveFrame:
    """Per-frame artifacts captured by ``run_streaming_live``."""

    frame_idx: int
    outlier_frac: float = float("nan")
    step_sec: float = float("nan")
    blob_a: Optional[np.ndarray] = None
    hyperblob_a: Optional[np.ndarray] = None
    positions: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    blob_weights: Optional[np.ndarray] = None   # (n_blobs,) posterior-mean weights
    n_blobs: Optional[int] = None               # blob count (outlier-fold id for J-mean)


@dataclass
class LiveRunResult:
    indices: np.ndarray                # (N,) stride-8 subsample indices
    frames: List[LiveFrame] = field(default_factory=list)
    init_state_sec: float = float("nan")
    first_step_jit_sec: float = float("nan")
    matter_fps: float = float("nan")   # steady-state matter throughput
    num_frames: int = 0
    error: Optional[str] = None

    @property
    def outlier_frac_history(self) -> np.ndarray:
        return np.asarray([f.outlier_frac for f in self.frames
                           if np.isfinite(f.outlier_frac)], dtype=np.float32)


def compute_frame0_features(source: Path, *,
                             stride: int = genmatter_rt.STRIDE,
                             n_keep: int = genmatter_rt.N_KEEP,
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return frame-0 DINO features at the streaming-PCA basis.

    Used by the σ_F phase of the calibrator: we only need the per-datapoint
    feature vectors at frame 0, not depth / flow / the tracker.

    Returns ``(features (N, FEATURE_DIM), pca_basis, pca_mean, pca_std)`` plus
    ``indices`` separately via the caller's choice (the caller may re-derive them
    or reuse a canonical subsample).
    """
    for _, bgr in iter_frames(Path(source), max_frames=1):
        feat_np, grid_hw = _features_forward(bgr)
        h, w = bgr.shape[:2]
        indices = genmatter_rt.subsample_indices(h=h, w=w, stride=stride, n_keep=n_keep, seed=0)
        features, basis, mean, std = genmatter_rt.dino_features_to_datapoints(
            feat_np, indices,
            pca_basis=None, pca_mean=None, pca_std=None,
            stride=stride, image_hw=(h, w),
            target_dim=genmatter_rt.FEATURE_DIM,
            feat_grid_hw=grid_hw,
        )
        return features, basis, mean, std
    raise RuntimeError(f"No frames yielded from {source!r}")


def run_streaming_live(
    source: Path | str,
    *,
    yaml_cfg: dict,
    max_frames: int = -1,
    depth_ema_alpha: float = 0.6,
    intrinsics=None,
    capture_inputs: bool = False,
    capture_assignments: bool = True,
    capture_blob_weights: bool = False,
    num_gibbs_sweeps_per_frame: int = 1,
    verbose: bool = False,
    key_seed: int = 0,
) -> LiveRunResult:
    """Drive the full live tracker over ``source`` sequentially.

    Mirrors the live demo's worker stack on a single thread:
    Depth + EMA → Flow(prev,cur) → DINO patches → unproject + PCA →
    ``init_state`` (frame 1) → ``step`` thereafter → ``extract_assignments``.

    Parameters
    ----------
    source            : path to source.mp4/.MOV OR a directory of jpg/png frames.
    yaml_cfg          : dict loaded via ``genmatter_rt.load_yaml_hypers``.
    max_frames        : -1 for all frames, else cap.
    depth_ema_alpha   : matches ``render_demo.DepthWorker`` default (0.6).
    capture_inputs    : if True, store positions/velocities/features per frame.
    capture_assignments : if True, store ``(blob_a, hyperblob_a)`` per frame.

    Returns a ``LiveRunResult`` with per-frame ``outlier_frac`` + timing + the
    optional captured inputs/assignments.
    """
    source = Path(source)
    intrinsics = intrinsics or genmatter_rt.DEFAULT_INTRINSICS
    num_blobs = int(yaml_cfg["tracking"]["num_blobs"])
    num_hyperblobs = int(yaml_cfg["tracking"]["num_hyperblobs"])
    stride = genmatter_rt.STRIDE
    n_keep = genmatter_rt.N_KEEP

    indices = genmatter_rt.subsample_indices(h=WORK_HW[0], w=WORK_HW[1],
                                              stride=stride, n_keep=n_keep, seed=0)

    result = LiveRunResult(indices=indices)
    prev_tensor: Optional[torch.Tensor] = None
    depth_ema: Optional[np.ndarray] = None
    pca_basis = pca_mean = pca_std = None
    state = None
    key = jax.random.PRNGKey(key_seed)

    step_walls: List[float] = []
    t_init: float = float("nan")
    t_first_step: float = float("nan")
    frames_processed = 0

    try:
        for i, bgr in iter_frames(source, max_frames):
            t_frame_start = time.monotonic()

            # 1. Depth + EMA
            d_raw = _depth_forward(bgr).astype(np.float32)
            depth_ema = d_raw if depth_ema is None else (
                depth_ema_alpha * d_raw + (1.0 - depth_ema_alpha) * depth_ema)
            depth_use = depth_ema

            # 2. Flow (requires a previous frame); first frame has no flow yet.
            cur_tensor, hw = _bgr_to_raft_tensor(bgr)
            if prev_tensor is None:
                prev_tensor = cur_tensor
                frames_processed += 1
                # No tracker step on the very first frame — wait for flow.
                if verbose:
                    print(f"[run_streaming_live] frame {i:3d}: depth warm (no flow yet)",
                          flush=True)
                # Capture nothing for this frame except the index.
                result.frames.append(LiveFrame(frame_idx=i))
                continue
            flow_np = _flow_forward(prev_tensor, cur_tensor, hw)
            prev_tensor = cur_tensor

            # 3. DINO patches
            feat_raw, grid_hw = _features_forward(bgr)

            # 4. unproject + feature gather (same paths as MatterWorker)
            positions, velocities = genmatter_rt.unproject(
                depth_use, flow_np, indices, intrinsics, stride)
            features, pca_basis, pca_mean, pca_std = genmatter_rt.dino_features_to_datapoints(
                feat_raw, indices, pca_basis, pca_mean, pca_std,
                stride=stride, image_hw=bgr.shape[:2],
                target_dim=genmatter_rt.FEATURE_DIM,
                feat_grid_hw=grid_hw,
            )

            # 5. init_state on the first valid (depth, flow, features) tuple,
            # else one Gibbs step.
            t_track_start = time.monotonic()
            if state is None:
                state, key = genmatter_rt.init_state(
                    positions, velocities, features, key,
                    yaml_cfg=yaml_cfg,
                    num_blobs=num_blobs, num_hyperblobs=num_hyperblobs,
                    sam_segmentation=None, subsample_indices=None,
                    verbose=verbose,
                )
                t_init = time.monotonic() - t_track_start
                t0_step = time.monotonic()
                state, key = genmatter_rt.step_multi_sweep(
                    state, positions, velocities, features, key,
                    num_sweeps=num_gibbs_sweeps_per_frame)
                state.datapoints_state.blob_assignments.block_until_ready()
                t_first_step = time.monotonic() - t0_step
            else:
                state, key = genmatter_rt.step_multi_sweep(
                    state, positions, velocities, features, key,
                    num_sweeps=num_gibbs_sweeps_per_frame)
                state.datapoints_state.blob_assignments.block_until_ready()
                step_walls.append(time.monotonic() - t_track_start)

            blob_a, hyperblob_a = genmatter_rt.extract_assignments(state)
            outlier_frac = float(np.mean(blob_a == -1))

            # J-mean scorer (calibration) needs per-frame posterior-mean blob
            # weights + the blob count (outlier-fold id). One host transfer/frame,
            # after the fully on-device step — never inside an inner kernel.
            blob_w = None
            n_blobs_out = None
            if capture_blob_weights:
                blob_w, key = genmatter_rt.extract_blob_weights(state, key)
                n_blobs_out = int(blob_w.shape[0])

            lf = LiveFrame(
                frame_idx=i,
                outlier_frac=outlier_frac,
                step_sec=time.monotonic() - t_frame_start,
                blob_a=blob_a.copy() if capture_assignments else None,
                hyperblob_a=hyperblob_a.copy() if capture_assignments else None,
                positions=positions.copy() if capture_inputs else None,
                velocities=velocities.copy() if capture_inputs else None,
                features=features.copy() if capture_inputs else None,
                blob_weights=blob_w,
                n_blobs=n_blobs_out,
            )
            result.frames.append(lf)
            frames_processed += 1

            if verbose:
                print(f"[run_streaming_live] frame {i:3d} outlier_frac={outlier_frac:.4f} "
                      f"step={lf.step_sec*1000:.1f}ms", flush=True)
    except Exception as exc:  # surface errors so the calibrator can log + skip
        result.error = repr(exc)
        import traceback
        traceback.print_exc()

    result.init_state_sec = t_init
    result.first_step_jit_sec = t_first_step
    result.num_frames = frames_processed
    if step_walls:
        result.matter_fps = float(1.0 / np.median(step_walls))
    return result


# ----------------------------------------------------------------------
# CLI smoke harness
# ----------------------------------------------------------------------

def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=str, help="Path to source.mp4 or RGB frames dir.")
    p.add_argument("--config", type=str,
                   default=str(_REPO_ROOT / "configs" / "streaming_default.yaml"))
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    yaml_cfg = genmatter_rt.load_yaml_hypers(Path(args.config))
    result = run_streaming_live(args.source, yaml_cfg=yaml_cfg,
                                max_frames=args.max_frames, verbose=args.verbose)
    hist = result.outlier_frac_history
    summary = {
        "frames": result.num_frames,
        "init_state_sec": result.init_state_sec,
        "first_step_jit_sec": result.first_step_jit_sec,
        "matter_fps": result.matter_fps,
        "outlier_frac_n": int(hist.size),
        "outlier_frac_mean": float(hist.mean()) if hist.size else float("nan"),
        "outlier_frac_max": float(hist.max()) if hist.size else float("nan"),
        "outlier_frac_p95": float(np.percentile(hist, 95)) if hist.size else float("nan"),
        "error": result.error,
    }
    print(json.dumps(summary, indent=2))
    return 0 if result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(_cli())

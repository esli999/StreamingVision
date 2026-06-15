"""Run the streaming pipeline on assets/test.mp4 and render an MP4 that
shows everything happening in real-time, side-by-side, with per-stream FPS
counters and a total system FPS.

Output: assets/streaming_demo.mp4 — 2320×720, 30 fps, h264.

The video is captured at the source FPS (so playback is real-time). Each
tile shows one stream with its own measured FPS in the corner; the right
sidebar shows TOTAL FPS plus per-worker latency / staleness.

Performance metrics glossary
----------------------------

**lat** — *latency, in milliseconds*. How long the worker takes to process
one frame's output, wall-clock end-to-end. `depth lat 35.5ms` means each
depth inference took 35.5 ms from the moment the worker started on an RGB
frame until it published its result. RGB shows `0.0ms` because the
FrameSource isn't doing any inference — it just reads and resizes.

**stale** — *staleness, in source frames*. How many RGB frames behind "now"
this stream's latest output is. Computed at recorder snapshot time as
`latest_rgb_gid − this_slot_gid`. `depth stale 1f` means the depth slot's
most recent value was derived from an RGB frame that is 1 frame older than
the current RGB frame. RGB is always `0f` because it IS the latest source
frame; matter is typically `1–3f` because each Gibbs sweep depends on the
current depth/flow/features slots, which themselves are ≥1f stale.

Together they tell you two different things:
- **lat**: how heavy each model is on the GPU.
- **stale**: how far behind real-time each stream is when the recorder grabs
  a snapshot.

The colored dot to the left of each sidebar row, and the colored bar at the
top of each tile, encode staleness visually:
- green: ≤1 frame
- yellow: 2–4 frames
- red: ≥5 frames

**drift** — *source pace drift, in %*. Negative means the source thread is
falling behind its target FPS (e.g., GPU contention causing the
FrameSource's pacing sleep to overrun). 0 means source is keeping its pace
exactly. Computed from frames published since recording started:
`(target_fps − achieved_fps) / target_fps × 100`. In `--uncapped-source`
mode the sidebar shows raw achieved FPS instead.

Causal-honesty contract
-----------------------

- All slot snapshots in a single tile composite are taken at one wall-clock
  instant — the causal cut. Each tile's `gid=N` badge tells you which source
  frame its data was derived from.
- The recorder never spin-writes fresh-looking frames it didn't have time
  to capture: under load it pads with the previous tile + a
  `RECORDER LAG +Nms` overlay so playback wall-time stays accurate AND the
  lag is visible.
- Sidebar reports recorder dropped_ticks and max_lag_ms so any cheating is
  immediately auditable.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
# Pin BLAS/OpenMP/MKL thread pools to ONE thread BEFORE importing numpy /
# sklearn / torch.  The streaming pipeline runs 5 Python threads concurrently
# (FrameSource, DepthWorker, FlowWorker, FeatureWorker, MatterWorker); when
# MatterWorker calls sklearn KMeans (during init_state) while the other
# workers are mid-flight with torch CUDA + JAX work, OpenMP nesting can
# deadlock the KMeans thread pool acquisition.  Symptom: process sits at
# 0-1% CPU forever after the "entering K-means hierarchical init" log line.
# Single-threaded BLAS is a tiny perf hit (heavy work is on the GPU) and
# eliminates the deadlock entirely.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, json, argparse, time, threading, subprocess, shutil
import faulthandler, signal
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Dict, List, Tuple

# Dump tracebacks for every Python thread when we receive SIGUSR1.  This is
# how we debug hangs in MatterWorker init: `kill -USR1 <pid>` writes
# tracebacks to stderr (and our log file) so we can see exactly which line
# is blocked without needing sudo for py-spy / gdb.
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

sys.path.insert(0, os.path.abspath("third_party/SEA-RAFT/core"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp

from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModel
from torchvision.utils import flow_to_image
from raft import RAFT

import genmatter_rt
import genmatter_viz
import streaming_dino


# ---------- Primitives ----------

@dataclass
class Slot:
    name: str
    payload: Any = None
    source_global_id: int = -1
    source_frame_idx: int = -1
    source_time_sec: float = 0.0
    wall_start: float = 0.0
    wall_complete: float = 0.0
    latency: float = 0.0
    version: int = 0
    valid: bool = False
    error: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._lock = threading.Lock()

    def publish(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.version += 1
            self.valid = True

    def snapshot(self) -> "Slot":
        with self._lock:
            return replace(self)


class GPUSemaphore:
    def __init__(self, max_concurrent=1):
        self._sem = threading.Semaphore(max_concurrent)
    def __enter__(self):
        self._sem.acquire()
        return self
    def __exit__(self, *a):
        torch.cuda.synchronize()
        self._sem.release()


class MonotonicId:
    def __init__(self):
        self._n = 0
        self._lock = threading.Lock()
    def next(self) -> int:
        with self._lock:
            self._n += 1
            return self._n


# ---------- Constants ----------

DEVICE = "cuda"
RESIZE = (640, 360)       # working res (W, H)
STOP   = threading.Event()
GPU    = GPUSemaphore(max_concurrent=1)
GID    = MonotonicId()


# ---------- Models ----------

DEPTH_CKPT = "depth-anything/Depth-Anything-V2-Small-hf"
SEA_CFG    = "third_party/SEA-RAFT/config/eval/spring-M.json"


def _load_depth_model(device: str = DEVICE):
    """Load DepthAnythingV2-Small (HF) processor + fp16 model. Returns (proc, model)."""
    proc  = AutoImageProcessor.from_pretrained(DEPTH_CKPT)
    model = AutoModelForDepthEstimation.from_pretrained(
                DEPTH_CKPT, torch_dtype=torch.float16).to(device).eval()
    return proc, model


def _load_flow_model(device: str = DEVICE):
    """Load SEA-RAFT (Tartan-C-T-TSKH-spring540x960-M). Returns the model only."""
    with open(SEA_CFG) as f:
        cfg = json.load(f)
    sea_args = argparse.Namespace(**cfg); sea_args.iters = 4; sea_args.scale = 0
    return RAFT.from_pretrained("MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
                                args=sea_args).to(device).eval()


# Perf note: torch.compile + cudnn.benchmark are intentionally NOT used here. The
# depth / DINO ViTs are matmul-bound (cudnn.benchmark is a no-op) and the torch
# forwards are already only ~22 ms total; enabling torch.compile AND
# cudnn.benchmark together makes the co-resident JAX Gibbs tracker much slower
# because the expanded torch GPU footprint starves XLA at
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.25. The dominant cost is the JAX tracker, which
# torch compilation cannot touch.


print("loading models...")
depth_proc, depth_model = _load_depth_model(DEVICE)
flow_model = _load_flow_model(DEVICE)
dino_model = streaming_dino.load_dino(DEVICE)
print("models loaded.")


# ---------- DINO feature kernels ----------

# Denser DINO feature map (Workstream A): DINOv2-S/14 at DINO_H x DINO_W with
# interpolated position embeddings -> DINO_GH x DINO_GW patches, aspect-matched
# to the 640x360 frame (vs. the HF processor's 16x16 224 crop, which upsampled
# ~5x onto the 80x45 tracking grid). Geometry + the forward kernel live in
# streaming_dino so the offline calibration/debug scripts share them exactly.
from streaming_dino import (
    DINO_PATCH, DINO_H, DINO_W, DINO_GH, DINO_GW, N_DINO_PATCHES, DINO_DIM,
)

@jax.jit
def fx_dino_pca(feats):
    """PCA-to-3 of dense DINO patch features.

    feats: (N_DINO_PATCHES, DINO_DIM) float32
    Returns (proj, pcs):
      proj: (N_DINO_PATCHES, 3) float32 — projection onto top-3 components
      pcs:  (DINO_DIM, 3) float32 — top-3 right-eigenvectors of the covariance,
            returned so the caller can sign-align against the previous frame's
            basis (PCA basis is ±-ambiguous).
    """
    mu  = feats.mean(0, keepdims=True)
    X   = feats - mu
    # 384x384 covariance is cheaper than SVD on (256, 384) and deterministic.
    C   = (X.T @ X) / (X.shape[0] - 1)
    _, V = jnp.linalg.eigh(C)            # eigenvectors ascending eigenvalues
    pcs  = V[:, -3:]                      # top-3 → (384, 3)
    proj = X @ pcs                        # (256, 3)
    return proj, pcs


# Warmup the PCA kernel at the patch grid the FeatureWorker actually feeds it.
_zfeat = jnp.zeros((N_DINO_PATCHES, DINO_DIM), dtype=jnp.float32)
_p, _q = fx_dino_pca(_zfeat); _p.block_until_ready(); _q.block_until_ready()


# ---------- Workers ----------

class FrameSource(threading.Thread):
    def __init__(self, src, out_slot: Slot, throttle_fps=None, resize=(640, 360)):
        super().__init__(daemon=True, name="FrameSource")
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {src!r}")
        self.is_file = isinstance(src, str)
        # Explicit False / 0 disables pacing — used by --uncapped-source.
        # None on a file means "auto-detect native FPS".
        if throttle_fps is False or throttle_fps == 0:
            throttle_fps = None
        elif throttle_fps is None and self.is_file:
            throttle_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.throttle_fps = throttle_fps
        self.out, self.resize = out_slot, resize
        self.loop_idx, self.source_idx = 0, -1
        self._next_t = 0.0
        self.start_wall: Optional[float] = None   # set on first iteration

    def run(self):
        self.start_wall = time.monotonic()
        try:
            while not STOP.is_set():
                if self.throttle_fps:
                    now = time.monotonic()
                    if self._next_t == 0: self._next_t = now
                    wait = self._next_t - now
                    if wait > 0: time.sleep(min(wait, 0.05))
                    self._next_t += 1.0 / self.throttle_fps
                ok, frame = self.cap.read()
                if not ok:
                    if self.is_file:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.loop_idx += 1
                        self.source_idx = -1
                        continue
                    break
                self.source_idx += 1
                frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)
                gid = GID.next()
                wall = time.monotonic()
                pace = self.throttle_fps if self.throttle_fps else 30.0
                src_t = self.source_idx / pace
                self.out.publish(
                    payload={"bgr": frame},
                    source_global_id=gid,
                    source_frame_idx=self.source_idx,
                    source_time_sec=src_t,
                    wall_start=wall, wall_complete=wall, latency=0.0,
                    extras={"loop_idx": self.loop_idx},
                )
        finally:
            self.cap.release()


class DepthWorker(threading.Thread):
    def __init__(self, rgb, out, ema_alpha=0.6, poll=0.001):
        super().__init__(daemon=True, name="DepthWorker")
        self.rgb, self.out, self.alpha, self.poll = rgb, out, ema_alpha, poll
        self._last_id = -1
        self._ema = None

    def run(self):
        while not STOP.is_set():
            snap = self.rgb.snapshot()
            if (not snap.valid) or snap.source_global_id == self._last_id:
                time.sleep(self.poll); continue
            bgr = snap.payload["bgr"]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t0 = time.monotonic()
            with GPU, torch.inference_mode():
                d_in = depth_proc(images=rgb, return_tensors="pt").to(DEVICE, dtype=torch.float16)
                d = depth_model(**d_in).predicted_depth
                d = F.interpolate(d[:, None], size=rgb.shape[:2],
                                   mode="bilinear", align_corners=False)[0, 0].float()
                d_np = d.cpu().numpy()
            self._ema = d_np if self._ema is None else self.alpha*d_np + (1-self.alpha)*self._ema
            d_use = self._ema
            dn = (d_use - d_use.min()) / (d_use.max() - d_use.min() + 1e-6)
            viz = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            d_jax = jnp.asarray(d_use.astype(np.float32))
            t1 = time.monotonic()
            self.out.publish(
                payload={"raw": d_use, "viz_bgr": viz, "jax": d_jax},
                source_global_id=snap.source_global_id,
                source_frame_idx=snap.source_frame_idx,
                source_time_sec=snap.source_time_sec,
                wall_start=t0, wall_complete=t1, latency=t1-t0,
                error=None,
            )
            self._last_id = snap.source_global_id


class FlowWorker(threading.Thread):
    def __init__(self, rgb, out, iters=4, poll=0.001):
        super().__init__(daemon=True, name="FlowWorker")
        self.rgb, self.out, self.iters, self.poll = rgb, out, iters, poll
        self._prev_t = None
        self._prev_id = -1

    @staticmethod
    def _to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(DEVICE).float()
        _, _, h, w = t.shape
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        return F.pad(t, (0, pw, 0, ph)), (h, w)

    def run(self):
        while not STOP.is_set():
            snap = self.rgb.snapshot()
            if (not snap.valid) or snap.source_global_id == self._prev_id:
                time.sleep(self.poll); continue
            cur_t, (h, w) = self._to_tensor(snap.payload["bgr"])
            if self._prev_t is None:
                self._prev_t, self._prev_id = cur_t, snap.source_global_id
                continue
            t0 = time.monotonic()
            with GPU, torch.inference_mode():
                out = flow_model(self._prev_t, cur_t,
                                 iters=self.iters, test_mode=True)
                flow = out["flow"][-1][..., :h, :w]
                fv   = flow_to_image(flow[0]).permute(1,2,0).contiguous().cpu().numpy()
                flow_np = flow[0].cpu().numpy()
            viz = cv2.cvtColor(fv, cv2.COLOR_RGB2BGR)
            flow_jax = jnp.asarray(flow_np.astype(np.float32))
            t1 = time.monotonic()
            gap = snap.source_global_id - self._prev_id
            self.out.publish(
                payload={"raw": flow_np, "viz_bgr": viz, "jax": flow_jax},
                source_global_id=snap.source_global_id,
                source_frame_idx=snap.source_frame_idx,
                source_time_sec=snap.source_time_sec,
                wall_start=t0, wall_complete=t1, latency=t1-t0,
                extras={"prev_global_id": self._prev_id, "frame_gap": gap},
                error=None,
            )
            self._prev_t, self._prev_id = cur_t, snap.source_global_id


class FeatureWorker(threading.Thread):
    def __init__(self, rgb, out, viz_size=(640,360), poll=0.001):
        super().__init__(daemon=True, name="FeatureWorker")
        self.rgb, self.out, self.viz_size, self.poll = rgb, out, viz_size, poll
        self._last_id = -1
        # PCA basis from previous frame, used to sign-align this frame's basis
        # so the dense-feature viz colors don't flip arbitrarily.
        self._prev_pcs: Optional[jnp.ndarray] = None

    def run(self):
        while not STOP.is_set():
            snap = self.rgb.snapshot()
            if (not snap.valid) or snap.source_global_id == self._last_id:
                time.sleep(self.poll); continue
            bgr = snap.payload["bgr"]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t0 = time.monotonic()
            with GPU:
                # Dense DINOv2-S patches at DINO_H x DINO_W via the shared
                # streaming_dino kernel (manual resize + ImageNet normalize +
                # interpolate_pos_encoding -> DINO_GH x DINO_GW grid).
                feat_np, (gh, gw) = streaming_dino.dino_patches(dino_model, rgb, DEVICE)

            # Move features to JAX device and run PCA. Both operations release
            # the GIL; we explicitly block before publishing so wall_complete
            # is the true wall-clock when work finished.
            feat_jax = jnp.asarray(feat_np)
            proj, pcs = fx_dino_pca(feat_jax)
            proj.block_until_ready()
            pcs.block_until_ready()

            # Sign-align this frame's basis against last frame's. The PCA basis
            # is determined only up to ± per column, so without alignment the
            # color channels in the viz would flip arbitrarily.
            if self._prev_pcs is not None:
                dots = jnp.sum(pcs * self._prev_pcs, axis=0)   # (3,)
                signs = jnp.where(dots >= 0, 1.0, -1.0)         # (3,)
                pcs  = pcs  * signs[None, :]
                proj = proj * signs[None, :]
                proj.block_until_ready()
                pcs.block_until_ready()
            self._prev_pcs = pcs

            # Build dense viz: 16x16x3 from projection, normalized per-channel
            # to uint8, upsampled to (640, 360).
            proj_np = np.array(proj).reshape(gh, gw, 3).astype(np.float32)
            pmin = proj_np.reshape(-1, 3).min(0)
            pmax = proj_np.reshape(-1, 3).max(0)
            dense_small = ((proj_np - pmin) / (pmax - pmin + 1e-6) * 255).astype(np.uint8)
            dense_viz = cv2.resize(dense_small, self.viz_size, interpolation=cv2.INTER_CUBIC)

            # Keep the cheap norm viz too — still useful for the sidebar legend
            # and harmless to compute. Cost is negligible (~50 us at 16x16).
            norms = np.linalg.norm(feat_np, axis=1).reshape(gh, gw)
            norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-6)
            norm_img = cv2.resize((norms*255).astype(np.uint8), self.viz_size,
                                  interpolation=cv2.INTER_CUBIC)
            norm_viz = cv2.applyColorMap(norm_img, cv2.COLORMAP_VIRIDIS)

            t1 = time.monotonic()
            self.out.publish(
                payload={"raw": feat_np, "norm_viz_bgr": norm_viz,
                         "dense_viz_bgr": dense_viz, "jax": feat_jax,
                         "grid_hw": (gh, gw)},
                source_global_id=snap.source_global_id,
                source_frame_idx=snap.source_frame_idx,
                source_time_sec=snap.source_time_sec,
                wall_start=t0, wall_complete=t1, latency=t1-t0,
                error=None,
            )
            self._last_id = snap.source_global_id


# ---------- GenMatter++ Gibbs tracker ----------

class MatterWorker(threading.Thread):
    """Per-frame GenMatter++ DINO Gibbs sweep.

    Consumes depth + flow + features, unprojects to 3D points + 3D velocities
    at a 1/8 subsampled grid (2925 datapoints), and runs one Gibbs sweep
    using the persistent GenMatter_State_DINO from the previous frame.

    Publishes all ROW2 + ROW3 tiles in one payload per Gibbs sweep:
      * clusters             — hyperblob (~4) Gaussian ellipses on RGB
      * particles            — blob (~64) Gaussian ellipses on RGB
      * pixel_by_cluster     — per-pixel mask colored by hyperblob assignment
      * pixel_by_particle    — per-pixel mask colored by blob assignment
      * pointcloud_by_cluster   — 3D-view point cloud, colored by hyperblob
      * pointcloud_by_particle  — 3D-view point cloud, colored by blob

    The two pointcloud tiles are CPU splats from the same per-pixel mask
    images above, so they're guaranteed visually consistent with the 2D
    pixel-mask tiles (same coloring, same frame of data) but show the depth
    structure that the 2D views collapse away.

    First valid frame triggers K-means init + warmup Gibbs sweeps + JIT
    compile — expect 30-90 s of blocking on cold start before any output
    appears.  Subsequent calls hit the JIT cache (~10 ms per sweep on RTX
    5090).
    """

    def __init__(self, depth, flow, features, rgb, out, poll=0.001,
                  intrinsics=None, config_path=None, sam_frame0_path=None):
        super().__init__(daemon=True, name="MatterWorker")
        self.depth, self.flow, self.features, self.rgb = depth, flow, features, rgb
        self.out, self.poll = out, poll
        self.intrinsics = intrinsics or genmatter_rt.DEFAULT_INTRINSICS
        self.yaml_cfg = genmatter_rt.load_yaml_hypers(config_path)
        self._num_blobs = int(self.yaml_cfg["tracking"]["num_blobs"])
        self._num_hyperblobs = int(self.yaml_cfg["tracking"]["num_hyperblobs"])
        # Per-frame Gibbs sweeps: >1 stabilizes the particles (the 1-sweep live
        # path is noisy — particles "fly"). Read from config (default 1 keeps the
        # original real-time behavior; the render config sets ~4).
        self._gibbs_sweeps = int(self.yaml_cfg["tracking"].get(
            "num_gibbs_sweeps_per_frame", 1))
        # "Fixed cluster view" tracking flags (Python bools, jit-static in
        # genmatter_rt's step_multi_sweep -> one compile per combo). step() /
        # step_multi_sweep() DEFAULT these to False, so we MUST read them from the
        # YAML and thread them through explicitly or the frozen-cluster behavior
        # never reaches the live loop. See blob_tracking_gibbs_dino_streaming for
        # semantics.
        _trk = self.yaml_cfg["tracking"]
        self._feature_aware_final = bool(_trk.get(
            "feature_aware_final_assignment",
            genmatter_rt._FEATURE_AWARE_FINAL_DEFAULT))
        self._final_outlier = bool(_trk.get(
            "final_assignment_outlier", genmatter_rt._FINAL_OUTLIER_DEFAULT))
        self._freeze_hyperblob_assignment = bool(_trk.get(
            "freeze_hyperblob_assignment",
            genmatter_rt._FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT))
        # SAM-anchored semantic init: if tracking.use_sam_frame0 is set and the
        # cached frame-0 SAM mask exists, load it (RGB) so init_state can seed one
        # hyperblob per SAM instance.  Downsampled to the stride-8 grid at init
        # time (once the frame size is known).  Missing mask -> graceful fallback
        # to the flat k-means init.
        self._sam_rgb_full = None
        if self.yaml_cfg["tracking"].get("use_sam_frame0", False) and sam_frame0_path:
            p = os.path.abspath(sam_frame0_path)
            if os.path.isfile(p):
                self._sam_rgb_full = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR),
                                                  cv2.COLOR_BGR2RGB)
                print(f"[MatterWorker] SAM-frame-0 init enabled ({p})", flush=True)
            else:
                print(f"[MatterWorker] use_sam_frame0 set but mask missing ({p}); "
                      f"falling back to k-means init", flush=True)
        self._last_d_v = self._last_f_v = self._last_ft_v = -1
        self._state = None
        self._indices = None
        self._pca_basis = None
        self._pca_mean = None
        self._pca_std = None
        self._key = jax.random.PRNGKey(0)
        # Per-frame outlier_frac from the Gibbs assignment, recorded for the
        # end-of-run summary.
        self._outlier_history: List[float] = []
        # Honest FPS accounting: the matter worker runs the Gibbs tracker AND then
        # draws 8 debug tiles before publishing, so its raw publish rate conflates
        # INFERENCE + RENDER. Track the two separately so the sidebar can show the
        # tracker's TRUE rate as the headline and render as its own line.
        self._infer_ms_hist: deque = deque(maxlen=30)
        self._render_ms_hist: deque = deque(maxlen=30)
        self._last_loop_idx = 0
        # Pointcloud focal length is auto-fit on the first matter frame and
        # then FROZEN — recomputing it per frame makes the cloud jitter
        # in/out even when depth is stable, because the percentile XY extents
        # fluctuate with depth-normalization noise.
        self._pointcloud_focal: Optional[float] = None

    def run(self):
        while not STOP.is_set():
            ds   = self.depth.snapshot()
            fs   = self.flow.snapshot()
            ftsn = self.features.snapshot()
            rs   = self.rgb.snapshot()
            if not (ds.valid and fs.valid and ftsn.valid and rs.valid):
                time.sleep(self.poll); continue
            if (ds.version == self._last_d_v and fs.version == self._last_f_v
                    and ftsn.version == self._last_ft_v):
                time.sleep(self.poll); continue

            t0 = time.monotonic()
            try:
                # CPU prep: unproject + feature gather (these are NumPy /
                # cv2 ops and cheap).
                depth_np = ds.payload["raw"].astype(np.float32)
                flow_np  = fs.payload["raw"]
                feat_np  = ftsn.payload["raw"]

                if self._indices is None:
                    h, w = depth_np.shape
                    self._indices = genmatter_rt.subsample_indices(
                        h=h, w=w, stride=genmatter_rt.STRIDE,
                        n_keep=genmatter_rt.N_KEEP, seed=0)
                    # Downsample the SAM mask to the stride-8 grid that
                    # self._indices indexes into, so the per-datapoint instance
                    # labels line up with `positions`.
                    if self._sam_rgb_full is not None:
                        gh, gw = h // genmatter_rt.STRIDE, w // genmatter_rt.STRIDE
                        self._sam_grid_rgb = cv2.resize(
                            self._sam_rgb_full, (gw, gh), interpolation=cv2.INTER_NEAREST)
                    else:
                        self._sam_grid_rgb = None
                positions, velocities = genmatter_rt.unproject(
                    depth_np, flow_np, self._indices, self.intrinsics,
                    genmatter_rt.STRIDE)
                features, self._pca_basis, self._pca_mean, self._pca_std = \
                    genmatter_rt.dino_features_to_datapoints(
                        feat_np, self._indices, self._pca_basis, self._pca_mean,
                        self._pca_std,
                        stride=genmatter_rt.STRIDE,
                        image_hw=depth_np.shape,
                        target_dim=genmatter_rt.FEATURE_DIM,
                        feat_grid_hw=ftsn.payload["grid_hw"])

                with GPU:
                    if self._state is None:
                        print(f"[MatterWorker] init: K-means + warmup Gibbs + "
                              f"JIT compile (this takes ~30-90s on cold start)...",
                              flush=True)
                        self._state, self._key = genmatter_rt.init_state(
                            positions, velocities, features, self._key,
                            yaml_cfg=self.yaml_cfg,
                            num_blobs=self._num_blobs,
                            num_hyperblobs=self._num_hyperblobs,
                            sam_segmentation=self._sam_grid_rgb,
                            subsample_indices=self._indices,
                            verbose=True)
                        # Pay the per-step JIT cost up front so steady-state
                        # latency below is honest.
                        self._state, self._key = genmatter_rt.step_multi_sweep(
                            self._state, positions, velocities, features, self._key,
                            num_sweeps=self._gibbs_sweeps,
                            feature_aware_final=self._feature_aware_final,
                            final_outlier=self._final_outlier,
                            freeze_hyperblob_assignment=self._freeze_hyperblob_assignment)
                        print(f"[MatterWorker] init complete.")
                    else:
                        self._state, self._key = genmatter_rt.step_multi_sweep(
                            self._state, positions, velocities, features, self._key,
                            num_sweeps=self._gibbs_sweeps,
                            feature_aware_final=self._feature_aware_final,
                            final_outlier=self._final_outlier,
                            freeze_hyperblob_assignment=self._freeze_hyperblob_assignment)
                    self._state.datapoints_state.blob_assignments.block_until_ready()

                # End of INFERENCE (the Gibbs step). Everything below is rendering
                # the debug tiles — timed separately so the FPS counter is honest.
                t_infer = time.monotonic()
                blob_a, hyperblob_a = genmatter_rt.extract_assignments(self._state)

                # Outlier fraction this frame + loop-wrap marker, logged per frame.
                frac = float(np.mean(blob_a == -1))
                self._outlier_history.append(frac)
                loop_idx = int(rs.extras.get("loop_idx", 0)) if rs.extras else 0
                wrap = "  <WRAP>" if loop_idx != self._last_loop_idx else ""
                self._last_loop_idx = loop_idx
                print(f"[MatterWorker] gid={rs.source_global_id:4d} "
                      f"src_idx={rs.source_frame_idx:3d} loop={loop_idx} "
                      f"outlier_frac={frac:.4f}{wrap}", flush=True)

                pixel_by_particle_bgr, pixel_by_cluster_bgr = genmatter_viz.render_matter_tile(
                    blob_a, hyperblob_a, self._indices,
                    h=depth_np.shape[0], w=depth_np.shape[1],
                    stride=genmatter_rt.STRIDE)
                # 3D-ellipse overlays atop the live RGB frame.  Hyperblob ellipses
                # tend to be large (whole-scene) — render them at a lower sigma_scale
                # than blobs.
                base_bgr = rs.payload["bgr"]
                blob_means, blob_covs = genmatter_rt.extract_blob_means_and_covs(self._state)
                hb_means, hb_covs = genmatter_rt.extract_hyperblob_means_and_covs(self._state)
                particles_bgr = genmatter_viz.render_centroid_tile(
                    blob_means, blob_covs,
                    genmatter_viz.BLOB_PALETTE[:blob_means.shape[0]],
                    base_bgr, self.intrinsics, sigma_scale=1.0, alpha=0.55)
                clusters_bgr = genmatter_viz.render_centroid_tile(
                    hb_means, hb_covs,
                    genmatter_viz.HYPERBLOB_PALETTE[:hb_means.shape[0]],
                    base_bgr, self.intrinsics, sigma_scale=0.5, alpha=0.55)

                # ROW3: 3D point-cloud splats colored by the same masks above.
                # Same depth + intrinsics → spatially consistent with the 2D
                # pixel mask tiles; the rotation just makes depth visible.
                # The two tiles share unproject/rotate/project/sort — only the
                # per-pixel colors differ — so amortize through the pair API.
                # Focal length is auto-fit on the first frame, then frozen
                # via self._pointcloud_focal so the framing stays stable.
                pointcloud_by_cluster_bgr, pointcloud_by_particle_bgr, f_used, pc_proj = \
                    genmatter_viz.render_pointcloud_tiles_pair(
                        depth_np, pixel_by_cluster_bgr, pixel_by_particle_bgr,
                        self.intrinsics,
                        focal_length=self._pointcloud_focal)
                if self._pointcloud_focal is None and f_used > 0.0:
                    self._pointcloud_focal = f_used

                # ROW4: the probabilistic particles/clusters as 3D covariance
                # ellipsoids, drawn in the SAME rotated view as ROW3's cloud
                # (via pc_proj) so they line up — each particle's 3D mean + spread.
                particles_3d_bgr = genmatter_viz.render_particle_ellipsoid_tile(
                    blob_means, blob_covs,
                    genmatter_viz.BLOB_PALETTE[:blob_means.shape[0]], pc_proj,
                    sigma_scale=1.0)
                clusters_3d_bgr = genmatter_viz.render_particle_ellipsoid_tile(
                    hb_means, hb_covs,
                    genmatter_viz.HYPERBLOB_PALETTE[:hb_means.shape[0]], pc_proj,
                    sigma_scale=0.6)

                t1 = time.monotonic()
                self._infer_ms_hist.append((t_infer - t0) * 1000.0)
                self._render_ms_hist.append((t1 - t_infer) * 1000.0)
                self.out.publish(
                    payload={
                        "clusters":              clusters_bgr,
                        "particles":             particles_bgr,
                        "pixel_by_cluster":      pixel_by_cluster_bgr,
                        "pixel_by_particle":     pixel_by_particle_bgr,
                        "pointcloud_by_cluster": pointcloud_by_cluster_bgr,
                        "pointcloud_by_particle":pointcloud_by_particle_bgr,
                        "particles_3d":          particles_3d_bgr,
                        "clusters_3d":           clusters_3d_bgr,
                        "blob_assignments":      blob_a,
                        "hyperblob_assignments": hyperblob_a,
                    },
                    source_global_id=rs.source_global_id,
                    source_frame_idx=rs.source_frame_idx,
                    source_time_sec=rs.source_time_sec,
                    wall_start=t0, wall_complete=t1, latency=t1-t0,
                    extras={"depth_global_id":    ds.source_global_id,
                            "flow_global_id":     fs.source_global_id,
                            "features_global_id": ftsn.source_global_id,
                            "infer_ms":  float(np.median(self._infer_ms_hist)),
                            "render_ms": float(np.median(self._render_ms_hist))},
                    error=None,
                )
            except Exception as e:
                t1 = time.monotonic()
                # Publish an error so the recorder still has *something* and
                # the sidebar shows the failure rather than going silent.
                self.out.publish(
                    payload=None,
                    source_global_id=rs.source_global_id,
                    wall_start=t0, wall_complete=t1, latency=t1-t0,
                    error=repr(e),
                )
                import traceback; traceback.print_exc()
            self._last_d_v  = ds.version
            self._last_f_v  = fs.version
            self._last_ft_v = ftsn.version


# ---------- Recorder ----------

class FpsTracker:
    """Rolling FPS from (timestamp, version) samples."""
    def __init__(self, maxlen=30):
        self.hist = deque(maxlen=maxlen)
    def update(self, version):
        self.hist.append((time.monotonic(), version))
    def fps(self) -> float:
        if len(self.hist) < 2: return 0.0
        (t0, v0), (t1, v1) = self.hist[0], self.hist[-1]
        dt = t1 - t0
        return (v1 - v0) / dt if dt > 1e-6 else 0.0


def _staleness_color(stale: int):
    """Green (≤1), yellow (2-4), red (≥5). BGR."""
    if stale <= 1:  return (  0, 200,   0)
    if stale <= 4:  return (  0, 220, 220)
    return                  ( 40,  40, 255)


def _draw_tile_label(img, name: str, fps: float, gid: int = -1,
                     stale: int = 0, color=(60, 255, 255)):
    """Per-tile overlay: name top-left, FPS top-right, gid bottom-left, staleness bar.

    The gid + staleness bar are the causal-honesty markers: each tile shows
    which source frame its data was derived from, plus a colored bar that
    immediately tells the viewer how far behind RGB this stream is.
    """
    H, W = img.shape[:2]
    # 6px staleness bar across the top.
    cv2.rectangle(img, (0, 0), (W, 6), _staleness_color(stale), -1)
    # Stream name top-left (slightly lower so it doesn't overlap the bar).
    cv2.putText(img, name, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    # FPS top-right with a black backdrop.
    fps_text = f"{fps:5.1f} FPS"
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    x0, y0 = W - tw - 16, 14
    cv2.rectangle(img, (x0 - 8, y0), (x0 + tw + 8, y0 + th + 12), (0, 0, 0), -1)
    cv2.putText(img, fps_text, (x0, y0 + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    # gid bottom-left so viewer can audit causal ordering directly.
    if gid >= 0:
        gid_text = f"gid={gid}  stale={stale}"
        (gtw, gth), _ = cv2.getTextSize(gid_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (8, H - gth - 14), (12 + gtw + 8, H - 4), (0, 0, 0), -1)
        cv2.putText(img, gid_text, (12, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


def _fmt_clock(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


def _perf_sidebar(slots, trackers, t_rec_start, source_thread, total_fps,
                  gpu_name: str, args_state: dict,
                  dropped_ticks: int = 0, max_lag_ms: float = 0.0,
                  src_idx_at_rec_start: int = 0,
                  height: int = 1080):
    """Right-side 400×{height} performance panel.

    Replaces the in-grid stats tile. Shows TOTAL FPS, recorder wall-clock,
    source-pace drift (% behind target FPS), per-stream FPS/latency/
    staleness with a color dot, and the recorder lag accounting.

    ``height`` matches the grid height so this can be hstacked into the
    final composite without padding.
    """
    W, H = 400, height
    img = np.zeros((H, W, 3), dtype=np.uint8)
    rs = slots["rgb"].snapshot()
    now = time.monotonic()

    # Header: TRACKER FPS — the model's TRUE inference rate (the Gibbs step only),
    # NOT the matter worker's raw publish rate. The matter worker also draws 8
    # debug tiles before publishing, so its publish rate (``total_fps``) is
    # gibbs+render; we surface gibbs / render / the actual output rate as their
    # own line below so the headline reflects the tracker and nothing is hidden.
    mex = (slots["matter"].snapshot().extras) or {}
    infer_ms  = float(mex.get("infer_ms", 0.0) or 0.0)
    render_ms = float(mex.get("render_ms", 0.0) or 0.0)
    tracker_fps = (1000.0 / infer_ms) if infer_ms > 0 else total_fps
    cv2.putText(img, "TRACKER FPS", (W//2 - 95, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
    big = f"{tracker_fps:4.1f}"
    (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 2.4, 5)
    cv2.putText(img, big, (W//2 - tw//2, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, (60, 255, 255), 5, cv2.LINE_AA)
    # Inference vs render vs actual output, broken out so the counter is honest:
    # the tracker runs at TRACKER FPS; the on-screen 8-tile render adds render_ms,
    # so the matter worker actually emits frames at `out`.
    sub = f"gibbs {infer_ms:4.0f}ms  render {render_ms:4.0f}ms  out {total_fps:4.1f}fps"
    cv2.putText(img, sub, (W//2 - 150, 124),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # Recorder wall-clock (informational).
    rec_elapsed = now - t_rec_start if t_rec_start else 0.0
    cv2.putText(img, f"REC  {_fmt_clock(rec_elapsed)}", (16, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Source pace drift: how is the source thread keeping up with its target
    # FPS over the recording window? Computed from frames-published-since-rec-
    # start using the monotonic source_global_id (GID), which doesn't reset
    # across MP4 loop iterations (unlike source_idx).
    target_fps = args_state.get("source_fps", 30)
    uncapped   = args_state.get("uncapped_source", False)
    if rec_elapsed > 0.1:
        src_gid_now = rs.source_global_id
        src_frames_window = max(0, src_gid_now - src_idx_at_rec_start)
        achieved = src_frames_window / rec_elapsed
    else:
        achieved = 0.0

    if uncapped:
        line  = f"src: {achieved:5.1f} fps (uncapped)"
        color = (60, 255, 255)
    else:
        drift_pct = ((target_fps - achieved) / target_fps * 100) if target_fps > 0 else 0.0
        line  = f"src: {achieved:5.1f}/{target_fps} fps  drift {drift_pct:+5.1f}%"
        a = abs(drift_pct)
        color = (0, 200, 0) if a < 5 else ((0, 220, 220) if a < 15 else (40, 40, 255))
    cv2.putText(img, line, (16, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Per-stream table header
    y = 220
    cv2.putText(img, "stream      FPS    lat     stale", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    y += 24

    rows = [("rgb", "rgb"), ("depth", "depth"), ("flow", "flow"),
            ("features", "features"), ("matter", "matter")]
    for label, slot_name in rows:
        s = slots[slot_name].snapshot()
        stale_f = rs.source_global_id - s.source_global_id if s.valid else -1
        lat_ms  = s.latency * 1000
        # Color dot for staleness
        dot_col = _staleness_color(max(stale_f, 0))
        cv2.circle(img, (24, y - 6), 6, dot_col, -1)
        line = f"{label:<10}{trackers[slot_name].fps():5.1f}  {lat_ms:5.1f}ms  {stale_f:>3}f"
        cv2.putText(img, line, (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        y += 30

    # Legend
    y += 12
    cv2.putText(img, "staleness:", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA); y += 24
    cv2.circle(img, (24, y - 5), 6, (0, 200, 0), -1)
    cv2.putText(img, "<= 1 frame", (40, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA); y += 22
    cv2.circle(img, (24, y - 5), 6, (0, 220, 220), -1)
    cv2.putText(img, "2-4 frames", (40, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA); y += 22
    cv2.circle(img, (24, y - 5), 6, (40, 40, 255), -1)
    cv2.putText(img, ">= 5 frames", (40, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA); y += 28

    # GPU + paced/uncapped state
    cv2.putText(img, gpu_name[:38], (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA); y += 24
    pace_str = "source: UNCAPPED" if args_state.get("uncapped_source") else \
               f"source: paced  {args_state.get('source_fps', 30):.0f} fps"
    cv2.putText(img, pace_str, (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA); y += 24
    cv2.putText(img, f"recorder:      {args_state.get('rec_fps', 30):.0f} fps", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA); y += 26

    # Recorder lag accounting
    lag_col = (255, 255, 255) if dropped_ticks == 0 else (40, 40, 255)
    cv2.putText(img, f"dropped ticks: {dropped_ticks}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, lag_col, 1, cv2.LINE_AA); y += 22
    cv2.putText(img, f"max rec lag:   {max_lag_ms:5.1f} ms", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, lag_col, 1, cv2.LINE_AA); y += 22

    return img


class Recorder(threading.Thread):
    """Capture tiles at the recorder FPS and write them to an MP4.

    Geometry (output width × height = 2320 × 1080):
      - 4×2 grid of 480×360 tiles on the upper-left (1920×720).
      - 1×2 grid of 960×360 point-cloud tiles below the grid (1920×360).
      - 400×1080 performance sidebar on the right.

    Pacing is deadline-skip: under load the recorder pads with the last
    captured tile + a "RECORDER LAG +Nms" overlay rather than spin-writing
    fresh-looking frames it didn't actually have time to capture.
    """

    TILE_W, TILE_H = 480, 360
    WIDE_TILE_W    = 960          # ROW3 tiles span 2 cells each
    SIDEBAR_W      = 400
    # ROW1: streaming inputs / per-frame model outputs.
    # ROW2: GenMatter++ Gibbs 2D outputs (ellipse overlays + per-pixel masks).
    # ROW3: same per-pixel-mask data but unprojected via depth into a 3/4 view
    #        3D point cloud — depth structure visible, identical coloring.
    ROW1 = ["rgb",      "depth",    "flow",            "dense_features"]
    ROW2 = ["clusters", "particles", "pixel_by_cluster", "pixel_by_particle"]
    ROW3 = ["pointcloud_by_cluster", "pointcloud_by_particle"]
    # ROW4: the PROBABILISTIC particles/clusters as 3D covariance ellipsoids,
    #        drawn in the same rotated view as ROW3 (each particle's 3D mean +
    #        spread, not hard per-pixel colors).
    ROW4 = ["particles_3d", "clusters_3d"]
    # Map each tile name to the slot whose version drives its FPS counter
    # and gid badge.  All ROW2 + ROW3 + ROW4 tiles come from MatterWorker.
    TILE_STREAM = {
        "rgb":               "rgb",
        "depth":             "depth",
        "flow":              "flow",
        "dense_features":    "features",
        "clusters":          "matter",
        "particles":         "matter",
        "pixel_by_cluster":  "matter",
        "pixel_by_particle": "matter",
        "pointcloud_by_cluster":  "matter",
        "pointcloud_by_particle": "matter",
        "particles_3d":      "matter",
        "clusters_3d":       "matter",
    }

    def __init__(self, slots, out_path, fps=30, duration_s=10.0,
                 source_thread=None, gpu_name="?", args_state=None):
        super().__init__(daemon=True, name="Recorder")
        self.slots, self.out_path = slots, out_path
        self.fps, self.duration   = fps, duration_s
        self.source_thread        = source_thread
        self.gpu_name             = gpu_name
        self.args_state           = args_state or {}
        self.W = self.TILE_W * 4 + self.SIDEBAR_W
        # 4 rows of TILE_H each: ROW1 + ROW2 (4 tiles wide), then ROW3 (2 wide
        # point-cloud tiles) + ROW4 (2 wide 3D-ellipsoid tiles), each 960 wide.
        self.H = self.TILE_H * 4
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, (self.W, self.H))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {out_path}")
        self.trackers = {k: FpsTracker() for k in ["rgb","depth","flow","features","matter"]}
        self.frames_written = 0
        self.dropped_ticks  = 0
        self.max_lag_ms     = 0.0
        self.t_start        = 0.0
        # Captured at recorder start so source-pace drift is measured over
        # the recording window, not from epoch.
        self.src_idx_at_rec_start = 0

    def _draw_lag_overlay(self, img, lag_ms: float):
        text = f"RECORDER LAG +{lag_ms:.0f} ms"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        H, W = img.shape[:2]
        x0, y0 = (W - tw) // 2, H // 2 - 20
        cv2.rectangle(img, (x0 - 16, y0 - th - 12),
                            (x0 + tw + 16, y0 + 14), (0, 0, 0), -1)
        cv2.putText(img, text, (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 255), 3, cv2.LINE_AA)

    def _tile(self) -> Optional[np.ndarray]:
        rs = self.slots["rgb"].snapshot()
        if not rs.valid:
            return None

        # Snapshot all slots at one wall-clock instant — this is the causal
        # cut. Anything written afterwards belongs to a future frame.
        snaps = {k: self.slots[k].snapshot()
                  for k in ["rgb","depth","flow","features","matter"]}

        # Update FPS trackers from the snapshot versions (one observation per
        # recorder tick — honest from the recorder's frame of reference).
        for k, s in snaps.items():
            self.trackers[k].update(s.version)

        blank = np.zeros((self.TILE_H, self.TILE_W, 3), dtype=np.uint8)

        ds, flsn, ftsn, mat = snaps["depth"], snaps["flow"], snaps["features"], snaps["matter"]

        # Resolve each tile's image AND the source_global_id it derives from.
        # `cells_imgs[name] = (img, gid)`.  All ROW2 tiles come from MatterWorker.
        def _payload_or_blank(snap, key):
            return snap.payload[key] if snap.valid and snap.payload and key in snap.payload else blank

        wide_blank = np.zeros((self.TILE_H, self.WIDE_TILE_W, 3), dtype=np.uint8)
        cells_imgs = {
            "rgb":               (rs.payload["bgr"],                              rs.source_global_id),
            "depth":             (_payload_or_blank(ds,   "viz_bgr"),             ds.source_global_id),
            "flow":              (_payload_or_blank(flsn, "viz_bgr"),             flsn.source_global_id),
            "dense_features":    (_payload_or_blank(ftsn, "dense_viz_bgr"),       ftsn.source_global_id),
            "clusters":          (_payload_or_blank(mat,  "clusters"),            mat.source_global_id),
            "particles":         (_payload_or_blank(mat,  "particles"),           mat.source_global_id),
            "pixel_by_cluster":  (_payload_or_blank(mat,  "pixel_by_cluster"),    mat.source_global_id),
            "pixel_by_particle": (_payload_or_blank(mat,  "pixel_by_particle"),   mat.source_global_id),
        }

        # ROW3 tiles are wide (960×360) and may not be ready if matter slot is
        # empty. Use a wide blank in that case.
        def _wide_payload_or_blank(snap, key):
            if snap.valid and snap.payload and key in snap.payload:
                return snap.payload[key]
            return wide_blank
        cells_imgs["pointcloud_by_cluster"]  = (_wide_payload_or_blank(mat, "pointcloud_by_cluster"),  mat.source_global_id)
        cells_imgs["pointcloud_by_particle"] = (_wide_payload_or_blank(mat, "pointcloud_by_particle"), mat.source_global_id)
        cells_imgs["particles_3d"] = (_wide_payload_or_blank(mat, "particles_3d"), mat.source_global_id)
        cells_imgs["clusters_3d"]  = (_wide_payload_or_blank(mat, "clusters_3d"),  mat.source_global_id)

        # Resize each tile to its grid cell size, ensure writable, draw label.
        # ROW1+ROW2 cells are 480x360; ROW3 cells are 960x360.
        def _prep_tile(name, target_w):
            img, gid = cells_imgs[name]
            if img.shape[:2] != (self.TILE_H, target_w):
                img = cv2.resize(img, (target_w, self.TILE_H), interpolation=cv2.INTER_AREA)
            else:
                img = img.copy() if img.flags.writeable else img.copy()
            stale = max(0, rs.source_global_id - gid) if gid >= 0 else 0
            stream_name = self.TILE_STREAM[name]
            _draw_tile_label(img, name, self.trackers[stream_name].fps(),
                              gid=gid, stale=stale)
            cells_imgs[name] = (img, gid)
        for name in self.ROW1 + self.ROW2:
            _prep_tile(name, self.TILE_W)
        for name in self.ROW3 + self.ROW4:
            _prep_tile(name, self.WIDE_TILE_W)

        # Compose: ROW1 over ROW2 (both 4×480), then ROW3 (2×960 point cloud)
        # and ROW4 (2×960 3D ellipsoids). Equal widths since 4*480 == 2*960.
        row1 = np.hstack([cells_imgs[n][0] for n in self.ROW1])
        row2 = np.hstack([cells_imgs[n][0] for n in self.ROW2])
        row3 = np.hstack([cells_imgs[n][0] for n in self.ROW3])
        row4 = np.hstack([cells_imgs[n][0] for n in self.ROW4])
        grid = np.vstack([row1, row2, row3, row4])

        # Total FPS = matter rate (most downstream stage that depends on
        # depth + flow + features). All tile stales feed off rs.gid.
        total_fps = self.trackers["matter"].fps()
        sidebar = _perf_sidebar(
            self.slots, self.trackers,
            t_rec_start=self.t_start, source_thread=self.source_thread,
            total_fps=total_fps,
            gpu_name=self.gpu_name, args_state=self.args_state,
            dropped_ticks=self.dropped_ticks,
            max_lag_ms=self.max_lag_ms,
            src_idx_at_rec_start=self.src_idx_at_rec_start,
            height=self.H,
        )
        return np.hstack([grid, sidebar])

    def run(self):
        period = 1.0 / self.fps
        self.t_start = time.monotonic()
        # Baseline for source-pace drift: monotonic global_id of the latest
        # source frame at recording start. (source_idx resets on each MP4
        # loop; global_id does not.)
        self.src_idx_at_rec_start = self.slots["rgb"].snapshot().source_global_id
        next_t  = self.t_start
        last_tile = None
        while not STOP.is_set() and (time.monotonic() - self.t_start) < self.duration:
            # Wait until the next scheduled tick (or skip past missed ticks).
            now = time.monotonic()
            wait = next_t - now
            if wait > 0:
                time.sleep(wait)

            # If we're still behind by more than one full period after sleeping,
            # we missed N ticks. Pad them with the previous tile + a lag overlay
            # so playback wall-time stays accurate AND the viewer can see the
            # lag directly. This is the causal-honesty fix: without it we'd
            # spin-write fresh-looking frames at the wrong wall-clock rate.
            now = time.monotonic()
            lag_s = now - next_t
            if lag_s > period:
                skipped = int(lag_s / period)
                self.dropped_ticks += skipped
                self.max_lag_ms = max(self.max_lag_ms, lag_s * 1000)
                for _ in range(skipped):
                    pad = last_tile.copy() if last_tile is not None else \
                          np.zeros((self.H, self.W, 3), np.uint8)
                    self._draw_lag_overlay(pad, lag_s * 1000)
                    self.writer.write(pad)
                    self.frames_written += 1
                next_t += skipped * period

            tile = self._tile()
            if tile is not None:
                self.writer.write(tile)
                self.frames_written += 1
                last_tile = tile
            next_t += period
        self.writer.release()


# ---------- Entry point ----------

def _transcode_to_h264(in_path, out_path):
    """cv2.VideoWriter('mp4v') writes MPEG-4 simple profile which some
    players (and browsers) don't decode. Re-encode to H.264 yuv420p for
    universal playback."""
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found — skipping h264 transcode")
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", in_path,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart",
           out_path]
    subprocess.run(cmd, check=True)
    return True


def _warm_threadpool_controller():
    """Prime sklearn's threadpoolctl controller in the MAIN thread before any
    worker threads spawn.

    ``threadpoolctl`` lazy-initializes by walking every loaded ``.so`` via
    ``dl_iterate_phdr`` and ``realpath``-ing each path to detect OpenMP/BLAS
    pools.  With JAX + Torch + Numpy + cv2 + sklearn loaded the list runs to
    several hundred libraries; the first call can take minutes on this system
    and hangs the process at "entering K-means hierarchical init" if it
    happens inside MatterWorker's thread while that thread is holding the
    GPU semaphore (blocking depth/flow/feature workers indefinitely).

    Warming the controller from the main thread converts the hang into a
    one-time cold-start cost during ``main()``, after which sklearn KMeans in
    MatterWorker hits the cached controller and returns in ~0.1 s.
    """
    try:
        from sklearn.utils.parallel import _get_threadpool_controller
        _get_threadpool_controller()
    except Exception as e:
        print(f"[warm_threadpool_controller] non-fatal: {e!r}", flush=True)


def main(out_path="assets/streaming_demo.mp4", duration=10.0, fps=30,
         uncapped_source=False, source_fps=30, config_path=None,
         sam_frame0_path=None, source_path="assets/test.mp4"):
    _warm_threadpool_controller()
    slots = {k: Slot(name=k) for k in ["rgb","depth","flow","features","matter"]}

    STOP.clear()
    # Source pacing: throttle_fps=False disables — workers run flat-out and
    # the viewer can see true throughput. Default mimics a 30 fps camera.
    src_throttle = False if uncapped_source else source_fps
    source = FrameSource(source_path, slots["rgb"],
                         throttle_fps=src_throttle, resize=RESIZE)
    matter_worker = MatterWorker(slots["depth"], slots["flow"], slots["features"],
                                  slots["rgb"], slots["matter"],
                                  config_path=config_path,
                                  sam_frame0_path=sam_frame0_path)
    workers = [
        source,
        DepthWorker(slots["rgb"], slots["depth"]),
        FlowWorker(slots["rgb"], slots["flow"]),
        FeatureWorker(slots["rgb"], slots["features"]),
        matter_worker,
    ]
    for w in workers:
        w.start()

    # Let the perception workers warm up so the MatterWorker has every stream
    # available on its first iteration.  Then block until MatterWorker
    # finishes its (slow) K-means + Gibbs warm-up + JIT compile, so the
    # Recorder doesn't start with an empty matter tile.
    time.sleep(1.5)
    print("waiting for MatterWorker init (K-means + JIT compile)...")
    t_wait = time.monotonic()
    while not slots["matter"].snapshot().valid:
        if time.monotonic() - t_wait > 180:
            print("WARNING: MatterWorker did not produce output in 180s; "
                  "continuing anyway")
            break
        time.sleep(0.5)
    print(f"MatterWorker init done in {time.monotonic() - t_wait:.1f}s")

    args_state = {
        "uncapped_source": uncapped_source,
        "source_fps":      source_fps,
        "rec_fps":         fps,
    }
    gpu_name = torch.cuda.get_device_name(0)

    tmp_path = out_path + ".mpeg4.mp4"
    rec = Recorder(slots, out_path=tmp_path, fps=fps, duration_s=duration,
                   source_thread=source, gpu_name=gpu_name, args_state=args_state)
    print(f"recording {duration:.1f} s at {fps} fps → {out_path}")
    rec.start()
    rec.join()
    STOP.set()
    for w in workers:
        w.join(timeout=2)

    print(f"wrote {rec.frames_written} frames ({rec.frames_written / fps:.2f} s)")
    print(f"dropped ticks: {rec.dropped_ticks}, max lag: {rec.max_lag_ms:.1f} ms")
    print(f"final per-stream FPS:")
    for k in ["rgb","depth","flow","features","matter"]:
        print(f"  {k:<8} {rec.trackers[k].fps():5.2f}")

    hist = matter_worker._outlier_history
    if hist:
        arr = np.array(hist, dtype=np.float32)
        p95 = float(np.percentile(arr, 95))
        print(f"matter outlier_frac: n={len(arr)} mean={arr.mean():.4f} "
              f"max={arr.max():.4f} p95={p95:.4f}")

    print(f"transcoding to h264 → {out_path}")
    if _transcode_to_h264(tmp_path, out_path):
        os.remove(tmp_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"done. {out_path} ({size_mb:.1f} MB)")
    else:
        os.rename(tmp_path, out_path)
        print(f"left mpeg4 output at {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out",       default="assets/streaming_demo.mp4")
    p.add_argument("--source",    default="assets/test.mp4",
                   help="Source video file (mp4/mov) or RGB-frames dir to stream "
                        "through the live demo. Loops on EOF (real-time camera sim).")
    p.add_argument("--duration",  type=float, default=10.0)
    p.add_argument("--fps",       type=int,   default=30,
                   help="Recorder tick rate / output MP4 FPS.")
    p.add_argument("--source-fps", type=int, default=30,
                   help="Source pacing FPS (camera simulation).")
    p.add_argument("--uncapped-source", action="store_true",
                   help="Disable source pacing — workers run flat-out, "
                        "viewer sees true achievable throughput.")
    p.add_argument("--config", default=None,
                   help="YAML hyperparameter config for MatterWorker "
                        "(default: configs/streaming_default.yaml; pass "
                        "configs/streaming_general.yaml for the multi-video "
                        "live-calibrated config once it has been written).")
    p.add_argument("--sam-frame0", default="",
                   help="Cached SAM-frame-0 mask for semantic init (opt-in; only "
                        "used when tracking.use_sam_frame0 is also set). OFF by "
                        "default: the live demo's slow init lands on a late/looped "
                        "frame, so a frame-0 mask would be spatially misaligned. "
                        "Pass e.g. assets/custom_videos/test/SAM_frame0/"
                        "test_SAM_frame0.png to experiment.")
    args = p.parse_args()
    main(out_path=args.out, duration=args.duration, fps=args.fps,
         uncapped_source=args.uncapped_source, source_fps=args.source_fps,
         config_path=args.config, sam_frame0_path=args.sam_frame0 or None,
         source_path=args.source)

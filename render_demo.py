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
frame; fusion is usually `1–2f` because it depends on depth/flow/features
which themselves are ≥1f stale.

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

import sys, json, argparse, time, threading, subprocess, shutil
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Dict, List, Tuple

sys.path.insert(0, os.path.abspath("third_party/SEA-RAFT/core"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModel
from torchvision.utils import flow_to_image
from raft import RAFT


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

print("loading models...")
depth_ckpt  = "depth-anything/Depth-Anything-V2-Small-hf"
depth_proc  = AutoImageProcessor.from_pretrained(depth_ckpt)
depth_model = AutoModelForDepthEstimation.from_pretrained(
                  depth_ckpt, torch_dtype=torch.float16).to(DEVICE).eval()

SEA_CFG = "third_party/SEA-RAFT/config/eval/spring-M.json"
with open(SEA_CFG) as f:
    cfg = json.load(f)
sea_args = argparse.Namespace(**cfg); sea_args.iters = 4; sea_args.scale = 0
flow_model = RAFT.from_pretrained("MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
                                  args=sea_args).to(DEVICE).eval()

dino_ckpt  = "facebook/dinov2-small"
dino_proc  = AutoImageProcessor.from_pretrained(dino_ckpt)
dino_model = AutoModel.from_pretrained(dino_ckpt, torch_dtype=torch.float16).to(DEVICE).eval()
print("models loaded.")


# ---------- JAX fusion kernels ----------

SOBEL_X = jnp.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=jnp.float32)
SOBEL_Y = SOBEL_X.T

def _gauss1d(sigma, radius):
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    k = jnp.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()

def _blur2d(img, sigma=3.0, radius=9):
    k = _gauss1d(sigma, radius)
    img = convolve2d(img, k[None, :], mode="same")
    img = convolve2d(img, k[:, None], mode="same")
    return img

@jax.jit
def fx_neon_glow(depth, flow_mag, frame_bgr):
    dx = convolve2d(depth, SOBEL_X, mode="same")
    dy = convolve2d(depth, SOBEL_Y, mode="same")
    edges = jnp.sqrt(dx**2 + dy**2)
    edges = edges / (edges.max() + 1e-6)
    motion = edges * (flow_mag / (flow_mag.max() + 1e-6))
    glow = jnp.clip(_blur2d(motion, sigma=2.5, radius=7) * 6.0, 0., 1.)[..., None]
    neon = jnp.array([220., 50., 255.], dtype=jnp.float32)
    return jnp.clip(frame_bgr.astype(jnp.float32) + glow * neon, 0., 255.).astype(jnp.uint8)

@jax.jit
def fx_salience(depth, flow_mag, frame_bgr):
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    f_norm = flow_mag / (flow_mag.max() + 1e-6)
    s = jnp.clip(_blur2d((1. - d_norm) * f_norm, sigma=3.0, radius=9) * 4.0, 0., 1.)
    r = jnp.clip(3.*s,       0., 1.)
    g = jnp.clip(3.*s - 1.,  0., 1.)
    b = jnp.clip(3.*s - 2.,  0., 1.)
    hot_bgr = jnp.stack([b, g, r], axis=-1) * 255.
    alpha = s[..., None] * 0.7
    return jnp.clip(frame_bgr.astype(jnp.float32) * (1 - alpha)
                    + hot_bgr * alpha, 0., 255.).astype(jnp.uint8)

@jax.jit
def fx_topo(depth, flow_x, flow_y, frame_bgr):
    n_bands = 12
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    bands = jnp.floor(d_norm * n_bands)
    dv = jnp.abs(jnp.diff(bands, axis=0, prepend=bands[:1, :]))
    dh = jnp.abs(jnp.diff(bands, axis=1, prepend=bands[:, :1]))
    edge = ((dv + dh) > 0).astype(jnp.float32)
    edge = jnp.clip(_blur2d(edge, sigma=1.0, radius=3) * 2.0, 0., 1.)
    hue = (jnp.arctan2(flow_y, flow_x) + jnp.pi) / (2. * jnp.pi)
    h6 = hue * 6.0
    x = 1. - jnp.abs((h6 % 2.) - 1.)
    sec = jnp.floor(h6).astype(jnp.int32) % 6
    r = jnp.choose(sec, [jnp.ones_like(x), x, jnp.zeros_like(x),
                         jnp.zeros_like(x), x, jnp.ones_like(x)], mode="clip")
    g = jnp.choose(sec, [x, jnp.ones_like(x), jnp.ones_like(x),
                         x, jnp.zeros_like(x), jnp.zeros_like(x)], mode="clip")
    b = jnp.choose(sec, [jnp.zeros_like(x), jnp.zeros_like(x), x,
                         jnp.ones_like(x), jnp.ones_like(x), x], mode="clip")
    rgb_bgr = jnp.stack([b, g, r], axis=-1) * 255.
    return jnp.clip(frame_bgr.astype(jnp.float32) + rgb_bgr * edge[..., None],
                    0., 255.).astype(jnp.uint8)

# --- DINO feature kernels ---

# Number of patches DINOv2-S/14 produces at the 224x224 input the HF processor
# resizes to: gh = gw = 16, so 256 patches. Fixed for jit shape stability.
N_DINO_PATCHES = 256
DINO_DIM       = 384
KMEANS_K       = 8
KMEANS_ITERS   = 5

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


@jax.jit
def fx_kmeans_clusters(feats, centers):
    """K-means on dense DINO features. Warm-started from previous frame's centers
    so feature-cluster identity is stable enough to permute on the host.

    feats:   (N_DINO_PATCHES, DINO_DIM) float32
    centers: (KMEANS_K, DINO_DIM) float32
    Returns (labels, new_centers).
    """
    def body(_, c):
        d2 = ((feats[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        labels = jnp.argmin(d2, axis=1)
        oh = jax.nn.one_hot(labels, c.shape[0])
        counts = oh.sum(0)
        new = (oh.T @ feats) / jnp.maximum(counts, 1.0)[:, None]
        # If a cluster is empty (count == 0), keep its previous center.
        return jnp.where(counts[:, None] > 0, new, c)

    centers = jax.lax.fori_loop(0, KMEANS_ITERS, body, centers)
    d2 = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    labels = jnp.argmin(d2, axis=1)
    return labels, centers


# Warmup at the working resolution.
_z2 = jnp.zeros((RESIZE[1], RESIZE[0]), dtype=jnp.float32)
_z3 = jnp.zeros((RESIZE[1], RESIZE[0], 3), dtype=jnp.uint8)
fx_neon_glow(_z2, _z2, _z3).block_until_ready()
fx_salience(_z2, _z2, _z3).block_until_ready()
fx_topo(_z2, _z2, _z2, _z3).block_until_ready()

_zfeat   = jnp.zeros((N_DINO_PATCHES, DINO_DIM), dtype=jnp.float32)
_zcenters = jnp.zeros((KMEANS_K, DINO_DIM), dtype=jnp.float32)
_p, _q = fx_dino_pca(_zfeat); _p.block_until_ready(); _q.block_until_ready()
_l, _c = fx_kmeans_clusters(_zfeat, _zcenters); _l.block_until_ready(); _c.block_until_ready()


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
            with GPU, torch.inference_mode():
                inp = dino_proc(images=rgb, return_tensors="pt").to(DEVICE, dtype=torch.float16)
                out = dino_model(**inp).last_hidden_state[0]
                patches = out[1:]
                feat_np = patches.float().cpu().numpy()
                n = patches.shape[0]
                gh = gw = int(round(np.sqrt(n)))

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


# Tab10-style 8-color BGR palette for k-means clusters. Chosen for high
# inter-cluster contrast and reasonable luminance.
_KMEANS_PALETTE = np.array([
    [180,  50,  31],   # blue
    [ 14, 127, 255],   # orange
    [ 44, 160,  44],   # green
    [ 40,  39, 214],   # red
    [189, 103, 148],   # purple
    [ 75,  86, 140],   # brown
    [194, 119, 227],   # pink
    [127, 127, 127],   # gray
], dtype=np.uint8)


def _permute_kmeans_labels(new_centers, prev_centers):
    """Greedy match new clusters → prev clusters by cosine similarity.

    Returns perm: array of length K where perm[i] = the previous-cluster
    slot that new cluster i should be reassigned to. Used to remap labels
    and reorder centers so cluster colors track regions across frames.
    """
    K = new_centers.shape[0]
    a = new_centers / (np.linalg.norm(new_centers, axis=1, keepdims=True) + 1e-9)
    b = prev_centers / (np.linalg.norm(prev_centers, axis=1, keepdims=True) + 1e-9)
    S = a @ b.T   # (K, K)
    perm = np.full(K, -1, dtype=np.int32)
    used_prev = set()
    used_new  = set()
    Sm = S.copy()
    for _ in range(K):
        idx = int(np.argmax(Sm))
        i, j = idx // K, idx % K
        perm[i] = j
        used_new.add(i); used_prev.add(j)
        Sm[i, :] = -np.inf
        Sm[:, j] = -np.inf
    return perm


class FusionWorker(threading.Thread):
    def __init__(self, depth, flow, rgb, features, out, poll=0.001):
        super().__init__(daemon=True, name="FusionWorker")
        self.depth, self.flow, self.rgb, self.features = depth, flow, rgb, features
        self.out, self.poll = out, poll
        self._last_d_v = self._last_f_v = self._last_ft_v = -1
        # K-means warm-start state. None on first frame → init from feats.
        self._km_centers: Optional[jnp.ndarray] = None

    def run(self):
        while not STOP.is_set():
            ds   = self.depth.snapshot()
            fs   = self.flow.snapshot()
            rs   = self.rgb.snapshot()
            ftsn = self.features.snapshot()
            # We need depth + rgb + features to produce a useful tile.
            # Flow is allowed to be invalid (we'll zero it).
            if not (ds.valid and rs.valid and ftsn.valid):
                time.sleep(self.poll); continue
            if (ds.version == self._last_d_v and fs.version == self._last_f_v
                    and ftsn.version == self._last_ft_v):
                time.sleep(self.poll); continue
            t0 = time.monotonic()
            depth_j = ds.payload["jax"]
            frame_j = jnp.asarray(rs.payload["bgr"])
            if fs.valid:
                flow_raw = fs.payload["raw"]
                fx_j = jnp.asarray(flow_raw[0].astype(np.float32))
                fy_j = jnp.asarray(flow_raw[1].astype(np.float32))
                fm_j = jnp.sqrt(fx_j**2 + fy_j**2)
            else:
                H, W = depth_j.shape
                fx_j = jnp.zeros((H, W), dtype=jnp.float32)
                fy_j = jnp.zeros((H, W), dtype=jnp.float32)
                fm_j = jnp.zeros((H, W), dtype=jnp.float32)

            feat_jax = ftsn.payload["jax"]
            gh, gw   = ftsn.payload["grid_hw"]
            # Initialize k-means centers on the very first frame from this
            # frame's features (evenly-spaced rows for diversity). On
            # subsequent frames, warm-start from the previous run's centers.
            if self._km_centers is None:
                idx = jnp.linspace(0, N_DINO_PATCHES - 1, KMEANS_K).astype(jnp.int32)
                init_centers = feat_jax[idx]
            else:
                init_centers = self._km_centers

            with GPU:
                neon = fx_neon_glow(depth_j, fm_j, frame_j).block_until_ready()
                sal  = fx_salience(depth_j, fm_j, frame_j).block_until_ready()
                topo = fx_topo(depth_j, fx_j, fy_j, frame_j).block_until_ready()
                labels_j, new_centers_j = fx_kmeans_clusters(feat_jax, init_centers)
                labels_j.block_until_ready()
                new_centers_j.block_until_ready()

            new_centers_np = np.array(new_centers_j)
            labels_np      = np.array(labels_j).astype(np.int32)

            # Greedy permutation against previous centers so cluster colors
            # stay attached to the same semantic region across frames. On
            # the first frame, perm is identity (initial state).
            if self._km_centers is not None:
                prev_np = np.array(self._km_centers)
                perm    = _permute_kmeans_labels(new_centers_np, prev_np)
                # remap labels: each label i maps to perm[i]
                labels_np = perm[labels_np]
                # reorder centers so slot j holds the cluster that now represents
                # what slot j represented before (used as warm-start next frame).
                reordered = np.zeros_like(new_centers_np)
                reordered[perm] = new_centers_np
                new_centers_np = reordered

            self._km_centers = jnp.asarray(new_centers_np)

            # Build cluster colormap: (gh, gw) → upsample NEAREST → palette index.
            labels_grid = labels_np.reshape(gh, gw)
            labels_full = cv2.resize(labels_grid.astype(np.uint8),
                                     (frame_j.shape[1], frame_j.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            kmeans_bgr = _KMEANS_PALETTE[labels_full]   # (H, W, 3) uint8

            payload = {
                "neon":     np.array(neon),
                "salience": np.array(sal),
                "topo":     np.array(topo),
                "kmeans":   kmeans_bgr,
            }
            t1 = time.monotonic()
            self.out.publish(
                payload=payload,
                source_global_id=rs.source_global_id,
                wall_start=t0, wall_complete=t1, latency=t1-t0,
                extras={
                    "depth_global_id":    ds.source_global_id,
                    "flow_global_id":     fs.source_global_id if fs.valid else -1,
                    "features_global_id": ftsn.source_global_id,
                },
                error=None,
            )
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
                  src_idx_at_rec_start: int = 0):
    """Right-side 400x720 performance panel.

    Replaces the in-grid stats tile. Shows TOTAL FPS, recorder wall-clock,
    source-pace drift (% behind target FPS), per-stream FPS/latency/
    staleness with a color dot, and the recorder lag accounting.
    """
    W, H = 400, 720
    img = np.zeros((H, W, 3), dtype=np.uint8)
    rs = slots["rgb"].snapshot()
    now = time.monotonic()

    # Header: TOTAL FPS
    cv2.putText(img, "TOTAL FPS", (W//2 - 80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
    big = f"{total_fps:4.1f}"
    (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 2.4, 5)
    cv2.putText(img, big, (W//2 - tw//2, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, (60, 255, 255), 5, cv2.LINE_AA)

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
            ("features", "features"), ("fusion", "fusion")]
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

    Geometry (output width × height = 2320 × 720):
      - 4×2 grid of 480×360 tiles on the left (1920×720).
      - 400×720 performance sidebar on the right.

    Pacing is deadline-skip: under load the recorder pads with the last
    captured tile + a "RECORDER LAG +Nms" overlay rather than spin-writing
    fresh-looking frames it didn't actually have time to capture.
    """

    TILE_W, TILE_H = 480, 360
    SIDEBAR_W      = 400
    # Tile grid layout: row 1 keeps inputs/per-frame model outputs; row 2 is
    # JAX-accelerated overlays. Last cell of row 2 is the new k-means tile.
    ROW1 = ["rgb",  "depth",    "flow", "dense_features"]
    ROW2 = ["neon", "salience", "topo", "kmeans"]
    # Map each tile name to the slot whose version drives its FPS counter
    # and gid badge.
    TILE_STREAM = {
        "rgb":            "rgb",
        "depth":          "depth",
        "flow":           "flow",
        "dense_features": "features",
        "neon":           "fusion",
        "salience":       "fusion",
        "topo":           "fusion",
        "kmeans":         "fusion",
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
        self.H = self.TILE_H * 2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, (self.W, self.H))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {out_path}")
        self.trackers = {k: FpsTracker() for k in ["rgb","depth","flow","features","fusion"]}
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
        snaps = {k: self.slots[k].snapshot() for k in ["rgb","depth","flow","features","fusion"]}

        # Update FPS trackers from the snapshot versions (one observation per
        # recorder tick — honest from the recorder's frame of reference).
        for k, s in snaps.items():
            self.trackers[k].update(s.version)

        blank = np.zeros((self.TILE_H, self.TILE_W, 3), dtype=np.uint8)

        ds, flsn, ftsn, fus = snaps["depth"], snaps["flow"], snaps["features"], snaps["fusion"]

        # Resolve each tile's image AND the source_global_id it derives from.
        # `cells_imgs[name] = (img, gid)`.
        def _payload_or_blank(snap, key):
            return snap.payload[key] if snap.valid and snap.payload and key in snap.payload else blank

        cells_imgs = {
            "rgb":            (rs.payload["bgr"],                            rs.source_global_id),
            "depth":          (_payload_or_blank(ds,   "viz_bgr"),           ds.source_global_id),
            "flow":           (_payload_or_blank(flsn, "viz_bgr"),           flsn.source_global_id),
            "dense_features": (_payload_or_blank(ftsn, "dense_viz_bgr"),     ftsn.source_global_id),
            "neon":           (_payload_or_blank(fus,  "neon"),              fus.source_global_id),
            "salience":       (_payload_or_blank(fus,  "salience"),          fus.source_global_id),
            "topo":           (_payload_or_blank(fus,  "topo"),              fus.source_global_id),
            "kmeans":         (_payload_or_blank(fus,  "kmeans"),            fus.source_global_id),
        }

        # Resize each tile to the grid cell size, ensure writable, draw label.
        for name in self.ROW1 + self.ROW2:
            img, gid = cells_imgs[name]
            if img.shape[:2] != (self.TILE_H, self.TILE_W):
                img = cv2.resize(img, (self.TILE_W, self.TILE_H), interpolation=cv2.INTER_AREA)
            else:
                img = img.copy() if img.flags.writeable else img.copy()
            stale = max(0, rs.source_global_id - gid) if gid >= 0 else 0
            stream_name = self.TILE_STREAM[name]
            _draw_tile_label(img, name, self.trackers[stream_name].fps(),
                              gid=gid, stale=stale)
            cells_imgs[name] = (img, gid)

        # Compose 4x2 grid then h-stack the sidebar.
        row1 = np.hstack([cells_imgs[n][0] for n in self.ROW1])
        row2 = np.hstack([cells_imgs[n][0] for n in self.ROW2])
        grid = np.vstack([row1, row2])

        # Total FPS = fusion rate (most downstream stage that depends on
        # depth + flow + features). All tile stales feed off rs.gid.
        total_fps = self.trackers["fusion"].fps()
        sidebar = _perf_sidebar(
            self.slots, self.trackers,
            t_rec_start=self.t_start, source_thread=self.source_thread,
            total_fps=total_fps,
            gpu_name=self.gpu_name, args_state=self.args_state,
            dropped_ticks=self.dropped_ticks,
            max_lag_ms=self.max_lag_ms,
            src_idx_at_rec_start=self.src_idx_at_rec_start,
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


def main(out_path="assets/streaming_demo.mp4", duration=10.0, fps=30,
         uncapped_source=False, source_fps=30):
    slots = {k: Slot(name=k) for k in ["rgb","depth","flow","features","fusion"]}

    STOP.clear()
    # Source pacing: throttle_fps=False disables — workers run flat-out and
    # the viewer can see true throughput. Default mimics a 30 fps camera.
    src_throttle = False if uncapped_source else source_fps
    source = FrameSource("assets/test.mp4", slots["rgb"],
                         throttle_fps=src_throttle, resize=RESIZE)
    workers = [
        source,
        DepthWorker(slots["rgb"], slots["depth"]),
        FlowWorker(slots["rgb"], slots["flow"]),
        FeatureWorker(slots["rgb"], slots["features"]),
        FusionWorker(slots["depth"], slots["flow"], slots["rgb"],
                     slots["features"], slots["fusion"]),
    ]
    for w in workers:
        w.start()

    # Let the pipeline warm up so first frames already have every stream alive.
    time.sleep(1.5)

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
    for k in ["rgb","depth","flow","features","fusion"]:
        print(f"  {k:<8} {rec.trackers[k].fps():5.2f}")

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
    p.add_argument("--duration",  type=float, default=10.0)
    p.add_argument("--fps",       type=int,   default=30,
                   help="Recorder tick rate / output MP4 FPS.")
    p.add_argument("--source-fps", type=int, default=30,
                   help="Source pacing FPS (camera simulation).")
    p.add_argument("--uncapped-source", action="store_true",
                   help="Disable source pacing — workers run flat-out, "
                        "viewer sees true achievable throughput.")
    args = p.parse_args()
    main(out_path=args.out, duration=args.duration, fps=args.fps,
         uncapped_source=args.uncapped_source, source_fps=args.source_fps)

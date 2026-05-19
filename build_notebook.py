"""Build streaming_demo.ipynb from the plan."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------- Title ----------
md(r"""
# Real-Time Streaming Perception — Milestone 1

Async multi-worker substrate running depth (Depth Anything V2 Small), optical flow
(SEA-RAFT-M), and DINOv2 feature extraction off a single looping MP4. Latest-value
slots, GPU semaphore for kernel serialization, JAX fusion overlays, 2D image grid
and k3d 3D point cloud, plus a 60-second soak test.

Run top-to-bottom. Each cell asserts something before the next layer is built. If a
checkpoint fails, stop and fix it.
""")

# ---------- Group A — Setup ----------
md(r"""
## Group A — Setup

Cell A1 must run *before* any `import jax` happens elsewhere — XLA reads the env
vars only at first JAX import.
""")

# A1 — env vars, imports, GPU sanity
code(r"""
# Cell A1 — env vars + imports + GPU sanity (checkpoint #0)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

import sys, json, argparse, time, threading, signal, gc
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Dict, List, Tuple

sys.path.insert(0, os.path.abspath("third_party/SEA-RAFT/core"))

import cv2, numpy as np, torch
import torch.nn.functional as F
from IPython.display import display, Image, clear_output
import ipywidgets as widgets

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

import k3d
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModel
from torchvision.utils import flow_to_image
from raft import RAFT

# Sanity prints
print("torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0),
      "| capability:", torch.cuda.get_device_capability())
print("jax:", jax.__version__, "| devices:", jax.devices())
assert torch.cuda.is_available(), "CUDA not available"
assert torch.cuda.get_device_capability() == (12, 0), \
    f"Expected sm_120 (RTX 5090); got {torch.cuda.get_device_capability()}"
assert "cuda" in str(jax.devices()[0]).lower(), \
    f"JAX not on CUDA: {jax.devices()}"
print("✓ checkpoint #0: env OK")
""")

# A2 — Slot, GPUSemaphore, MonotonicId + smoke test
code(r"""
# Cell A2 — Slot + GPUSemaphore + MonotonicId + smoke test (checkpoint #1)

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
        # Torch CUDA submissions are async — sync before releasing so the next
        # acquirer doesn't race against in-flight kernels. JAX users must also
        # call .block_until_ready() on their result inside the with-block.
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


# Smoke test: two writers, one reader; expect no torn reads, version monotonic.
def _smoke_slot():
    s = Slot(name="test")
    stop = threading.Event()
    def writer(payload):
        while not stop.is_set():
            s.publish(payload=payload, source_global_id=payload["v"])
    t1 = threading.Thread(target=writer, args=({"v": 1},), daemon=True); t1.start()
    t2 = threading.Thread(target=writer, args=({"v": 2},), daemon=True); t2.start()
    versions = []
    for _ in range(2000):
        snap = s.snapshot()
        assert snap.payload in ({"v": 1}, {"v": 2}, None), f"torn read: {snap.payload}"
        versions.append(snap.version)
    stop.set(); t1.join(); t2.join()
    assert versions == sorted(versions), "versions not monotonic"
    print(f"✓ checkpoint #1: Slot atomic, final version={s.version}")
_smoke_slot()
""")

# A3 — load PyTorch models + smoke test
code(r"""
# Cell A3 — load Depth-Anything V2 + SEA-RAFT-M + DINOv2-Small (checkpoint #2)
DEVICE = "cuda"

# --- Depth Anything V2 (Small) — fp16 ---
depth_ckpt  = "depth-anything/Depth-Anything-V2-Small-hf"
depth_proc  = AutoImageProcessor.from_pretrained(depth_ckpt)
depth_model = AutoModelForDepthEstimation.from_pretrained(
                  depth_ckpt, torch_dtype=torch.float16).to(DEVICE).eval()

# --- SEA-RAFT-M from Hugging Face. fp32: fp16 hits a grid_sample dtype mismatch
# in bilinear_sampler (coords are always float32). fp32 still runs ~14ms@360p on a 5090.
SEA_CFG = "third_party/SEA-RAFT/config/eval/spring-M.json"
with open(SEA_CFG) as f: cfg = json.load(f)
sea_args = argparse.Namespace(**cfg); sea_args.iters = 4; sea_args.scale = 0
flow_model = RAFT.from_pretrained("MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
                                  args=sea_args).to(DEVICE).eval()

# --- DINOv2 (Small) — fp16 ---
dino_ckpt  = "facebook/dinov2-small"
dino_proc  = AutoImageProcessor.from_pretrained(dino_ckpt)
dino_model = AutoModel.from_pretrained(dino_ckpt, torch_dtype=torch.float16).to(DEVICE).eval()

# Smoke test — fixed-size dummy frame.
with torch.inference_mode():
    dummy = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

    d_in = depth_proc(images=dummy, return_tensors="pt").to(DEVICE, dtype=torch.float16)
    d_out = depth_model(**d_in).predicted_depth
    assert d_out.ndim == 3 and d_out.shape[0] == 1, f"depth shape {d_out.shape}"

    t1 = torch.from_numpy(dummy).permute(2,0,1).unsqueeze(0).to(DEVICE).float()
    o = flow_model(t1, t1, iters=4, test_mode=True)
    assert "flow" in o and o["flow"][-1].shape[1] == 2, f"flow keys/shape {o['flow'][-1].shape}"

    f_in = dino_proc(images=dummy, return_tensors="pt").to(DEVICE, dtype=torch.float16)
    f_out = dino_model(**f_in).last_hidden_state
    assert f_out.ndim == 3 and f_out.shape[-1] == 384, f"dino shape {f_out.shape}"

print(f"✓ checkpoint #2: depth/flow/dino loaded "
      f"({sum(p.numel() for p in depth_model.parameters())/1e6:.1f}M / "
      f"{sum(p.numel() for p in flow_model.parameters())/1e6:.1f}M / "
      f"{sum(p.numel() for p in dino_model.parameters())/1e6:.1f}M params)")
""")

# A4 — JAX fusion kernels + warmup
code(r"""
# Cell A4 — JAX fusion kernels + warmup (checkpoint #3)
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

# Warmup at the working resolution.
H_VIZ, W_VIZ = 360, 640
_z2 = jnp.zeros((H_VIZ, W_VIZ), dtype=jnp.float32)
_z3 = jnp.zeros((H_VIZ, W_VIZ, 3), dtype=jnp.uint8)
_ = fx_neon_glow(_z2, _z2, _z3).block_until_ready()
_ = fx_salience(_z2, _z2, _z3).block_until_ready()
_ = fx_topo(_z2, _z2, _z2, _z3).block_until_ready()

t0 = time.monotonic()
for _ in range(20):
    fx_neon_glow(_z2, _z2, _z3).block_until_ready()
    fx_salience(_z2, _z2, _z3).block_until_ready()
    fx_topo(_z2, _z2, _z2, _z3).block_until_ready()
elapsed_ms = (time.monotonic() - t0) / 20 * 1000
print(f"✓ checkpoint #3: JAX warmup OK, 3 fx total = {elapsed_ms:.2f} ms/frame")
assert elapsed_ms < 30, f"JAX fusion too slow ({elapsed_ms} ms) — JAX may be on CPU"
""")

# A5 — global state
code(r"""
# Cell A5 — global state: GPU semaphore, frame counter, slots, STOP, RESIZE
GPU  = GPUSemaphore(max_concurrent=1)
GID  = MonotonicId()
slots: Dict[str, Slot] = {k: Slot(name=k) for k in
                          ["rgb","depth","flow","features","fusion"]}
STOP = threading.Event()
RESIZE = (640, 360)  # (W, H)
print("✓ slots, GPU semaphore, stop event, RESIZE set")
""")

# ---------- Group B — Workers ----------
md(r"""
## Group B — FrameSource + workers

Each cell defines a worker class and runs a short integration test that asserts
the worker produces fresh output at the expected rate. The first test that fails
is your bug — fix it before continuing.
""")

# B1 — FrameSource + test
code(r"""
# Cell B1 — FrameSource + test (checkpoint #4)
class FrameSource(threading.Thread):
    def __init__(self, src, out_slot: Slot, throttle_fps=None, resize=(640,360)):
        super().__init__(daemon=True, name="FrameSource")
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {src!r}")
        self.is_file = isinstance(src, str)
        if throttle_fps is None and self.is_file:
            throttle_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.throttle_fps = throttle_fps
        self.out, self.resize = out_slot, resize
        self.loop_idx, self.source_idx = 0, -1
        self._next_t = 0.0

    def run(self):
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
                src_t = self.source_idx / max(self.throttle_fps, 1.0)
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

# --- Test: run source alone for 3 s ---
STOP.clear()
slots["rgb"] = Slot(name="rgb")
src = FrameSource("assets/test.mp4", slots["rgb"], resize=RESIZE)
src.start()
time.sleep(3.0)
STOP.set(); src.join(timeout=2)
snap = slots["rgb"].snapshot()
fps  = snap.source_global_id / 3.0
print(f"✓ checkpoint #4: FrameSource ran {snap.source_global_id} frames "
      f"in 3 s ({fps:.1f} FPS); loops={snap.extras.get('loop_idx',0)}")
assert snap.valid and snap.payload["bgr"].shape == (RESIZE[1], RESIZE[0], 3)
""")

# B2 — DepthWorker + test
code(r"""
# Cell B2 — DepthWorker + test (checkpoint #5)
class DepthWorker(threading.Thread):
    def __init__(self, rgb: Slot, out: Slot, ema_alpha=0.6, poll=0.001):
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
            try:
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
            except Exception as e:
                self.out.publish(error=str(e))
                raise

# --- Test ---
STOP.clear()
slots["rgb"]   = Slot(name="rgb")
slots["depth"] = Slot(name="depth")
src = FrameSource("assets/test.mp4", slots["rgb"], resize=RESIZE); src.start()
dep = DepthWorker(slots["rgb"], slots["depth"]); dep.start()
time.sleep(4.0)
STOP.set(); src.join(2); dep.join(2)
ds = slots["depth"].snapshot(); rs = slots["rgb"].snapshot()
print(f"  rgb v={rs.version}, depth v={ds.version}, depth latency={ds.latency*1000:.1f} ms")
print(f"  staleness frames = {rs.source_global_id - ds.source_global_id}")
assert ds.valid and ds.version > 10, "depth slot didn't update enough"
assert ds.latency < 0.2, f"depth too slow: {ds.latency*1000:.0f} ms"
assert ds.payload["viz_bgr"].shape == (RESIZE[1], RESIZE[0], 3)
print("✓ checkpoint #5: DepthWorker OK")
""")

# B3 — FlowWorker + test
code(r"""
# Cell B3 — FlowWorker + test (checkpoint #6)
class FlowWorker(threading.Thread):
    def __init__(self, rgb: Slot, out: Slot, iters=4, poll=0.001):
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
            try:
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
            except Exception as e:
                self.out.publish(error=str(e))
                raise

# --- Test ---
STOP.clear()
slots["rgb"]  = Slot(name="rgb")
slots["flow"] = Slot(name="flow")
src = FrameSource("assets/test.mp4", slots["rgb"], resize=RESIZE); src.start()
flw = FlowWorker(slots["rgb"], slots["flow"]); flw.start()
time.sleep(4.0)
STOP.set(); src.join(2); flw.join(2)
fs = slots["flow"].snapshot(); rs = slots["rgb"].snapshot()
gap = fs.extras.get("frame_gap", 0)
print(f"  flow v={fs.version}, latency={fs.latency*1000:.1f} ms, last frame_gap={gap}")
assert fs.valid and fs.version > 5
assert fs.latency < 0.15, f"flow too slow: {fs.latency*1000:.0f} ms"
assert gap >= 1, "frame_gap should be >= 1"
print("✓ checkpoint #6: FlowWorker OK")
""")

# B4 — FeatureWorker + test
code(r"""
# Cell B4 — FeatureWorker (DINOv2-S) + test (checkpoint #7)
class FeatureWorker(threading.Thread):
    def __init__(self, rgb: Slot, out: Slot, viz_size=(640,360), poll=0.001):
        super().__init__(daemon=True, name="FeatureWorker")
        self.rgb, self.out, self.viz_size, self.poll = rgb, out, viz_size, poll
        self._last_id = -1

    def run(self):
        while not STOP.is_set():
            snap = self.rgb.snapshot()
            if (not snap.valid) or snap.source_global_id == self._last_id:
                time.sleep(self.poll); continue
            bgr = snap.payload["bgr"]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t0 = time.monotonic()
            try:
                with GPU, torch.inference_mode():
                    inp = dino_proc(images=rgb, return_tensors="pt").to(DEVICE, dtype=torch.float16)
                    out = dino_model(**inp).last_hidden_state[0]
                    patches = out[1:]
                    feat_np = patches.float().cpu().numpy()
                    n = patches.shape[0]
                    gh = gw = int(round(np.sqrt(n)))
                    assert gh*gw == n, f"non-square patch grid {n}"
                norms = np.linalg.norm(feat_np, axis=1).reshape(gh, gw)
                norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-6)
                norm_img = cv2.resize((norms*255).astype(np.uint8), self.viz_size,
                                      interpolation=cv2.INTER_CUBIC)
                viz = cv2.applyColorMap(norm_img, cv2.COLORMAP_VIRIDIS)
                feat_jax = jnp.asarray(feat_np)
                t1 = time.monotonic()
                self.out.publish(
                    payload={"raw": feat_np, "norm_viz_bgr": viz, "jax": feat_jax,
                             "grid_hw": (gh, gw)},
                    source_global_id=snap.source_global_id,
                    source_frame_idx=snap.source_frame_idx,
                    source_time_sec=snap.source_time_sec,
                    wall_start=t0, wall_complete=t1, latency=t1-t0,
                    error=None,
                )
                self._last_id = snap.source_global_id
            except Exception as e:
                self.out.publish(error=str(e))
                raise

# --- Test ---
STOP.clear()
slots["rgb"]      = Slot(name="rgb")
slots["features"] = Slot(name="features")
src = FrameSource("assets/test.mp4", slots["rgb"], resize=RESIZE); src.start()
ftw = FeatureWorker(slots["rgb"], slots["features"]); ftw.start()
time.sleep(4.0)
STOP.set(); src.join(2); ftw.join(2)
fs = slots["features"].snapshot()
print(f"  feat v={fs.version}, latency={fs.latency*1000:.1f} ms, "
      f"grid={fs.payload['grid_hw']}, D={fs.payload['raw'].shape[1]}")
assert fs.valid and fs.version > 5
assert fs.latency < 0.1, f"DINO too slow: {fs.latency*1000:.0f} ms"
assert fs.payload["raw"].shape[1] == 384
print("✓ checkpoint #7: FeatureWorker OK")
""")

# B5 — FusionWorker + test
code(r"""
# Cell B5 — FusionWorker (JAX overlays) + test (checkpoint #8)
class FusionWorker(threading.Thread):
    def __init__(self, depth: Slot, flow: Slot, rgb: Slot, out: Slot, poll=0.001):
        super().__init__(daemon=True, name="FusionWorker")
        self.depth, self.flow, self.rgb, self.out, self.poll = depth, flow, rgb, out, poll
        self._last_d_v = self._last_f_v = -1

    def run(self):
        while not STOP.is_set():
            ds = self.depth.snapshot()
            fs = self.flow.snapshot()
            rs = self.rgb.snapshot()
            if not (ds.valid and rs.valid):
                time.sleep(self.poll); continue
            if ds.version == self._last_d_v and fs.version == self._last_f_v:
                time.sleep(self.poll); continue
            t0 = time.monotonic()
            try:
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
                # JAX has its own CUDA stream — block_until_ready before semaphore release.
                with GPU:
                    neon = fx_neon_glow(depth_j, fm_j, frame_j).block_until_ready()
                    sal  = fx_salience(depth_j, fm_j, frame_j).block_until_ready()
                    topo = fx_topo(depth_j, fx_j, fy_j, frame_j).block_until_ready()
                # np.asarray() on a JAX device array returns a readonly view —
                # copy so downstream consumers (e.g. cv2.putText) can draw on it.
                payload = {
                    "neon":     np.array(neon),
                    "salience": np.array(sal),
                    "topo":     np.array(topo),
                }
                t1 = time.monotonic()
                self.out.publish(
                    payload=payload,
                    source_global_id=rs.source_global_id,
                    wall_start=t0, wall_complete=t1, latency=t1-t0,
                    extras={
                        "depth_global_id": ds.source_global_id,
                        "flow_global_id":  fs.source_global_id if fs.valid else -1,
                    },
                    error=None,
                )
                self._last_d_v, self._last_f_v = ds.version, fs.version
            except Exception as e:
                self.out.publish(error=str(e))
                raise

# --- Test ---
STOP.clear()
for k in ["rgb","depth","flow","fusion"]:
    slots[k] = Slot(name=k)
src = FrameSource("assets/test.mp4", slots["rgb"], resize=RESIZE); src.start()
dep = DepthWorker(slots["rgb"], slots["depth"]); dep.start()
flw = FlowWorker(slots["rgb"], slots["flow"]); flw.start()
fus = FusionWorker(slots["depth"], slots["flow"], slots["rgb"], slots["fusion"]); fus.start()
time.sleep(5.0)
STOP.set()
for t in [src, dep, flw, fus]: t.join(2)
fz = slots["fusion"].snapshot()
print(f"  fusion v={fz.version}, latency={fz.latency*1000:.1f} ms")
print(f"  depth_id used={fz.extras['depth_global_id']}, flow_id used={fz.extras['flow_global_id']}")
for k in ["neon","salience","topo"]:
    img = fz.payload[k]
    assert img.shape == (RESIZE[1], RESIZE[0], 3) and img.dtype == np.uint8
    assert img.max() > 10, f"{k} looks blank (max={img.max()})"
print("✓ checkpoint #8: FusionWorker OK — all three overlays alive")
""")

# ---------- Group C — Visualization ----------
md(r"""
## Group C — Visualization (2D grid + k3d 3D cloud)

Both viz threads are pure consumers — they snapshot, never publish.
""")

# C1 — 2D grid + test
code(r"""
# Cell C1 — Notebook2DGrid + test (checkpoint #9)
# Realtime 2D video panel.
# Top-left tile is the live RGB stream with a big ingest-FPS counter
# overlaid in the top-right corner. The remaining 7 tiles show depth,
# flow, DINOv2 feature norm, the three JAX fusion overlays, and a stats
# panel with per-worker latency / staleness.
# Widget is an ipywidgets.Image — the worker thread writes JPEG bytes to
# .value and ipywidgets propagates the update to the frontend.
class Notebook2DGrid(threading.Thread):
    def __init__(self, slots: Dict[str, Slot], cell_size=(640,360), fps=30):
        super().__init__(daemon=True, name="Notebook2DGrid")
        self.slots, self.cs, self.dt = slots, cell_size, 1.0/fps
        self.widget = widgets.Image(format="jpeg",
                                    width=cell_size[0]*4, height=cell_size[1]*2)
        # Rolling windows for FPS measurement.
        # ingest_fps  = rate of new RGB frames from the source thread
        # display_fps = rate at which this thread re-encodes a tile
        self._rgb_hist     = deque(maxlen=30)  # (wall_time, source_global_id)
        self._display_hist = deque(maxlen=30)  # wall_time

    @staticmethod
    def _rolling_fps(hist) -> float:
        if len(hist) < 2: return 0.0
        if isinstance(hist[0], tuple):
            (t0, v0), (t1, v1) = hist[0], hist[-1]
            dt = t1 - t0
            return (v1 - v0) / dt if dt > 1e-6 else 0.0
        t0, t1 = hist[0], hist[-1]
        dt = t1 - t0
        return (len(hist) - 1) / dt if dt > 1e-6 else 0.0

    def _stats_panel(self, ingest_fps: float, display_fps: float) -> np.ndarray:
        rs = self.slots["rgb"].snapshot()
        img = np.zeros((self.cs[1], self.cs[0], 3), dtype=np.uint8)
        y = 30
        cv2.putText(img, f"ingest  {ingest_fps:5.1f} FPS",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 255), 2); y += 32
        cv2.putText(img, f"display {display_fps:5.1f} FPS",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 255), 2); y += 36
        cv2.putText(img, f"rgb v={rs.version} gid={rs.source_global_id}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1); y += 26
        for k in ["depth","flow","features","fusion"]:
            s = self.slots[k].snapshot()
            stale_f = rs.source_global_id - s.source_global_id if s.valid else -1
            line = f"{k:<8} v={s.version:<5} stale={stale_f:<3}f lat={s.latency*1000:5.1f}ms"
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y += 24
            if k == "flow":
                gap = s.extras.get("frame_gap", 0)
                cv2.putText(img, f"  gap={gap}f", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1); y += 22
        return img

    @staticmethod
    def _draw_fps_badge(img: np.ndarray, fps: float):
        # Big FPS counter overlay, top-right of the RGB tile.
        text = f"{fps:4.1f} FPS"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        H, W = img.shape[:2]
        x0, y0 = W - tw - 24, 16
        # Black drop-shadow rectangle behind the text for legibility.
        cv2.rectangle(img, (x0 - 10, y0), (x0 + tw + 10, y0 + th + 14), (0, 0, 0), -1)
        cv2.putText(img, text, (x0, y0 + th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 255, 255), 3, cv2.LINE_AA)

    def _tile(self, ingest_fps: float, display_fps: float) -> Optional[np.ndarray]:
        rs = self.slots["rgb"].snapshot()
        if not rs.valid: return None
        blank = np.zeros((self.cs[1], self.cs[0], 3), dtype=np.uint8)
        ds   = self.slots["depth"].snapshot()
        flsn = self.slots["flow"].snapshot()
        ftsn = self.slots["features"].snapshot()
        fus  = self.slots["fusion"].snapshot()
        cells = {
            "rgb":      rs.payload["bgr"],
            "depth":    ds.payload["viz_bgr"]      if ds.valid   and ds.payload   is not None else blank,
            "flow":     flsn.payload["viz_bgr"]    if flsn.valid and flsn.payload is not None else blank,
            "features": ftsn.payload["norm_viz_bgr"] if ftsn.valid and ftsn.payload is not None else blank,
            "neon":     fus.payload["neon"]        if fus.valid  and fus.payload  is not None else blank,
            "salience": fus.payload["salience"]    if fus.valid  and fus.payload  is not None else blank,
            "topo":     fus.payload["topo"]        if fus.valid  and fus.payload  is not None else blank,
            "stats":    self._stats_panel(ingest_fps, display_fps),
        }
        for k, v in cells.items():
            if v.shape[:2] != (self.cs[1], self.cs[0]):
                cells[k] = cv2.resize(v, self.cs, interpolation=cv2.INTER_AREA)
            elif not v.flags.writeable:
                # JAX-derived arrays come in readonly; cv2.putText would refuse.
                cells[k] = v.copy()
            cv2.putText(cells[k], k, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # Big FPS counter on the live RGB tile.
        self._draw_fps_badge(cells["rgb"], ingest_fps)
        row1 = np.hstack([cells["rgb"],  cells["depth"],    cells["flow"], cells["features"]])
        row2 = np.hstack([cells["neon"], cells["salience"], cells["topo"], cells["stats"]])
        return np.vstack([row1, row2])

    def run(self):
        while not STOP.is_set():
            now = time.monotonic()
            rs  = self.slots["rgb"].snapshot()
            if rs.valid:
                self._rgb_hist.append((now, rs.source_global_id))
            ingest_fps  = self._rolling_fps(self._rgb_hist)
            display_fps = self._rolling_fps(self._display_hist)
            tile = self._tile(ingest_fps, display_fps)
            if tile is not None:
                ok, buf = cv2.imencode(".jpg", tile, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    self.widget.value = buf.tobytes()
                    self._display_hist.append(now)
            time.sleep(self.dt)

# --- Test: synthetic publishes, render once, check shape ---
STOP.clear()
for k, payload in [
    ("rgb",      {"bgr": np.full((360,640,3), 64, np.uint8)}),
    ("depth",    {"viz_bgr": np.full((360,640,3), 96, np.uint8)}),
    ("flow",     {"viz_bgr": np.full((360,640,3), 128, np.uint8)}),
    ("features", {"norm_viz_bgr": np.full((360,640,3), 160, np.uint8)}),
    ("fusion",   {"neon": np.full((360,640,3), 50, np.uint8),
                  "salience": np.full((360,640,3), 80, np.uint8),
                  "topo": np.full((360,640,3), 110, np.uint8)}),
]:
    slots[k] = Slot(name=k)
    slots[k].publish(payload=payload, source_global_id=1)
grid_test = Notebook2DGrid(slots)
tile = grid_test._tile(ingest_fps=30.0, display_fps=29.5)
assert tile is not None and tile.shape == (720, 2560, 3), f"tile shape {tile.shape}"
# Sanity: the FPS badge changes pixels in the RGB tile's top-right corner.
rgb_top_right = tile[16:64, 640-240:640-10]
assert rgb_top_right.std() > 5, "FPS badge does not appear to be drawn on RGB tile"
print(f"✓ checkpoint #9: 2D grid renders tile of shape {tile.shape} with FPS overlay")
""")

# C2 — K3DPointCloud + test
code(r"""
# Cell C2 — K3DPointCloud + test (checkpoint #10)
class K3DPointCloud(threading.Thread):
    def __init__(self, depth_slot: Slot, rgb_slot: Slot, step=4, point_size=0.015, fps=20):
        super().__init__(daemon=True, name="K3DPointCloud")
        self.depth, self.rgb, self.step, self.dt = depth_slot, rgb_slot, step, 1.0/fps
        H, W = RESIZE[1], RESIZE[0]
        self.H, self.W = H, W
        self.fx = self.fy = W
        self.cx, self.cy = W/2, H/2
        ys, xs = np.mgrid[0:H:step, 0:W:step].astype(np.float32)
        self.us = xs.flatten(); self.vs = ys.flatten()
        self._last_v = -1
        N = self.us.size
        self.plot = k3d.plot(camera_auto_fit=True, grid_visible=False, background_color=0x101010)
        self.points = k3d.points(
            positions=np.zeros((N, 3), dtype=np.float32),
            colors=np.zeros(N, dtype=np.uint32),
            point_size=point_size, shader="flat",
        )
        self.plot += self.points

    def _back_project(self, depth, bgr):
        d = depth[::self.step, ::self.step].flatten().astype(np.float32)
        b = bgr[::self.step, ::self.step]
        Z = d
        X = (self.us - self.cx) * Z / self.fx
        Y = (self.vs - self.cy) * Z / self.fy
        pos = np.stack([X, -Y, -Z], axis=1).astype(np.float32)
        r  = b[..., 2].astype(np.uint32)
        g  = b[..., 1].astype(np.uint32)
        bb = b[..., 0].astype(np.uint32)
        col = ((r << 16) | (g << 8) | bb).flatten().astype(np.uint32)
        return pos, col

    def run(self):
        while not STOP.is_set():
            ds = self.depth.snapshot()
            rs = self.rgb.snapshot()
            if ds.valid and rs.valid and ds.version != self._last_v:
                pos, col = self._back_project(ds.payload["raw"], rs.payload["bgr"])
                self.points.positions = pos
                self.points.colors    = col
                self._last_v = ds.version
            time.sleep(self.dt)

# --- Test: synthetic depth + rgb, check back-projection ---
STOP.clear()
slots["rgb"]   = Slot(name="rgb")
slots["depth"] = Slot(name="depth")
slots["rgb"].publish(
    payload={"bgr": np.random.randint(0,255,(360,640,3),np.uint8)},
    source_global_id=1)
slots["depth"].publish(
    payload={"raw": np.random.uniform(0.1, 5.0, (360,640)).astype(np.float32),
             "viz_bgr": np.zeros((360,640,3),np.uint8),
             "jax": jnp.zeros((360,640))},
    source_global_id=1)
cloud_test = K3DPointCloud(slots["depth"], slots["rgb"], step=4)
pos, col = cloud_test._back_project(slots["depth"].snapshot().payload["raw"],
                                     slots["rgb"].snapshot().payload["bgr"])
print(f"  point count = {pos.shape[0]}, X range = ({pos[:,0].min():.2f}, {pos[:,0].max():.2f})")
assert pos.shape[1] == 3 and col.shape[0] == pos.shape[0]
assert pos[:,2].std() > 0.01, "Z all the same — backprojection broken"
print("✓ checkpoint #10: K3D backprojection OK; display the cloud with cloud.plot.display()")
""")

# ---------- Group D — Orchestrator ----------
md(r"""
## Group D — Orchestrator + 60 s soak test
""")

# D1 — System class
code(r"""
# Cell D1 — System orchestrator
class System:
    _instance: Optional["System"] = None

    def __init__(self, src_path="assets/test.mp4"):
        if System._instance is not None:
            System._instance.stop()
        System._instance = self
        self.src_path = src_path
        global slots
        slots = {k: Slot(name=k) for k in ["rgb","depth","flow","features","fusion"]}
        self.slots = slots
        self.threads: List[threading.Thread] = []
        self.grid: Optional[Notebook2DGrid] = None
        self.cloud: Optional[K3DPointCloud] = None

    def start(self):
        STOP.clear()
        self.threads = [
            FrameSource(self.src_path, self.slots["rgb"], resize=RESIZE),
            DepthWorker(self.slots["rgb"], self.slots["depth"]),
            FlowWorker(self.slots["rgb"], self.slots["flow"]),
            FeatureWorker(self.slots["rgb"], self.slots["features"]),
            FusionWorker(self.slots["depth"], self.slots["flow"],
                         self.slots["rgb"], self.slots["fusion"]),
        ]
        for t in self.threads: t.start()
        self.grid  = Notebook2DGrid(self.slots);                       self.grid.start()
        self.cloud = K3DPointCloud(self.slots["depth"], self.slots["rgb"]); self.cloud.start()
        return self

    def stop(self, timeout=3.0):
        STOP.set()
        for t in self.threads + [self.grid, self.cloud]:
            if t is not None: t.join(timeout=timeout)
        torch.cuda.empty_cache()
        System._instance = None
""")

# D2 — start system + show widgets
code(r"""
# Cell D2 — start the system, show widgets
#
# Order matters in JupyterLab: the grid widget and the k3d plot must be
# display()'d from the cell that owns the output area. The worker threads
# can then update the grid widget's .value and the k3d plot's traits in
# place — those updates render correctly because the widgets are already
# attached to a live output.
sys_ = System("assets/test.mp4").start()

# Stop button (top).
stop_btn = widgets.Button(description="Stop", button_style="danger")
stop_btn.on_click(lambda _: sys_.stop())
display(stop_btn)

# 2D image grid (8 panels). The worker thread writes JPEG bytes to
# sys_.grid.widget.value; ipywidgets propagates the change to the frontend.
display(sys_.grid.widget)

# 3D point cloud. k3d.plot.display() registers the plot with the current
# output area and starts the WebGL viewer.
sys_.cloud.plot.display()
""")

# D3 — soak test
code(r"""
# Cell D3 — 60-second soak test (checkpoint #11)
#
# Memory measurement strategy: PyTorch's caching allocator grows in chunks
# (typically 20-200 MB at a time) as new kernel shapes / cudnn workspaces get
# requested. A naive start-vs-end delta therefore conflates real leaks with
# normal allocator growth. We let the system run a 10-s warmup so the allocator
# reaches steady state, then take a baseline, then run 50 s of steady-state
# soak. The leak check then targets the steady-state window only.
def soak(seconds=60, stale_limit_frames=30, warmup=10.0):
    sys2 = System("assets/test.mp4").start()
    t0 = time.monotonic()
    samples = []
    # Warmup phase: let allocator stabilize.
    while time.monotonic() - t0 < warmup:
        time.sleep(0.5)
    torch.cuda.synchronize()
    mem_baseline = torch.cuda.memory_allocated()
    mem_peak     = mem_baseline
    try:
        while time.monotonic() - t0 < seconds:
            time.sleep(0.5)
            rs = sys2.slots["rgb"].snapshot()
            row = {"t": time.monotonic() - t0, "rgb_gid": rs.source_global_id}
            for k in ["depth","flow","features","fusion"]:
                s = sys2.slots[k].snapshot()
                row[f"{k}_v"]     = s.version
                row[f"{k}_stale"] = rs.source_global_id - s.source_global_id if s.valid else None
                row[f"{k}_lat"]   = s.latency
            mem_now = torch.cuda.memory_allocated()
            row["mem_mb"] = mem_now / 1e6
            mem_peak = max(mem_peak, mem_now)
            samples.append(row)
    finally:
        torch.cuda.synchronize()
        mem_end = torch.cuda.memory_allocated()
        sys2.stop()
    threads_ok = all(s[f"{k}_v"] > samples[0][f"{k}_v"]
                     for s in samples[-3:] for k in ["depth","flow","features","fusion"])
    last_stale = {k: samples[-1][f"{k}_stale"] for k in ["depth","flow","features","fusion"]}
    last_lat   = {k: samples[-1][f"{k}_lat"]   for k in ["depth","flow","features","fusion"]}
    steady_delta = (mem_end - mem_baseline) / 1e6
    peak_delta   = (mem_peak - mem_baseline) / 1e6
    print(f"  steady-state mem: baseline={mem_baseline/1e6:.1f} MB → end={mem_end/1e6:.1f} MB "
          f"(Δ={steady_delta:+.1f} MB, peak Δ={peak_delta:+.1f} MB)")
    print(f"  final stale frames: {last_stale}")
    print(f"  final latencies ms: {{ {', '.join(f'{k}: {v*1000:.1f}' for k,v in last_lat.items())} }}")
    assert threads_ok, "a worker stopped producing"
    # Steady-state leak threshold: 200 MB over 50 s of soak is conservative —
    # anything growing this fast in steady state is a real leak.
    assert steady_delta < 200, f"GPU memory growing in steady state: {steady_delta:.1f} MB"
    for k, sf in last_stale.items():
        assert sf is not None and sf < stale_limit_frames, f"{k} too stale: {sf} frames"
    print("✓ checkpoint #11: 60-second soak passed")
soak(seconds=60)
""")

# ---------- Group E — Cleanup ----------
md(r"""
## Group E — Cleanup
""")

# E1 — manual teardown
code(r"""
# Cell E1 — idempotent teardown
STOP.set()
if System._instance is not None: System._instance.stop()
torch.cuda.empty_cache()
print("clean.")
""")

# ---------- Write ----------
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (streamingvision)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
with open("/home/esli/StreamingVision/streaming_demo.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"Wrote streaming_demo.ipynb with {len(cells)} cells.")

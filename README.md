# StreamingVision

Async multi-worker real-time perception substrate (milestone 1):
RGB source → depth (Depth-Anything V2 Small) + optical flow (SEA-RAFT-M)
+ DINOv2-S features → JAX fusion overlays → 2D grid + k3d 3D point cloud,
all running off a single looping MP4 with latest-value slots and a GPU
semaphore. The full JAX inference model is deferred to milestone 2 — this
milestone proves the substrate.

## Hardware

Built and tested on:
- NVIDIA RTX 5090 (sm_120), driver 580.
- CUDA 12.8+ required for sm_120 — earlier toolkits crash with
  "no kernel image is available for execution on the device".

## Environment

Conda env: `streamingvision` (Python 3.11).

```bash
conda create -n streamingvision python=3.11 -y
conda activate streamingvision
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the GPU stack works before anything else:

```bash
python -c "
import os; os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.25'
import torch, jax
print(torch.__version__, torch.cuda.get_device_capability(), jax.devices())
"
# Expect: '2.12.x+cu13x (12, 0) [CudaDevice(id=0)]'
```

XLA env vars **must** be set before `import jax` (the XLA backend is
initialized at first import; setting them after has no effect). The notebook
takes care of this in Cell A1.

## One-time setup

```bash
# 1. Move and transcode the test video.
mkdir -p assets
mv purple_jacket.MOV assets/purple_jacket.MOV     # if not already moved
ffmpeg -y -i assets/purple_jacket.MOV \
       -c:v libx264 -pix_fmt yuv420p -crf 18 -an \
       assets/test.mp4

# 2. Clone SEA-RAFT (the optical-flow model code lives there).
mkdir -p third_party
git clone --depth 1 https://github.com/princeton-vl/SEA-RAFT.git third_party/SEA-RAFT
```

The SEA-RAFT weights and the Depth-Anything/DINOv2 checkpoints are pulled
from Hugging Face on first use (cached under `~/.cache/huggingface/`).

## Running

```bash
jupyter lab streaming_demo.ipynb
```

Run cells top-to-bottom. Each cell asserts a checkpoint (`✓ checkpoint #N`)
before the next layer is built on top. The 60-second soak in Cell D3
exercises the full pipeline and verifies bounded memory, low staleness, and
all five workers continue to produce output.

Headless validation (no GUI required):

```bash
jupyter nbconvert --to notebook --execute streaming_demo.ipynb \
                  --output streaming_demo_executed.ipynb \
                  --ExecutePreprocessor.timeout=300
```

## Architecture

9 threads: 1 frame source, 5 workers (depth, flow, features, fusion, and
the future inference model slot), 3 viz threads (2D grid, k3d cloud, stats).
Workers communicate through latest-value slots; a `threading.Semaphore(1)`
serializes GPU submissions. See `~/.claude/plans/real-time-streaming-depth-iterative-bee.md`
for the full design doc.

Measured on RTX 5090 at 360p:

| stream    | latency (ms) | staleness (frames) |
|-----------|--------------|--------------------|
| depth     | 9–28         | 0–1                |
| flow      | 16–31        | 1                  |
| features  | 9–32         | 0–1                |
| fusion    | 15–32        | 1                  |

GPU during 60 s soak: 40 % avg / 57 % peak utilization, ~2.3 GB VRAM
(stable, no leak), ~180 W power, ~46 °C. Steady-state memory delta ≈ 0 MB.

## Migrating to a webcam

Change one line in Cell D2:

```python
sys_ = System(0).start()   # 0 = first USB camera
```

`FrameSource` already handles `isinstance(src, int)` — no other code changes.

# GenMatterPlusPlus

Probabilistic 3D particle tracking for motion segmentation. This fork keeps **TAP-Vid DAVIS** paper benchmarks and adds a **custom MP4** workflow (`genmatter` CLI) for realtime preprocess, tracking, Rerun visualization, and SAM2 pseudo-GT.

## Requirements

- CUDA 12.4+ and a compatible NVIDIA driver (recommended for preprocessing)
- GPU with ≤ 24 GB VRAM for full-resolution pipelines
- Python 3.11 (pinned via `.python-version`; uv will download it if missing)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Install

From the repository root:

```bash
uv sync
```

This registers the **`genmatter`** CLI entry point.

## Custom MP4 — preprocess, track, viz

```bash
uv run genmatter preprocess /path/to/your_video.mp4
uv run genmatter track --video-id <video_id>
uv run genmatter viz --video-id <video_id>
```

Optional flags:

```bash
uv run genmatter preprocess clip.mp4 \
  --config configs/custom_default.yaml \
  --video-id my_clip \
  --set preprocess.motion_3d.max_len=50
```

Preprocess runs four GPU stages sequentially: RGB frames → 3D motion (VDA + RAFT) → DINO PCA → SAM frame 0. Outputs default to `assets/custom_videos/<video_id>/` (override via `paths.custom_videos_root` or `GENMATTER_CUSTOM_VIDEOS_DIR`).

Tracking runs Gibbs initialization, temporal tracking (JIT-compiled, compile-once across videos), and dense evaluation. JAX compile time is reported separately from FPS.

### Pseudo-GT (SAM2)

```bash
uv run genmatter pseudo-gt --video-id <video_id>
```

See **[configs/custom_default.yaml](configs/custom_default.yaml)** for hyperparameters.

## TAP-Vid DAVIS benchmarks

Put DAVIS data under **`assets/`** (gitignored). Full download and experiment commands are in **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**.

Quick start after data is in place:

```bash
uv run python run_experiments.py davis-tracking-sam
uv run python run_experiments.py postprocess-davis
```

Paths and constants: **`config.py`** (or `GENMATTER_DAVIS_DIR`, `GENMATTER_RESULTS_DIR`, etc.).

## Tests

```bash
uv sync --extra dev
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).

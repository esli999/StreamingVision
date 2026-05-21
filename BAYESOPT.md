# Offline hyperparameter optimization on `assets/test.mp4`

This document records how the SAM2-pseudo-GT bayesopt pipeline is run for the
StreamingVision streaming demo.  The optimization is **offline**: it
produces `configs/streaming_tuned.yaml`, which the real-time demo then
consumes.

## Prerequisites

- A clean conda env matching `requirements.txt`.
- `assets/test.mp4` present.
- ~600 MB free in `assets/deeplearning_weights/` for SAM2.1-Large weights.
- The vendored GenMatter++ realtime-demo lives under `genmatterpp/`; see
  `genmatterpp/VENDORED.md` for the source SHA and local patches.

## Pipeline (run in order from the repo root)

```bash
# 0. Layout: the branch CLI expects `assets/custom_videos/<id>/source.mp4`.
mkdir -p assets/custom_videos/test
ln -sf "$(pwd)/assets/test.mp4" assets/custom_videos/test/source.mp4

# 1. Preprocess (frames + Video-Depth-Anything 3D motion + DINO PCA + SAM
# frame-0).  ~5-10 min on RTX 5090.  Auto-clones Video-Depth-Anything into
# `genmatterpp/external/video-depth-anything/` and downloads VDA-vits weights
# (~70 MB).  Uses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to
# accommodate 4K source.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m \
    genmatterpp.genmatter.custom.cli preprocess \
    assets/custom_videos/test/source.mp4 \
    --video-id test \
    --config configs/streaming_default.yaml

# 2. SAM2 pseudo-GT per frame.  ~5-15 min for 56 frames.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m \
    genmatterpp.genmatter.custom.cli pseudo-gt \
    --video-id test \
    --config configs/streaming_default.yaml

# 3. Bayesopt: 64 Sobol + LCB on 17 hyperparameters.  ~30-60 min wall clock.
# GENMATTER_DEFAULT_MAX_FRAMES is set because the vendored bayesopt's compile
# helper otherwise tries to look up a DAVIS dataset path that doesn't exist;
# see the local-patch note in `genmatterpp/VENDORED.md`.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GENMATTER_DEFAULT_MAX_FRAMES=64 \
    python -m genmatterpp.genmatter.custom.cli bayesopt \
    --video-id test \
    --config configs/streaming_default.yaml \
    --bayesopt-config configs/streaming_bayesopt.yaml

# 4. After bayesopt finishes, copy the winning trial yaml into the active
# streaming config.
cp assets/custom_videos/test/bayesopt/<RUN_ID>/best_trial.yaml \
   configs/streaming_tuned.yaml
```

## Accuracy gate

The bayesopt objective is mean per-GT-instance IoU (Hungarian-matched against
SAM2 pseudo-GT, threshold 0.5).  After Step 3 finishes:

- Mean per-instance IoU should be **comfortably above 0.15** (the branch's
  Sobol-phase top valid trials on test.mp4 land around 0.17).  Higher is
  better; the LCB phase typically lifts this further.
- Mean outlier % must stay **below 10 %** (the bayesopt's hard validity
  threshold) — invalid trials score 0.0 in Ax.

If the gate fails: inspect `assets/custom_videos/test/bayesopt/<RUN_ID>/`:
- `trials.jsonl` — per-trial parameters + metrics
- `trials_summary.csv` — flat table; sort by `ax_objective` to see the top
  valid trials
- `best_trial.yaml` — the bayesopt's chosen winner

## Disposable artifacts

The preprocessing outputs under `assets/custom_videos/test/{rgb_frames,
npzs, dino, SAM_frame0, pseudo_gt_sam}/` are **temporary**.  Once
`configs/streaming_tuned.yaml` exists, those can be deleted.  Keep the
`source.mp4` symlink for reproducibility.

`.gitignore` excludes `assets/custom_videos/*` so these don't get
committed.

## How the streaming demo consumes the tuned yaml

```bash
python render_demo.py --duration 12 --fps 30 \
    --out assets/matter_demo.mp4 \
    --config configs/streaming_tuned.yaml
```

`MatterWorker.__init__` reads the yaml and threads `sigma_F`, `outlier_prob`,
all the `sigma_*` / `Psi_*` / discrete-motion priors, and
`init_gibbs_sweeps` straight into the `GenMatter_Hyperparams_DINO` it
constructs.  There is no in-demo hyperparameter tuning — the demo is
strictly causal real-time consumption of pre-tuned values.

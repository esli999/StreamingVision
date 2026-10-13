# Legacy particle visualization (`render_gaussian_demo.py`)

> For the project overview + architecture, see the [README](../README.md). This doc is
> the deep dive on the particle figure: layout, CLI flags, and the render-only knobs.
> Focused mode is the unified figure in `scripts/render_unified.py`.

A 2×3 video visualization of the GenMatter tracker's latent state: the input frame, the
per-pixel particle/cluster segmentation, and the tracker's **actual 3-D Gaussian
particles** rendered two ways (true color + by object) plus a cluster-colored 3-D point
cloud, under a slow camera pan. It renders the model's **own** `blob_means`/`blob_covs`
(no per-frame re-fit), so the visualization can never diverge from the tracking — every
overlay is render-only and the tracker stays bit-identical to the validated run.

- **Script:** `scripts/render_gaussian_demo.py`
- **One-command launcher:** `scripts/render_particles_demo.sh`
- **Config:** `configs/streaming_render_v2.yaml` (the validated, shipped tracking config)
- **Output:** `runs/calibrate_consistency/viz_gaussian/<video>_live.mp4` (H.264)

---

## Quickstart

```bash
# Current unified figures, with initial DAVIS masks:
python scripts/render_unified.py --videos dog breakdance horsejump-high blackswan \
  --preset quality --out-dir runs/figures

# Legacy 2x3 on a raw video, using a k-means seed:
scripts/render_particles_demo.sh --source assets/test.mp4 --target-duration 0

# a single video, quick 24-frame smoke (no looping)
python scripts/render_gaussian_demo.py --videos dog --max-frames 24 --target-duration 0

# any DAVIS or custom video by name, to a custom out dir
scripts/render_particles_demo.sh --videos dog camel --out-dir runs/my_renders
```

`python scripts/render_gaussian_demo.py --help` prints every CLI flag **and** the
environment-variable tuning knobs with their defaults.

---

## The 2×3 layout

Each caption names the **type of visualization** first (`2D pixels` / `3D particles` /
`3D point cloud`), then the coloring.

```
        col1                          col2                       col3
 2D │ RGB camera frame            │ 2D pixels, by particle    │ 2D pixels, by cluster      │
 3D │ 3D particles, by avg color  │ 3D particles, by cluster  │ 3D point cloud, by cluster │
    │   (panning camera)          │   (panning camera)        │   (panning camera)         │
                                   (+ a stats row)
```

| Tile | What you're looking at |
|------|------------------------|
| `RGB camera frame` | the raw input image (with a tracker-FPS badge) |
| `2D pixels, by particle` | the image segmentation: each pixel painted its **particle's average color** (a posterized scene) |
| `2D pixels, by cluster` | the image segmentation: each pixel painted its **object/cluster** color (vivid foreground, muted background) |
| `3D particles, by avg color` | the tracker's **actual 3-D Gaussian particles** as depth-shaded balls, each tinted its average color |
| `3D particles, by cluster` | the **same** particles, tinted by which object (cluster) they belong to |
| `3D point cloud, by cluster` | the depth-projected pixels lifted to 3-D, colored by cluster ("scene clusters as the pixels") |

The two 3-D particle tiles share one geometry and one alpha — they differ **only in
tint**. The camera pans slowly and monotonically (a shallow yaw sweep, no double-back).

---

## Focused mode (`RENDER_MODE=focused`)

Focused mode uses `scripts/render_unified.py`: one causal inference pass,
saved state captures, then a CPU-only 5120×2080 figure render. Its configuration
is `configs/unified.yaml`; legacy environment knobs below apply only to the 2×3
renderer. It requires an initial object mask.

```bash
python scripts/render_unified.py --videos test --preset quality --out-dir runs/figures
python scripts/render_unified.py --videos test --preset quality --stage render --out-dir runs/figures
python scripts/render_unified.py --videos test --preset fast --stage benchmark --out-dir runs/benchmark
```

MP4s are saved as `<out-dir>/<clip>_unified.mp4`. `--stage render` redraws a
saved capture, and `--stage benchmark` times inference without writing video.
The sections below document the legacy non-focused renderer.

## Rendering any video / adding a new one

Videos are looked up by name in `scripts/render_live_grid.py::_resolve()`:

- **DAVIS videos** (ground-truth seeded) — any of the 30 names under
  `assets/tapvid_davis_30_videos_processed/tapvid_davis_segmasks/` (e.g. `blackswan`,
  `car-roundabout`, `judo`, `parkour`). Source is the pre-cut mp4 in
  `runs/calibrate_consistency/_davis_src/` if present, else the RGB-frames directory.
- **Custom videos** (SAM2 frame-0 seeded) — any subdir of `assets/custom_videos/`
  (e.g. `wine_swirl`, `gray_jacket`, `purple_jacket`).

**To add a new custom video,** create this layout and pass `--videos <name>`:

```
assets/custom_videos/<name>/
├── source.mp4                 # (or source.mov / source.MOV) — any OpenCV-readable video
└── pseudo_gt_sam/segmasks/<name>/00000.png   # frame-0 SAM instance mask, uint16, native res
```

The frame-0 mask is a **uint16 instance-id PNG** at the video's native resolution: id `0`
= background, ids `1..N` = object instances (`genmatter_rt.instance_mask_to_rgb_grid`
encodes it for the tracker's SAM-seed branch). To generate it (and the rest of the
preprocessing) with SAM2, use the vendored CLI:

```bash
python -m genmatter.custom.cli pseudo-gt <name> --config configs/custom_default.yaml
```

(See `genmatterpp/README.md` for the full custom-video pipeline. For a manual mask, any
uint16 PNG with the id convention above is enough to render.)

---

## CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--videos` | the 4 featured videos | video names (DAVIS or custom); `all-davis` / `all-custom` / `all` enumerate every test video from disk |
| `--source` | (none) | render a RAW video file (mp4/mov) or RGB-frames dir directly, k-means-seeded (no GT/SAM data) — overrides `--videos` |
| `--config` | `configs/streaming_render_v2.yaml` | tracking YAML (the validated config) |
| `--out-dir` | `runs/calibrate_consistency/viz_gaussian` | where `<video>_live.mp4` is written |
| `--num-sweeps` | from config, otherwise 4 | Gibbs sweeps per frame (shipped config omits this key) |
| `--max-frames` | -1 (whole clip) | cap source frames (small = quick smoke) |
| `--out-fps` | 30 | output frame rate |
| `--frame-stride` | 1 | keep every Nth source frame before looping |
| `--target-duration` | 6 | loop the (short) clip until N seconds emitted (0 = play once) |

## Environment-variable tuning knobs (advanced, all render-only)

| Env var | Default | Effect |
|---------|---------|--------|
| `RENDER_MODE` | (unset) | `focused` = the unified pipeline figure (see above); unset = the 2×3 grid |
| `FORCE_SAM_SEED` | 0 | `1` = seed EVERY clip with self-supervised SAM (no GT). Default off = custom→SAM, DAVIS→GT |
| `PC_PAN_AMP` | 7.0 | (2×3 grid) camera pan half-range in degrees |
| `PC_CENTROID_ALPHA` | 0.15 | orbit-center EMA (lower = steadier cloud) |
| `MARBLE_GEOM_BETA` | 0.25 | per-particle mean/cov EMA (lower = steadier particles) |
| `MARBLE_SIGMA` | 1.15 | particle size, in sigmas (smaller separates a dense 256-blob cluster) |
| `SEG_EMA` | 0.3 | 2-D seg-color temporal EMA (lower = less edge flicker) |
| `SEG_EDGE_AWARE` | 1 | RGB-guided upsample (0 = blocky but stable) |
| `SPURIOUS_EXTENT_FENCE` | 0=auto | override the size-outlier cut (cov max-eigenvalue) |
| `SPURIOUS_DENSITY_PCT` | 15 | "diffuse" cut: density percentile among drawable particles |
| `SPURIOUS_EXTENT_PCT` | 70 | "big" gate paired with the density cut |
| `SPURIOUS_EXTENT` | 0=auto | override the background size backstop |
| `SPURIOUS_HARD_CAP` | 0=off | absolute extent cap |
| `SPURIOUS_DEBUG` | 1 | print per-particle extent/count/density at frame 0 |

The **spurious-particle filter** drops big-yet-sparse particles (low `density =
datapoint-count / cov-extent`) from the 3-D views — this is what removes the flickery
"big diffuse background blob." Thresholds are frozen at frame 0 and applied to the EMA'd
geometry, with a kill-mask EMA so a dropped particle fades rather than pops.

---

## Pipeline & reusable functions

Per frame, `render_video()` runs the live perception + tracker, then renders:

```
depth (Depth-Anything) ─┐
optical flow (SEA-RAFT) ─┼─> unproject ─> init_state / step_multi_sweep  (the JAX Gibbs tracker)
DINO features ──────────┘                         │
                                                  ├─ extract_assignments        (per-datapoint blob/cluster ids)
                                                  ├─ extract_blob_means_and_covs (the real Gaussians, no re-fit)
                                                  └─ _blob_weights_mean_jit      (deterministic weights -> alpha)
```

**Code layout.** The tracker and the renderer are separate modules:
`genmatter_rt.py` is the streaming **tracker** (perception → `init_state` /
`step_multi_sweep` → the state extractors), and `genmatter_viz.py` is the
**renderer**. Every render function draws the tracker's REAL state (blob
means/covariances, cluster assignments) and never re-fits or mutates anything,
so tuning or swapping the visualization can't change tracking. The dependency is
strictly one-directional — `genmatter_viz` imports a handful of constants +
`_depth_to_Z` from `genmatter_rt`; the tracker never imports the renderer.

**`num_blobs` rule.** Every per-blob LUT / label-grid call passes `num_blobs`
(the real blob count). `compute_blob_color_lut`, `render_matter_label_grids`, and
`render_matter_tile` default their blob-id range to the 128-entry `BLOB_PALETTE`;
at the demo's 256 blobs, ids ≥ 128 would otherwise be silently clamped onto blob
127 — a one-colour smear (a colour no pixel in the scene actually has).
`tests/test_palette_sizing.py` pins this invariant.

The render helpers live in `genmatter_viz.py` and are reusable for other viz:

| Function | Purpose |
|----------|---------|
| `render_matter_label_grids(blob_a, hyperblob_a, indices, …)` | upsample per-datapoint labels to dense `(H,W)` blob/cluster id grids + outlier mask |
| `compute_blob_color_lut(blob_a, indices, rgb, …)` | per-particle **average-color** BGR LUT (lock at frame 0) |
| `build_cluster_palette(num_fg, n_total)` / `_cluster_palette_for(...)` | vivid-foreground / muted-background **cluster** palette (used for both the 2-D cluster tile and the 3-D cluster particles) |
| `render_particle_marbles_tile(means, covs, colors_bgr, proj, …)` | draw the Gaussians as shaded 3-D balls (painter-occluded), tinted per particle |
| `render_pointcloud_tiles_multi(depth, (tile,…), …)` | splat the depth map to a 3-D point cloud, sampling each color source; returns a shared `proj` for the particles |
| `make_lit_sphere_sprite(...)` | the precomputed lit-sphere sprite warped onto each particle |
| `blob_alpha_from_weights(w_ema, …)` | smoothstep per-particle visibility from posterior weights (weak particles fade) |
| `compute_blob_feature_lut` / `compute_blob_motion_lut` | alternative per-particle colorings (DINO-feature PCA; ego-compensated velocity wheel) — not used by this demo but available |

The tracking config (`configs/streaming_render_v2.yaml`, `tracking:` block) sets
`num_blobs: 128`, `num_hyperblobs: 10`, `freeze_hyperblob_assignment: true`,
`feature_aware_final_assignment: true`, `blob_means_updates_per_frame: 1`,
`use_sam_frame0: true` — it is validated on a held-out split, so do not re-tune it from
demo observations.

---

## Seed quality (the "background clusters move" artifact)

When background clusters appear to move, the dominant cause is the **frame-0 SAM seed
over-segmenting the static scene**: a static surface (e.g. a table) is seeded as foreground
instances and only 2–4 % of datapoints are background, so background clusters exist and move
because the tracker was never told that surface is background. This is a seed-quality issue,
not a per-frame assignment bug; GT-seeded clips (single clean object) do not show it.

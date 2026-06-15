# Particle visualization demo (`render_gaussian_demo.py`)

> For the project overview + architecture, see the [README](../README.md). This doc is
> the deep dive on the particle figure: layout, CLI flags, and the render-only knobs.

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
# render the featured deforming subjects (dog, breakdance, horsejump-high, blackswan — DAVIS/GT)
# as the unified pipeline figure (<vid>_unified.mp4, white-paper theme)
RENDER_MODE=focused DEMO_NUM_BLOBS=256 STICKINESS=12 ROBUST_DELTA=6 \
  scripts/render_particles_demo.sh --target-duration 0 --out-fps 12

# any RAW video (no dataset / GT / SAM needed — k-means seed) — works on the bundled sample
RENDER_MODE=focused scripts/render_particles_demo.sh --source assets/test.mp4 --target-duration 0

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

## Focused mode (`RENDER_MODE=focused`) — ONE unified pipeline figure per clip

`RENDER_MODE=focused` writes **`<vid>_unified.mp4`** from one tracker pass (the 2×3 path
above is used when `RENDER_MODE` is unset): a **white-paper conceptual figure**

**`1. RGB input → 2. 3-D Gaussian particles → 3a. RGB reconstruction + 3b. object segmentation`**

with floating panel cards (soft shadows), real typography (DejaVu via PIL), and pipeline arrows.

```bash
RENDER_MODE=focused DEMO_NUM_BLOBS=256 STICKINESS=12 ROBUST_DELTA=6 \
  python scripts/render_gaussian_demo.py \
  --target-duration 0 --out-fps 12 --out-dir runs/calibrate_consistency/viz_unified
```

### The hero: ONE cloud, both attributes multiplexed per particle — pure particles
Each particle has two attributes — **appearance** (average color) and **object membership**
(semantic cluster) — and the hero shows ONE set of particles carrying both at once. There
are **no point-cloud splats anywhere** in the figure:

- **fill = average RGB** — the particle's true appearance (lit-sphere shaded marble, hi-res
  160-px sprite for crisp shading; `MARBLE_SIGMA` 1.15 so a dense 256-blob cluster separates).
  Colors come from a **slow per-blob EMA** (`LUT_EMA` 0.15, only blobs owning datapoints
  update): on a hard pan a blob re-binds to new scene content, so a frame-0-locked color LUT
  would keep painting its old color; the EMA tracks what each blob currently covers while
  keeping identity stable. BOTH per-blob tables must be sized by
  `num_blobs`: `compute_blob_color_lut(..., num_blobs=nb)` AND
  `render_matter_label_grids(..., num_blobs=nb)` — the label grid's default palette-sized
  clamp silently relabels every blob ≥ 128 as blob 127 at 256 blobs, painting half the
  datapoints with one blob's color (see the **`num_blobs` rule** below);
- **glow = an under-pass halo in its segmentation color** (`GLOW_ALPHA` 0.9,
  `GLOW_SCALE` 2.4): all halos composite far→near *before* any fill, so they union into a soft
  highlighter field behind a dense cluster instead of being covered by the fills. Background
  glows are doubly faint (muted palette × `BG_FADE`), and the OBJECT's glow is **exempt from
  the fog fade** (the subject marker stays full-strength at any depth);
- **depth fog** (`FOG` 0.45, atmospheric perspective): fills mix toward the paper tone and
  glows fade with each view's normalized (10th–90th pct) depth — near particles full-strength,
  far ones recede into the page, so the cloud reads as a 3-D scene from particles alone;
- **world stabilization** (`EGO_STABILIZE` 1): the background of the world tends to be static
  and the **camera moves** — implemented computationally from the clustering. Each frame the
  camera's ego-motion is solved by **trimmed weighted Kabsch** over the background blob means
  (blob indices persist → correspondences are free; anchors = the blobs' previous world
  positions, the worst-residual 25 % trimmed so a re-bound blob can't vote) and everything
  renders in the **frame-0 world frame**: the background cloud holds still, the object moves
  through it, and the camera rig visibly travels/pans with the real camera. The hero itself is
  a fixed world-frame meta-camera (`_make_proj`) pivoting on the object's slow world EMA;
- **three camera frustums at REAL positions, depth-interleaved with the marbles**: the
  **original/input camera** (azure accent — the same border color as every panel imaged at
  that pose) sits **exactly at the real camera position** (the camera-frame origin mapped
  through the ego pose — it traces the true camera path), its plane is the real image plane,
  and the two **novel** poses (gray, ±`FRUSTUM_YAW` 22°) are that camera orbited about the
  object pivot on the **true-radius arc**. Rig scalars are **frozen in the moving camera
  frame** at calibration end and mapped through the ego pose each frame, so the three cameras
  stay **mutually rigid** (deriving them from per-frame object stats lets them drift apart).
  Every camera is AIMED AT THE SCENE CENTER (`_frustum_cams` lookAt: the yaw-0 apex is the
  real camera position, the novel apexes orbit it about the followed `aim_t`, and each image
  plane is ⊥ its own view axis). Windows are TRUE perspective-warped planes (the novel views
  shear), with the apex rays drawn UNDER the picture; the plane half-width is capped against
  the aimed arc's plane-center spacing so adjacent planes can never fuse. Windows and marbles
  composite in ONE painter's far→near ordering (marbles bucketed around each window's view
  depth with a small HYSTERESIS deadband so matter at window depth can't flicker over/under,
  passes chained through `base_tile`; labels drawn last): matter in front of a camera correctly
  occludes it, matter behind it is hidden — no punch-through in either direction. Each
  window shows **the rendered RGB representation as pixels**: the input plane carries the
  per-pixel RGB reconstruction (= panel 3a, the azure tie), the novel planes carry the Gaussian
  state **volumetrically splatted** to a dense soft image (`_render_gaussian_rgb_view`:
  EWA-style footprints, front-to-back compositing, normalized by occlusion weight so paper
  white doesn't bleed through between Gaussians; `GS_ALPHA_MAX` 0.9);
- **FOLLOW hero camera — two-pass whole-clip solve**: the hero keeps the scene tightly framed
  while NEVER zooming or dollying. PASS 0 runs the tracker over the source
  once, collecting per-frame framing stats and freezing the rig scalars;
  `_solve_follow_camera` then fits: the SMOOTHED EGO-POSE track (σ 5, rotations re-projected
  to SO(3)) — the raw live Kabsch pose stream's noise rocks the whole view, visually identical
  to viewer-camera jitter, so pass 1 renders the smoothed poses and never the raw solve; TWO
  near-static tracks with FIXED depths (σ 25) — `aim_t` (the object/scene track every camera
  points at and the novel apexes orbit) and `pivot_t` (the hero framing center, 0.7·object +
  0.3·apex-mean); ONE focal (min over all frames of the soft-object / hard-rig fit at that
  frame's pivot); the pullback (`PULLBACK` 0.6 — a pure view-depth retreat so the real camera
  apexes frame); and the per-pose splat focals. PASS 1 replays the deterministic tracker and
  renders with that camera: essentially still, drifting only when the action genuinely
  migrates, constant scale, **everything fitting every frame by construction**. The aim and
  framing tracks are deliberately SEPARATE — blending the apexes into the aim pulls the orbit
  center toward the cameras, halving the arc radius (foreshortened poses) and fusing the planes;
- **the inference-stats row** spans the bottom of the whole diagram (`_stats_row`): big
  **inference FPS**, real-time, per-stage latency (depth / flow / dino), and tracker columns.

The **object segmentation** output panel is the INFERRED flat cluster tile (vivid fg / muted
bg + dark-red outliers), same palette as the glow (`_cluster_palette_for`): the green-glowing
dog particles ARE the green segment.

### Hero camera + frustum geometry
- **`HERO_YAW` (12°) / `HERO_PITCH` (14°)** — gentle look-down; steeper (22–30°) turns the scene
  into a floating specimen jar.
- **`FRUSTUM_IMG_FRAC` (0.45)** — plane size: half-width from the object radius, floored at
  0.06·d₀ and **capped by the camera spacing** (0.14·d₀, d₀ = the true camera-object distance)
  so a large object can't fuse the three planes into a screen; the image
  plane sits at 0.30·d₀ from the apex (a long camera-like pyramid).
- **`RECON_FIT_MARGIN` (0.5)** — the video-independent fit (`_calibrate_marble_focal`): soft
  97th-pct fit for object particles, **hard** 0.88 fit for frustum corners/apexes, solved
  ONCE over the whole clip by the two-pass `_solve_fixed_camera` (min over all pass-0 frames),
  so the camera is constant and everything fits every frame (no live calibration or
  zoom guard). The novel-plane focals are object-fit (margin 0.75, scene-wide fits
  collapse on near-camera background blobs), **median**-accumulated during pass 0, then frozen.

All render-only (tracker state never touched → bit-identical tracking).

### Figure-ground + featured subjects
Background fills mix toward near-white (`BG_DESAT` 0.45) and fade (`BG_FADE` 0.3) — clearly
transparent ghosts that keep the scene structure readable while the full-color, green-glowing
object is unmistakably the subject. A cluster counts as "background" if it's seed/GT background
(`hyperblob >= num_fg`) **or** — SAM seeds only — it covers more than `BG_SIZE_FRAC` (0.33) of
the particles (a SAM seed mislabels the dominant couch/sky/ground as foreground; a GT seed is
clean, and the size test would wrongly fade a large GT object like the breakdancer).

The featured subjects (`DEFAULT_VIDEOS`) are deforming, low-occlusion DAVIS clips
whose track holds across the whole clip (GT-seeded for the demo): **dog**, **breakdance**,
**horsejump-high**, **blackswan**. Small subjects read poorly (too few particles land on them).
To add your own: a single clearly-deforming object with low occlusion reads best; validate the
track by rendering and checking the segmentation overlay follows the object cleanly across the
clip before featuring it.

### Seeds in focused mode (custom = SAM, DAVIS = GT)
The demo seeds **custom** clips from their self-supervised SAM frame-0 map and **DAVIS** clips from
their clean **GT** segmask. This is a *visualization* choice: a DAVIS GT mask is a single clean
object, whereas the SAM-propagated DAVIS seed **over-segments** (e.g. blackswan → ~12 instances)
and clutters the cluster lens. It is **not** an eval shortcut — the honest self-supervised path is
SAM everywhere (`FORCE_SAM_SEED=1`). Custom clips have no GT and are always SAM-seeded either way.

---

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
| `--num-sweeps` | from config (1) | Gibbs sweeps per frame |
| `--max-frames` | -1 (whole clip) | cap source frames (small = quick smoke) |
| `--out-fps` | 30 | output frame rate |
| `--frame-stride` | 1 | keep every Nth source frame before looping |
| `--target-duration` | 6 | loop the (short) clip until N seconds emitted (0 = play once) |

## Environment-variable tuning knobs (advanced, all render-only)

| Env var | Default | Effect |
|---------|---------|--------|
| `RENDER_MODE` | (unset) | `focused` = the unified pipeline figure (see below); unset = the 2×3 grid |
| `FORCE_SAM_SEED` | 0 | `1` = seed EVERY clip with self-supervised SAM (no GT). Default off = custom→SAM, DAVIS→GT |
| `HERO_YAW` | 12 | focused: hero meta-camera yaw (deg) |
| `HERO_PITCH` | 14 | focused: hero elevation — gentle look-down (steeper = floating specimen jar) |
| `HERO_DEPTH_SCALE` | 1.0 | focused: depth smoosh in the hero (1 = off; <1 compresses depth thickness) |
| `FRUSTUM_YAW` | 22 | focused: ± yaw of the two novel frustum cameras (their images re-render from these poses; too wide and the fit shrinks the view windows) |
| `PULLBACK` | 0.6 | focused: hero retreat (fraction of the true camera-object distance) so the REAL camera positions fit in frame |
| `FRUSTUM_IMG_FRAC` | 0.4 | focused: frustum image-plane half-width (fraction of the object radius; spacing-capped) |
| `RECON_FIT_MARGIN` | 0.5 | focused: soft object fit (97th pct) in the hero; frustums get a hard 0.88 fit, then frozen |
| `FOCAL_CALIB_FRAMES` | 8 | focused: frames to min-accumulate the hero focal over before freezing |
| `DEMO_NUM_BLOBS` | (config) | focused: override num_blobs (256 gives dense object coverage in the demo) |
| `EGO_STABILIZE` | 1 | focused: world stabilization — Kabsch ego-motion from the background blobs (0 = camera frame) |
| `LUT_EMA` | 0.15 | focused: per-blob avg-RGB color EMA rate (colors track a pan; the 2×3 grid keeps the frame-0 lock) |
| `GS_ALPHA_MAX` | 0.9 | focused: peak per-Gaussian opacity in the novel planes' volumetric splat renders |
| `GLOW_ALPHA` | 0.9 | focused: the semantic glow strength (object glow exempt from fog; 0 = off) |
| `GLOW_SCALE` | 2.4 | focused: glow radius in units of the marble axes |
| `FOG` | 0.45 | focused: depth fog — fills toward paper, glows fade with view depth (0 = off) |
| `BG_FADE` | 0.3 | focused: background-particle alpha (1.0 = no fade) |
| `BG_DESAT` | 0.45 | focused: mix background fills toward near-white (figure-ground on paper; 0 = off) |
| `BG_SIZE_FRAC` | 0.33 | focused, SAM seeds only: a cluster covering > this fraction of particles is faded as background |
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

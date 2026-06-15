#!/usr/bin/env python3
"""Focused 2x3 demo: clusters + FAITHFUL particle Gaussians, GT/SAM-seeded.

A pared-down sibling of render_live_grid_v2.py for the featured deforming subjects
(dog, breakdance, horsejump-high, blackswan). Same live tracker +
identical per-frame config; the bottom row draws the tracker's OWN latent Gaussians
(`blob_means`/`blob_covs`, no per-frame re-fit) as depth-shaded 3-D particles, so the
VISUALIZATION cannot diverge from the TRACKING.

Layout — SEMANTIC 2x3. Captions name the TYPE of visualization first (2D pixels /
3D particles / 3D point cloud), then the coloring; rows are 2-D (top) and the 3-D
camera-pan render (bottom); ONE camera PANS (yaw sweeps slowly, monotonic). The six tiles:

         col1                         col2                      col3
    2D | RGB camera frame           | 2D pixels, by particle  | 2D pixels, by cluster      |
    3D | 3D particles, by avg color | 3D particles, by cluster| 3D point cloud, by cluster |
    (+ a stats row)

The two 3-D PARTICLE lenses (col1/col2, bottom) are the SAME Gaussians from the SAME
geometry, differing ONLY in tint: col1 by per-blob AVG color (each particle = the average
color of the pixels it covers — "what it is"), col2 highlighted by CLUSTER (vivid
foreground / muted background — "which object it's in", the same palette as the cluster
seg). col3 lifts the depth-projected pixels into 3-D as a point cloud, colored by cluster.
The 2-D top row is the per-pixel assignment of the image to particles (avg color) and to
clusters. Particles are solid depth-shaded balls, painter-occluded (no 2-D outline). The
3-D view is ONE slow shallow pan (no double-back). Several RENDER-ONLY temporal EMAs keep
it calm (blob means/covs + orbit centre kill the particle jitter; a 2-D seg-color EMA
kills boundary flicker), big diffuse background particles are filtered, outliers are dark
red, and per-particle alpha is smoothed off the posterior weight trace so weak particles
fade. Tracking is untouched (bit-identical).

RENDER_MODE=focused retires this 2x3 and ships ONE unified pipeline FIGURE per clip
(<vid>_unified.mp4), white-paper themed: "RGB input -> 3-D Gaussian particles -> outputs".
The hero is WORLD-STABILIZED: per-frame ego-motion is estimated from the BACKGROUND blobs
(trimmed weighted Kabsch — the clustering hands us the static background, so the rigid
motion it shares in the camera frame IS the camera's) and everything renders in the
frame-0 world frame: the background cloud holds still, the object moves through it, and
the camera rig (three frustums, frozen mutually-rigid scalars mapped through the ego pose)
visibly TRAVELS/PANS with the real camera. The hero shows ONE particle cloud, PURE
PARTICLES (no point-cloud splats anywhere), with both attributes MULTIPLEXED per particle
— fill = the particle's true average color (appearance, a slow per-blob EMA so colors
track what each blob currently covers), a faint under-pass GLOW = its cluster color
(semantics) — plus DEPTH FOG (atmospheric perspective) so the cloud reads as a 3-D scene.
The frustums sit at REAL camera positions, every one AIMED AT THE SCENE CENTER — the
input camera's apex IS the actual camera (the ego map sends the camera-frame origin to
the true camera position), the novel cameras orbit it about the followed scene pivot, and
each camera's orientation is lookAt(apex -> scene center). The hero is a FOLLOW camera
solved over the whole clip: a TWO-PASS render (pass 0 = tracker only, collecting framing
stats; pass 1 = deterministic replay + render) fits smoothed aim + framing TRACKS with
FIXED depth and ONE focal — the view pans steadily with the action, keeps everything
interesting tightly in frame, and never zooms or dollies. Windows are TRUE
perspective-warped image planes (they may shear — the honest 3-D look), DEPTH-INTERLEAVED
with the marbles (matter in front of a camera occludes it, matter behind is hidden). The
windows show the RENDERED RGB representation: the input camera carries the per-pixel
reconstruction (= panel 3a, the azure tie), the two novel poses carry the Gaussian state
rendered DENSELY as an RGB image from exactly those cameras. Outputs: 3a RGB
reconstruction + 3b INFERRED object segmentation (co-equal renders of the same state),
plus the classic INFERENCE-STATS row spanning the bottom. Real typography (DejaVu via
PIL), floating cards with soft shadows. Seeds route custom=SAM / DAVIS=GT
(FORCE_SAM_SEED=1 forces all-SAM). See docs/RENDER_PARTICLES.md.

Output is H.264. Run:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 python scripts/render_gaussian_demo.py \
        --target-duration 6 --out-dir runs/calibrate_consistency/viz_gaussian
    # the unified figure (DAVIS=GT seeds; the featured deforming subjects):
    RENDER_MODE=focused DEMO_NUM_BLOBS=256 STICKINESS=12 ROBUST_DELTA=6 \
        python scripts/render_gaussian_demo.py --target-duration 0
"""
from __future__ import annotations
import argparse, os, sys, time
from collections import deque
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
try:                                   # real typography for the unified figure (PIL + DejaVu);
    from PIL import Image, ImageDraw, ImageFont   # cv2 Hershey is the (ugly) fallback
    _PIL_OK = True
except Exception:                      # noqa: BLE001
    _PIL_OK = False

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))

import run_streaming_live as live
import genmatter_rt
import genmatter_viz
from render_demo import _transcode_to_h264
import render_live_grid as classic
from render_live_grid import (_resolve, _seed_grid, _cap_seed_clusters,
                              _count_seed_instances, _iter_frames_looped, _stats_row,
                              MAX_SAM_CLUSTERS)
from render_live_grid_v2 import _color_cluster_tile

# SEED ROUTING for the DEMO. `render_live_grid._resolve` routes CUSTOM clips to their
# self-supervised SAM frame-0 seed (`is_gt=False`) and DAVIS clips to their clean GT segmask
# (`is_gt=True`). The visualization demo uses exactly that — custom=SAM, DAVIS=GT — because a
# DAVIS GT mask is a single clean object whereas the SAM-propagated DAVIS seed over-segments
# (e.g. blackswan = ~12 instances), which clutters the cluster lens. This is a DEMO choice for
# clean visuals, NOT an eval/calibration shortcut: the honest self-supervised path is SAM
# everywhere (reachable via FORCE_SAM_SEED=1 -> `_resolve_sam`).
#
# `_resolve_sam` is the no-GT-leakage variant: it swaps a DAVIS GT result for the SAM2-propagated
# frame-0 seed (same uint16 id-map format as custom clips -> is_gt=False). Used only when
# FORCE_SAM_SEED=1.
_DAVIS_SAM = _REPO / "assets/tapvid_davis_30_videos_processed/sam2_propagated"
FORCE_SAM_SEED = os.environ.get("FORCE_SAM_SEED", "") == "1"


def _resolve_sam(vid):
    src, mask_path, is_gt = _resolve(vid)
    if is_gt:
        sam = _DAVIS_SAM / vid / "00000.png"
        if sam.is_file():
            return src, sam, False
    return src, mask_path, is_gt


def _resolve_seed(vid):
    """Demo seed routing: custom=SAM, DAVIS=GT (the clean-visual default). FORCE_SAM_SEED=1
    forces the honest all-SAM path (`_resolve_sam`)."""
    return _resolve_sam(vid) if FORCE_SAM_SEED else _resolve(vid)


# Systemic over-seg fix: self-supervised MOTION-coherence merge of SAM frame-0 instances.
# SEED_MERGE_VEL (a world-velocity distance threshold, e.g. 0.02) enables it; unset = off.
# SEED_MERGE_K = warmup frames of velocity to average (a moving object needs a few frames to
# separate from a static/coherent background; frame-0 velocity alone is degenerate).
SEED_MERGE_VEL = os.environ.get("SEED_MERGE_VEL")
SEED_MERGE_K = int(os.environ.get("SEED_MERGE_K", "12"))


def merge_seed_instances(seed_grid, vels, indices, gh, gw, *, vel_thr):
    """Agglomeratively MERGE spatially-ADJACENT SAM frame-0 instances that MOVE TOGETHER
    (mean multi-frame velocity within ``vel_thr``) -> a rigid/coherent background (table
    fragments at ~0 velocity; swan/water sharing the camera-induced velocity) collapses to
    ONE cluster, while a genuinely moving object (jacket, swan) keeps a distinct velocity and
    stays separate. So the tracker builds fewer hyperblobs AND the coherent background is held
    rigid by its hyperblob. White [255,255,255] background is never fused into an object.
    Self-supervised, ONE global threshold (no per-video tuning); replaces the blunt count-cap
    `_cap_seed_clusters`. ``vels`` = (N,3) per-datapoint mean velocity over the warmup frames."""
    sg = np.asarray(seed_grid).reshape(-1, 3)
    colors, inv = np.unique(sg, axis=0, return_inverse=True)        # (C,3), per-cell color id
    C = len(colors)
    is_white = np.all(colors == 255, axis=1)
    dp_cid = inv[indices.astype(np.int64)]                          # (N,) instance id per datapoint
    vsum = np.zeros((C, 3), np.float64); cnt = np.zeros(C)
    np.add.at(vsum, dp_cid, np.asarray(vels, np.float64)); np.add.at(cnt, dp_cid, 1.0)
    mvel = vsum / np.maximum(cnt[:, None], 1.0)                     # (C,3) per-instance mean velocity
    idg = inv.reshape(gh, gw)
    parent = list(range(C))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    seen = set()
    for di, dj in ((0, 1), (1, 0)):                                 # 4-neighbour adjacency on the id map
        a = idg[:gh - di, :gw - dj].ravel(); b = idg[di:, dj:].ravel()
        for x, y in zip(a, b):
            if x == y: continue
            k = (x, y) if x < y else (y, x)
            if k in seen: continue
            seen.add(k)
            if is_white[x] or is_white[y] or cnt[x] < 1 or cnt[y] < 1: continue
            if float(np.linalg.norm(mvel[x] - mvel[y])) < vel_thr:  # move together -> merge
                parent[find(x)] = find(y)
    merged = np.array([find(c) for c in range(C)])
    return colors[merged[inv]].reshape(gh, gw, 3).astype(np.uint8)


def _warmup_total_displacement(src, indices, intr, stride, H, W, k):
    """Pre-pass: SUM per-datapoint 3-D velocity over the first ``k`` frames = each point's
    TOTAL DISPLACEMENT (depth + flow only, no tracker). Total displacement (not per-frame
    velocity) is what separates a SLOW-moving object — which accumulates real displacement
    over k frames — from a TRULY-static background that stays ~0, so the motion-coherence
    merge doesn't fuse slow objects (wine, jello) into the background. Reuses the per-frame
    perception kernels."""
    import pathlib
    dacc = None; prev = None; dstate = None
    for _i, bgr in live.iter_frames(pathlib.Path(src), k):
        d_raw = live._depth_forward(bgr).astype(np.float32)
        depth_use, dlo, dhi, dstate = genmatter_rt.stabilize_depth(d_raw, dstate)
        cur, hw = live._bgr_to_raft_tensor(bgr)
        if prev is None:
            prev = cur; continue                                   # need a frame pair for flow
        flow = live._flow_forward(prev, cur, hw); prev = cur
        _pos, vel = genmatter_rt.unproject(depth_use, flow, indices, intr, stride,
                                           depth_lo=dlo, depth_hi=dhi)
        dacc = vel.copy() if dacc is None else dacc + vel          # accumulate displacement
    return dacc

TILE_H, TILE_W = 432, 768
STATS_H = classic.STATS_H
NCOL = 3
SRC_FPS = 30.0
# 3-D camera PAN: ONE slow MONOTONIC sweep -amp -> +amp across the whole clip (a
# single gentle pan from a shallow angle; never doubles back). yaw uses the output
# frame index so it's smooth even across loop seams.
# DEPTH STABILIZATION (the fix for background drift): robust affine-align each
# frame's Depth-Anything disparity to a FROZEN frame-0 reference + fixed-bound
# normalization (genmatter_rt.stabilize_depth), so static geometry keeps a CONSTANT Z
# instead of swimming with per-frame min/max. Feeds BOTH the tracker (unproject) and the
# point cloud, so particles AND the cloud stop drifting. DEPTH_STABILIZE=0 falls back to
# per-frame-norm + 0.6/0.4 pre-norm EMA. Tracker-affecting -> shipping it into the validated
# config needs the held-out region-J gate; the demo carries it as an override.
DEPTH_STABILIZE = os.environ.get("DEPTH_STABILIZE", "1") == "1"
PC_PAN_AMP = float(os.environ.get("PC_PAN_AMP", "20.0"))      # yaw half-range (deg) — wider so the 3-D reads clearly
PC_PITCH_AMP = float(os.environ.get("PC_PITCH_AMP", "9.0"))   # one gentle pitch ARC (sin(pi*frac)) adds vertical parallax
# RENDER_MODE=focused ships TWO clean videos per clip from ONE tracker pass (retiring the busy
# 2x3): <vid>_particles.mp4 = the particle field in BOTH colorings (avg-RGB + cluster) from THREE
# STATIC views (the parallax between fixed angles reads as genuinely 3-D without a nauseating pan),
# and <vid>_pixels.mp4 = the 2-D RGB reconstruction + cluster segmentation beside the input (the
# generative model "used for rendering"). Default (unset) keeps the legacy 2x3 path byte-for-byte.
RENDER_MODE = os.environ.get("RENDER_MODE", "").strip().lower()
# ---- focused-mode HERO camera + frustum diagram (all RENDER-ONLY; tracker untouched) ----
# The particles video is a PIPELINE TRIPTYCH that reads as a conceptual diagram:
#     RGB input  ->  ONE BIG elevated 3/4 "hero" view of the 3-D Gaussian particles
#                    with THREE camera frustums drawn IN the scene  ->  RGB reconstruction.
# (Replaces the unreadable 2x3: six small tiles, three near-identical views, fg ~ bg color,
# and a naked blob-cloud with no spatial reference. One big view + in-scene cameras showing
# their rendered images + accent-tied input/output panels IS the story: "cameras looking at
# the 3-D Gaussian scene the tracker maintains".) Knobs:
HERO_YAW = float(os.environ.get("HERO_YAW", "12.0"))       # hero meta-camera yaw (deg)
HERO_PITCH = float(os.environ.get("HERO_PITCH", "14.0"))   # hero elevation: look DOWN on scene + camera
# arc. Higher tilts the ground plane up behind the object; a gentle look-down lets the ground
# recede low so the object cluster stands against the backdrop.
HERO_DEPTH_SCALE = float(os.environ.get("HERO_DEPTH_SCALE", "1.0"))  # depth smoosh in the hero (1 = OFF:
# the elevated view + the frustums already carry the depth story)
FRUSTUM_YAW = float(os.environ.get("FRUSTUM_YAW", "22.0"))  # +/- yaw of the two NOVEL frustum cameras (deg).
# The novel images are honestly RE-RENDERED from exactly this yaw (drawn pose == rendered
# content). Wider yaw throws the novel apexes further off-axis, so the everything-fits zoom
# shrinks the view windows; this value keeps the arc readable AND the planes big.
# The frustum locations are REAL: the input camera's apex sits at the ACTUAL camera position
# (the camera-frame origin mapped through the ego pose) and the novel cameras are that real
# camera rotated about the object pivot on the TRUE-radius arc. The real camera position can't
# be engulfed by the matter it images. PULLBACK retreats the hero meta-camera (pure view-depth
# shift) so the real camera + its image plane + the whole particle scene all FIT in the hero frame.
PULLBACK = float(os.environ.get("PULLBACK", "0.6"))   # hero retreat, as a fraction of the true
# camera-object distance (bigger = wider stage, smaller object)
FRUSTUM_IMG_FRAC = float(os.environ.get("FRUSTUM_IMG_FRAC", "0.45"))    # image-plane half-width as a fraction
# of the object's (EMA'd) particle radius
RECON_FIT_MARGIN = float(os.environ.get("RECON_FIT_MARGIN", "0.5"))    # object spans ~this frac of the hero;
# the rest is HEADROOM so a moving/deforming object doesn't pop out of frame. The frustum
# corners/apexes get a separate HARD fit (0.88) — the cameras are the outermost diagram elements.
# Make the DEFORMING OBJECT the star: the background plane (couch / ground / sky) is most of the
# image, so the 3-D particle field is otherwise dominated by background and the foreground object is
# lost. BG_FADE multiplies the alpha of BACKGROUND-cluster particles (hyperblob id >= num_fg) in the
# focused 3-D views so the background drops to a faint haze and the vivid foreground object pops.
# 1.0 = no fade; lower = stronger emphasis (more transparent background). (2-D panels stay full.)
BG_FADE = float(os.environ.get("BG_FADE", "0.3"))
# A cluster covering more than this fraction of the particles is treated as the background plane and
# faded too (catches the dominant couch/sky/ground that a SAM seed mislabels as a foreground instance).
BG_SIZE_FRAC = float(os.environ.get("BG_SIZE_FRAC", "0.33"))
# FIGURE-GROUND: besides the alpha fade, mix background marble FILLS toward near-white (BG_GHOST_BGR)
# so they recede into the paper while the object keeps its true color. 0 = off, 1 = full ghost.
BG_DESAT = float(os.environ.get("BG_DESAT", "0.45"))
# SEMANTIC GLOW (the unification): EVERY particle keeps its TRUE avg-RGB fill and carries a FAINT
# radial halo in its SEGMENTATION color behind it (the cluster palette — vivid fg / muted bg, the
# same colors as the segmentation overlay). Object glows union into a soft highlighter field; bg
# glows are additionally scaled by BG_FADE so they whisper. GLOW_ALPHA is the blend parameter
# (higher = stronger highlight glow); the OBJECT's glow is exempt from the depth-fog fade (only
# fills + background glows fog out).
GLOW_ALPHA = float(os.environ.get("GLOW_ALPHA", "0.9"))
GLOW_SCALE = float(os.environ.get("GLOW_SCALE", "2.4"))    # glow radius, in units of the marble
# axes (higher = larger visible fringe around each particle)
# DEPTH FOG (atmospheric perspective): each view's particles mix toward the paper tone with
# that view's NORMALIZED depth (10th-90th pct),
# and glows fade with it — near particles full-strength, far ones recede into the page, so the
# cloud reads as a 3-D scene from particles ALONE. 0 = off.
FOG = float(os.environ.get("FOG", "0.45"))
# WORLD STABILIZATION: the background of the world tends to be STATIC and the CAMERA
# MOVES — implemented computationally from the clustering. Per frame the camera's ego-motion is
# solved by trimmed weighted Kabsch over the BACKGROUND blob means (blob indices persist, so
# correspondences are free) and the whole cloud is rendered in the frame-0 WORLD frame; the
# frustum camera rig is frozen in the (moving) camera frame and mapped through the ego pose, so
# the three cameras stay mutually rigid (no drift-apart) and visibly travel with the real camera.
EGO_STABILIZE = os.environ.get("EGO_STABILIZE", "1") == "1"   # 0 = camera-frame (no stabilization)
# Per-blob avg-RGB color EMA rate (focused mode). A frame-0 LOCKED LUT keeps painting a re-bound
# blob with its OLD content's color when the camera pans hard; a slow EMA tracks what each blob
# currently covers while keeping identity stable (no flicker). 1.0 = per-frame raw, ~0.15 = slow follow.
LUT_EMA = float(os.environ.get("LUT_EMA", "0.15"))
# The NOVEL frustum planes show the Gaussian state RENDERED AS AN RGB IMAGE ("the frustum windows
# show the RENDERED rgb representation" — PIXELS, not a marble scatter): a tiny volumetric
# Gaussian-splatting rasterizer (front-to-back per-pixel alpha compositing of the REAL
# blob_means/blob_covs/avg-colors) produces a dense soft image of the representation from each
# novel camera. Pure model state — no depth-map re-projection, no point clouds.
GS_ALPHA_MAX = float(os.environ.get("GS_ALPHA_MAX", "0.9"))   # peak per-Gaussian opacity
# OBJECT-CENTRIC framing (focused mode): the camera ORBITS and FRAMES on the deforming object (not the
# whole scene), so a small object FILLS the view and is well-scaled (a scene-wide fit makes small
# objects tiny + badly scaled). The object centroid is EMA'd across frames so the camera follows it smoothly.
OBJ_CENTROID_ALPHA = float(os.environ.get("OBJ_CENTROID_ALPHA", "0.3"))   # higher = camera tracks the moving object faster
FOCAL_CALIB_FRAMES = int(os.environ.get("FOCAL_CALIB_FRAMES", "8"))       # frames to accumulate the widest object framing over
# Particle DENSITY: the object is a small fraction of the image, so with the shipped 128 blobs only a
# handful land on it (sparse). DEMO_NUM_BLOBS overrides num_blobs for the demo so MORE particles land on
# the object (densely visualized); the background gets more too but is faded + framed out. "" = use config.
DEMO_NUM_BLOBS = os.environ.get("DEMO_NUM_BLOBS", "")
# Jitter fix (render-only EMAs; tracking state untouched -> bit-identical):
PC_CENTROID_ALPHA = float(os.environ.get("PC_CENTROID_ALPHA", "0.15"))  # orbit-center EMA (kills cloud swim)
MARBLE_GEOM_BETA = float(os.environ.get("MARBLE_GEOM_BETA", "0.25"))    # blob mean/cov EMA (kills particle wobble)
# Temporal assignment STICKINESS (real tracker fix for background squish): kappa added
# to each datapoint's prev-frame blob logit so static points stop hopping. Env overrides
# tracking.final_assignment_stickiness. "" / 0 = OFF (bit-exact).
STICKINESS = os.environ.get("STICKINESS")
# Occlusion/outlier-robust blob-mean update (Huber IRLS delta in sigma units, ~3): an
# occluded background point can't pull its blob. Env overrides tracking.mean_update_robust_delta.
ROBUST_DELTA = os.environ.get("ROBUST_DELTA")
MARBLE_SIGMA = float(os.environ.get("MARBLE_SIGMA", "1.15"))  # particle size, in sigmas (smaller
# separates a dense object cluster into distinct marbles; larger fuses them together)
SEG_EMA = float(os.environ.get("SEG_EMA", "0.3"))            # render-only EMA of the 2D seg tiles (edge-flicker fix; 0.3 ~halves boundary oscillation)
SEG_EDGE_AWARE = os.environ.get("SEG_EDGE_AWARE", "1") == "1"  # 0 => plain (blocky but temporally STABLE) upsample
# Spurious-particle filter (systematic, fg/bg-AGNOSTIC). The robust signal is
# DENSITY = effective datapoint count / 3-D covariance extent: a particle that is
# BIG yet owns FEW datapoints is diffuse/spurious (the "big gray background blob"),
# regardless of whether it was mis-assigned to a foreground cluster. Thresholds are
# frozen at frame 0 (so the cull set doesn't itself flicker) and applied per-frame to
# the EMA'd (smooth) geometry. Env knobs override the auto cuts:
SPURIOUS_EXTENT_FENCE = float(os.environ.get("SPURIOUS_EXTENT_FENCE", "0.0"))  # 0 => auto (Q75+1.5*IQR boxplot fence on extent)
SPURIOUS_DENSITY_PCT = float(os.environ.get("SPURIOUS_DENSITY_PCT", "15.0"))   # density percentile (among alive blobs) for the "diffuse" cut
SPURIOUS_EXTENT_PCT = float(os.environ.get("SPURIOUS_EXTENT_PCT", "70.0"))     # extent percentile gate for "big" (paired with the density cut)
SPURIOUS_EXTENT = float(os.environ.get("SPURIOUS_EXTENT", "0.0"))   # bg backstop: 0 => auto (88th pct of bg extents)
SPURIOUS_HARD_CAP = float(os.environ.get("SPURIOUS_HARD_CAP", "0.0"))  # 0 => disabled
SPURIOUS_DEBUG = os.environ.get("SPURIOUS_DEBUG", "1") == "1"  # print the per-blob extent/count/density distribution at frame 0
WEMA_ALPHA = 0.30                                             # temporal EMA on posterior weights
DARK_RED_BGR = np.array([30, 30, 130], dtype=np.uint8)       # outlier color
CULL_MIN_PTS, CULL_FULL_PTS = 3.0, 11.0                       # smoothstep cull thresholds (datapoints)
# Featured subjects = VALIDATED richly-deforming, low-occlusion DAVIS clips whose track holds across
# the whole clip (GT-seeded for the demo; eval stays SAM). Small subjects are excluded because too
# few particles land on them (sparse, unreadable).
DEFAULT_VIDEOS = ["dog", "breakdance", "horsejump-high", "blackswan"]


def _enumerate_davis():
    """All DAVIS video names, enumerated from disk (a segmask dir with a 00000.png
    frame-0 mask) — never a hardcoded list, so it tracks the assets/ tree."""
    root = classic._DAVIS_GT
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "00000.png").is_file())


def _enumerate_custom():
    """All custom video names (assets/custom_videos/<name>/source.{mp4,mov,MOV}),
    enumerated from disk — matches what _resolve() can actually load."""
    root = classic._CUSTOM
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and any((p / f"source{e}").is_file()
                                        for e in (".mp4", ".mov", ".MOV")))


def _expand_video_args(videos):
    """Expand the sentinels ``all-davis`` / ``all-custom`` / ``all`` into concrete
    names enumerated from disk; any other token passes through unchanged (so
    ``--videos dog all-custom`` works). Order-preserving + de-duplicated."""
    expand = {"all-davis": _enumerate_davis, "all-custom": _enumerate_custom,
              "all": lambda: _enumerate_davis() + _enumerate_custom()}
    out, seen = [], set()
    for v in videos:
        for name in (expand[v]() if v in expand else [v]):
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _is_raw_source(vid: str) -> bool:
    """True if ``vid`` is a raw video path (a --source file/frames dir) rather than
    a bare DAVIS/custom video name resolved via _resolve_seed. DAVIS/custom names
    are bare tokens, so a path separator or an existing file disambiguates."""
    return (os.sep in str(vid)) or Path(vid).is_file()


# ---- unified pipeline-diagram layout (the ONE focused-mode video per clip) ----
# WHITE-PAPER theme: the diagram reads like a figure, not a night scene — on white the object's
# colors + the vivid semantic palette pop and the frustum wireframes are crisp (dark marbles and
# wireframes are muddy on black). Layout (2576 x 720):
#
#                      [2. 3D Gaussian particles                ]  ->  [3a. RGB reconstruction]
#   [1. RGB input] ->  [   fill = avg RGB, glow = segmentation  ]
#                      [   + input camera & 2 novel cameras     ]  ->  [3b. object segmentation]
#
# ONE video per clip (<vid>_unified.mp4): the recon/seg tiles are the diagram's OUTPUT panels.
# ONE particle cloud carries BOTH attributes multiplexed per particle: FILL = true avg color
# (appearance), faint GLOW = cluster color (semantics).
UNI_SIDE_W, UNI_SIDE_H = 576, 324      # side panels: input / recon / segmentation (16:9)
UNI_CAP_H = 48                         # caption strip above each side panel (big readable type)
HERO_W, HERO_H = 1280, 720             # THE hero view (16:9)
UNI_GUT = 72                           # arrow gutters between the columns
UNI_MARGIN = 24                        # outer canvas margin (cards float on the paper)
UNI_W = UNI_MARGIN * 2 + UNI_SIDE_W * 2 + UNI_GUT * 2 + HERO_W   # 2624
UNI_H = 768                            # hero card (720) + breathing room; right col = 2x372 + 24
CANVAS_BGR = (247, 247, 247)           # the page tone — white CARDS float on it w/ soft shadows
FRUSTUM_IMG_HW = (180, 320)            # render size of the little novel-view images (16:9)
PAPER_BGR = (255, 255, 255)            # the white-paper canvas + hero background
INK_BGR = (30, 30, 30)                 # captions / diagram text
ACCENT_BGR = (255, 140, 0)             # azure-blue: ORIGINAL-camera frustum + the borders of every
# panel imaged at that pose (input / reconstruction / segmentation) — gray = the novel poses
FRUSTUM_GRAY = (110, 110, 110)         # the two novel-view frustums (dark gray reads on white)
PANEL_EDGE = (185, 185, 185)           # neutral 1-px panel border (non-accented panels)
BG_GHOST_BGR = 232.0                   # bg marbles mix toward this near-white (faint ghosts on paper)


def _log(m): print(f"[render_gaussian {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _label(img, name, fps=None):
    # Larger, bolder caption bar so the per-tile type is easy to read.
    cv2.rectangle(img, (0, 0), (TILE_W, 42), (0, 0, 0), -1)
    cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if fps is not None:
        t = f"{fps:4.1f} FPS"
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        cv2.putText(img, t, (TILE_W - tw - 10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 255, 255), 1, cv2.LINE_AA)
    return img


def _fit(img):
    return img if img.shape[:2] == (TILE_H, TILE_W) else cv2.resize(img, (TILE_W, TILE_H))


def _smoosh_depth(means, centroid, z_scale):
    """Compress each particle's DEPTH offset from the view centroid by ``z_scale`` (<1) — the
    'smoosh the axis into the camera'. Returns a COPY (means never mutated -> tracking + the
    next-frame EMA untouched). z_scale==1 is a no-op (returns the input)."""
    if z_scale == 1.0 or means is None or len(means) == 0:
        return means
    cz = float(np.asarray(centroid, np.float32)[2])
    out = np.asarray(means, np.float32).copy()
    out[:, 2] = cz + (out[:, 2] - cz) * float(z_scale)
    return out


def _calibrate_marble_focal(means, covs, alpha, proj, *, z_scale, sigma_scale,
                            margin, out_hw, fallback, extra_pts=None, extra_margin=0.88):
    """VIDEO-INDEPENDENT focal calibration. Size ONE focal (frozen on frame 0) so the ACTUAL
    displayed particles — each centre PLUS its projected radius — fit the tile at this (widest)
    view, AFTER the same pitch + depth ``z_scale`` smoosh the marbles are drawn with. Framing the
    real particles (not the depth-point cloud, which over-frames) at a high percentile of their
    centre+radius extent means every clip auto-frames itself with no per-video tuning and nothing
    spills off the edge. ``extra_pts`` (M,3 world — the frustum corners/apexes) get a HARD
    max-fit at ``extra_margin``: diagram elements must NEVER clip, while the particle fit stays
    a soft percentile. Returns ``fallback`` when no particles are visible."""
    out_H, out_W = out_hw
    means = None if means is None else np.asarray(means, np.float32)
    if proj is None or means is None or means.shape[0] == 0:
        return fallback
    vis = np.asarray(alpha, np.float32) > 0.02
    if not vis.any():
        return fallback
    R = np.asarray(proj["R"], np.float32); centroid = np.asarray(proj["centroid"], np.float32)
    cz = float(centroid[2])
    d = means[vis] - centroid
    d[:, 2] *= float(z_scale)                       # same smoosh as the marble draw
    mr = d @ R.T; mr[:, 2] += cz
    z = mr[:, 2]; ok = z > 1e-3
    if not ok.any():
        return fallback
    mr = mr[ok]; z = z[ok]
    lam = np.linalg.eigvalsh(np.asarray(covs, np.float64)[vis][ok])[:, -1]  # world radius^2
    r = float(sigma_scale) * np.sqrt(np.maximum(lam, 0.0))                  # particle world radius
    ext_x = (np.abs(mr[:, 0]) + r) / z              # angular half-extent incl. the marble disk
    ext_y = (np.abs(mr[:, 1]) + r) / z
    fx = margin * 0.5 * out_W / (float(np.percentile(ext_x, 97)) + 1e-6)
    fy = margin * 0.5 * out_H / (float(np.percentile(ext_y, 97)) + 1e-6)
    f = float(min(fx, fy))
    if f <= 0.0:
        return fallback
    if extra_pts is not None and len(extra_pts):
        e = np.asarray(extra_pts, np.float32) - centroid
        e[:, 2] *= float(z_scale)                   # same depth smoosh as the particles
        er = e @ R.T; er[:, 2] += cz
        ez = er[:, 2]; eok = ez > 1e-3
        if eok.any():
            ex = float(np.max(np.abs(er[eok, 0]) / ez[eok])) + 1e-6
            ey = float(np.max(np.abs(er[eok, 1]) / ez[eok])) + 1e-6
            f = float(min(f, extra_margin * 0.5 * out_W / ex, extra_margin * 0.5 * out_H / ey))
    return f if f > 0.0 else fallback


def _project_world_pts(pts, proj, *, z_scale=1.0):
    """Project WORLD points through the SAME rotate->perspective transform the marble tile
    uses (``proj``'s R/centroid/focal, plus the optional depth smoosh), so frustum geometry
    lands EXACTLY in the marbles' view. Returns ``(uv (N,2) float, view-depth z (N,))``;
    points at/behind the camera give non-finite uv — callers must guard."""
    R = np.asarray(proj["R"], np.float32); c = np.asarray(proj["centroid"], np.float32)
    p = np.asarray(pts, np.float32) - c
    if z_scale != 1.0:
        p = p.copy(); p[:, 2] *= float(z_scale)
    mr = p @ R.T
    mr[:, 2] += float(c[2])
    z = mr[:, 2]
    f = float(proj["focal"])
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(z > 1e-3, z, np.nan)
        u = f * mr[:, 0] / safe + float(proj["out_cx"])
        v = f * mr[:, 1] / safe + float(proj["out_cy"])
    return np.stack([u, v], 1), z


def _make_proj(yaw_deg, pitch_deg, pivot, focal, out_hw):
    """A MANUAL marble-renderer projection — exactly the ``{centroid, R, focal, out_cx,
    out_cy}`` fields ``render_particle_marbles_tile`` / ``_project_world_pts`` consume:
    rotate about ``pivot`` by yaw then pitch, keep the pivot's depth as the view distance.
    Replaces the focused path's per-frame ``_build_pointcloud_projection`` calls (whose
    centroid came from the DEPTH MAP, i.e. the moving camera) so the hero is a FIXED
    world-frame meta-camera and the background no longer swims with the real camera."""
    out_H, out_W = out_hw
    yaw, pitch = np.deg2rad(float(yaw_deg)), np.deg2rad(float(pitch_deg))
    Ry = np.array([[np.cos(yaw), 0., np.sin(yaw)],
                   [0., 1., 0.],
                   [-np.sin(yaw), 0., np.cos(yaw)]], np.float32)
    Rx = np.array([[1., 0., 0.],
                   [0., np.cos(pitch), -np.sin(pitch)],
                   [0., np.sin(pitch), np.cos(pitch)]], np.float32)
    return {"centroid": np.asarray(pivot, np.float32).copy(), "R": (Rx @ Ry).astype(np.float32),
            "focal": float(focal), "out_cx": out_W * 0.5, "out_cy": out_H * 0.5}


def _smooth_track(x, sigma=8.0):
    """Gaussian-smooth a per-frame track (n,) offline (reflect-padded) — the FOLLOW
    camera's pivot path: perfectly steady, no causal lag, no jitter."""
    x = np.asarray(x, np.float64)
    if x.shape[0] < 3:
        return x.astype(np.float32)
    rad = max(int(3 * sigma), 1)
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / float(sigma)) ** 2)
    k /= k.sum()
    xp = np.pad(x, rad, mode="reflect")
    return np.convolve(xp, k, mode="valid").astype(np.float32)


def _solve_follow_camera(stats, rig, hero_R):
    """Solve the whole-clip FOLLOW camera from the pass-0 trajectories — "keep everything
    interesting in the frame, no weird blank space, follow the actual scene" under the
    earlier hard rule "the camera must not go in and out":
    - ``ego`` = the SMOOTHED ego-pose track. The live incremental-Kabsch pose stream
      carries per-frame noise — rotation wobble rocks the whole view, t_z noise makes it
      BREATHE in and out (~±1.8%/frame scale + 0.8 deg/frame rotation) — and world jitter
      is visually identical to VIEWER-camera jitter. Offline Gaussian smoothing (sigma 5;
      R re-projected to SO(3) by SVD) keeps the real camera path and kills the noise; pass
      1 renders with THESE poses, not the live solve;
    - ``aim_t`` = the smoothed OBJECT/scene track (fixed depth) — what every camera POINTS
      AT and what the novel apexes ORBIT. The aim must be the scene itself: blending the
      apex positions in here pulls the orbit center toward the cameras, HALVING the arc
      radius and shrinking the plane spacing below the plane width (so the frustums would
      overlap);
    - ``pivot_t`` = the hero FRAMING track = 0.7·object + 0.3·apex-mean (apexes from the
      SMOOTHED poses), NEAR-STATIC smoothing (sigma 25 — the calmer the camera, the
      better; the focal fit absorbs the extra margin a lazy follow needs) with DEPTH
      FIXED to the whole-clip median — a moving pivot-z is a dolly, the exact "in and
      out";
    - ONE ``focal`` = the min over all frames of the fit at that frame's framing pivot
      (soft object margin RECON_FIT_MARGIN + HARD 0.92 fit of the frame's aimed-rig
      corners/apexes) — tight framing that never clips, constant scale;
    - per-pose SPLAT focals for the novel windows (object-fit margin 0.75, median over
      frames) — the lookAt poses depend on the solved aim track, so they're fitted here.
    ``stats`` carries CAMERA-frame geometry (bm/bc/piv_cam) + the live poses, so the
    solver can re-map everything consistently under the smoothed poses. Returns
    ``{"pivot_t", "aim_t", "ego", "focal", "push", "view_focals"}``; conservative
    fallbacks for degenerate clips (no object / no rig)."""
    n = len(stats["bm"])
    # ---- smoothed ego poses (kill the per-frame pose noise) ----
    Rs = (np.stack([np.asarray(stats["ego"][t][0], np.float32) for t in range(n)])
          if n else np.zeros((0, 3, 3), np.float32))
    Ts = (np.stack([np.asarray(stats["ego"][t][1], np.float32) for t in range(n)])
          if n else np.zeros((0, 3), np.float32))
    Rs_s, Ts_s = Rs.copy(), Ts.copy()
    if n >= 3:
        for a in range(3):
            Ts_s[:, a] = _smooth_track(Ts[:, a], sigma=5.0)
            for b in range(3):
                Rs_s[:, a, b] = _smooth_track(Rs[:, a, b], sigma=5.0)
        for t in range(n):                       # back to SO(3): nearest rotation by SVD
            U, _S, Vt = np.linalg.svd(Rs_s[t].astype(np.float64))
            Rk = U @ Vt
            if np.linalg.det(Rk) < 0:
                Rk = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
            Rs_s[t] = Rk.astype(np.float32)
    ego_fix = [(Rs_s[t], Ts_s[t]) for t in range(n)]
    # ---- world-frame geometry under the SMOOTHED poses ----
    bm_w = [np.asarray(stats["bm"][t], np.float32) @ Rs_s[t].T + Ts_s[t] for t in range(n)]
    bc_w = [np.einsum("ab,nbc,dc->nad", Rs_s[t],
                      np.asarray(stats["bc"][t], np.float32), Rs_s[t]) for t in range(n)]
    pivs = [(None if stats["piv_cam"][t] is None
             else np.asarray(stats["piv_cam"][t], np.float32) @ Rs_s[t].T + Ts_s[t])
            for t in range(n)]
    valid = [p for p in pivs if p is not None]
    center0 = (np.mean(np.asarray(valid, np.float32), 0) if valid
               else (bm_w[0].mean(0) if n else np.array([0.0, 0.0, 2.0], np.float32)))
    yawd = np.deg2rad(FRUSTUM_YAW)
    Rp = np.array([[np.cos(yawd), 0., np.sin(yawd)], [0., 1., 0.],
                   [-np.sin(yawd), 0., np.cos(yawd)]], np.float32)
    obj_tr = np.zeros((max(n, 1), 3), np.float32)
    interest = np.zeros((max(n, 1), 3), np.float32)
    for t in range(n):
        a0 = Ts_s[t]                              # the (smoothed) real camera position
        p = pivs[t] if pivs[t] is not None else center0
        obj_tr[t] = p
        am = (a0 + ((a0 - center0) @ Rp + center0) + ((a0 - center0) @ Rp.T + center0)) / 3.0
        interest[t] = 0.7 * p + 0.3 * am   # object-weighted: the rig is hard-fit anyway,
        # and chasing the apexes injects the handheld camera's wander into the framing
    aim_t = obj_tr.copy()
    pivot_t = interest.copy()
    if n:
        for arr, src in ((aim_t, obj_tr), (pivot_t, interest)):
            # sigma 25 = a NEAR-STATIC camera that only drifts when the action genuinely
            # migrates — any livelier follow read as "jerky, floats around, vertigo";
            # the focal fit below absorbs whatever margin the lazy track needs.
            arr[:, 0] = _smooth_track(src[:, 0], sigma=25.0)
            arr[:, 1] = _smooth_track(src[:, 1], sigma=25.0)
            arr[:, 2] = float(np.median(src[:, 2]))   # FIXED depths: no dolly, ever
    push = (PULLBACK * float(rig["d0"])) if rig is not None else 0.0
    push_vec = (push * np.asarray(hero_R, np.float32)[2, :]).astype(np.float32)
    f = None
    vf_acc = {-FRUSTUM_YAW: [], FRUSTUM_YAW: []}
    for t in range(n):
        cams = (_frustum_cams(aim_t[t], Ts_s[t], rig["g0"], rig["wh0"],
                              novel_yaw=FRUSTUM_YAW) if rig is not None else [])
        fp = (np.vstack([np.vstack([c["corners"], c["apex"][None]]) for c in cams])
              if cams else None)
        proj0 = _make_proj(HERO_YAW, HERO_PITCH, pivot_t[t], 1.0, (HERO_H, HERO_W))
        f_t = _calibrate_marble_focal(
            bm_w[t] + push_vec, bc_w[t], stats["oa"][t], proj0,
            z_scale=HERO_DEPTH_SCALE, sigma_scale=MARBLE_SIGMA,
            margin=RECON_FIT_MARGIN, out_hw=(HERO_H, HERO_W), fallback=None,
            extra_pts=(None if fp is None else fp + push_vec), extra_margin=0.92)
        if f_t is not None and f_t > 0.0:
            f = f_t if f is None else min(f, f_t)
        for c in (cams or []):
            if c["orig"]:
                continue
            proj_v = {"centroid": aim_t[t], "R": c["R_view"], "focal": 1.0,
                      "out_cx": FRUSTUM_IMG_HW[1] * 0.5, "out_cy": FRUSTUM_IMG_HW[0] * 0.5}
            f_v = _calibrate_marble_focal(
                bm_w[t], bc_w[t], stats["oa"][t], proj_v, z_scale=1.0,
                sigma_scale=2.0, margin=0.75, out_hw=FRUSTUM_IMG_HW, fallback=None)
            if f_v is not None and f_v > 0.0:
                vf_acc[c["yaw"]].append(f_v)
    view_focals = {yk: float(np.median(v)) for yk, v in vf_acc.items() if v}
    return {"pivot_t": pivot_t, "aim_t": aim_t, "ego": ego_fix,
            "focal": float(f if f else 600.0), "push": push_vec,
            "view_focals": view_focals}


def _kabsch(P, Q, w):
    """Weighted least-squares rigid alignment ``argmin_{R,t} sum_i w_i ||R p_i + t - q_i||^2``
    (Kabsch/Umeyama with the det(R)=+1 reflection guard). Returns ``(R (3,3), t (3,))``."""
    w = np.asarray(w, np.float64)[:, None]
    P = np.asarray(P, np.float64); Q = np.asarray(Q, np.float64)
    ws = float(w.sum()) + 1e-12
    mp = (P * w).sum(0) / ws; mq = (Q * w).sum(0) / ws
    Hm = ((P - mp) * w).T @ (Q - mq)
    U, _S, Vt = np.linalg.svd(Hm)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T))) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R.astype(np.float32), (mq - R @ mp).astype(np.float32)


def _estimate_ego_rigid(cur, anchors, w, prev):
    """EGO-MOTION from the clustering (the world-stabilization core): the camera pose
    ``(R_t, t_t)`` mapping this frame's CAMERA-frame background blob means onto their
    (static) WORLD anchors, by weighted Kabsch with ONE trim pass — drop the worst-residual
    25% and re-solve, because a blob that RE-BOUND to new scene content during a hard pan
    is an outlier, not ego-motion evidence. "The background of the world tends to be
    static" is exactly the gauge: whatever rigid motion the background blobs share is
    attributed to the camera. Falls back to ``prev`` (last frame's pose) on too few
    anchors or a degenerate solve."""
    cur = np.asarray(cur, np.float32); anchors = np.asarray(anchors, np.float32)
    if cur.shape[0] < 4 or not (np.isfinite(cur).all() and np.isfinite(anchors).all()):
        return prev
    try:
        R, t = _kabsch(cur, anchors, w)
        r = np.linalg.norm(cur @ R.T + t - anchors, axis=1)
        keep = r <= np.percentile(r, 75)
        if int(keep.sum()) >= 4:
            R, t = _kabsch(cur[keep], anchors[keep], np.asarray(w, np.float32)[keep])
        if not (np.isfinite(R).all() and np.isfinite(t).all()):
            return prev
        return R, t
    except np.linalg.LinAlgError:
        return prev


def _render_gaussian_rgb_view(means, covs, colors, alpha, proj, out_hw):
    """Render the Gaussian state AS AN RGB IMAGE from camera ``proj`` — a tiny volumetric
    splatting rasterizer ("the frustum windows show the RENDERED rgb representation"):
    each REAL blob (mean/cov/avg-color, no re-fit) is projected to a 2-D Gaussian
    footprint (EWA-style linearization ``S2 = J R cov R^T J^T``) and front-to-back
    alpha-composited per pixel with transmittance ``T``, then NORMALIZED by the total
    occlusion weight: a covered pixel takes the full weighted-mean color instead of
    bleeding paper white through the gaps between Gaussians (raw compositing left mean
    transmittance ~0.7 -> a washed-out plane), so the planes show dense PIXELS (a soft
    reconstruction-like image), not a marble scatter. Pure model state — no depth-map
    re-projection, no point clouds. Genuinely uncovered pixels keep the paper white."""
    out_H, out_W = out_hw
    paper = np.full((out_H, out_W, 3), 255.0, np.float32)
    if proj is None or means is None or len(means) == 0:
        return paper.astype(np.uint8)
    R = np.asarray(proj["R"], np.float32); c = np.asarray(proj["centroid"], np.float32)
    f = float(proj["focal"])
    m = (np.asarray(means, np.float32) - c) @ R.T
    m[:, 2] += float(c[2])
    z = m[:, 2]
    rgb = np.zeros((out_H, out_W, 3), np.float32)
    wsum = np.zeros((out_H, out_W), np.float32)        # total occlusion-weighted coverage
    T = np.ones((out_H, out_W), np.float32)            # per-pixel transmittance
    alpha = np.asarray(alpha, np.float32)
    colors = np.asarray(colors, np.float32)
    for i in np.argsort(z):                            # FRONT -> BACK (near occludes far)
        a0 = float(alpha[i]) if i < alpha.shape[0] else 0.0
        zi = float(z[i])
        if a0 <= 0.02 or zi <= 1e-3 or i >= colors.shape[0]:
            continue
        x, y = float(m[i, 0]), float(m[i, 1])
        u = f * x / zi + float(proj["out_cx"]); v = f * y / zi + float(proj["out_cy"])
        cov_r = R.astype(np.float64) @ np.asarray(covs[i], np.float64) @ R.astype(np.float64).T
        J = np.array([[f / zi, 0.0, -f * x / zi ** 2],
                      [0.0, f / zi, -f * y / zi ** 2]], np.float64)
        S2 = J @ cov_r @ J.T
        S2[0, 0] += 0.35; S2[1, 1] += 0.35             # anti-alias floor (px^2)
        det = S2[0, 0] * S2[1, 1] - S2[0, 1] * S2[1, 0]
        if not np.isfinite(det) or det <= 1e-9:
            continue
        r_max = 3.5 * float(np.sqrt(max(S2[0, 0], S2[1, 1])))
        r_max = min(r_max, 0.35 * float(max(out_H, out_W)))   # cap a runaway blob
        x0 = int(max(np.floor(u - r_max), 0)); x1 = int(min(np.ceil(u + r_max) + 1, out_W))
        y0 = int(max(np.floor(v - r_max), 0)); y1 = int(min(np.ceil(v + r_max) + 1, out_H))
        if x1 <= x0 or y1 <= y0:
            continue
        Si = np.array([[S2[1, 1], -S2[0, 1]], [-S2[1, 0], S2[0, 0]]], np.float64) / det
        X = np.arange(x0, x1, dtype=np.float32) - np.float32(u)
        Y = (np.arange(y0, y1, dtype=np.float32) - np.float32(v))[:, None]
        q = (Si[0, 0] * X * X)[None, :] + 2.0 * Si[0, 1] * X[None, :] * Y + Si[1, 1] * Y * Y
        a = (GS_ALPHA_MAX * a0) * np.exp(-0.5 * q).astype(np.float32)
        Treg = T[y0:y1, x0:x1]
        w = a * Treg
        rgb[y0:y1, x0:x1] += w[..., None] * colors[i][None, None, :]
        wsum[y0:y1, x0:x1] += w
        T[y0:y1, x0:x1] = Treg * (1.0 - a)
    m_cov = np.clip(wsum / 0.08, 0.0, 1.0)[..., None]  # soft coverage ramp -> paper
    dense = rgb / np.maximum(wsum, 1e-6)[..., None]
    out = m_cov * dense + (1.0 - m_cov) * paper
    return out.clip(0, 255).astype(np.uint8)


def _lookat_rows(apex, target):
    """Row-convention view rotation (rows = right / image-down / forward — the marble
    renderer's ``view = R (p - c)`` form) for a camera at ``apex`` LOOKING AT ``target``,
    world +y = down. Reduces to the identity when the target sits along +z from the apex;
    guarded for the (non-occurring) vertical-forward degeneracy."""
    f = np.asarray(target, np.float32) - np.asarray(apex, np.float32)
    n = float(np.linalg.norm(f))
    if n < 1e-6:
        return np.eye(3, dtype=np.float32)
    f = f / n
    r0 = np.cross(np.array([0., 1., 0.], np.float32), f)
    nr = float(np.linalg.norm(r0))
    r0 = (np.array([1., 0., 0.], np.float32) if nr < 1e-6 else r0 / nr)
    r1 = np.cross(f, r0)
    return np.stack([r0, r1, f]).astype(np.float32)


def _frustum_cams(pivot, apex0, g, wh, *, novel_yaw):
    """The THREE diagram cameras, every one AIMED AT THE SCENE CENTER ("the orientation of
    each of the cameras should always be on the center of the scene"). The yaw-0 apex is
    the REAL camera position (the ego-mapped camera origin — the azure frustum traces the
    true camera path); the two NOVEL apexes are that position rotated ``±novel_yaw`` about
    the VERTICAL axis through the (followed) scene pivot — spacing is rigid by construction
    (rotations preserve distance). Each camera's orientation = lookAt(apex -> pivot); its
    image plane is ⊥ that view axis at depth ``g``, half-width ``wh`` (frozen scalars).
    Aim varies smoothly (both endpoints are smooth trajectories) — no popping. Returns
    world-space ``apex`` + ``corners`` (TL,TR,BR,BL — the warp's source-corner order) + the
    lookAt ``R_view`` (the SPLAT renders REUSE it, so drawn pose == rendered pose) +
    ``orig``/``yaw`` tags."""
    piv = np.asarray(pivot, np.float32)
    a0 = np.asarray(apex0, np.float32)
    wh = float(wh)
    hh = wh * 9.0 / 16.0
    corners_v = np.array([[-wh, -hh, g], [wh, -hh, g],
                          [wh, hh, g], [-wh, hh, g]], np.float32)
    cams = []
    for yaw_deg in (-float(novel_yaw), 0.0, float(novel_yaw)):
        yaw = np.deg2rad(yaw_deg)
        Ry = np.array([[np.cos(yaw), 0., np.sin(yaw)],
                       [0., 1., 0.],
                       [-np.sin(yaw), 0., np.cos(yaw)]], np.float32)
        apex = (a0 - piv) @ Ry + piv                # orbit the REAL camera about the pivot
        R_v = _lookat_rows(apex, piv)
        corners = corners_v @ R_v + apex            # view -> world: p = apex + R_v^T v
        cams.append({"apex": apex.astype(np.float32), "corners": corners.astype(np.float32),
                     "R_view": R_v, "orig": yaw_deg == 0.0, "yaw": float(yaw_deg)})
    return cams


def _draw_frustum(canvas, proj, cam, img, *, color, z_scale=1.0, img_alpha=0.92):
    """Draw ONE camera frustum INTO the hero view: the wireframe rays + apex dot FIRST,
    then its rendered image perspective-WARPED onto the image plane over them, then the
    plane outline. The novel views CAN shear (the warped quad is the honest 3-D image
    plane); drawing the rays UNDER the image keeps them from crossing the picture's face.
    Returns the composited uint8 canvas; degenerate/behind-camera
    projections are skipped entirely."""
    H, W = canvas.shape[:2]
    pts = np.vstack([cam["corners"], cam["apex"][None]])
    uv, z = _project_world_pts(pts, proj, z_scale=z_scale)
    if not (np.isfinite(uv).all() and (z > 1e-2).all()):
        return canvas
    if float(np.abs(uv).max()) > 8.0 * max(H, W):   # blown-up projection -> skip, unreadable
        return canvas
    quad = uv[:4].astype(np.float32)
    qi = np.round(quad).astype(np.int32)
    ax, ay = int(round(float(uv[4, 0]))), int(round(float(uv[4, 1])))
    # rays UNDER the image ("dim" on the white-paper theme = mix toward white, so the
    # apex->corner rays recede behind the full-strength plane outline)
    dim = tuple(int(255 - 0.5 * (255 - ch)) for ch in color)
    for k in range(4):
        cv2.line(canvas, (ax, ay), (int(qi[k, 0]), int(qi[k, 1])), dim, 2, cv2.LINE_AA)
    cv2.circle(canvas, (ax, ay), 6, color, -1, cv2.LINE_AA)
    out = canvas
    if img is not None and float(cv2.contourArea(quad)) > 16.0:
        ih, iw = img.shape[:2]
        src = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], np.float32)
        M = cv2.getPerspectiveTransform(src, quad)
        warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)
        mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(mask, qi, 255, lineType=cv2.LINE_AA)
        m = (mask.astype(np.float32) * (float(img_alpha) / 255.0))[..., None]
        out = (canvas.astype(np.float32) * (1.0 - m)
               + warped.astype(np.float32) * m).clip(0, 255).astype(np.uint8)
    cv2.polylines(out, [qi], True, color, 3, cv2.LINE_AA)
    return out


_FONT_CACHE: dict = {}
_DEJAVU_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(px, bold=True):
    """Memoized DejaVu TTF font; None when PIL/the font is unavailable (cv2 fallback)."""
    key = (int(px), bool(bold))
    f = _FONT_CACHE.get(key)
    if f is None and _PIL_OK:
        try:
            f = ImageFont.truetype(_DEJAVU_BOLD if bold else _DEJAVU, int(px))
        except Exception:  # noqa: BLE001
            f = None
        _FONT_CACHE[key] = f
    return f


def _text(img, org, s, *, px=28, color=INK_BGR, bold=True, halo=None):
    """Draw REAL (TTF, anti-aliased) text onto a BGR image at ``org=(x, y_top)``.
    The figure's typography: cv2's Hershey fonts read as plotter output ("text too small
    and unreadable"); DejaVu at proper sizes reads like a figure. Only a small crop around
    the text round-trips through PIL, so the cost is ~free. ``halo`` draws a stroke outline
    (label legibility over the particle field). Falls back to Hershey without PIL."""
    x, y = int(org[0]), int(org[1])
    f = _font(px, bold)
    if f is None:
        cv2.putText(img, s, (x, y + int(px * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, px / 26.0,
                    color, 2, cv2.LINE_AA)
        return img
    sw = max(2, px // 9) if halo is not None else 0
    bb = f.getbbox(s)
    pad = sw + 4
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + (bb[2] - bb[0]) + 2 * pad + 6)
    y1 = min(img.shape[0], y + (bb[3] + 2) + 2 * pad + 6)
    if x1 <= x0 or y1 <= y0:
        return img
    crop = img[y0:y1, x0:x1]
    pil = Image.fromarray(crop[..., ::-1].copy())
    d = ImageDraw.Draw(pil)
    kw = dict(font=f, fill=(int(color[2]), int(color[1]), int(color[0])))
    if sw:
        kw.update(stroke_width=sw, stroke_fill=(int(halo[2]), int(halo[1]), int(halo[0])))
    d.text((x - x0, y - y0), s, **kw)
    crop[:] = np.asarray(pil)[..., ::-1]
    return img


def _uni_panel(img, caption, *, accent=None):
    """A SIDE panel card of the unified diagram (input / reconstruction / segmentation):
    a white caption strip (large bold ink type) over the resized image. An ACCENT border
    marks every panel imaged at the ORIGINAL camera pose (matching the accented frustum in
    the hero); others get a neutral edge."""
    blk = np.full((UNI_CAP_H + UNI_SIDE_H, UNI_SIDE_W, 3), 255, np.uint8)
    _text(blk, (6, 6), caption, px=30, color=INK_BGR)
    blk[UNI_CAP_H:] = cv2.resize(img, (UNI_SIDE_W, UNI_SIDE_H))
    if accent is not None:
        cv2.rectangle(blk, (0, UNI_CAP_H), (UNI_SIDE_W - 1, UNI_CAP_H + UNI_SIDE_H - 1), accent, 4)
    else:
        cv2.rectangle(blk, (0, UNI_CAP_H), (UNI_SIDE_W - 1, UNI_CAP_H + UNI_SIDE_H - 1), PANEL_EDGE, 2)
    return blk


def _blit_card(canvas, blk, x, y):
    """Paste a panel card with a SOFT DROP SHADOW (paper-figure depth). The shadow is a
    blurred dark rectangle composited under the card's lower-right; blur runs on a local
    crop only."""
    h, w = blk.shape[:2]
    H, W = canvas.shape[:2]
    pad = 26
    cx0 = max(0, x - pad); cy0 = max(0, y - pad)
    cx1 = min(W, x + w + pad + 14); cy1 = min(H, y + h + pad + 14)
    m = np.zeros((cy1 - cy0, cx1 - cx0), np.float32)
    cv2.rectangle(m, (x + 7 - cx0, y + 9 - cy0), (x + w + 7 - cx0, y + h + 9 - cy0), 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 8) * 0.20
    reg = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    canvas[cy0:cy1, cx0:cx1] = (reg * (1.0 - m[..., None])).astype(np.uint8)
    canvas[y:y + h, x:x + w] = blk


def _uni_arrow(canvas, x0, x1, y):
    """A clean pipeline arrow: soft-gray shaft + solid triangular head."""
    col = (150, 150, 150)
    x0, x1, y = int(x0), int(x1), int(y)
    cv2.line(canvas, (x0, y), (x1 - 18, y), col, 6, cv2.LINE_AA)
    tip = np.array([[x1, y], [x1 - 24, y - 14], [x1 - 24, y + 14]], np.int32)
    cv2.fillConvexPoly(canvas, tip, col, lineType=cv2.LINE_AA)


def render_video(vid, out_path, *, yaml_cfg, num_sweeps, max_frames=-1,
                 target_duration=0.0, out_fps=30.0, frame_stride=1):
    if _is_raw_source(vid):                      # --source: a raw video file / frames dir
        src, mask_path, is_gt = Path(vid), None, False   # no GT/SAM -> k-means frame-0 seed
        vid = Path(vid).stem                     # clean label for logs + output names
    else:
        src, mask_path, is_gt = _resolve_seed(vid)  # custom=SAM, DAVIS=GT (FORCE_SAM_SEED=1 -> all SAM)
    if src is None or not Path(src).exists():
        return {"vid": vid, "status": "no_source"}
    stride, n_keep = genmatter_rt.STRIDE, genmatter_rt.N_KEEP
    H, W = live.WORK_HW
    indices = genmatter_rt.subsample_indices(h=H, w=W, stride=stride, n_keep=n_keep, seed=0)
    intr = genmatter_rt.DEFAULT_INTRINSICS
    nb = int(yaml_cfg["tracking"]["num_blobs"]); nh = int(yaml_cfg["tracking"]["num_hyperblobs"])
    if DEMO_NUM_BLOBS not in (None, ""):          # demo override: more particles -> denser object coverage
        nb = int(DEMO_NUM_BLOBS)
    seed_grid = _seed_grid(mask_path, is_gt, H // stride, W // stride)
    if seed_grid is not None and not is_gt:
        seed_grid = _cap_seed_clusters(seed_grid, MAX_SAM_CLUSTERS)
    seed_kind = ("GT" if is_gt else "SAM") if seed_grid is not None else "kmeans"
    num_fg = _count_seed_instances(seed_grid)
    if is_gt:   # GT is the clean DEMO seed for DAVIS; the self-supervised SAM seed over-segments
        _log(f"{vid}: DAVIS GT seed (demo; num_fg={num_fg}). "
             f"(self-supervised SAM seed over-segments DAVIS; FORCE_SAM_SEED=1 to use it)")

    _trk = yaml_cfg["tracking"]
    feat_final = bool(_trk.get("feature_aware_final_assignment", genmatter_rt._FEATURE_AWARE_FINAL_DEFAULT))
    final_outlier = bool(_trk.get("final_assignment_outlier", genmatter_rt._FINAL_OUTLIER_DEFAULT))
    freeze_hb = bool(_trk.get("freeze_hyperblob_assignment", genmatter_rt._FREEZE_HYPERBLOB_ASSIGNMENT_DEFAULT))
    blob_means_updates = int(_trk.get("blob_means_updates_per_frame", genmatter_rt._BLOB_MEANS_UPDATES_DEFAULT))
    freeze_blob_features = bool(_trk.get("freeze_blob_features", genmatter_rt._FREEZE_BLOB_FEATURES_DEFAULT))
    _damp = _trk.get("feature_update_damping", None)
    feature_update_damping = float(_damp) if _damp is not None else None
    use_damp = (feature_update_damping is not None) and (feature_update_damping < 1.0)
    _stick_cfg = _trk.get("final_assignment_stickiness", None)
    _stick = (float(STICKINESS) if STICKINESS not in (None, "")
              else (float(_stick_cfg) if _stick_cfg is not None else None))
    use_stick = (_stick is not None) and (_stick > 0)
    _rd_cfg = _trk.get("mean_update_robust_delta", None)
    _rdelta = (float(ROBUST_DELTA) if ROBUST_DELTA not in (None, "")
               else (float(_rd_cfg) if _rd_cfg is not None else None))
    use_robust = (_rdelta is not None) and (_rdelta > 0)
    blob_feat_anchor = hb_feat_anchor = None
    _log(f"{vid}: src={Path(src).name} seed={seed_kind} num_blobs={nb} "
         f"freeze_blob_features={freeze_blob_features} damping={feature_update_damping} "
         f"stickiness={_stick} robust_delta={_rdelta}")

    import jax
    raw = sum(1 for _ in live.iter_frames(Path(src), max_frames))
    src_frames = (raw + frame_stride - 1) // max(frame_stride, 1)
    target_frames = int(round(target_duration * out_fps)) if target_duration > 0 else 0
    total_frames = max(src_frames, target_frames) if target_frames > 0 else src_frames
    cw = TILE_W * NCOL
    focused = (RENDER_MODE == "focused")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if focused:
        # ONE unified pipeline-diagram video per clip (white-paper theme):
        #   <vid>_unified.mp4   RGB input -> two stacked hero 3-D views (true-color particles
        #                       + camera frustums / semantic-cluster particles) -> RGB
        #                       reconstruction + object segmentation. (The separate pixels
        #                       video is retired — its tiles are the output panels here.)
        pc_path = out_path.parent / f"{vid}_unified.mp4"
        pc_tmp = str(pc_path) + ".mp4v.mp4"
        writer_pc = cv2.VideoWriter(pc_tmp, fourcc, out_fps, (UNI_W, UNI_H + STATS_H))
        writer = None
        if not writer_pc.isOpened():
            return {"vid": vid, "status": "writer_fail"}
    else:
        ch = TILE_H * 2 + STATS_H
        writer = cv2.VideoWriter(str(out_path) + ".mp4v.mp4", fourcc, out_fps, (cw, ch))
        if not writer.isOpened():
            return {"vid": vid, "status": "writer_fail"}

    key = jax.random.PRNGKey(0)
    state = seed_state = prev = depth_ema = None
    depth_state = None               # stabilize_depth carry (frame-0 reference + frozen bounds + EMA)
    pca = [None, None, None]
    pc_focal = None                  # FROZEN focal (sized at the pan extreme -> no overflow)
    _hero_R = _make_proj(HERO_YAW, HERO_PITCH, np.zeros(3, np.float32), 1.0,
                         (HERO_H, HERO_W))["R"]   # the (constant) hero rotation; row 2 = the
    #                                  view axis the PULLBACK retreats along
    obj_rad_ema = None               # EMA'd object particle radius (sizes the frustum planes)
    centroid_ema = None              # EMA'd object pivot (focused: WORLD frame; legacy: orbit center)
    ego_state = None                 # ego-motion carry: {"R","t" (camera->world), "world" (last
    #                                  world-frame blob means = the incremental Kabsch anchors), "ok"}
    fr_rig = None                    # FROZEN frustum-rig scalars {c0,d0,wh0,y0} in the CAMERA frame
    cam_pivot_ema = None             # CAMERA-frame object pivot EMA (only feeds the rig freeze)
    rgb_lut = None                   # legacy: LOCKED frame-0 per-particle AVG-RGB BGR
    lut_ema = None                   # focused: SLOW per-blob avg-RGB EMA (colors track the pan)
    w_ema = None                     # temporal EMA of posterior blob weights (cull smoothing)
    bm_ema = bc_ema = None           # EMA'd blob means/covs for RENDER ONLY (kills particle jitter)
    clu_ema = rgb_ema = None         # EMA'd 2D seg color tiles for RENDER ONLY (edge-flicker fix)
    spurious_cuts = None             # FROZEN frame-0 {extent fence, density cut, ...} for the spurious filter
    kill_ema = None                  # EMA of the spurious kill-mask (fades a rejected particle in/out, no pop)
    fps_hist = deque(maxlen=12)
    drop = wall = worst = 0.0; nwrote = 0
    # TWO-PASS follow camera (focused mode): the hero must never zoom or dolly ("the
    # camera goes in and out of the scene, which shouldn't happen") yet keep everything
    # interesting tightly framed ("no weird blank space — follow the actual scene").
    # Streaming calibration can't see the future, so PASS 0 runs the tracker over the
    # source once collecting tiny per-frame framing stats (and freezing the rig scalars),
    # _solve_follow_camera then fits the smoothed pivot TRACK (fixed depth) + ONE focal +
    # the splat-plane focals, and PASS 1 replays the (deterministic) tracker and renders
    # with that camera — a smooth pan that follows the action, constant scale, nothing
    # ever clipping. Legacy 2x3 = single pass, untouched.
    cam_fix = None                   # the solved camera {"pivot_t","focal","push","view_focals"}
    cam_stats = {"bm": [], "bc": [], "oa": [], "piv_cam": [], "ego": []}  # pass-0 framing stats
    plane_side = None                # bucket HYSTERESIS carry: sticky in-front/behind side per
    #                                  (plane, blob) — matter at window depth can't flicker

    def _pass_frames():
        if focused:
            for _i, _b, _s in _iter_frames_looped(Path(src), max_frames, 0, frame_stride):
                yield 0, _i, _b, _s
        for _i, _b, _s in _iter_frames_looped(Path(src), max_frames, target_frames, frame_stride):
            yield 1, _i, _b, _s

    for render_pass, i, bgr, is_seam in _pass_frames():
        cam_locked = focused and (render_pass == 1)
        if cam_locked and cam_fix is None:
            # ---- between passes: lock the camera, then FULL-RESET for a bit-identical
            # tracker replay (fresh PRNG key + state; same inputs -> same stream). The
            # camera artifacts (cam_fix, fr_rig) are the ONLY carry-overs.
            cam_fix = _solve_follow_camera(cam_stats, fr_rig, _hero_R)
            _piv_t = cam_fix["pivot_t"]
            _log(f"{vid}: follow camera solved over {len(cam_stats['bm'])} pass-0 frames "
                 f"(focal={cam_fix['focal']:.1f}, pivot x-range "
                 f"[{_piv_t[:, 0].min():.3f}, {_piv_t[:, 0].max():.3f}], "
                 f"z={_piv_t[0, 2]:.3f} fixed)")
            cam_stats = None
            key = jax.random.PRNGKey(0)
            state = seed_state = prev = depth_ema = None
            depth_state = None
            pca = [None, None, None]
            w_ema = None; bm_ema = bc_ema = None
            centroid_ema = None; cam_pivot_ema = None; obj_rad_ema = None
            clu_ema = rgb_ema = None; kill_ema = None
            lut_ema = None; ego_state = None
            rgb_lut = None; spurious_cuts = None
            blob_feat_anchor = hb_feat_anchor = None
            plane_side = None
            fps_hist.clear(); drop = wall = worst = 0.0
        t0 = time.monotonic()
        if is_seam:
            prev = None
            if seed_state is not None:
                state = seed_state
            w_ema = None             # reset the smoothing EMAs at the loop seam
            bm_ema = bc_ema = None
            centroid_ema = None
            cam_pivot_ema = None
            obj_rad_ema = None
            clu_ema = rgb_ema = None
            kill_ema = None
            lut_ema = None           # colors re-anchor to the (identical) seam frame
            ego_state = None         # world re-anchors to the seam frame's camera (= frame 0's)
            plane_side = None
        d_raw = live._depth_forward(bgr).astype(np.float32)
        if DEPTH_STABILIZE:
            # Robust affine-align to frame-0 + frozen bounds -> static geometry keeps a
            # constant Z (instead of the per-frame-norm + pre-norm EMA).
            depth_use, dlo, dhi, depth_state = genmatter_rt.stabilize_depth(d_raw, depth_state)
        else:                        # baseline: per-frame-norm + pre-norm EMA
            depth_ema = d_raw if depth_ema is None else 0.6 * d_raw + 0.4 * depth_ema
            depth_use, dlo, dhi = depth_ema, None, None
        t_d = time.monotonic()
        cur, hw = live._bgr_to_raft_tensor(bgr)
        flow = np.zeros((2, H, W), np.float32) if prev is None else live._flow_forward(prev, cur, hw)
        prev = cur; t_f = time.monotonic()
        feat_raw, grid_hw = live._features_forward(bgr)
        positions, velocities = genmatter_rt.unproject(
            depth_use, flow, indices, intr, stride, depth_lo=dlo, depth_hi=dhi)
        features, pca[0], pca[1], pca[2] = genmatter_rt.dino_features_to_datapoints(
            feat_raw, indices, pca[0], pca[1], pca[2], stride=stride,
            image_hw=bgr.shape[:2], target_dim=genmatter_rt.FEATURE_DIM, feat_grid_hw=grid_hw)
        t_dn = time.monotonic()
        if state is None:
            if SEED_MERGE_VEL not in (None, "") and seed_grid is not None:
                # Systemic over-seg fix: a SHORT velocity warmup (so a moving object
                # separates from a static/coherent background) then merge co-moving
                # adjacent SAM instances -> fewer hyperblobs + a coherent rigid background.
                wvel = _warmup_total_displacement(src, indices, intr, stride, H, W, SEED_MERGE_K)
                if wvel is not None:
                    seed_grid = merge_seed_instances(
                        seed_grid, wvel, indices, H // stride, W // stride,
                        vel_thr=float(SEED_MERGE_VEL))
                    num_fg = _count_seed_instances(seed_grid)
                    _log(f"{vid}: motion seed-merge -> num_fg={num_fg} "
                         f"(vel<{SEED_MERGE_VEL}, K={SEED_MERGE_K})")
            state, key = genmatter_rt.init_state(
                positions, velocities, features, key, yaml_cfg=yaml_cfg, num_blobs=nb,
                num_hyperblobs=nh, sam_segmentation=seed_grid, subsample_indices=indices)
            seed_state = state
            if use_damp:
                import jax.numpy as _jnp
                blob_feat_anchor = _jnp.asarray(state.blobs_state.blob_features)
                hb_feat_anchor = _jnp.asarray(state.hyperblobs_state.hyperblob_features)
        state, key = genmatter_rt.step_multi_sweep(
            state, positions, velocities, features, key, num_sweeps=num_sweeps,
            feature_aware_final=feat_final, final_outlier=final_outlier,
            freeze_hyperblob_assignment=freeze_hb, blob_means_updates=blob_means_updates,
            freeze_blob_features=freeze_blob_features, blob_feat_anchor=blob_feat_anchor,
            hb_feat_anchor=hb_feat_anchor, feature_update_damping=feature_update_damping,
            use_final_stickiness=use_stick,
            final_assignment_stickiness=(_stick if use_stick else None),
            use_robust_mean=use_robust,
            mean_update_robust_delta=(_rdelta if use_robust else None))
        state.datapoints_state.blob_assignments.block_until_ready()
        blob_a, hyperblob_a = genmatter_rt.extract_assignments(state)
        outlier_frac = float(np.mean(blob_a < 0))
        t_g = time.monotonic()

        if rgb_lut is None and not focused:
            # LEGACY 2x3: per-blob AVG color (true scene color), LOCKED at frame 0 (stable
            # identity across frames; kept bit-identical). num_blobs=nb so the LUT covers EVERY
            # blob — a palette-sized LUT clips high blob ids, skipping their avg-color marbles
            # and smearing one color over every clipped-id pixel in the 2-D reconstruction.
            rgb_lut = genmatter_viz.compute_blob_color_lut(
                blob_a, indices, bgr, h=H, w=W, stride=stride, num_blobs=nb)
        if focused:
            # FOCUSED: slow per-blob COLOR EMA. On a hard pan blobs RE-BIND to new scene
            # content but a frame-0-LOCKED LUT keeps painting their OLD colors, smearing stale
            # color across the reconstruction. Recompute the per-blob avg color each frame and
            # EMA it; only blobs that OWN datapoints THIS frame update (an occluded/empty blob
            # keeps its last color instead of snapping to compute_blob_color_lut's global-mean
            # fallback). Slow rate = colors track what each blob currently covers, identity
            # stays stable (no flicker). Feeds the 2-D reconstruction AND the marble/plane tints.
            lut_cur = genmatter_viz.compute_blob_color_lut(
                blob_a, indices, bgr, h=H, w=W, stride=stride, num_blobs=nb).astype(np.float32)
            owned = np.bincount(np.clip(blob_a[blob_a >= 0], 0, nb - 1), minlength=nb) > 0
            if lut_ema is None:
                lut_ema = lut_cur
            else:
                lut_ema[owned] = LUT_EMA * lut_cur[owned] + (1.0 - LUT_EMA) * lut_ema[owned]
            rgb_lut = lut_ema

        # Smoothed weight culling from the probabilistic trace -> per-particle alpha.
        # _blob_weights_mean_jit is the DETERMINISTIC Dirichlet posterior mean (no
        # PRNG), so reading it does NOT advance the tracker key stream — the tracking
        # stays bit-identical to the validated run. Temporal EMA + smoothstep means
        # weak particles FADE, never pop (the anti-flicker behavior).
        w = np.asarray(genmatter_rt._blob_weights_mean_jit(state))
        w_ema = w if w_ema is None else WEMA_ALPHA * w + (1.0 - WEMA_ALPHA) * w_ema
        blob_alpha = genmatter_viz.blob_alpha_from_weights(
            w_ema, total_points=float(genmatter_rt.N_KEEP),
            min_points=CULL_MIN_PTS, full_points=CULL_FULL_PTS)
        bm, bc = genmatter_rt.extract_blob_means_and_covs(state)   # REAL means/covs (no re-fit)
        tint = np.asarray(rgb_lut)[:bm.shape[0]].astype(np.float32)   # 3-D particles: avg color (true scene color)
        # JITTER FIX: per-blob EMA of the particle GEOMETRY (means + covs), RENDER ONLY —
        # `state` is never written back, so tracking stays bit-identical. Blob index i is
        # the same particle every frame, so this index-keyed EMA is valid; a convex combo
        # of PSD covs stays PSD. Particles are drawn from these smoothed copies, not `bm`/`bc`,
        # so they stop wobbling frame-to-frame. (Reset on seam.)
        if bm_ema is None:
            bm_ema, bc_ema = bm.copy(), bc.copy()
        else:
            bm_ema = MARBLE_GEOM_BETA * bm + (1.0 - MARBLE_GEOM_BETA) * bm_ema
            bc_ema = MARBLE_GEOM_BETA * bc + (1.0 - MARBLE_GEOM_BETA) * bc_ema

        # SYSTEMATIC SPURIOUS-PARTICLE FILTER (fg/bg-AGNOSTIC). The robust discriminator is
        # DENSITY = effective datapoint count / 3-D covariance extent. A real object particle
        # is COMPACT for the matter it owns (high density); a spurious blob is BIG yet owns
        # FEW datapoints (low density) — e.g. the big diffuse gray blob smeared over the wine
        # background, even when it gets mis-assigned to a FOREGROUND cluster (the old is_bg-only
        # gate missed exactly that). Robust thresholds are FROZEN at frame 0 (so the cull SET
        # doesn't itself flicker) and applied each frame to the EMA'd (smooth) extent/count, so a
        # blob that drifts big+diffuse is faded out gradually. Render-only; tracking untouched.
        hb_per_blob = np.asarray(state.blobs_state.hyperblob_assignments)[:bm.shape[0]]
        extent = np.linalg.eigvalsh(np.asarray(bc_ema))[:, -1]        # (L,) max eigenvalue of 3D cov
        count = np.asarray(w_ema)[:bm.shape[0]] * float(genmatter_rt.N_KEEP)   # effective datapoints/blob
        density = count / (extent + 1e-6)                            # compact (real) vs diffuse (spurious)
        is_bg = hb_per_blob >= max(int(num_fg), 0)
        if spurious_cuts is None:          # FREEZE robust cuts once at frame 0
            alive = count > CULL_MIN_PTS                             # only characterise DRAWABLE particles
            ext_a = extent[alive] if alive.any() else extent
            den_a = density[alive] if alive.any() else density
            q25, q75 = np.percentile(ext_a, [25, 75])
            e_fence = (SPURIOUS_EXTENT_FENCE if SPURIOUS_EXTENT_FENCE > 0.0
                       else float(q75 + 1.5 * (q75 - q25)))          # boxplot upper-fence size outlier
            e_mid = float(np.percentile(ext_a, SPURIOUS_EXTENT_PCT)) # "big" gate (paired with density)
            d_lo = float(np.percentile(den_a, SPURIOUS_DENSITY_PCT)) # "diffuse" gate
            bg_ext = extent[is_bg]
            bg_cut = (SPURIOUS_EXTENT if SPURIOUS_EXTENT > 0.0
                      else (float(np.percentile(bg_ext, 88)) if bg_ext.size > 4 else np.inf))
            spurious_cuts = {"e_fence": e_fence, "e_mid": e_mid, "d_lo": d_lo, "bg_cut": bg_cut}
            if SPURIOUS_DEBUG:
                pe = np.percentile(ext_a, [50, 90, 99]); pd = np.percentile(den_a, [1, 10, 50])
                _log(f"{vid}: spurious cuts e_fence={e_fence:.4g} e_mid={e_mid:.4g} d_lo={d_lo:.4g} "
                     f"bg_cut={bg_cut:.4g} | extent p50/p90/p99={pe[0]:.3g}/{pe[1]:.3g}/{pe[2]:.3g} "
                     f"| density p1/p10/p50={pd[0]:.3g}/{pd[1]:.3g}/{pd[2]:.3g} (alive={int(alive.sum())})")
        c = spurious_cuts
        spurious = ((extent > c["e_fence"])                          # clear SIZE outlier (any fg/bg)
                    | ((extent > c["e_mid"]) & (density < c["d_lo"])) # BIG and DIFFUSE
                    | (is_bg & (extent > c["bg_cut"])))               # background backstop
        if SPURIOUS_HARD_CAP > 0.0:
            spurious = spurious | (extent > SPURIOUS_HARD_CAP)
        # Fade the kill in/out (temporal EMA of the spurious MASK) so a blob that crosses the
        # frozen threshold ramps to invisible over a few frames instead of POPPING — the last
        # anti-flicker guard. Render-only; reset on seam.
        kill = spurious.astype(np.float32)
        kill_ema = kill if kill_ema is None else 0.3 * kill + 0.7 * kill_ema
        blob_alpha = blob_alpha * (1.0 - kill_ema)
        if SPURIOUS_DEBUG and i == 0:
            _log(f"{vid}: frame0 spurious={int(spurious.sum())}/{int(spurious.size)} "
                 f"[fence={int((extent > c['e_fence']).sum())} "
                 f"diffuse={int(((extent > c['e_mid']) & (density < c['d_lo'])).sum())} "
                 f"bg={int((is_bg & (extent > c['bg_cut'])).sum())}]")

        if (not focused) or render_pass == 1:
            # 2-D segmentation tiles (top row): particles (AVG-RGB) + clusters (objects), both
            # from the SAME upsampled label grids. The avg-RGB particle seg posterizes each
            # pixel to its blob's true color (shows the Gaussian tessellation in real color).
            # Outliers -> DARK RED (local override so the live demo's OUTLIER_BGR is
            # untouched). Skipped in the focused PASS 0 (framing-stats only, nothing drawn).
            # num_blobs=nb so the label grid clips blob ids into the palette range correctly:
            # a default palette-sized clamp relabels every high blob id to the top palette
            # index, painting many datapoints with ONE blob's color (same clip the LUT needs).
            bf, hf, of = genmatter_viz.render_matter_label_grids(
                blob_a, hyperblob_a, indices, h=H, w=W, stride=stride,
                rgb_guide=(bgr if SEG_EDGE_AWARE else None), num_blobs=nb)
            px_cluster = _color_cluster_tile(hf, of, num_fg)
            rgb_dense = np.asarray(rgb_lut)[np.clip(bf, 0, rgb_lut.shape[0] - 1)].astype(np.uint8)
            if of.any():
                px_cluster[of] = DARK_RED_BGR
                rgb_dense[of] = DARK_RED_BGR
            # EDGE-FLICKER FIX: render-only temporal EMA of the 2-D seg COLOR tiles. The label
            # grids are recomputed fresh each frame (Gibbs assignments + NN-fill + edge-snap all
            # churn at object boundaries); blending toward the previous frame makes a boundary
            # pixel EASE between labels instead of snapping. The EMA'd tiles are what we display
            # AND what the 3-D cloud samples, so both calm down. (Reset on seam.)
            clu_ema = px_cluster.astype(np.float32) if clu_ema is None else \
                SEG_EMA * px_cluster + (1.0 - SEG_EMA) * clu_ema
            rgb_ema = rgb_dense.astype(np.float32) if rgb_ema is None else \
                SEG_EMA * rgb_dense + (1.0 - SEG_EMA) * rgb_ema
            px_cluster = clu_ema.clip(0, 255).astype(np.uint8)
            rgb_dense = rgb_ema.clip(0, 255).astype(np.uint8)

        # Per-blob CLUSTER color (vivid fg / muted bg) — the SAME palette _color_cluster_tile
        # paints the 2-D cluster tile with, so the cluster particles match the object segmentation.
        # (Shared by both render modes; independent of the camera projection.)
        nhyp = genmatter_viz.HYPERBLOB_PALETTE.shape[0]
        cluster_pal = (genmatter_viz._cluster_palette_for(num_fg, nhyp)
                       if num_fg and num_fg > 0 else genmatter_viz.HYPERBLOB_PALETTE)
        cluster_tint = cluster_pal[np.clip(hb_per_blob, 0, nhyp - 1)].astype(np.float32)
        if focused:
            # FOCUSED particles video = a PIPELINE DIAGRAM, not a tile grid: ONE BIG elevated
            # 3/4 "hero" view of the avg-RGB particle marbles, WORLD-STABILIZED (ego-motion
            # from the background blobs -> the background holds STILL, the camera rig MOVES)
            # and OBJECT-CENTRIC (the meta-camera pivots on the object; bg faded AND
            # desaturated so figure-ground reads), with THREE camera frustums at REAL
            # positions drawn INTO the 3-D scene, DEPTH-INTERLEAVED with the marbles
            # (painter's ordering — matter in front of a camera occludes it, matter behind
            # it is hidden; neither side can wrongly punch through the other), each showing
            # its RENDERED RGB image on its image plane: the ORIGINAL/input camera
            # (accented — the pose panels 1/3a/3b correspond to) carries the per-pixel RGB
            # reconstruction (= panel 3) and the two NOVEL poses at +/-FRUSTUM_YAW carry the
            # Gaussian state rendered DENSELY as an RGB image from exactly those cameras.
            # The frustums double as the spatial reference the old naked blob-cloud lacked.
            #
            # OBJECT vs BACKGROUND: a cluster is BACKGROUND if it's seed/GT background
            # (hyperblob >= num_fg) OR it's the DOMINANT plane (> BG_SIZE_FRAC of the particles) — the
            # size test catches the couch/sky/ground a SAM seed mislabels as a foreground instance. The
            # OBJECT = the visible rest. (If too few object particles, fall back to the whole scene.)
            bg_blob = hb_per_blob >= max(int(num_fg), 0)
            if not is_gt:
                # SIZE test only for SAM seeds: a SAM seed mislabels the big background plane (couch /
                # sky / ground) as a foreground instance, so the dominant cluster is faded too. A GT
                # seed already labels the object cleanly via num_fg — applying the size test there
                # wrongly fades a LARGE GT object (e.g. the breakdancer, > BG_SIZE_FRAC of the
                # particles) and breaks the object-centric framing.
                tot_ct = float(np.asarray(count).sum()) + 1e-6
                for _h in np.unique(hb_per_blob):
                    if float(np.asarray(count)[hb_per_blob == _h].sum()) / tot_ct > BG_SIZE_FRAC:
                        bg_blob = bg_blob | (hb_per_blob == _h)
            vis = np.asarray(blob_alpha, np.float32) > 0.02
            obj_blob = (~bg_blob) & vis
            if int(obj_blob.sum()) < 3:                  # degenerate -> frame the whole scene
                obj_blob = vis
            fg_alpha = np.asarray(blob_alpha, np.float32).copy()
            if BG_FADE < 1.0:
                fg_alpha[bg_blob] *= BG_FADE
            # FIGURE-GROUND tint (white-paper theme): mix background FILLS toward NEAR-WHITE
            # (render-only copy) so they recede into the paper while the object keeps its true
            # color. On a white canvas "receding" means LIGHTER (the old dark-gray mix receded
            # on black); pure desaturation alone can't separate a white object from a beige ground.
            tint_hero = tint
            if BG_DESAT > 0.0 and bg_blob.any():
                tint_hero = tint.copy()
                tint_hero[bg_blob] = ((1.0 - BG_DESAT) * tint_hero[bg_blob]
                                      + BG_DESAT * BG_GHOST_BGR)
            # SEMANTIC GLOW alphas: every particle glows faintly in its cluster color (fill =
            # appearance, glow = membership — the multiplex); background glows are doubly faint
            # (muted palette x BG_FADE) so they whisper, not shout.
            glow_alpha = GLOW_ALPHA * np.asarray(blob_alpha, np.float32)
            glow_alpha[bg_blob] *= BG_FADE
            # ---- WORLD STABILIZATION: ego-motion from the background blobs ----
            # The clustering hands us the background (bg_blob); background matter tends to
            # be STATIC in the world, so the rigid motion those blobs share in the camera
            # frame IS the camera's ego-motion. World frame := frame-0 camera frame. Each
            # frame solve (R_t, t_t) against the blobs' PREVIOUS world positions
            # (incremental anchors: a blob that re-binds to new content during a pan is
            # trimmed for a frame, then re-anchored — robust where frozen frame-0 anchors
            # break on long pans), then render EVERYTHING in world coordinates: the
            # background cloud holds still, the object moves through it, the camera rig
            # travels. Render-only (EMA'd copies in, nothing written back to the tracker).
            if not cam_locked:
                # PASS 0 (and legacy-off): the LIVE incremental-Kabsch solve. Its raw pose
                # stream is too noisy to RENDER with (rotation wobble rocks the view, t_z
                # noise breathes it in/out — "vertigo"); pass 0 records it, the solver
                # SMOOTHS it, and pass 1 renders with the smoothed track below.
                vis_now = np.asarray(blob_alpha, np.float32) > 0.05
                if ego_state is None:
                    ego_state = {"R": np.eye(3, dtype=np.float32), "t": np.zeros(3, np.float32),
                                 "world": np.asarray(bm_ema, np.float32).copy(), "ok": vis_now.copy()}
                else:
                    if EGO_STABILIZE:
                        use = bg_blob & vis_now & ego_state["ok"]
                        ego_state["R"], ego_state["t"] = _estimate_ego_rigid(
                            np.asarray(bm_ema, np.float32)[use], ego_state["world"][use],
                            np.asarray(blob_alpha, np.float32)[use],
                            prev=(ego_state["R"], ego_state["t"]))
                    ego_state["world"] = (np.asarray(bm_ema, np.float32) @ ego_state["R"].T
                                          + ego_state["t"])
                    ego_state["ok"] = vis_now.copy()
                R_ego, t_ego = ego_state["R"], ego_state["t"]
                bm_w = ego_state["world"]                   # blob means, WORLD frame
            else:
                # PASS 1: the SMOOTHED ego pose from the solver — world jitter is visually
                # identical to viewer-camera jitter, so the render NEVER uses raw poses.
                _ne = len(cam_fix["ego"])
                if _ne:
                    R_ego, t_ego = cam_fix["ego"][min(i % _ne, _ne - 1)]
                else:
                    R_ego, t_ego = np.eye(3, dtype=np.float32), np.zeros(3, np.float32)
                bm_w = np.asarray(bm_ema, np.float32) @ R_ego.T + t_ego
            bc_w = np.einsum("ab,nbc,dc->nad", R_ego, np.asarray(bc_ema, np.float32), R_ego)

            # object-only alpha (frames the meta-camera on the object) + the object's EMA'd
            # pivot in BOTH frames: WORLD (the hero pivot — slow follow of the real object
            # motion; the static background no longer swims with it) and CAMERA (feeds the
            # frustum-rig freeze only). Reset on seam.
            obj_alpha = np.where(obj_blob, np.asarray(blob_alpha, np.float32), 0.0)
            if obj_blob.any():
                wts = np.asarray(blob_alpha, np.float32)[obj_blob][:, None]
                obj_c_w = (bm_w[obj_blob] * wts).sum(0) / (float(wts.sum()) + 1e-6)
                centroid_ema = (obj_c_w if centroid_ema is None
                                else OBJ_CENTROID_ALPHA * obj_c_w + (1.0 - OBJ_CENTROID_ALPHA) * centroid_ema)
                obj_c_cam = (np.asarray(bm_ema, np.float32)[obj_blob] * wts).sum(0) / (float(wts.sum()) + 1e-6)
                cam_pivot_ema = (obj_c_cam if cam_pivot_ema is None
                                 else OBJ_CENTROID_ALPHA * obj_c_cam + (1.0 - OBJ_CENTROID_ALPHA) * cam_pivot_ema)
            obj_centroid = centroid_ema
            # object RADIUS (sizes the frustum image planes), EMA'd so the diagram doesn't
            # breathe; rotation-invariant, so the world-frame radius serves the rig too.
            if obj_blob.any() and obj_centroid is not None:
                rr = np.linalg.norm(bm_w[obj_blob] - obj_centroid, axis=1)
                r_now = float(np.percentile(rr, 90))
                obj_rad_ema = r_now if obj_rad_ema is None else 0.15 * r_now + 0.85 * obj_rad_ema

            # RIG SCALARS frozen at calib end (d0 = |c0| = the true camera-object distance
            # at calibration; g0 = plane depth along each camera's view axis; wh0 floored
            # so tiny objects still get a readable plane, CAPPED by the camera spacing so
            # a LARGE object can't size planes that fuse into one folding screen). The
            # AIM is no longer frozen — every camera looks at the followed scene pivot
            # (built per frame in pass 1 via the lookAt _frustum_cams).
            rig = fr_rig
            if rig is None and cam_pivot_ema is not None and obj_rad_ema:
                _d0 = float(np.linalg.norm(cam_pivot_ema))
                _g0 = 0.30 * _d0
                # plane half-width CAPPED against the PLANE-CENTER SPACING of the aimed
                # arc (2·sin(yaw/2)·(d0−g0)) so adjacent planes can never touch/fuse —
                # "the frustums collide with each other" at a fixed-fraction cap once the
                # yaw tightened; floored so tiny objects still get a readable plane.
                _sp = 2.0 * float(np.sin(np.deg2rad(FRUSTUM_YAW) / 2.0)) * (_d0 - _g0)
                rig = {"c0": np.asarray(cam_pivot_ema, np.float32).copy(), "d0": _d0,
                       "g0": _g0,
                       "wh0": min(max(FRUSTUM_IMG_FRAC * float(obj_rad_ema), 0.05 * _d0),
                                  0.40 * _sp)}
                if i >= FOCAL_CALIB_FRAMES - 1:
                    fr_rig = rig
            if render_pass == 0:
                # ---- PASS 0: collect the tiny whole-clip framing stats, render nothing.
                # CAMERA-frame geometry (bm/bc/piv_cam) + the LIVE ego pose: the solver
                # smooths the poses and re-maps everything to world consistently under
                # them — _solve_follow_camera rebuilds the aimed rig per frame from these.
                cam_stats["bm"].append(np.asarray(bm_ema, np.float32).copy())
                cam_stats["bc"].append(np.asarray(bc_ema, np.float32).copy())
                cam_stats["oa"].append(np.asarray(obj_alpha, np.float32).copy())
                cam_stats["piv_cam"].append(None if cam_pivot_ema is None
                                            else np.asarray(cam_pivot_ema, np.float32).copy())
                cam_stats["ego"].append((R_ego.copy(), t_ego.copy()))
                continue
            # ---- PASS 1: THE FOLLOW CAMERA — pivot_t = the offline-smoothed interest
            # track (object + camera rig; "keep everything interesting in the frame, no
            # weird blank space, follow the actual scene") with FIXED depth, ONE whole-
            # clip focal: the camera PANS smoothly with the action and never zooms or
            # dollies ("the camera must not go in and out"). push_vec = constant pure
            # view-depth retreat so the real camera apexes fit in frame.
            _npv = cam_fix["pivot_t"].shape[0]
            _ti = min(i % max(_npv, 1), _npv - 1)
            pivot_w = cam_fix["pivot_t"][_ti]          # hero FRAMING center (object+rig blend)
            aim_w = cam_fix["aim_t"][_ti]              # the SCENE point every camera looks at
            cams = (_frustum_cams(aim_w, t_ego.astype(np.float32), rig["g0"], rig["wh0"],
                                  novel_yaw=FRUSTUM_YAW) if rig is not None else [])
            push_vec = cam_fix["push"]
            bm_h = bm_w + push_vec[None, :]
            cams_h = [{**c, "apex": c["apex"] + push_vec, "corners": c["corners"] + push_vec}
                      for c in cams]
            proj_h = _make_proj(HERO_YAW, HERO_PITCH, pivot_w, cam_fix["focal"],
                                (HERO_H, HERO_W))
            _sprite = genmatter_viz.make_lit_sphere_sprite(160)  # hi-res: crisp marble shading

            # The hero particle renderer (fill+glow multiplex, depth FOG, figure-ground).
            # ``sel`` masks a DEPTH BUCKET (only those marbles draw — masked alpha also
            # suppresses their glows) and ``geom`` shares one precomputed (means, view-z)
            # across the bucket passes, so FOG normalization stays whole-cloud and colors
            # can't shift between buckets. FOG = atmospheric perspective from the view's
            # OWN depths: fills mix toward the paper tone and glows fade with normalized
            # (10th-90th pct) view depth, so the cloud reads as a 3-D scene from particles
            # alone (no point-cloud splats anywhere).
            def _render_particle_view(proj_v, out_hw_v, base=None, sel=None, geom=None):
                if geom is not None:
                    bm_v, zv = geom
                else:
                    bm_v = (_smoosh_depth(bm_h, proj_v["centroid"], HERO_DEPTH_SCALE)
                            if proj_v is not None else bm_h)
                    zv = (_project_world_pts(bm_v, proj_v)[1]
                          if (proj_v is not None and FOG > 0.0) else None)
                tint_v, glow_v = tint_hero, glow_alpha
                if zv is not None and FOG > 0.0:
                    z10, z90 = np.percentile(zv, [10, 90])
                    dn = np.clip((zv - z10) / max(z90 - z10, 1e-6), 0.0, 1.0).astype(np.float32)
                    tint_v = tint_hero * (1.0 - FOG * dn[:, None]) + 255.0 * FOG * dn[:, None]
                    # fog fades FILLS + BACKGROUND glows only — the OBJECT's highlight is
                    # the subject marker and stays full-strength at any depth ("the
                    # highlights need to be stronger")
                    glow_v = np.where(bg_blob, glow_alpha * (1.0 - FOG * dn), glow_alpha)
                a_v = fg_alpha if sel is None else np.where(sel, fg_alpha, 0.0)
                return genmatter_viz.render_particle_marbles_tile(
                    bm_v, bc_w, tint_v, proj_v, out_hw=out_hw_v,
                    alpha_per=a_v, sigma_scale=MARBLE_SIGMA, bg_bgr=PAPER_BGR,
                    base_tile=base, base_dim=1.0, sprite=_sprite,
                    glow_colors=(cluster_tint if GLOW_ALPHA > 0.0 else None),
                    glow_alpha_per=(glow_v if GLOW_ALPHA > 0.0 else None),
                    glow_scale=GLOW_SCALE)

            # FRUSTUM PLANES = the RENDERED RGB representation, as PIXELS. The input camera
            # (yaw 0) carries the model's per-pixel RGB reconstruction (identical content to
            # panel 3a — the azure tie); the two NOVEL poses carry the Gaussian state
            # VOLUMETRICALLY SPLATTED to a dense RGB image from exactly those cameras
            # (_render_gaussian_rgb_view: per-pixel front-to-back compositing of the real
            # blob means/covs/colors — a soft reconstruction from a new angle, not a marble
            # scatter). Poses = the SAME lookAt rotations as the drawn frustums (drawn pose
            # == rendered pose), pivoting on the followed scene pivot — whose DEPTH is
            # fixed, so the view-depth offset can't drift; per-pose focals were object-fit
            # in the solver (margin 0.75 — a scene-wide fit collapses on near-camera
            # background blobs), median over frames, frozen.
            if cams_h and proj_h is not None:
                blob_alpha_f = np.asarray(blob_alpha, np.float32)
                cam_imgs = []
                for ck, yk in enumerate((-FRUSTUM_YAW, 0.0, FRUSTUM_YAW)):
                    if yk == 0.0:
                        cam_imgs.append(rgb_dense)
                        continue
                    if yk in cam_fix["view_focals"]:
                        proj_v = {"centroid": aim_w, "R": cams[ck]["R_view"],
                                  "focal": float(cam_fix["view_focals"][yk]),
                                  "out_cx": FRUSTUM_IMG_HW[1] * 0.5,
                                  "out_cy": FRUSTUM_IMG_HW[0] * 0.5}
                        cam_imgs.append(_render_gaussian_rgb_view(
                            np.asarray(bm_w, np.float32), np.asarray(bc_w, np.float32),
                            tint, blob_alpha_f, proj_v, FRUSTUM_IMG_HW))
                    else:               # degenerate clip (no object) -> blank paper plane
                        cam_imgs.append(np.full((*FRUSTUM_IMG_HW, 3), 255, np.uint8))
                # DEPTH-CORRECT COMPOSITING: the planes participate in the painter's
                # far->near ordering WITH the marbles (drawing the frustums entirely under
                # or entirely over the cloud lets a particle on the wrong side of a plane
                # paint through it). Marbles are bucketed by view depth around each plane's
                # depth and the passes are chained through base_tile: farther matter first,
                # then the plane over it, then nearer matter over the plane — particles in
                # front of a camera correctly occlude it, particles behind it are correctly
                # hidden. (Plane depth = mean corner view-z; the thin wireframe inherits the
                # plane's depth — an accepted approximation.)
                bm_v = _smoosh_depth(bm_h, proj_h["centroid"], HERO_DEPTH_SCALE)
                _uvm, zm = _project_world_pts(bm_v, proj_h)
                geom_h = (bm_v, zm)
                plane_z = np.array([np.nanmean(_project_world_pts(
                    c["corners"], proj_h, z_scale=HERO_DEPTH_SCALE)[1]) for c in cams_h])
                # bucket HYSTERESIS: sticky in-front/behind side per (plane, blob) with a
                # deadband — matter sitting AT a window's depth would otherwise flicker
                # over/under it frame to frame. Real crossings still cross.
                side_now = zm[None, :] > plane_z[:, None]
                if plane_side is None or plane_side.shape != side_now.shape:
                    plane_side = side_now
                else:
                    band = np.abs(zm[None, :] - plane_z[:, None]) <= 0.02 * float(rig["d0"])
                    plane_side = np.where(band, plane_side, side_now)
                tile = np.full((HERO_H, HERO_W, 3), 255, np.uint8)
                drawn = np.zeros(zm.shape[0], dtype=bool)
                for k in np.argsort(-plane_z):                  # farthest plane first
                    sel = (~drawn) & plane_side[int(k)]
                    tile = _render_particle_view(proj_h, (HERO_H, HERO_W), base=tile,
                                                 sel=sel, geom=geom_h)
                    drawn |= sel
                    cam = cams_h[int(k)]
                    tile = _draw_frustum(
                        tile, proj_h, cam, cam_imgs[int(k)],
                        color=(ACCENT_BGR if cam["orig"] else FRUSTUM_GRAY),
                        z_scale=HERO_DEPTH_SCALE)
                hero_tile = _render_particle_view(proj_h, (HERO_H, HERO_W), base=tile,
                                                  sel=~drawn, geom=geom_h)
                # labels LAST — diagram annotations stay readable over everything (they
                # were buried inside the cloud when drawn with the planes)
                uvA, zA = _project_world_pts(np.stack([c["apex"] for c in cams_h]),
                                             proj_h, z_scale=HERO_DEPTH_SCALE)
                for k, cam in enumerate(cams_h):
                    if not (np.isfinite(uvA[k]).all() and zA[k] > 1e-2):
                        continue
                    col = ACCENT_BGR if cam["orig"] else FRUSTUM_GRAY
                    lab = ("input camera" if cam["orig"]
                           else f"novel view {cam['yaw']:+.0f} deg")
                    ax, ay = int(round(float(uvA[k, 0]))), int(round(float(uvA[k, 1])))
                    lx = int(np.clip(ax - 60, 6, HERO_W - 260))
                    ly = int(np.clip(ay + 14, 6, HERO_H - 40))
                    _text(hero_tile, (lx, ly), lab, px=26, color=col, halo=(255, 255, 255))
            else:
                hero_tile = _render_particle_view(proj_h, (HERO_H, HERO_W))
        else:
            # LEGACY BOTTOM ROW = ONE 3-D view whose camera PANS (yaw_i). The three bottom cells
            # share ONE projection this frame: col1/col2 are the two PARTICLE lenses (avg color +
            # cluster-highlight) drawn from the SAME geometry/alpha/proj so they differ ONLY in tint,
            # and col3 is the cluster-colored point cloud (the SAME object colors as the 2-D cluster
            # tile, splatted = "3-D scene clusters as the pixels"). Jitter fix: the orbit center is
            # EMA'd (prev_centroid/centroid_alpha) so the cloud doesn't swim, and the focal is frozen
            # at the pan EXTREME (computed once on frame 0) so nothing overflows mid-sweep.
            # ONE slow monotonic pan -amp -> +amp across the whole clip (shallow; no double-back).
            frac = i / max(total_frames - 1, 1)
            yaw_i = -PC_PAN_AMP + 2.0 * PC_PAN_AMP * float(frac)   # monotonic wide yaw (no double-back)
            pitch_i = PC_PITCH_AMP * float(np.sin(np.pi * frac))   # ONE smooth pitch arc 0->amp->0 (vertical parallax)
            if pc_focal is None:                 # frame 0: size the framing for the WIDEST yaw AND pitch extreme
                *_unused, f_ext, _pj = genmatter_viz._build_pointcloud_projection(
                    depth_use, intr, yaw_deg=PC_PAN_AMP, pitch_deg=PC_PITCH_AMP, point_subsample=1,
                    out_hw=(TILE_H, TILE_W), focal_length=None, depth_lo=dlo, depth_hi=dhi)
                pc_focal = f_ext if f_ext > 0.0 else None
            (pc_cluster3d,), _f, proj = genmatter_viz.render_pointcloud_tiles_multi(
                depth_use, (px_cluster,), intr, yaw_deg=yaw_i, pitch_deg=pitch_i,
                point_subsample=1, point_size=2, out_hw=(TILE_H, TILE_W),
                focal_length=pc_focal, prev_centroid=centroid_ema,
                centroid_alpha=PC_CENTROID_ALPHA, depth_lo=dlo, depth_hi=dhi)
            if proj is not None:
                centroid_ema = proj["centroid"]
            # Two particle lenses: SAME geometry/alpha/proj, differ ONLY in tint (avg color vs cluster).
            particles_color = genmatter_viz.render_particle_marbles_tile(
                bm_ema, bc_ema, tint, proj, out_hw=(TILE_H, TILE_W),
                alpha_per=blob_alpha, sigma_scale=MARBLE_SIGMA)
            particles_cluster = genmatter_viz.render_particle_marbles_tile(
                bm_ema, bc_ema, cluster_tint, proj, out_hw=(TILE_H, TILE_W),
                alpha_per=blob_alpha, sigma_scale=MARBLE_SIGMA)
        t_r = time.monotonic()

        dt = t_g - t0
        if i > 0:
            fps_hist.append(1.0 / max(dt, 1e-6)); wall += dt
            drop += max(0.0, dt * SRC_FPS - 1.0); worst = max(worst, dt * 1000.0)
        fps = float(np.mean(fps_hist)) if fps_hist else SRC_FPS
        st = {"fps": fps, "play_fps": out_fps, "frame": i, "total": total_frames - 1,
              "dropped": int(round(drop)), "lag_ms": max(0.0, wall - nwrote / SRC_FPS) * 1000.0,
              "worst_ms": worst, "depth_ms": (t_d - t0) * 1000, "flow_ms": (t_f - t_d) * 1000,
              "dino_ms": (t_dn - t_f) * 1000, "gibbs_ms": (t_g - t_dn) * 1000,
              "render_ms": (t_r - t_g) * 1000, "seed": seed_kind, "outlier_frac": outlier_frac,
              "n_clusters": int(np.unique(hyperblob_a[hyperblob_a >= 0]).size)}

        if focused:
            # THE UNIFIED FIGURE (white paper): input -> the fill+glow particle state (+ the
            # camera frustums) -> its two attribute renders: RGB reconstruction (appearance)
            # and the segmentation OVERLAY (semantics — the same alpha-wash language as the
            # hero's glow, same palette). Azure border = imaged at the original camera pose
            # (input / recon / seg), matching the accented frustum.
            _text(hero_tile, (20, 14), "2. 3D Gaussian particles", px=38, color=INK_BGR)
            _text(hero_tile, (20, 62), "fill = avg color   |   glow = object", px=26,
                  color=(110, 110, 110))
            cv2.rectangle(hero_tile, (0, 0), (HERO_W - 1, HERO_H - 1), PANEL_EDGE, 2)
            uni = np.full((UNI_H, UNI_W, 3), CANVAS_BGR[0], np.uint8)
            x_mid = UNI_MARGIN + UNI_SIDE_W + UNI_GUT
            y_h = (UNI_H - HERO_H) // 2
            _blit_card(uni, hero_tile, x_mid, y_h)
            blk_in = _uni_panel(bgr, "1. RGB input", accent=ACCENT_BGR)
            y_in = (UNI_H - blk_in.shape[0]) // 2
            _blit_card(uni, blk_in, UNI_MARGIN, y_in)
            # the two OUTPUT renders of the state (right column, stacked): the per-pixel RGB
            # reconstruction and the INFERRED object segmentation (the flat cluster tile —
            # not an overlay on the input). Labeled 3a/3b — neither pane is privileged
            # (both are co-equal renders of the same state).
            x_r = x_mid + HERO_W + UNI_GUT
            blk_rc = _uni_panel(rgb_dense, "3a. RGB reconstruction", accent=ACCENT_BGR)
            y_rc = (UNI_H // 2 - blk_rc.shape[0]) // 2
            _blit_card(uni, blk_rc, x_r, y_rc)
            blk_sg = _uni_panel(px_cluster, "3b. object segmentation", accent=ACCENT_BGR)
            y_sg = UNI_H // 2 + (UNI_H // 2 - blk_sg.shape[0]) // 2
            _blit_card(uni, blk_sg, x_r, y_sg)
            _uni_arrow(uni, UNI_MARGIN + UNI_SIDE_W + 10, x_mid - 12, UNI_H // 2)
            _uni_arrow(uni, x_mid + HERO_W + 10, x_r - 12, y_rc + blk_rc.shape[0] // 2)
            _uni_arrow(uni, x_mid + HERO_W + 10, x_r - 12, y_sg + blk_sg.shape[0] // 2)
            # INFERENCE-STATS ROW under the whole diagram — the EXACT legacy analytics
            # panel (big INFERENCE FPS + real-time / per-stage latency / tracker columns),
            # "just like in the older versions of the visualizer".
            uni = np.vstack([uni, _stats_row(UNI_W, st)])
            writer_pc.write(uni); nwrote += 1
        else:
            # SEMANTIC 2x3 (legacy). Captions name the visualization TYPE first
            # (2D pixels / 3D particles / 3D point cloud), then the coloring. Rows = 2-D (top) /
            # 3-D camera-pan (bottom).
            row1 = np.hstack([_label(_fit(bgr.copy()), "RGB camera frame", fps),
                              _label(_fit(rgb_dense), "2D pixels, by particle"),
                              _label(_fit(px_cluster), "2D pixels, by cluster")])
            row2 = np.hstack([_label(_fit(particles_color), "3D particles, by avg color (panning camera)"),
                              _label(_fit(particles_cluster), "3D particles, by cluster (panning camera)"),
                              _label(_fit(pc_cluster3d), "3D point cloud, by cluster (panning camera)")])
            writer.write(np.vstack([row1, row2, _stats_row(cw, st)])); nwrote += 1
    def _finish(w, tmp_name, final_path):
        w.release()
        if _transcode_to_h264(tmp_name, str(final_path)):
            Path(tmp_name).unlink(missing_ok=True)
        else:
            Path(tmp_name).rename(final_path)
    if focused:
        _finish(writer_pc, pc_tmp, pc_path)
        return {"vid": vid, "status": "ok", "frames": nwrote, "seed": seed_kind,
                "fps": round(float(np.mean(fps_hist)), 1) if fps_hist else None,
                "path": pc_path.name}
    _finish(writer, str(out_path) + ".mp4v.mp4", out_path)
    return {"vid": vid, "status": "ok", "frames": nwrote, "seed": seed_kind,
            "fps": round(float(np.mean(fps_hist)), 1) if fps_hist else None, "path": str(out_path)}


_ENV_KNOBS_EPILOG = """
environment-variable knobs (advanced tuning; all render-only, tracking is untouched):
  output mode
    RENDER_MODE=focused      ship ONE unified pipeline FIGURE per clip (<vid>_unified.mp4,
                             white-paper theme): RGB input -> hero 3-D particle cloud
                             (WORLD-STABILIZED: ego-motion from the background blobs ->
                             static background, moving camera rig; fill = avg color,
                             faint GLOW = object cluster, depth FOG; 3 camera frustums at
                             REAL positions all AIMED at the scene center, warped image
                             planes depth-interleaved with the particles, windows show
                             the RENDERED RGB representation; the hero is a FOLLOW camera
                             — two-pass-solved smoothed tracks, fixed depth + focal,
                             pans with the action, never zooms/dollies)
                             -> 3a RGB reconstruction + 3b inferred object segmentation
                             + the classic inference-stats row under the diagram
    FORCE_SAM_SEED=1         seed EVERY clip with self-supervised SAM (no GT leakage). Default off
                             = custom clips use SAM, DAVIS clips use clean GT (avoids over-seg in
                             the demo)
  focused-mode hero camera + frustum diagram
    HERO_YAW=12              hero meta-camera yaw (deg)
    HERO_PITCH=14            hero elevation — gentle look-down (steeper turns the scene into a
                             floating specimen jar; lower reads like a photo of the scene)
    HERO_DEPTH_SCALE=1.0     depth smoosh in the hero (1 = off; <1 compresses depth thickness)
    FRUSTUM_YAW=35           +/- yaw of the two NOVEL frustum cameras; their little images are the
                             reconstruction RE-RENDERED from those poses (the accented ORIGINAL
                             camera shows the original-pose reconstruction = panel 3)
    PULLBACK=0.6             hero meta-camera retreat (frac of the true camera-object distance):
                             the frustums sit at REAL camera positions, so the hero pulls back to
                             fit the camera + its image plane + the whole scene in frame
    FRUSTUM_IMG_FRAC=0.4     frustum image-plane half-width as a frac of the object radius (also
                             capped by the camera spacing so large objects can't fuse the planes)
    RECON_FIT_MARGIN=0.5     object spans ~this frac of the hero (soft 97th-pct fit; the frustums
                             get a hard 0.88 fit); min-accumulated over FOCAL_CALIB_FRAMES, frozen
                             (then an UNFLOORED soft guard keeps the traveling rig in frame —
                             everything always fits)
  focused-mode world stabilization (static background, moving cameras — from the clustering)
    EGO_STABILIZE=1          solve the camera ego-motion per frame from the BACKGROUND blobs
                             (trimmed weighted Kabsch vs their previous world positions) and
                             render in the frame-0 world frame; the frustum rig is FROZEN in the
                             camera frame and mapped through the pose (mutually rigid, travels
                             with the real camera). 0 = camera-frame (no stabilization)
    LUT_EMA=0.15             per-blob avg-RGB color EMA rate (colors track what a blob currently
                             covers on a pan; only blobs owning datapoints update). The legacy
                             2x3 keeps the frame-0 LOCKED LUT
    GS_ALPHA_MAX=0.9         peak per-Gaussian opacity in the NOVEL frustum planes' volumetric
                             splatting render (front-to-back compositing of the real Gaussians
                             into a dense PIXEL image — a soft recon from those poses)
  focused-mode hero look (fill+glow multiplex, depth fog, figure-ground; pure particles —
  no point-cloud splats anywhere; frustums DEPTH-INTERLEAVED with the marbles: correct
  mutual occlusion, no punch-through either way)
    GLOW_ALPHA=0.9           the SEMANTIC GLOW strength (blend parameter; higher = stronger glow):
                             every particle's under-pass halo in its cluster color; the OBJECT's
                             glow is exempt from the fog fade (subject marker); 0 = off
    GLOW_SCALE=2.4           glow radius in units of the marble axes
    FOG=0.45                 depth fog (atmospheric perspective): fills mix toward the paper and
                             glows fade with each view's normalized depth (0 = off)
    BG_FADE=0.3              background-particle alpha (1.0 = no fade) — clearly transparent ghosts
    BG_DESAT=0.45            mix background fills toward near-white (0 = off) — paper figure-ground
    BG_SIZE_FRAC=0.33        SAM seeds only: a cluster covering > this fraction of the particles is
                             treated as the background plane and faded (a SAM-mislabeled couch/sky)
  camera pan / jitter (LEGACY 2x3 path only)
    PC_PAN_AMP=7.0           camera pan half-range in degrees (the slow monotonic sweep)
    PC_CENTROID_ALPHA=0.15   orbit-center EMA (lower = steadier cloud, less swim)
    MARBLE_GEOM_BETA=0.25    per-particle mean/cov EMA (lower = steadier particles)
    MARBLE_SIGMA=1.3         particle (marble) size, in sigmas
  2-D segmentation tiles
    SEG_EMA=0.3              temporal EMA of the 2-D seg color (lower = less edge flicker)
    SEG_EDGE_AWARE=1         1 = RGB-guided upsample; 0 = blocky-but-stable
  spurious-particle filter (drops big diffuse particles from the 3-D views)
    SPURIOUS_EXTENT_FENCE=0  >0 overrides the auto size-outlier cut (cov max-eigenvalue)
    SPURIOUS_DENSITY_PCT=15  "diffuse" cut: density percentile among drawable particles
    SPURIOUS_EXTENT_PCT=70   "big" gate paired with the density cut (extent percentile)
    SPURIOUS_EXTENT=0        >0 overrides the background size backstop
    SPURIOUS_HARD_CAP=0      >0 = absolute extent cap, kills anything bigger
    SPURIOUS_DEBUG=1         print the per-particle extent/count/density at frame 0

examples:
  # the featured deforming subjects, pipeline-triptych particles video + pixels video
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 RENDER_MODE=focused DEMO_NUM_BLOBS=256 STICKINESS=12 \
      ROBUST_DELTA=6 python scripts/render_gaussian_demo.py --target-duration 0 --out-fps 12
  # one DAVIS video, a quick 24-frame smoke
  python scripts/render_gaussian_demo.py --videos blackswan --max-frames 24 --target-duration 0
  # a steeper hero view, wider novel cameras
  HERO_PITCH=38 FRUSTUM_YAW=45 RENDER_MODE=focused python scripts/render_gaussian_demo.py
"""


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, epilog=_ENV_KNOBS_EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS,
                   help="video names to render (DAVIS GT names or assets/custom_videos/<name>); "
                        "the sentinels all-davis / all-custom / all enumerate every test video "
                        f"from disk; default: {' '.join(DEFAULT_VIDEOS)}")
    p.add_argument("--source", default=None,
                   help="render the particle figure on a RAW video file (mp4/mov) or RGB-frames "
                        "directory instead of a named DAVIS/custom video. Frame 0 is seeded by "
                        "k-means (no GT/SAM data needed) — works on the bundled assets/test.mp4.")
    p.add_argument("--config", default=str(_REPO / "configs/streaming_render_v2.yaml"),
                   help="tracking YAML (the validated, shipped config; default streaming_render_v2.yaml)")
    p.add_argument("--out-dir", default=str(_REPO / "runs/calibrate_consistency/viz_gaussian"),
                   help="output directory for <video>_live.mp4 files")
    p.add_argument("--num-sweeps", type=int, default=None,
                   help="Gibbs sweeps/frame (default: config num_gibbs_sweeps_per_frame, else 4)")
    p.add_argument("--max-frames", type=int, default=-1,
                   help="cap source frames read (-1 = whole clip); use a small value for a quick smoke")
    p.add_argument("--out-fps", type=float, default=30.0, help="output video frame rate")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="keep every Nth source frame before looping (2 = half the frames)")
    p.add_argument("--target-duration", type=float, default=6.0,
                   help="loop the (short) clip until this many seconds are emitted; 0 = play once")
    args = p.parse_args(argv)
    # --source overrides the named-video list: render the one raw file/dir directly.
    # Resolve to an absolute path so _is_raw_source routes it to the raw k-means
    # branch even for a bare-name frames directory (and a sentinel-named file like
    # "all" is never expanded).
    args.videos = [str(Path(args.source).resolve())] if args.source else _expand_video_args(args.videos)
    cfg = genmatter_rt.load_yaml_hypers(Path(args.config))
    cfg["tracking"]["use_sam_frame0"] = True
    sweeps = args.num_sweeps or int(cfg["tracking"].get("num_gibbs_sweeps_per_frame", 4))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"rendering {len(args.videos)} videos (sweeps={sweeps} "
         f"target_dur={args.target_duration}s) -> {out_dir}")
    results = []
    for j, vid in enumerate(args.videos, 1):
        label = Path(vid).stem if _is_raw_source(vid) else vid
        out = out_dir / f"{label}_live.mp4"
        try:
            r = render_video(vid, out, yaml_cfg=cfg, num_sweeps=sweeps, max_frames=args.max_frames,
                             target_duration=args.target_duration, out_fps=args.out_fps,
                             frame_stride=args.frame_stride)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); r = {"vid": vid, "status": "error", "error": repr(e)}
        results.append(r)
        _log(f"[{j}/{len(args.videos)}] {vid}: {r.get('status')} seed={r.get('seed')} "
             f"frames={r.get('frames')} fps={r.get('fps')} -> {r.get('path','-')}")
    _log("DONE: " + ", ".join(f"{r['vid']}={r['status']}" for r in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Depth-stabilized particle-viz demos

2×3 GenMatter particle visualization (`scripts/render_gaussian_demo.py`) rendered with the new
**temporal depth stabilization** (`genmatter_rt.stabilize_depth`): each frame's Depth-Anything
disparity is robustly affine-aligned (scale+shift) to a frozen frame-0 reference and normalized
with fixed bounds, so the static background no longer swims with the per-frame min/max
normalization. Measured ~40–71 % less per-pixel Z drift (wine_swirl −71 %, blackswan −49 %,
gray_jacket −40 %).

Click a file to play it in GitHub's viewer. Layout per video:

```
 2D │ RGB camera frame            │ 2D pixels, by particle    │ 2D pixels, by cluster      │
 3D │ 3D particles, by avg color  │ 3D particles, by cluster  │ 3D point cloud, by cluster │   (panning camera)
```

- `wine_swirl`, `jello_trim`, `gray_jacket`, `purple_jacket` — SAM2-seeded custom clips
- `blackswan` — DAVIS (ground-truth seeded)

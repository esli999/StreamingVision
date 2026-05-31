# Vendored: GenMatter++ `arijit/realtime-demo`

This directory is a one-shot `git archive` of GenMatter++ at the commit:

- **Source**: `git@github.com:esli999/GenMatterPlusPlus.git`
- **Branch**: `arijit/realtime-demo`
- **Commit**: `1db60f716ea9d0f686c1ebf1897c4e7f04c07c7e`
  ("Realtime demo: custom CLI, JIT tracking, bayesopt, DAVIS integration")
- **Vendored on**: 2026-05-20

The intent is **self-contained shipping**: StreamingVision no longer depends on the sibling repo at `/home/esli/GenMatterPlusPlus`.

## How it was vendored

```
mkdir -p StreamingVision/genmatterpp
git -C /home/esli/GenMatterPlusPlus archive origin/arijit/realtime-demo \
    | tar -x -C StreamingVision/genmatterpp
# clean up
rm -f genmatterpp/uv.lock
rm -rf genmatterpp/docs
touch genmatterpp/__init__.py
```

## Policy

This is a one-shot vendor — no upstream-pull workflow. If GenMatter++ updates and you want the new code, manually re-run the archive command (and re-apply any local edits captured below).

## Local edits

- `preprocessing/motion_extraction_3d/vda_preprocess_video.py:37` — replaced `from torchvision.io import read_video` with a local `read_video` shim using `imageio.v2`. The original API was removed in torchvision 0.27 and the streaming env pins a newer torch. The shim returns `(frames_tensor, empty, {})` matching the old signature.
- `preprocessing/motion_extraction_3d/vda_preprocess_video.py:496` — `batch_size` for RAFT optical flow lowered from 16 → 2.  Our 4K test.mp4 even at `max_res: 480` produces correlation volumes that OOM on RTX 5090 at the original batch.
- `genmatter/tracking/dino.py:884` — `compile_dino_tracking_program()` now falls back to the `GENMATTER_DEFAULT_MAX_FRAMES` env var if the DAVIS-derived `tapvid_davis_max_frames()` raises FileNotFoundError. Custom-video bayesopt runs don't have a DAVIS install.
- `genmatter/instance_seg_metrics.py:103` — added `pixel_jaccard` (standardized binary foreground/background Jaccard) to the per-frame metrics dict, and `avg_pixel_jaccard` to the video-level aggregator.  Kept for diagnostics; not used as the objective (degenerate on multi-instance SAM2 GT).
- `genmatter/instance_seg_metrics.py:_persistent_identity_iou` — new persistent-identity IoU at HYPERBLOB granularity (not individual blobs).  Hungarian-match hyperblob IDs ↔ GT IDs ONCE on frame 0, then evaluate the fixed mapping at every subsequent frame.  Hyperblobs are the model's object-level persistent clusters (each = union of related blobs); matching at this scale rewards identity preservation and avoids the "tiny blob covers only fraction of GT" problem.  Added `rasterize_hyperblob_label_map` helper.
- **StreamingVision deletion (2026-05-27)** — the `genmatter/bayesopt/` package, `genmatter/custom/bayesopt_cmd.py`, the `genmatter bayesopt` CLI subcommand, and their tests/scripts/configs were removed because the per-video bayesopt output regressed live (outlier_frac → 1.00).  Multi-video live calibration via `scripts/calibrate_general.py` replaces them.  The original vendor SHA above still has the full bayesopt subsystem if it ever needs to be revived.

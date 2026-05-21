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
- `genmatter/bayesopt/objective.py:18` — `PRIMARY_METRIC_KEY` switched from `avg_mean_gt_iou` (Hungarian re-run per frame) to `avg_persistent_iou`.  Both metrics are still recorded in trial meta.
- `genmatter/bayesopt/objective.py` — added `MultiVideoTrackingObjective` + `build_multi_objective_contexts`.  Multi-video bayesopt: each trial runs on every preprocessed video, primary metric is the mean across videos, validity requires ALL videos under the outlier % cap.
- `genmatter/bayesopt/objective.py` — replaced the hard `ax_objective = 0.0` clamp for invalid trials with a soft `ax_objective = score - OUTLIER_PENALTY_WEIGHT * overshoot_frac` penalty (PENALTY_WEIGHT = 2.0).  Previously the GP saw a bimodal training set: all invalid trials collapsed to 0.0 regardless of how badly they overshot the cap, producing a discontinuous surface across the constraint boundary that the BoTorch default Matern-5/2 + ARD kernel could not smooth.  Soft penalty preserves raw-IoU signal for "invalid-but-close" trials (e.g. 22% outliers / 0.57 IoU) while still ordering them below valid trials.  `valid` flag retained for best-trial selection, so the WINNER is still strictly inside the cap.  Multi-video uses MAX over per-video outlier %s (most stringent video drives the penalty).
- `genmatter/bayesopt/ax_runner.py:run_bayesopt_forever` — `video_id` now accepts a comma-separated list; the first id provides the run-artifact directory.

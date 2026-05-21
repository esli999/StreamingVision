#!/bin/bash
# Preprocess + SAM2 pseudo-GT for every video under assets/custom_videos/.
# Idempotent: skip_existing in streaming_default.yaml means already-processed
# stages are no-ops.
set -e
cd "$(dirname "$0")/.."

VIDEOS=(gray_jacket jello_trim new_eagle_trim purple_jacket snake_trim wine_swirl two_blocks_centered)

for vid in "${VIDEOS[@]}"; do
  echo "===== $vid ====="
  src=$(ls assets/custom_videos/"$vid"/source.* 2>/dev/null | head -1)
  [ -z "$src" ] && { echo "skip: no source.* in $vid"; continue; }

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m genmatterpp.genmatter.custom.cli preprocess \
      "$src" --video-id "$vid" --config configs/streaming_default.yaml \
      2>&1 | tail -20

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m genmatterpp.genmatter.custom.cli pseudo-gt \
      --video-id "$vid" --config configs/streaming_default.yaml \
      2>&1 | tail -20
done

echo
echo "===== summary ====="
for vid in "${VIDEOS[@]}"; do
  pgt_dir="assets/custom_videos/$vid/pseudo_gt_sam/segmasks/$vid"
  if [ -d "$pgt_dir" ]; then
    n=$(ls "$pgt_dir" | wc -l)
    echo "$vid: $n pseudo-GT frames"
  else
    echo "$vid: NO pseudo-GT"
  fi
done

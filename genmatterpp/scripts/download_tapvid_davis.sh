#!/usr/bin/env bash
# Download DAVIS 2017 full-resolution trainval and copy TAP-Vid subset into
# tapvid_davis_rgb_frames / tapvid_davis_segmasks (paths from config.py).
#
# Usage: from repo root, ./scripts/download_tapvid_davis.sh
#   or: uv run python run_experiments.py download-tapvid-davis
# Env: GENMATTER_DAVIS_DIR (same as config), GENMATTER_DAVIS_FETCH_CACHE (default: .cache/davis_fullres)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ZIP_NAME="DAVIS-2017-trainval-Full-Resolution.zip"
ZIP_URL="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip"
CACHE="${GENMATTER_DAVIS_FETCH_CACHE:-$REPO_ROOT/.cache/davis_fullres}"
ZIP_PATH="$CACHE/$ZIP_NAME"
STAGING="$CACHE/extract"

run_py() {
  if command -v uv &>/dev/null; then
    uv run python scripts/davis_tapvid_fetch.py "$@"
  else
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python3 "$REPO_ROOT/scripts/davis_tapvid_fetch.py" "$@"
  fi
}

eval "$(run_py paths)"

mkdir -p "$CACHE" "$DAVIS_RGB_PATH" "$DAVIS_SEGMASKS_PATH"

echo "Repo:      $REPO_ROOT"
echo "RGB dest:  $DAVIS_RGB_PATH"
echo "Seg dest:  $DAVIS_SEGMASKS_PATH"
echo "Cache:     $CACHE"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Downloading $ZIP_NAME (large; resume supported)..."
  mkdir -p "$CACHE"
  tmp="${ZIP_PATH}.part"
  if command -v curl &>/dev/null; then
    if [[ -f "$tmp" ]]; then
      curl -L -C - -o "$tmp" "$ZIP_URL"
    else
      curl -L -o "$tmp" "$ZIP_URL"
    fi
  elif command -v wget &>/dev/null; then
    wget -c -O "$tmp" "$ZIP_URL"
  else
    echo "Need curl or wget" >&2
    exit 1
  fi
  mv -f "$tmp" "$ZIP_PATH"
  echo "Downloaded: $(du -h "$ZIP_PATH" | cut -f1)"
fi

set +e
NEEDS="$(run_py needs-copy "$ZIP_PATH" --rgb "$DAVIS_RGB_PATH" --seg "$DAVIS_SEGMASKS_PATH" 2>/tmp/davis_fetch_err.$$)"
NEED_RC=$?
set -e
if [[ "$NEED_RC" -eq 2 ]]; then
  cat /tmp/davis_fetch_err.$$ >&2
  rm -f /tmp/davis_fetch_err.$$
  exit 2
fi
rm -f /tmp/davis_fetch_err.$$

if [[ -z "${NEEDS// }" ]]; then
  echo "All TAP-Vid sequences already match the zip (frame counts + RGB + seg). Nothing to copy."
  run_py verify "$ZIP_PATH" --rgb "$DAVIS_RGB_PATH" --seg "$DAVIS_SEGMASKS_PATH"
  echo "Removing zip and any staging under cache..."
  rm -f "$ZIP_PATH"
  rm -rf "$STAGING"
  echo "Done."
  exit 0
fi

echo "Copying sequences:"
echo "$NEEDS"

echo "Extracting zip (one-time, large)..."
rm -rf "$STAGING"
mkdir -p "$STAGING"
unzip -q "$ZIP_PATH" -d "$STAGING"

eval "$(run_py find-roots "$STAGING")"

while IFS= read -r v; do
  [[ -z "$v" ]] && continue
  if [[ ! -d "$JPEG_FULL/$v" ]]; then
    echo "ERROR: missing sequence in extract: $JPEG_FULL/$v" >&2
    echo "Available:" >&2
    ls -1 "$JPEG_FULL" >&2
    exit 1
  fi
  if [[ ! -d "$ANN_FULL/$v" ]]; then
    echo "ERROR: missing sequence in extract: $ANN_FULL/$v" >&2
    exit 1
  fi
  echo "  -> $v"
  mkdir -p "$DAVIS_RGB_PATH/$v" "$DAVIS_SEGMASKS_PATH/$v"
  cp -a "$JPEG_FULL/$v/." "$DAVIS_RGB_PATH/$v/"
  cp -a "$ANN_FULL/$v/." "$DAVIS_SEGMASKS_PATH/$v/"
done <<< "$NEEDS"

run_py verify "$ZIP_PATH" --rgb "$DAVIS_RGB_PATH" --seg "$DAVIS_SEGMASKS_PATH"

echo "Removing zip and extracted tree..."
rm -f "$ZIP_PATH"
rm -rf "$STAGING"
echo "Done."

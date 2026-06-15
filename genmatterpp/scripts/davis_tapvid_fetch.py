#!/usr/bin/env python3
"""Helpers for download_tapvid_davis.sh: paths from config, zip manifest, copy needs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import zipfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import config  # noqa: E402

# Match any path prefix inside the official DAVIS zip.
_JPG = re.compile(r"(?:^|/)JPEGImages/Full-Resolution/([^/]+)/[^/]+\.jpg$", re.IGNORECASE)
_PNG = re.compile(r"(?:^|/)Annotations/Full-Resolution/([^/]+)/[^/]+\.png$", re.IGNORECASE)


def zip_sequence_counts(zip_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Return (jpg_counts_by_seq, png_counts_by_seq) from archive member names."""
    jpg: dict[str, int] = {}
    png: dict[str, int] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            mj = _JPG.search(name.replace("\\", "/"))
            if mj:
                s = mj.group(1)
                jpg[s] = jpg.get(s, 0) + 1
            mp = _PNG.search(name.replace("\\", "/"))
            if mp:
                s = mp.group(1)
                png[s] = png.get(s, 0) + 1
    return jpg, png


def disk_counts(video: str, rgb_root: Path, seg_root: Path) -> tuple[int, int]:
    """Count *.jpg and *.png under rgb_root/video and seg_root/video."""
    rd = rgb_root / video
    sd = seg_root / video
    nj = len(list(rd.glob("*.jpg"))) if rd.is_dir() else 0
    np_ = len(list(sd.glob("*.png"))) if sd.is_dir() else 0
    return nj, np_


def find_davis_roots(extract_dir: Path) -> tuple[Path, Path]:
    """Return (jpeg_fullres_dir, ann_fullres_dir), each .../Full-Resolution containing per-video folders."""
    jpeg_fr = list(extract_dir.rglob("JPEGImages/Full-Resolution"))
    ann_fr = list(extract_dir.rglob("Annotations/Full-Resolution"))
    if not jpeg_fr or not ann_fr:
        raise FileNotFoundError(
            f"No JPEGImages/Full-Resolution or Annotations/Full-Resolution under {extract_dir}"
        )
    return jpeg_fr[0], ann_fr[0]


def cmd_find_roots(args: argparse.Namespace) -> int:
    j, a = find_davis_roots(Path(args.extract_dir))
    print(f"export JPEG_FULL={shlex.quote(str(j.resolve()))}")
    print(f"export ANN_FULL={shlex.quote(str(a.resolve()))}")
    return 0


def cmd_paths(_args: argparse.Namespace) -> int:
    print(f"export DAVIS_RGB_PATH={shlex.quote(str(config.DAVIS_RGB_PATH.resolve()))}")
    print(f"export DAVIS_SEGMASKS_PATH={shlex.quote(str(config.DAVIS_SEGMASKS_PATH.resolve()))}")
    print(f"export REPO_ROOT={shlex.quote(str(_REPO.resolve()))}")
    return 0


def cmd_videos(_args: argparse.Namespace) -> int:
    for v in config.TAPVID_DAVIS_VIDEO_NAMES:
        print(v)
    return 0


def cmd_zip_manifest(args: argparse.Namespace) -> int:
    zp = Path(args.zip)
    jpg, png = zip_sequence_counts(zp)
    out = {"jpg": jpg, "png": png}
    print(json.dumps(out))
    return 0


def cmd_needs_copy(args: argparse.Namespace) -> int:
    zp = Path(args.zip)
    rgb = Path(args.rgb)
    seg = Path(args.seg)
    jpg_m, png_m = zip_sequence_counts(zp)
    missing_in_zip: list[str] = []
    need_copy: list[str] = []
    for v in config.TAPVID_DAVIS_VIDEO_NAMES:
        if v not in jpg_m or v not in png_m:
            missing_in_zip.append(v)
            continue
        dj, dp = disk_counts(v, rgb, seg)
        if dj == jpg_m[v] and dp == png_m[v]:
            continue
        need_copy.append(v)
    if missing_in_zip:
        avail = sorted(set(jpg_m.keys()) & set(png_m.keys()))
        print(
            "ERROR: these TAP-Vid names are not in the zip (check DAVIS folder names): "
            + ", ".join(missing_in_zip),
            file=sys.stderr,
        )
        print("Available sequences (intersection jpg/png): " + ", ".join(avail[:40]) + "...", file=sys.stderr)
        return 2
    for v in need_copy:
        print(v)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    zp = Path(args.zip)
    rgb = Path(args.rgb)
    seg = Path(args.seg)
    jpg_m, png_m = zip_sequence_counts(zp)
    bad = []
    for v in config.TAPVID_DAVIS_VIDEO_NAMES:
        dj, dp = disk_counts(v, rgb, seg)
        if dj != jpg_m.get(v, -1) or dp != png_m.get(v, -1):
            bad.append((v, dj, jpg_m.get(v), dp, png_m.get(v)))
    if bad:
        for row in bad:
            print(f"VERIFY_FAIL {row}", file=sys.stderr)
        return 1
    print("VERIFY_OK all 30 sequences match zip counts")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="DAVIS TAP-Vid fetch helpers")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("paths", help="Print DAVIS_RGB_PATH, DAVIS_SEGMASKS_PATH, REPO_ROOT for shell eval")
    sub.add_parser("videos", help="Print one TAP-Vid video name per line")

    p_m = sub.add_parser("zip-manifest", help="JSON jpg/png counts per sequence from zip")
    p_m.add_argument("zip", type=Path)

    p_n = sub.add_parser("needs-copy", help="Print video names that need copy (one per line)")
    p_n.add_argument("zip", type=Path)
    p_n.add_argument("--rgb", type=Path, required=True)
    p_n.add_argument("--seg", type=Path, required=True)

    p_v = sub.add_parser("verify", help="Exit 0 iff disk counts match zip for all TAP-Vid videos")
    p_v.add_argument("zip", type=Path)
    p_v.add_argument("--rgb", type=Path, required=True)
    p_v.add_argument("--seg", type=Path, required=True)

    p_f = sub.add_parser("find-roots", help="Print JPEG_FULL= and ANN_FULL= for extracted tree")
    p_f.add_argument("extract_dir", type=Path)

    args = p.parse_args()
    if args.cmd == "paths":
        return cmd_paths(args)
    if args.cmd == "videos":
        return cmd_videos(args)
    if args.cmd == "zip-manifest":
        return cmd_zip_manifest(args)
    if args.cmd == "needs-copy":
        return cmd_needs_copy(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    if args.cmd == "find-roots":
        return cmd_find_roots(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

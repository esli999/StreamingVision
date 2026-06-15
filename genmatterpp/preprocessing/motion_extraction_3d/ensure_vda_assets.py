"""Clone Video-Depth-Anything and download Hugging Face checkpoints on first use (3D motion preprocessing)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

GIT_URL = "https://github.com/DepthAnything/Video-Depth-Anything.git"

# Defaults to ``vitl`` (Large). Other encoders download on first use.
CHECKPOINTS: dict[str, tuple[str, str]] = {
    "vits": (
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth",
        "video_depth_anything_vits.pth",
    ),
    "vitb": (
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth",
        "video_depth_anything_vitb.pth",
    ),
    "vitl": (
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth",
        "video_depth_anything_vitl.pth",
    ),
}

DEFAULT_ENCODER = "vitl"

_UA = "genmatter-ensure-vda/1"


def _default_vda_dir(repo_root: Path) -> Path:
    return repo_root / "external" / "video-depth-anything"


def resolve_vda_dir(repo_root: Path) -> Path:
    override = os.environ.get("GENMATTER_VIDEO_DEPTH_ANYTHING_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return _default_vda_dir(repo_root)


def _download_one(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    print(f"Downloading {dest.name} …", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=1200) as resp:
            with open(partial, "wb") as out:
                shutil.copyfileobj(resp, out)
        partial.replace(dest)
    except BaseException:
        if partial.exists():
            partial.unlink()
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        raise


def ensure_checkpoint_for_encoder(vda_dir: Path, encoder: str) -> None:
    """Download the ``.pth`` for ``encoder`` (``vits`` / ``vitb`` / ``vitl``) if missing."""
    if encoder not in CHECKPOINTS:
        raise ValueError(f"Unknown VDA encoder: {encoder}")
    url, name = CHECKPOINTS[encoder]
    _download_one(url, vda_dir / "checkpoints" / name)


def _ensure_default_checkpoint_only(vda_dir: Path) -> None:
    """Only the default weight (``vitl``); matches upstream default depth model."""
    ensure_checkpoint_for_encoder(vda_dir, DEFAULT_ENCODER)


def ensure_video_depth_anything(repo_root: Path) -> Path:
    """
    Ensure ``video_depth_anything/`` and the default checkpoint exist under the resolved VDA root.

    Uses ``GENMATTER_VIDEO_DEPTH_ANYTHING_PATH`` when set; otherwise clones into
    ``<repo>/external/video-depth-anything`` when missing.
    """
    vda_dir = resolve_vda_dir(repo_root)
    marker = vda_dir / "video_depth_anything"
    if not marker.is_dir():
        if os.environ.get("GENMATTER_VIDEO_DEPTH_ANYTHING_PATH"):
            raise RuntimeError(
                f"GENMATTER_VIDEO_DEPTH_ANYTHING_PATH={vda_dir} does not contain a Video-Depth-Anything "
                "checkout (expected video_depth_anything/). Clone it there or unset the variable."
            )
        vda_dir.parent.mkdir(parents=True, exist_ok=True)
        if vda_dir.exists():
            shutil.rmtree(vda_dir)
        print("Cloning Video-Depth-Anything …", flush=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", GIT_URL, str(vda_dir)],
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    _ensure_default_checkpoint_only(vda_dir)
    return vda_dir

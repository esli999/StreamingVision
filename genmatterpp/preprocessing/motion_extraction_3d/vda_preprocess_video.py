#!/usr/bin/env python3
"""
Video preprocessing for 3D motion NPZ output.

Uses Video-Depth-Anything + RAFT to produce dense ``points_3d``, ``motion_vectors_3d``, ``colors``,
``intrinsics`` in ``.npz`` format (``float64`` arrays, ``np.savez`` uncompressed). JAX is not imported here.

VDA checkout: ``<repo>/external/video-depth-anything`` or ``GENMATTER_VIDEO_DEPTH_ANYTHING_PATH``.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_GENMATTER_ROOT = Path(__file__).resolve().parents[2]
from preprocessing.motion_extraction_3d.ensure_vda_assets import (  # noqa: E402
    ensure_checkpoint_for_encoder,
    ensure_video_depth_anything,
)

_VDA_ENSURED = False


def _ensure_vda_once() -> None:
    global _VDA_ENSURED
    if not _VDA_ENSURED:
        ensure_video_depth_anything(_GENMATTER_ROOT)
        _VDA_ENSURED = True


import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from typing import Callable, Optional, Tuple
import glob
import re
from PIL import Image

# StreamingVision local patch: torchvision.io.read_video was removed in
# torchvision 0.27. Provide a drop-in shim using imageio so this module keeps
# working under the streaming env without downgrading torch.
def read_video(input_path, pts_unit="sec"):
    import imageio.v2 as _iio
    reader = _iio.get_reader(input_path)
    frames = np.stack([np.asarray(f) for f in reader], axis=0)  # (N, H, W, C) uint8
    reader.close()
    return torch.from_numpy(frames), torch.empty(0), {}

# Video-Depth-Anything: GenMatter repo root is parents[2] of this file
_vda_override = os.environ.get("GENMATTER_VIDEO_DEPTH_ANYTHING_PATH")
vda_path = (
    str(Path(_vda_override).expanduser().resolve())
    if _vda_override
    else str(_GENMATTER_ROOT / "external" / "video-depth-anything")
)

if os.path.exists(vda_path):
    sys.path.insert(0, vda_path)
    try:
        from video_depth_anything.video_depth import VideoDepthAnything
        from utils.dc_utils import read_video_frames

        VDA_AVAILABLE = True
    except ImportError:
        VDA_AVAILABLE = False
else:
    VDA_AVAILABLE = False


def _vlog(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


def preprocess_frames(batch):
    """Preprocess frames for optical flow computation."""
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 960)),
        ]
    )
    batch = transforms(batch)
    return batch


def is_video_file(path: str) -> bool:
    """Check if path is a video file (MP4, AVI, etc.)."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}
    return (
        os.path.isfile(path) and os.path.splitext(path.lower())[1] in video_extensions
    )


def is_image_folder(path: str) -> bool:
    """Check if path is a folder containing image frames."""
    if not os.path.isdir(path):
        return False

    # Check for common image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))

    return len(image_files) > 0


def read_image_frames(
    folder_path: str,
    max_len: int = -1,
    target_fps: int = -1,
    max_res: int = -1,
    skip_frames: int = 1,
) -> Tuple[np.ndarray, float]:
    """
    Read image frames from a folder.

    Args:
        folder_path: Path to folder containing image frames
        max_len: Maximum number of frames to process (-1 for no limit)
        target_fps: Target FPS (not used for image folders, but kept for compatibility)
        max_res: Maximum resolution for processing
        skip_frames: Skip every N frames (1 = process all frames)

    Returns:
        frames: numpy array of frames (N, H, W, C)
        fps: FPS (default 30 for image folders)
    """
    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))

    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")

    # Sort files by number in filename (extract numbers and sort)
    def extract_number(filename):
        """Extract number from filename for sorting."""
        numbers = re.findall(r"\d+", os.path.basename(filename))
        return int(numbers[0]) if numbers else 0

    image_files.sort(key=extract_number)

    # Apply max_len limit
    if max_len > 0:
        image_files = image_files[:max_len]

    # Apply frame skipping
    if skip_frames > 1:
        image_files = image_files[::skip_frames]

    # Read images
    frames = []
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        # Resize if needed
        if max_res > 0:
            h, w = img_array.shape[:2]
            if max(h, w) > max_res:
                scale = max_res / max(h, w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                img_array = np.array(img)

        frames.append(img_array)

    frames = np.stack(frames, axis=0)

    # Default FPS for image folders (can be overridden)
    fps = target_fps if target_fps > 0 else 30.0

    return frames, fps


@dataclass
class Motion3DTimings:
    """Wall-clock timings for 3D motion preprocessing substeps."""

    depth_seconds: float
    depth_post_seconds: float
    load_rgb_seconds: float
    flow_seconds: float
    save_seconds: float
    num_frames: int
    num_frame_pairs: int

    @property
    def depth_fps(self) -> float:
        return self.num_frames / self.depth_seconds if self.depth_seconds > 0 else 0.0

    @property
    def flow_fps(self) -> float:
        return (
            self.num_frame_pairs / self.flow_seconds if self.flow_seconds > 0 else 0.0
        )


def unproject_points(x, y, z, fx=520, fy=520, cx=None, cy=None):
    """
    Unproject 2D points to 3D using depth and camera intrinsics
    x, y: pixel coordinates
    z: depth values
    fx, fy: focal lengths
    cx, cy: principal point (if None, will be calculated from image dimensions)
    """
    # If cx and cy are not provided, calculate them from the image dimensions
    if cx is None:
        cx = x.shape[1] / 2  # Width / 2
    if cy is None:
        cy = x.shape[0] / 2  # Height / 2

    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    return np.stack([x_3d, y_3d, z], axis=-1)


def extract_depths_with_vda(
    input_path: str,
    encoder: str = "vitl",
    input_size: int = 518,
    max_res: int = 1280,
    max_len: int = -1,
    target_fps: int = -1,
    fp32: bool = False,
    device: str = "cuda",
    skip_frames: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """
    Extract depths using Video-Depth-Anything.

    Args:
        input_path: Path to video file or folder containing image frames
    """
    if not VDA_AVAILABLE:
        raise ImportError("Video-Depth-Anything is not available")

    _ensure_vda_once()

    # Load Video-Depth-Anything model
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    checkpoint_name = "video_depth_anything"
    checkpoint_path = os.path.join(
        vda_path, "checkpoints", f"{checkpoint_name}_{encoder}.pth"
    )

    ensure_checkpoint_for_encoder(Path(vda_path), encoder)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    video_depth_anything = VideoDepthAnything(**model_configs[encoder], metric=False)
    video_depth_anything.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu"), strict=True
    )
    video_depth_anything = video_depth_anything.to(device).eval()

    # Load frames - check if input is video file or image folder
    if is_video_file(input_path):
        frames, target_fps = read_video_frames(input_path, max_len, target_fps, max_res)
        _vlog(verbose, f"Loaded {len(frames)} frames from video")

        # Apply frame skipping
        if skip_frames > 1:
            frames = frames[::skip_frames]
            _vlog(
                verbose,
                f"After skipping every {skip_frames} frames: {len(frames)} frames remaining",
            )
    elif is_image_folder(input_path):
        frames, target_fps = read_image_frames(
            input_path, max_len, target_fps, max_res, skip_frames
        )
        _vlog(verbose, f"Loaded {len(frames)} frames from image folder")
    else:
        raise ValueError(
            f"Input path must be either a video file or a folder containing image frames: {input_path}"
        )

    # Get depth estimates
    _vlog(verbose, "Computing depth estimates...")
    _vlog(verbose, f"Processing {len(frames)} frames with Video-Depth-Anything...")

    depths, fps = video_depth_anything.infer_video_depth(
        frames, target_fps, input_size=input_size, device=device, fp32=fp32
    )

    _vlog(verbose, f"✅ Computed depths with shape: {depths.shape}")

    return depths


def process_video_to_3d_data(
    input_path: str,
    output_path: Optional[str] = None,
    encoder: str = "vitl",
    input_size: int = 518,
    max_res: int = 1280,
    max_len: int = -1,
    target_fps: int = -1,
    fp32: bool = False,
    device: Optional[str] = None,
    skip_frames: int = 1,
    subsample: float = 1.0,
    output_format: str = "npz",
    verbose: bool = False,
    user_log: Optional[Callable[[str], None]] = None,
    on_step_start: Optional[Callable[[str], None]] = None,
    on_frame_progress: Optional[Callable[[int, int], None]] = None,
    return_timings: bool = False,
    quiet: bool = False,
) -> str | tuple[str, Motion3DTimings]:
    """
    Process a video or image folder to extract 3D motion vectors using Video-Depth-Anything and optical flow.

    Args:
        input_path: Path to video file or folder containing image frames
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Check if input is video file or image folder
    if not (is_video_file(input_path) or is_image_folder(input_path)):
        raise ValueError(
            f"Input path must be either a video file or a folder containing image frames: {input_path}"
        )

    if not VDA_AVAILABLE:
        raise ImportError(
            "Video-Depth-Anything is not available. Please ensure the submodule is properly initialized."
        )

    _ensure_vda_once()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate subsample parameter
    if not (0.0 < subsample <= 1.0):
        raise ValueError(f"subsample must be between 0.0 and 1.0, got {subsample}")

    # Validate output format parameter
    if output_format not in ["npz", "pkl"]:
        raise ValueError(f"output_format must be 'npz' or 'pkl', got {output_format}")

    # Set up output path
    if output_path is None:
        if is_video_file(input_path):
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_dir = os.path.dirname(input_path)
        else:
            base_name = os.path.basename(input_path.rstrip("/"))
            output_dir = os.path.dirname(input_path.rstrip("/")) or "."
        output_path = os.path.join(output_dir, f"{base_name}_3d_data.{output_format}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log = user_log if user_log is not None else (lambda _m: None if quiet else print)
    total_steps = 5 + (1 if subsample < 1.0 else 0)

    _STEP_EVENT = {
        1: "depth_vda",
        2: "depth_post",
        3: "load_rgb",
        4: "optical_flow",
        5: "save_npz",
    }

    def u(msg: str) -> None:
        if not quiet:
            log(msg)

    def step(i: int, msg: str) -> None:
        if on_step_start is not None:
            on_step_start(_STEP_EVENT.get(i, f"step_{i}"))
        if not quiet:
            log(f"  [{i}/{total_steps}] {msg}")

    input_type = "video" if is_video_file(input_path) else "image folder"
    if not quiet:
        u("")
        u(f"3D motion  ({input_type})  {os.path.basename(input_path.rstrip('/'))}")
        u(f"  → {output_path}")
    _vlog(verbose, f"Processing {input_type}: {input_path}")
    _vlog(verbose, f"Output will be saved to: {output_path}")

    # Step 1: Extract depths using Video-Depth-Anything
    step(1, "Depth — Video-Depth-Anything")
    _vlog(verbose, "Step 1: Running Video-Depth-Anything depth extraction...")
    t_depth0 = time.perf_counter()
    depths = extract_depths_with_vda(
        input_path=input_path,
        encoder=encoder,
        input_size=input_size,
        max_res=max_res,
        max_len=max_len,
        target_fps=target_fps,
        fp32=fp32,
        device=device,
        skip_frames=skip_frames,
        verbose=verbose,
    )
    depth_seconds = time.perf_counter() - t_depth0
    num_depth_frames = int(depths.shape[0]) if hasattr(depths, "shape") else 0

    # Step 2: Process depths (convert to actual depth values)
    step(2, "Depth — clip inverse depth, convert to metric, median-scale")
    t_post0 = time.perf_counter()
    _vlog(verbose, "Step 2: Processing depth data...")
    # Clip the inverse depth to between 200 and 1200
    clipped_inverse_depth = np.clip(depths, 200, 1200)

    # Convert inverse depth to actual depth (add small epsilon to avoid division by zero)
    epsilon = 1e-6
    actual_depths = 1.0 / (clipped_inverse_depth + epsilon)

    # Scale the depth so that the median is around 5
    current_median = np.median(actual_depths)
    scaling_factor = 5.0 / current_median
    actual_depths = actual_depths * scaling_factor
    depth_post_seconds = time.perf_counter() - t_post0

    # Step 3: Load frames for optical flow processing
    step(3, "RGB — load frames (aligned with depth)")
    t_load_rgb0 = time.perf_counter()
    _vlog(verbose, "Step 3: Loading frames for optical flow...")
    if is_video_file(input_path):
        frames, _, _ = read_video(input_path, pts_unit="sec")
        frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        _vlog(verbose, f"Loaded {len(frames)} frames from video")

        # Apply frame skipping to match depth frames
        if skip_frames > 1:
            frames = frames[::skip_frames]
            _vlog(
                verbose,
                f"After skipping every {skip_frames} frames: {len(frames)} frames remaining",
            )
    else:
        # Load image frames
        frames_array, _ = read_image_frames(
            input_path, max_len, target_fps, max_res, skip_frames
        )
        # Convert to torch tensor: (N, H, W, C) -> (N, C, H, W)
        frames = torch.from_numpy(frames_array).permute(0, 3, 1, 2)
        _vlog(verbose, f"Loaded {len(frames)} frames from image folder")

    load_rgb_seconds = time.perf_counter() - t_load_rgb0

    # Step 4: Load optical flow model + compute flow and 3D motion
    step(4, "Motion — RAFT optical flow, unproject, 3D motion vectors")
    _vlog(verbose, "Step 4: Loading optical flow model...")
    from torchvision.models.optical_flow import Raft_Large_Weights

    t_flow0 = time.perf_counter()
    flow_model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False).to(
        device
    )
    flow_model = flow_model.eval()

    # Determine number of frames to process (up to second-to-last frame)
    num_frames = min(len(frames) - 1, len(actual_depths) - 1)
    _vlog(verbose, f"Processing {num_frames} frames for optical flow")

    all_points_3d = []
    all_motion_vectors_3d = []
    all_colors = []

    # Process frames in batches for efficiency
    batch_size = 2  # StreamingVision patch: was 16; 4K test.mp4 OOMs RAFT corr volume above ~4

    # Add progress bar for optical flow processing
    from tqdm import tqdm

    with tqdm(
        total=num_frames,
        desc="Optical flow",
        unit="frame",
        disable=not verbose,
    ) as pbar:
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)

            # Prepare batches of current and next frames
            img1_batch = preprocess_frames(frames[batch_start:batch_end]).to(device)
            img2_batch = preprocess_frames(frames[batch_start + 1 : batch_end + 1]).to(
                device
            )

            # Get optical flow between current and next frames (batched)
            with torch.no_grad():
                flow_batch = flow_model(img1_batch, img2_batch)

            # Process each frame in the batch
            for i in range(batch_end - batch_start):
                frame_idx = batch_start + i

                # Get the flow for this frame
                flow = (
                    flow_batch[-1][i].detach().cpu().numpy()
                )  # Get the final flow prediction

                # Get depth for current frame
                depth_data = actual_depths[frame_idx]

                # Resize depth to match flow dimensions if needed
                if depth_data.shape != (flow.shape[1], flow.shape[2]):
                    from skimage.transform import resize

                    depth_data = resize(
                        depth_data, (flow.shape[1], flow.shape[2]), preserve_range=True
                    )

                # Create coordinate grids
                h, w = flow.shape[1], flow.shape[2]
                y_grid, x_grid = np.mgrid[0:h, 0:w]

                # Calculate principal points from image dimensions
                cx = w / 2
                cy = h / 2

                # Get 3D points for current frame
                points_3d = unproject_points(x_grid, y_grid, depth_data, cx=cx, cy=cy)

                # Extract color information from the current frame
                current_frame = (
                    frames[frame_idx].permute(1, 2, 0).cpu().numpy()
                )  # Convert to (H, W, C)

                # Resize color frame to match flow dimensions if needed
                if current_frame.shape[:2] != (h, w):
                    from skimage.transform import resize

                    current_frame = resize(
                        current_frame, (h, w, 3), preserve_range=True
                    )

                # Get color values for each point
                colors = current_frame.astype(np.uint8)

                # Calculate destination coordinates using optical flow
                x_dest = x_grid + flow[0]
                y_dest = y_grid + flow[1]

                # Sample depth at destination points (need to handle out-of-bounds)
                x_dest_clipped = np.clip(x_dest, 0, w - 1).astype(int)
                y_dest_clipped = np.clip(y_dest, 0, h - 1).astype(int)

                # Get depth for next frame
                next_depth = actual_depths[frame_idx + 1]
                if next_depth.shape != (h, w):
                    from skimage.transform import resize

                    next_depth = resize(next_depth, (h, w), preserve_range=True)

                # Sample depth at destination points
                dest_depth = next_depth[y_dest_clipped, x_dest_clipped]

                # Get 3D points for destination
                dest_points_3d = unproject_points(
                    x_dest, y_dest, dest_depth, cx=cx, cy=cy
                )

                # Calculate 3D motion vectors
                motion_vectors_3d = dest_points_3d - points_3d

                all_points_3d.append(points_3d)
                all_motion_vectors_3d.append(motion_vectors_3d)
                all_colors.append(colors)

                # Update progress bar
                pbar.update(1)
                if on_frame_progress is not None:
                    on_frame_progress(frame_idx + 1, num_frames)

    flow_seconds = time.perf_counter() - t_flow0
    if not verbose:
        u(f"  ... finished optical flow + 3D motion  ({num_frames} frame pairs)")

    # Apply subsampling if requested
    if subsample < 1.0:
        step(5, f"Subsample spatial grid (rate={subsample})")
        _vlog(verbose, f"Step 6: Applying subsampling with rate {subsample}...")

        # Calculate step sizes for even sampling
        h_step = max(1, int(1.0 / np.sqrt(subsample)))
        w_step = max(1, int(1.0 / np.sqrt(subsample)))

        # Create subsampling indices
        h_indices = np.arange(0, h, h_step)
        w_indices = np.arange(0, w, w_step)

        # Apply subsampling to all data
        all_points_3d = [
            points_3d[h_indices][:, w_indices] for points_3d in all_points_3d
        ]
        all_motion_vectors_3d = [
            motion_3d[h_indices][:, w_indices] for motion_3d in all_motion_vectors_3d
        ]
        all_colors = [colors[h_indices][:, w_indices] for colors in all_colors]

        # Update dimensions after subsampling
        h_subsampled, w_subsampled = len(h_indices), len(w_indices)
        cx_subsampled = w_subsampled / 2
        cy_subsampled = h_subsampled / 2

        _vlog(verbose, f"   Subsampled from {h}x{w} to {h_subsampled}x{w_subsampled}")
    else:
        h_subsampled, w_subsampled = h, w
        cx_subsampled, cy_subsampled = cx, cy

    # Store camera intrinsics for reprojection
    fx, fy = 520, 520  # Focal lengths used in unproject_points
    intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx_subsampled,
        "cy": cy_subsampled,
        "width": w_subsampled,
        "height": h_subsampled,
    }

    # Final step: save (four keys, float64, np.savez uncompressed)
    save_idx = total_steps
    step(save_idx, "Save — write NPZ (float64, uncompressed)")
    t_save0 = time.perf_counter()
    _vlog(verbose, "Step 5: Saving results...")

    data_to_save = {
        "points_3d": np.array(all_points_3d, dtype=np.float64),
        "motion_vectors_3d": np.array(all_motion_vectors_3d, dtype=np.float64),
        "colors": np.array(all_colors, dtype=np.uint8),
        "intrinsics": intrinsics,
    }

    if output_format == "npz":
        np.savez(
            output_path,
            points_3d=data_to_save["points_3d"],
            motion_vectors_3d=data_to_save["motion_vectors_3d"],
            colors=data_to_save["colors"],
            intrinsics=data_to_save["intrinsics"],
        )
        u(f"  ✓ {output_path}")
        _vlog(verbose, f"✅ Saved NPZ file to {output_path}")
    elif output_format == "pkl":
        # Save as pickle format
        import pickle

        with open(output_path, "wb") as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        u(f"  ✓ {output_path}")
        _vlog(verbose, f"✅ Saved pickle file to {output_path}")

    save_seconds = time.perf_counter() - t_save0
    timings = Motion3DTimings(
        depth_seconds=depth_seconds,
        depth_post_seconds=depth_post_seconds,
        load_rgb_seconds=load_rgb_seconds,
        flow_seconds=flow_seconds,
        save_seconds=save_seconds,
        num_frames=num_depth_frames,
        num_frame_pairs=num_frames,
    )

    _vlog(verbose, f"Results saved to {output_path}")
    _vlog(verbose, "Output contains:")
    _vlog(verbose, f"  - points_3d: {np.array(all_points_3d).shape}")
    _vlog(verbose, f"  - motion_vectors_3d: {np.array(all_motion_vectors_3d).shape}")
    _vlog(verbose, f"  - colors: {np.array(all_colors).shape}")
    _vlog(verbose, f"  - intrinsics: {intrinsics}")

    if return_timings:
        return output_path, timings
    return output_path


def process_directory(
    input_dir: str,
    output_dir: str,
    encoder: str = "vitl",
    input_size: int = 518,
    max_res: int = 1280,
    max_len: int = -1,
    target_fps: int = -1,
    fp32: bool = False,
    device: str = "cuda",
    skip_frames: int = 1,
    subsample: float = 1.0,
    output_format: str = "npz",
    verbose: bool = False,
) -> list:
    """
    Process all video files and image folders in a directory.

    Args:
        input_dir: Directory containing MP4 files and/or folders with image frames
        output_dir: Directory to save output NPZ/PKL files
        encoder: Video-Depth-Anything encoder to use
        input_size: Input size for depth estimation
        max_res: Maximum resolution for processing
        max_len: Maximum number of frames to process (-1 for no limit)
        target_fps: Target FPS for processing (-1 for original FPS)
        fp32: Whether to use float32 precision
        device: Device to use for processing
        skip_frames: Skip every N frames (1 = process all frames)
        subsample: Fraction of points to subsample (0.0 to 1.0, 1.0 = no subsampling)

    Returns:
        List of successfully processed output files
    """
    # Find all video files in the input directory
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

    # Find all directories that contain image frames
    image_folders = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and is_image_folder(item_path):
            image_folders.append(item_path)

    all_inputs = video_files + image_folders

    if not all_inputs:
        print(f"❌ No video files or image folders found in {input_dir}")
        return []

    _vlog(
        verbose,
        f"📁 Found {len(video_files)} video file(s) and {len(image_folders)} image folder(s) in {input_dir}",
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    successful_outputs = []
    failed_files = []

    for i, input_path in enumerate(all_inputs, 1):
        input_name = os.path.basename(input_path)
        input_type = "video" if is_video_file(input_path) else "image folder"
        _vlog(verbose, f"\n🎬 Processing {i}/{len(all_inputs)} ({input_type}): {input_name}")

        # Generate output filename
        if is_video_file(input_path):
            base_name = os.path.splitext(os.path.basename(input_path))[0]
        else:
            base_name = os.path.basename(input_path.rstrip("/"))
        output_path = os.path.join(output_dir, f"{base_name}_3d_data.{output_format}")

        try:
            result_path = process_video_to_3d_data(
                input_path=input_path,
                output_path=output_path,
                encoder=encoder,
                input_size=input_size,
                max_res=max_res,
                max_len=max_len,
                target_fps=target_fps,
                fp32=fp32,
                device=device,
                skip_frames=skip_frames,
                subsample=subsample,
                output_format=output_format,
                verbose=verbose,
            )
            successful_outputs.append(result_path)
            _vlog(verbose, f"✅ Success: {result_path}")

        except Exception as e:
            failed_files.append((input_path, str(e)))
            print(f"❌ Failed: {e}")

    # Summary
    if verbose:
        print("\n📊 Processing Summary:")
        print(f"   ✅ Successful: {len(successful_outputs)}")
        print(f"   ❌ Failed: {len(failed_files)}")
    elif successful_outputs or failed_files:
        print(
            f"Done: {len(successful_outputs)} ok, {len(failed_files)} failed → {output_dir}"
        )

    if failed_files and verbose:
        print("\n❌ Failed files:")
        for input_path, error in failed_files:
            print(f"   - {os.path.basename(input_path)}: {error}")
    elif failed_files and not verbose:
        for input_path, error in failed_files:
            print(f"❌ {os.path.basename(input_path)}: {error}")

    return successful_outputs


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Preprocess video(s) or image folder(s) to extract 3D motion vectors"
    )
    parser.add_argument(
        "input_path",
        help="Path to input video file, image folder, or directory containing videos/image folders",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Path to output NPZ/PKL file or directory (optional)",
    )
    parser.add_argument(
        "--encoder",
        choices=["vits", "vitb", "vitl"],
        default="vitl",
        help="Video-Depth-Anything encoder to use",
    )
    parser.add_argument(
        "--input-size", type=int, default=518, help="Input size for depth estimation"
    )
    parser.add_argument(
        "--max-res", type=int, default=1280, help="Maximum resolution for processing"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=-1,
        help="Maximum number of frames to process (-1 for no limit)",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=-1,
        help="Target FPS for processing (-1 for original FPS)",
    )
    parser.add_argument("--fp32", action="store_true", help="Use float32 precision")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use for processing",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all video files and image folders in input directory",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Skip every N frames (1 = process all frames, 2 = process every other frame)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of points to subsample (0.0 to 1.0, 1.0 = no subsampling, 0.1 = 10% of points)",
    )
    parser.add_argument(
        "--output-format",
        choices=["npz", "pkl"],
        default="npz",
        help="Output file format: 'npz' (NumPy archive, legacy layout) or 'pkl' (default: npz)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-step progress (depth, optical flow bar, NPZ details)",
    )

    args = parser.parse_args()

    # Validate subsample parameter
    if not (0.0 < args.subsample <= 1.0):
        print(f"❌ Error: subsample must be between 0.0 and 1.0, got {args.subsample}")
        sys.exit(1)

    try:
        if args.batch or os.path.isdir(args.input_path):
            # Batch processing mode
            if not os.path.isdir(args.input_path):
                print(f"❌ Error: {args.input_path} is not a directory")
                sys.exit(1)

            output_dir = args.output_path or os.path.join(args.input_path, "processed")

            successful_outputs = process_directory(
                input_dir=args.input_path,
                output_dir=output_dir,
                encoder=args.encoder,
                input_size=args.input_size,
                max_res=args.max_res,
                max_len=args.max_len,
                target_fps=args.target_fps,
                fp32=args.fp32,
                device=args.device,
                skip_frames=args.skip_frames,
                subsample=args.subsample,
                output_format=args.output_format,
                verbose=args.verbose,
            )

            if successful_outputs:
                if args.verbose:
                    print(
                        f"\n🎉 Batch processing completed! {len(successful_outputs)} files processed successfully."
                    )
            else:
                print("\n❌ No files were processed successfully.")
                sys.exit(1)

        else:
            # Single file/folder processing mode
            if not (is_video_file(args.input_path) or is_image_folder(args.input_path)):
                print(
                    f"❌ Error: {args.input_path} is not a video file or image folder"
                )
                sys.exit(1)

            output_path = process_video_to_3d_data(
                input_path=args.input_path,
                output_path=args.output_path,
                encoder=args.encoder,
                input_size=args.input_size,
                max_res=args.max_res,
                max_len=args.max_len,
                target_fps=args.target_fps,
                fp32=args.fp32,
                device=args.device,
                skip_frames=args.skip_frames,
                subsample=args.subsample,
                output_format=args.output_format,
                verbose=args.verbose,
            )
            print(f"Saved: {output_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

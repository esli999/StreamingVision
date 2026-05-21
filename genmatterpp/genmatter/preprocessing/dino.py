"""DINOv2 feature extraction + per-pixel PCA (shared by DAVIS and custom pipeline)."""

from __future__ import annotations

import gc
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

# DAVIS defaults (match experiments/davis/dino_extractor.py)
DEFAULT_TARGET_H = 520
DEFAULT_TARGET_W = 960
DEFAULT_PATCH_SIZE = 14
DEFAULT_N_COMPONENTS = 10


@dataclass
class DinoParams:
    target_h: int = DEFAULT_TARGET_H
    target_w: int = DEFAULT_TARGET_W
    patch_size: int = DEFAULT_PATCH_SIZE
    n_components: int = DEFAULT_N_COMPONENTS
    device: str = "cuda"


@dataclass
class DinoStepTimings:
    load_frames_seconds: float
    extract_seconds: float
    pca_seconds: float
    resize_save_seconds: float
    num_frames: int

    @property
    def extract_fps(self) -> float:
        return (
            self.num_frames / self.extract_seconds if self.extract_seconds > 0 else 0.0
        )

    @property
    def resize_fps(self) -> float:
        return (
            self.num_frames / self.resize_save_seconds
            if self.resize_save_seconds > 0
            else 0.0
        )


@dataclass
class DinoResult:
    output_path: Path
    num_frames: int
    elapsed_seconds: float
    fps: float
    file_size_bytes: int
    timings: DinoStepTimings


def load_dino_model(device: str = "cuda") -> tuple[torch.nn.Module, torch.device]:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="xFormers is available*",
            category=UserWarning,
        )
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(dev)
    return model, dev


def load_and_preprocess_frames(
    video_path: str | Path,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    progress: bool = True,
    on_frame: Optional[Callable[[int, int], None]] = None,
) -> tuple[list[np.ndarray], int, int] | tuple[None, None, None]:
    """Load frames from directory and resize to be divisible by patch size."""
    video_path = Path(video_path)
    frame_files = sorted(
        f for f in os.listdir(video_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    if not frame_files:
        return None, None, None

    frames: list[np.ndarray] = []
    iterator = frame_files
    if progress:
        iterator = tqdm(frame_files, desc="Loading frames")
    for i, frame_file in enumerate(iterator):
        frame_path = video_path / frame_file
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if on_frame is not None:
            on_frame(i + 1, len(frame_files))

    if not frames:
        return None, None, None

    h_orig, w_orig = frames[0].shape[:2]
    h = (h_orig // patch_size) * patch_size
    w = (w_orig // patch_size) * patch_size

    frames_resized: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        frames_resized.append(cv2.resize(frame, (w, h)))
        if on_frame is not None:
            on_frame(i + 1, len(frames))

    return frames_resized, h, w


def extract_dino_features(
    frames: list[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    *,
    on_frame: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """Extract DINO patch features for all frames."""
    all_patch_features: list[torch.Tensor] = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    n = len(frames)
    for i, frame in enumerate(frames):
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_tensor = (img_tensor - mean) / std
        with torch.no_grad():
            features = model.forward_features(img_tensor)
            patch_features = features["x_norm_patchtokens"]
        all_patch_features.append(patch_features.cpu())
        if on_frame is not None:
            on_frame(i + 1, n)

    return torch.cat(all_patch_features, dim=1)


def apply_pca(features_np: np.ndarray, n_components: int = 10):
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features_np)
    pca_unnormalized = pca_features
    pca_rgb = pca_features[:, :3].copy()
    for i in range(3):
        col_min = pca_rgb[:, i].min()
        col_max = pca_rgb[:, i].max()
        denom = col_max - col_min
        if denom > 0:
            pca_rgb[:, i] = (pca_rgb[:, i] - col_min) / denom
    return pca, pca_unnormalized, pca_rgb


def save_npz_memory_efficient(
    video_name: str,
    pca_unnormalized: np.ndarray,
    num_frames: int,
    h: int,
    w: int,
    pca: PCA,
    output_dir: str | Path,
    *,
    target_h: int = DEFAULT_TARGET_H,
    target_w: int = DEFAULT_TARGET_W,
    patch_size: int = DEFAULT_PATCH_SIZE,
    on_frame: Optional[Callable[[int, int], None]] = None,
) -> Path | None:
    patches_h = h // patch_size
    patches_w = w // patch_size
    num_patches_per_frame = patches_h * patches_w
    final_shape = (num_frames, target_h, target_w, pca_unnormalized.shape[-1])
    try:
        pca_frames_resized = np.empty(final_shape, dtype=np.float32)
        for i in range(num_frames):
            start_idx = i * num_patches_per_frame
            end_idx = start_idx + num_patches_per_frame
            frame_pca_unnorm = pca_unnormalized[start_idx:end_idx].reshape(patches_h, patches_w, -1)
            frame_pca_unnorm_upsampled = cv2.resize(
                frame_pca_unnorm, (w, h), interpolation=cv2.INTER_LINEAR
            )
            frame_resized = cv2.resize(
                frame_pca_unnorm_upsampled,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )
            pca_frames_resized[i] = frame_resized
            if on_frame is not None:
                on_frame(i + 1, num_frames)
    except Exception:
        traceback.print_exc()
        return None

    pca_flat = pca_frames_resized.reshape(-1, pca_frames_resized.shape[-1])
    gaussian_means = np.mean(pca_flat, axis=0)
    gaussian_stds = np.std(pca_flat, axis=0)
    del pca_flat

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_name}_dino_pca_per_pixel.npz"
    np.savez(
        output_path,
        pca_features_unnormalized=pca_frames_resized,
        gaussian_means=gaussian_means,
        gaussian_stds=gaussian_stds,
        components=pca.components_,
        mean=pca.mean_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    return output_path


def process_video_dino(
    video_name: str,
    base_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: str | Path,
    params: DinoParams | None = None,
) -> bool:
    """DAVIS-style batch helper: process one video folder. Returns success."""
    params = params or DinoParams()
    output_path = Path(output_dir) / f"{video_name}_dino_pca_per_pixel.npz"
    if output_path.is_file():
        print(f"  ✓ {video_name}: Already processed, skipping")
        return True

    video_path = Path(base_path) / video_name
    if not video_path.is_dir():
        print(f"  ✗ {video_name}: Directory not found")
        return False

    try:
        frames, h, w = load_and_preprocess_frames(
            video_path, patch_size=params.patch_size
        )
        if frames is None:
            print(f"  ✗ {video_name}: No frames found")
            return False

        all_patch_features = extract_dino_features(frames, model, device)
        features_np = all_patch_features.squeeze(0).numpy()
        pca, pca_unnormalized, _pca_rgb = apply_pca(features_np, params.n_components)
        del all_patch_features, features_np
        gc.collect()

        out = save_npz_memory_efficient(
            video_name,
            pca_unnormalized,
            len(frames),
            h,
            w,
            pca,
            output_dir,
            target_h=params.target_h,
            target_w=params.target_w,
            patch_size=params.patch_size,
        )
        del pca_unnormalized, pca
        gc.collect()
        if out is None:
            return False
        print(f"  ✓ {video_name}: Processing complete → {out}")
        return True
    except Exception as e:
        print(f"  ✗ {video_name}: Error - {e}")
        traceback.print_exc()
        return False


def run_dino_extraction(
    video_id: str,
    frames_dir: Path,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    params: DinoParams,
    *,
    on_extract_frame: Optional[Callable[[int, int], None]] = None,
    on_save_frame: Optional[Callable[[int, int], None]] = None,
) -> DinoResult:
    """Run DINO + PCA for one video; returns timing and output path."""
    t0 = time.perf_counter()
    t_load0 = time.perf_counter()
    frames, h, w = load_and_preprocess_frames(
        frames_dir,
        patch_size=params.patch_size,
        progress=False,
    )
    load_frames_seconds = time.perf_counter() - t_load0
    if frames is None or h is None or w is None:
        raise RuntimeError(f"No frames in {frames_dir}")

    t_extract0 = time.perf_counter()
    all_patch_features = extract_dino_features(
        frames, model, device, on_frame=on_extract_frame
    )
    extract_seconds = time.perf_counter() - t_extract0
    features_np = all_patch_features.squeeze(0).numpy()
    t_pca0 = time.perf_counter()
    pca, pca_unnormalized, _ = apply_pca(features_np, params.n_components)
    pca_seconds = time.perf_counter() - t_pca0
    del all_patch_features, features_np
    gc.collect()

    n_frames = len(frames)
    t_save0 = time.perf_counter()
    output_path = save_npz_memory_efficient(
        video_id,
        pca_unnormalized,
        n_frames,
        h,
        w,
        pca,
        output_dir,
        target_h=params.target_h,
        target_w=params.target_w,
        patch_size=params.patch_size,
        on_frame=on_save_frame,
    )
    resize_save_seconds = time.perf_counter() - t_save0
    del frames, pca_unnormalized, pca
    gc.collect()

    if output_path is None:
        raise RuntimeError("DINO NPZ save failed")

    elapsed = time.perf_counter() - t0
    step_timings = DinoStepTimings(
        load_frames_seconds=load_frames_seconds,
        extract_seconds=extract_seconds,
        pca_seconds=pca_seconds,
        resize_save_seconds=resize_save_seconds,
        num_frames=n_frames,
    )
    return DinoResult(
        output_path=Path(output_path),
        num_frames=n_frames,
        elapsed_seconds=elapsed,
        fps=n_frames / elapsed if elapsed > 0 else 0.0,
        file_size_bytes=Path(output_path).stat().st_size,
        timings=step_timings,
    )

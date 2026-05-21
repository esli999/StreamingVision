# GenMatter Tracking with DINO Features - Batch Processing
# Clean implementation for running experiments on multiple videos

import os
import json
import time
import gc
import jax
import jax.numpy as jnp
from jax.random import key as jkey

import numpy as np
import pickle
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import config

from genmatter.datatypes import *
from genmatter.model_3d import *
from genmatter.inference import *
from genmatter.dataloader import *
from genmatter.utils import *
from genmatter.evaluation import *
from genmatter.bootstrap_stats import (
    BOOTSTRAP_N_SAMPLES,
    BOOTSTRAP_RANDOM_SEED,
    bootstrap_mean_ci_95,
)

import genjax
from genjax import Const, gen, Pytree

from genmatter.tracking.dino import (
    DinoTrackingHyperparams,
    DinoTrackingInputs,
    DinoTrackingParams,
    compile_dino_tracking_program,
    configure_jax_cache,
    run_dino_tracking,
)

# ============================================================================
# Configuration
# ============================================================================

VIDEO_NAMES = list(config.TAPVID_DAVIS_VIDEO_NAMES)

USE_SAM_FRAME0 = True
SAVE_3WIDE_VIDEO = False  # overridden by davis_run_cli when run as __main__
MEASURE_FPS = True
SKIP_COMPLETED = False  # overwritten by davis_run_cli.configure_experiment_module

# Paths
DAVIS_3D_MOTION_PATH = str(config.DAVIS_3D_MOTION_PATH)
DAVIS_SEGMASKS_PATH = str(config.DAVIS_SEGMASKS_PATH)
DAVIS_RGB_PATH = str(config.DAVIS_RGB_PATH)
DINO_PATH_TEMPLATE = str(config.DAVIS_DINO_PATH / '{}_dino_pca_per_pixel.npz')
SAM_FRAME0_PATH_TEMPLATE = str(config.DAVIS_SAM_FRAME0_PATH / '{}_SAM_frame0.png')

# Output directory
EXPERIMENT_SAVE_DIR = str(config.DAVIS_TRACKING_OUTPUT_DIR)

# Model hyperparameters
NUM_BLOBS = 500
NUM_HYPERBLOBS_ORIGINAL = 9
FOCAL_LENGTH = 520.0
BLOB_COUNTING_THRESHOLD = 0
RANDOM_SEED = 42
_DATAPOINT_RETAIN_PCT = 0.78125

configure_jax_cache()


def _davis_tracking_params() -> DinoTrackingParams:
    return DinoTrackingParams(
        num_blobs=NUM_BLOBS,
        num_hyperblobs=NUM_HYPERBLOBS_ORIGINAL,
        datapoint_retain_pct=_DATAPOINT_RETAIN_PCT,
        random_seed=RANDOM_SEED,
        focal_length=FOCAL_LENGTH,
        use_sam_frame0=USE_SAM_FRAME0,
        init_gibbs_sweeps=15,
        tracking_outlier_prob=1e-28,
        measure_fps=MEASURE_FPS,
        hyperparams=DinoTrackingHyperparams(
            sigma_F=0.2,
            outlier_prob=5.0,
            outlier_velocity_gamma_shape=5.0,
            outlier_velocity_gamma_rate=1.0,
            alpha=1.0,
            beta=1.0,
            sigma_H=(10 * 0.5) ** 2,
            sigma_V=10e14,
            translation_gaussian_scale=0.2,
            translation_max_radius=0.35,
            translation_num_radii_cells=15,
            translation_theta_step_deg=15.0,
            rotation_vmf_kappa=100.0,
            rotation_angle_max_deg=25.0,
            rotation_angle_step_deg=0.375,
        ),
    )


def create_3wide_video(tracking_data, video_name, rgb_path, img_dims, segmentation_masks,
                       num_blobs, save_path):
    """Create 3-wide synchronized video."""
    from glob import glob

    rgb_dir = os.path.join(rgb_path, video_name)
    rgb_files = sorted(glob(os.path.join(rgb_dir, "*.jpg")))

    if len(rgb_files) == 0:
        print(f"Warning: No RGB frames found for {video_name}")
        return

    img_height, img_width = img_dims
    # Seed numpy RNG for consistent visualization colors
    np.random.seed(RANDOM_SEED)
    blob_colors = np.random.randint(0, 255, size=(num_blobs, 3), dtype=np.uint8)
    combined_frames = []

    num_frames = min(len(tracking_data), len(rgb_files))

    # Determine object particles from frame 0 using segmentation mask overlap
    frame0 = tracking_data[0]
    blob_assignments_frame0 = frame0['blob_assignments']
    n_blobs_frame0 = frame0['n_blobs']
    gt_mask_frame0 = segmentation_masks[0]
    
    # Count pixels per blob in frame 0
    blob_pixel_counts_frame0 = np.bincount(
        blob_assignments_frame0[blob_assignments_frame0 < n_blobs_frame0],
        minlength=n_blobs_frame0
    )
    
    # Determine which blobs are object vs background based on mask overlap
    # Pixels assigned to each blob vs mask overlap
    object_blobs_frame0 = set()
    background_blobs_frame0 = set()
    
    for blob_idx in range(n_blobs_frame0):
        # Find pixels assigned to this blob
        blob_pixel_mask = (blob_assignments_frame0 == blob_idx)
        
        # Check if any of these pixels are on the segmentation mask
        pixels_on_mask = np.sum(gt_mask_frame0[blob_pixel_mask])
        
        if pixels_on_mask > 0:
            object_blobs_frame0.add(blob_idx)
        else:
            background_blobs_frame0.add(blob_idx)

    for frame_idx in tqdm(range(num_frames), desc="Rendering 3-wide video frames"):
        rgb_frame = cv2.imread(rgb_files[frame_idx])
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        frame = tracking_data[frame_idx]
        blob_assignments = frame['blob_assignments']
        n_blobs = frame['n_blobs']

        valid_mask = blob_assignments < n_blobs
        
        # Create binary mask based on object particles identified from frame 0
        # Convert object_blobs_frame0 set to array for vectorized operations
        object_blobs_array = np.array(list(object_blobs_frame0))
        
        # Vectorized check: for each blob_assignment, check if it's in object_blobs_frame0
        object_blob_mask = valid_mask & np.isin(blob_assignments, object_blobs_array)
        binary_mask = np.where(object_blob_mask, 255, 0).astype(np.uint8)
        
        background_color = np.array([128, 128, 128], dtype=np.uint8)

        blob_assignment_frame = np.zeros((len(blob_assignments), 3), dtype=np.uint8)
        blob_assignment_frame[~object_blob_mask] = background_color
        blob_assignment_frame[object_blob_mask] = blob_colors[blob_assignments[object_blob_mask]]
        
        # Reshape to image dimensions
        binary_image = binary_mask.reshape(img_height, img_width)
        blob_assignment_frame = blob_assignment_frame.reshape(img_height, img_width, 3)

        # Create segmentation frame from binary image
        segmentation_frame = np.stack([binary_image, binary_image, binary_image], axis=-1)

        rgb_h, rgb_w = rgb_frame.shape[:2]
        if (rgb_h, rgb_w) != (img_height, img_width):
            segmentation_frame = cv2.resize(segmentation_frame, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
            blob_assignment_frame = cv2.resize(blob_assignment_frame, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)

        SCALE_FACTOR = 0.5
        target_w = int(rgb_w * SCALE_FACTOR)
        target_h = int(rgb_h * SCALE_FACTOR)
        rgb_frame = cv2.resize(rgb_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        segmentation_frame = cv2.resize(segmentation_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        blob_assignment_frame = cv2.resize(blob_assignment_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        combined_frame = np.concatenate([rgb_frame, segmentation_frame, blob_assignment_frame], axis=1)
        combined_frames.append(combined_frame)

    # Write frames to temporary location, then use ffmpeg subprocess to encode with H.264
    # This avoids JAX fork issues since we call ffmpeg AFTER all JAX computation is done
    if len(combined_frames) == 0:
        print(f"Warning: No frames to write for {save_path}")
        return
    else:
        print(f"Writing {len(combined_frames)} frames to temporary location")

    import tempfile
    import subprocess
    import shutil

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    try:
        # Write frames as PNGs
        for i, frame in tqdm(enumerate(combined_frames), desc="Writing frames to temporary location"):
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)

        # Use ffmpeg to create H.264 video
        # This is safe to call after JAX initialization since all compute is done
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',
            '-i', os.path.join(temp_dir, 'frame_%05d.png'),
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Force even dimensions
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            save_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Saved 3-wide video: {save_path}")
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)

# ============================================================================
# Cleanup Function
# ============================================================================

def cleanup_between_runs():
    """Clear JAX state, environment variables, and memory between experimental runs."""
    # This function helps prevent memory leaks that can accumulate across runs
    
    # Clear matplotlib figure cache to prevent memory leaks
    plt.close('all')
    
    # Clear OpenCV cache if any
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    # Clear JAX device memory explicitly
    # Note: PJRT backend doesn't support explicit memory release operations
    # (device.clear_cache() and defragment() both cause fatal errors).
    # JAX's automatic memory management + Python GC will handle memory cleanup.
    # Explicit deletion of variables + multiple GC cycles is sufficient.
    
    # Do not call jax.clear_caches() here: it forces JIT recompilation on every video.
    # Persistent + in-memory JAX caches are reused via compile_dino_tracking_program().

    # Force garbage collection multiple times to ensure cleanup
    # Run multiple times because some objects may have circular references
    for _ in range(5):  # Increased from 3 to 5 for more thorough cleanup
        gc.collect()

# ============================================================================
# Main Processing Function
# ============================================================================

def process_video(video_name):
    """Process a single video and return results."""
    print(f"\n{'='*80}")
    print(f"Processing: {video_name}")
    print(f"{'='*80}")

    try:
        dino_path = Path(DINO_PATH_TEMPLATE.format(video_name))
        if not dino_path.is_file():
            print(f"DINO features not found: {dino_path}")
            return None

        motion_npz = Path(DAVIS_3D_MOTION_PATH) / f"{video_name}_3d_motion.npz"
        if not motion_npz.is_file():
            alt = Path(DAVIS_3D_MOTION_PATH) / f"{video_name}_3d_data.npz"
            motion_npz = alt if alt.is_file() else motion_npz

        with np.load(motion_npz) as motion_probe:
            _T, _H, _W, _ = motion_probe["points_3d"].shape
        img_dims_probe = (_H, _W)

        first_frame_seg = get_segmentation_mask(
            video_name, 0, DAVIS_SEGMASKS_PATH, img_dims=img_dims_probe, flatten=True
        )

        max_frames = None
        if video_name == "jello_trim":
            max_frames = 50

        sam_png = None
        if USE_SAM_FRAME0:
            sam_png = Path(SAM_FRAME0_PATH_TEMPLATE.format(video_name))

        inputs = DinoTrackingInputs(
            video_id=video_name,
            motion_npz=motion_npz,
            dino_npz=dino_path,
            sam_frame0_png=sam_png,
            segmentation_mask_frame0=first_frame_seg,
            max_frames=max_frames,
        )

        print("Running DINO tracking pipeline...")
        track_result = run_dino_tracking(
            inputs, _davis_tracking_params(), compiled=compiled_program
        )
        tracking_data = track_result.tracking_data
        img_dims = track_result.img_dims
        fps = track_result.timings.tracking_fps if MEASURE_FPS else None

        if video_name == "jello_trim":
            print(f"Limited to first {len(tracking_data)} frames for jello_trim")

        print("Computing error rates...")
        segmentation_masks = []
        for frame_idx in range(len(tracking_data)):
            seg_mask = get_segmentation_mask(
                video_name, frame_idx, DAVIS_SEGMASKS_PATH, img_dims=img_dims, flatten=True
            )
            segmentation_masks.append(seg_mask)

        all_results = {video_name: [tracking_data]}
        experiment_metrics, best_visualization_data = evaluate_single_davis_video(
            davis_name=video_name,
            multiple_genmatter_list=all_results[video_name],
            annotations_path=DAVIS_SEGMASKS_PATH,
            counting_threshold=BLOB_COUNTING_THRESHOLD,
            img_dims=img_dims,
            fps_list=None,
            render_results_video=False,
            experiment_save_dir=None,
            force_below_count_thresh_as_outlier=True,
            subsampled_indices=None,
        )

        result = {
            'video_name': video_name,
            'pixel_metrics': experiment_metrics,
            'tracking_data': tracking_data,
            'segmentation_masks': segmentation_masks,
            'img_dims': img_dims,
            'fps': fps,
        }

        print(f"Completed: {video_name}")

        if MEASURE_FPS and fps is not None:
            print(f"  FPS: {fps:.2f} frames/second")

        print(f"\n  MATTER-WEIGHTED (frame-0 blob weights) — primary DAVIS metric:")
        pm = experiment_metrics
        print(f"    Recall:     {pm['avg_matter_weighted_recall_fixed']:.3f}")
        print(f"    Precision:  {pm['avg_matter_weighted_precision_fixed']:.3f}")
        print(f"    Jaccard:    {pm['avg_matter_weighted_jaccard_fixed']:.3f}")

        return result

    except Exception as e:
        print(f"Error processing {video_name}: {e}")
        import traceback
        traceback.print_exc()
        return None



# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    _davis_dir = _Path(__file__).resolve().parent
    if str(_davis_dir) not in sys.path:
        sys.path.insert(0, str(_davis_dir))
    import davis_run_cli

    _parser = argparse.ArgumentParser(description="DAVIS DINO tracking")
    davis_run_cli.add_frame0_init_args(_parser)
    davis_run_cli.add_save_3wide_video_args(_parser)
    davis_run_cli.add_skip_completed_args(_parser)
    _args = _parser.parse_args()
    davis_run_cli.configure_experiment_module(sys.modules[__name__], _args, "tracking")

    os.makedirs(EXPERIMENT_SAVE_DIR, exist_ok=True)

    all_results = []
    all_accuracies = {}

    print(f"{'='*80}")
    print(f"DINO TRACKING EXPERIMENT - Processing {len(VIDEO_NAMES)} videos")
    print(f"  SAM frame-0 init: {USE_SAM_FRAME0}")
    print(f"  Save 3-wide videos: {SAVE_3WIDE_VIDEO}")
    if MEASURE_FPS:
        print(f"  FPS measurement: ENABLED")
    else:
        print(f"  FPS measurement: DISABLED")
    print(f"  Output directory: {EXPERIMENT_SAVE_DIR}")
    if SKIP_COMPLETED:
        print(f"  Skip completed videos: YES (existing json_results/*_results.json)")
    print(f"{'='*80}\n")

    json_results_dir = os.path.join(EXPERIMENT_SAVE_DIR, "json_results")
    os.makedirs(json_results_dir, exist_ok=True)

    compiled_program = compile_dino_tracking_program()
    print(f"  Compiled program max_frames: {compiled_program.max_frames}")
    print(f"  JAX compile-once: enabled (no per-video jax.clear_caches)")

    for video_name in tqdm(VIDEO_NAMES, desc="Overall Progress", position=0):
        per_run_json_path = os.path.join(json_results_dir, f"{video_name}_results.json")
        if SKIP_COMPLETED and os.path.isfile(per_run_json_path):
            with open(per_run_json_path, "r") as f:
                all_accuracies[video_name] = json.load(f)
            print(f"[skip-completed] {video_name}")
            json_path = os.path.join(EXPERIMENT_SAVE_DIR, "all_videos_experiment_results.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    existing_results = json.load(f)
                existing_results.update(all_accuracies)
                all_accuracies_to_save = existing_results
            else:
                all_accuracies_to_save = dict(all_accuracies)
            with open(json_path, "w") as f:
                json.dump(all_accuracies_to_save, f, indent=2)
            continue

        result = process_video(video_name)

        if result is not None:
            all_results.append(result)

            if SAVE_3WIDE_VIDEO:
                videos_dir = os.path.join(EXPERIMENT_SAVE_DIR, "3wide_videos")
                os.makedirs(videos_dir, exist_ok=True)
                video_path = os.path.join(videos_dir, f"{video_name}_3wide_synchronized.mp4")
                create_3wide_video(
                    result['tracking_data'], video_name, DAVIS_RGB_PATH,
                    result['img_dims'], result['segmentation_masks'],
                    NUM_BLOBS, video_path
                )
                print(f"Saved 3-wide video: {video_path}")

            pm = result['pixel_metrics']
            tracking_data = result['tracking_data']

            # Store metrics (GenMatter / DAVIS benchmark: matter-weighted fixed R/P/J)
            all_accuracies[video_name] = {
                'pixel_metrics': pm,

                'particle_count_matter_fixed_recall': pm['avg_matter_weighted_recall_fixed'],
                'particle_count_matter_fixed_precision': pm['avg_matter_weighted_precision_fixed'],
                'particle_count_matter_fixed_jaccard': pm['avg_matter_weighted_jaccard_fixed'],
                'particle_count_matter_fixed_accuracy': pm['avg_matter_weighted_accuracy_fixed'],
                'fps': result.get('fps'),
            }

            # Save per-run JSON
            per_run_json_path = os.path.join(json_results_dir, f"{video_name}_results.json")
            with open(per_run_json_path, 'w') as f:
                json.dump(all_accuracies[video_name], f, indent=2)
            print(f"Saved per-run JSON: {per_run_json_path}")

            # Save aggregated JSON after each video completes (incremental updates)
            json_path = os.path.join(EXPERIMENT_SAVE_DIR, "all_videos_experiment_results.json")

            # Read existing results if file exists, then merge with new results
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    existing_results = json.load(f)
                existing_results.update(all_accuracies)
                all_accuracies_to_save = existing_results
            else:
                all_accuracies_to_save = all_accuracies

            with open(json_path, 'w') as f:
                json.dump(all_accuracies_to_save, f, indent=2)
            print(f"Saved experiment results (updated): {json_path}")

            # Remove large tracking_data from result to free memory
            # Since 3wide video is disabled, we don't need to keep it in memory
            # The metrics are already saved to JSON
            del result['tracking_data']
            del tracking_data

        # Clean up environment variables and memory between runs
        # This prevents memory leaks from accumulating across experimental runs
        cleanup_between_runs()

    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Total videos processed: {len(all_results)}/{len(VIDEO_NAMES)}")
    print(f"Failed videos: {len(VIDEO_NAMES) - len(all_results)}")
    print(f"Results saved to: {EXPERIMENT_SAVE_DIR}")

    # Load all results from JSON file to compute aggregate statistics
    json_path = os.path.join(EXPERIMENT_SAVE_DIR, "all_videos_experiment_results.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            all_accuracies_from_json = json.load(f)
    else:
        all_accuracies_from_json = all_accuracies

    # Filter out incomplete results (videos that failed to process)
    valid_results = {
        k: v
        for k, v in all_accuracies_from_json.items()
        if 'particle_count_matter_fixed_jaccard' in v
    }

    print(f"\n{'='*80}")
    print(f"SUMMARY - {len(valid_results)} Videos")
    print(f"{'='*80}")
    print(f"\nMetric guide: matter-weighted recall / precision / Jaccard use frame-0 blob weights (primary DAVIS metric).")
    print(f"\n")

    # Aggregate statistics
    
    # if len(valid_results) == 0:
    #     print("⚠️  No valid results found in JSON file. Cannot compute aggregate statistics.")
    #     return
    
    particle_count_matter_fixed_recall = [m['particle_count_matter_fixed_recall'] for m in valid_results.values()]
    particle_count_matter_fixed_precision = [m['particle_count_matter_fixed_precision'] for m in valid_results.values()]
    particle_count_matter_fixed_jaccard = [m['particle_count_matter_fixed_jaccard'] for m in valid_results.values()]
    particle_count_matter_fixed_accuracy = [m['particle_count_matter_fixed_accuracy'] for m in valid_results.values()]

    fps_values = [m.get('fps') for m in valid_results.values() if m.get('fps') is not None]

    print(f"AGGREGATE STATISTICS ACROSS ALL VIDEOS:")
    print(
        f"  (95% CIs: percentile bootstrap on video means, B={BOOTSTRAP_N_SAMPLES}, seed={BOOTSTRAP_RANDOM_SEED})"
    )

    print(f"\n  MATTER-WEIGHTED FIXED (primary DAVIS metric):")
    for label, arr in [
        ("Recall", particle_count_matter_fixed_recall),
        ("Precision", particle_count_matter_fixed_precision),
        ("Jaccard", particle_count_matter_fixed_jaccard),
        ("Accuracy", particle_count_matter_fixed_accuracy),
    ]:
        m, lo, hi = bootstrap_mean_ci_95(arr)
        print(f"    {label:18s} {m:.3f} [{lo:.3f}, {hi:.3f}]")

    if fps_values:
        print(f"\n  5. PERFORMANCE METRICS:")
        fm, flo, fhi = bootstrap_mean_ci_95(fps_values)
        print(f"    FPS:                     {fm:.2f} [{flo:.2f}, {fhi:.2f}] frames/second")
        print(f"    Videos with FPS data:    {len(fps_values)}/{len(valid_results)}")
    else:
        print(f"\n  5. PERFORMANCE METRICS:")
        print(f"    FPS:                     No FPS data available")

    print(f"\n\nPER-VIDEO RESULTS:")
    print(f"{'='*80}\n")

    for video_name, metrics in valid_results.items():
        print(f"  {video_name}:")

        print(f"    Matter-weighted fixed:")
        print(f"      Recall:                  {metrics['particle_count_matter_fixed_recall']:.3f}")
        print(f"      Precision:               {metrics['particle_count_matter_fixed_precision']:.3f}")
        print(f"      Jaccard:                 {metrics['particle_count_matter_fixed_jaccard']:.3f}")
        print(f"      Accuracy:                {metrics['particle_count_matter_fixed_accuracy']:.3f}")

        video_fps = metrics.get('fps')
        if video_fps is not None:
            print(f"\n    5. PERFORMANCE METRICS:")
            print(f"      FPS:                     {video_fps:.2f} frames/second")
        else:
            print(f"\n    5. PERFORMANCE METRICS:")
            print(f"      FPS:                     Not measured")

        print(f"")

"""
CoTracker Baseline: Run on all DAVIS videos
Initializes 500 random points in first frame and tracks them through each video.
Computes particle-based accuracy metrics matching dino_tracking.py methodology.
"""

import os
import sys
import numpy as np
import torch
import cv2
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import json
import tempfile
import subprocess
import shutil
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))))
import config
from genmatter.bootstrap_stats import (
    BOOTSTRAP_N_SAMPLES,
    BOOTSTRAP_RANDOM_SEED,
    bootstrap_mean_ci_95,
)
from genmatter.evaluation import get_segmentation_mask

# ============================================================================
# Configuration
# ============================================================================

VIDEO_NAMES = list(config.TAPVID_DAVIS_VIDEO_NAMES)

DAVIS_RGB_PATH = str(config.DAVIS_RGB_PATH)
DAVIS_SEGMASKS_PATH = str(config.DAVIS_SEGMASKS_PATH)
MOTION_DATA_PATH = str(config.DAVIS_3D_MOTION_PATH)
NUM_RANDOM_POINTS = 500
RANDOM_SEED = 42
OUTPUT_DIR = str(config.COTRACKER_OUTPUT_DIR)

# New parameter: Number of particles to initialize on mask
NUM_INIT_PARTICLES_ON_MASK = None  # Set to e.g. 300 for "object", None for default behavior
SAVE_COTRACKER_OVERLAY_ON_SEGMASK_VIDEO = False

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "videos"), exist_ok=True)

# Set random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ============================================================================
# Load CoTracker Model Once
# ============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading CoTracker model on {device}...")
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
print("CoTracker model loaded successfully\n")

# ============================================================================
# Helper Functions for Video Generation
# ============================================================================

def batch_compute_classifications(pred_tracks_np, pred_visibility_np, object_particles_frame0, 
                                background_particles_frame0, video_name, img_dims, batch_size=32):
    """Batch compute TP/TN/FP/FN classifications for all frames."""
    num_frames = len(pred_tracks_np)
    img_height, img_width = img_dims
    
    # Pre-allocate arrays for all classifications
    all_classifications = np.zeros((num_frames, len(pred_tracks_np[0])), dtype=int)
    # 0: TN, 1: TP, 2: FP, 3: FN, -1: Not counted (occluded)
    
    def process_batch(batch_start, batch_end):
        batch_classifications = []
        
        for frame_idx in range(batch_start, min(batch_end, num_frames)):
            frame_coords = pred_tracks_np[frame_idx]
            frame_visibility = pred_visibility_np[frame_idx]
            
            # Load segmentation mask for this frame
            seg_mask = get_segmentation_mask(
                video_name, frame_idx, DAVIS_SEGMASKS_PATH, img_dims=img_dims, flatten=True
            )
            
            # Get pixel coordinates
            x_2d = np.clip(frame_coords[:, 0].astype(int), 0, img_width - 1)
            y_2d = np.clip(frame_coords[:, 1].astype(int), 0, img_height - 1)
            pixel_indices = y_2d * img_width + x_2d
            
            # Check if particles are in mask
            in_mask = np.zeros(len(frame_coords), dtype=bool)
            valid_indices = pixel_indices < len(seg_mask)
            in_mask[valid_indices] = seg_mask[pixel_indices[valid_indices]]
            
            # Classify each particle
            classifications = np.zeros(len(frame_coords), dtype=int)
            
            # Object particles (only consider visible points for TP/FN)
            obj_visibility = frame_visibility[object_particles_frame0]
            obj_visible = obj_visibility >= 0.5
            obj_in_mask = in_mask[object_particles_frame0]
            
            # Only classify visible object particles
            classifications[object_particles_frame0] = np.where(
                obj_visible,
                np.where(obj_in_mask, 1, 3),  # TP or FN for visible
                -1  # Not counted for occluded
            )
            
            # Background particles (exclude occluded points from FP/TN)
            bg_visibility = frame_visibility[background_particles_frame0]
            bg_visible = bg_visibility >= 0.5
            bg_in_mask = in_mask[background_particles_frame0]
            
            # Only classify visible background particles
            classifications[background_particles_frame0] = np.where(
                bg_visible,
                np.where(bg_in_mask, 2, 0),  # FP or TN for visible
                -1  # Not counted for occluded
            )
            
            batch_classifications.append(classifications)
        
        return np.array(batch_classifications)
    
    # Process in batches
    for batch_start in tqdm(range(0, num_frames, batch_size), desc="Computing classifications"):
        batch_end = batch_start + batch_size
        batch_results = process_batch(batch_start, batch_end)
        actual_end = min(batch_end, num_frames)
        all_classifications[batch_start:actual_end] = batch_results
    
    return all_classifications

def create_tracking_video(video_name, frames_np, pred_tracks_np, pred_visibility_np, 
                         all_classifications, output_path):
    """Create animated video showing tracking results with TP/TN/FP/FN colored points overlaid on segmentation masks."""
    
    # Color mapping with high contrast colors for visibility on segmentation masks
    # TN=cyan, TP=lime, FP=red, FN=yellow
    colors = ['cyan', 'lime', 'red', 'yellow']
    labels = ['TN', 'TP', 'FP', 'FN']
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    def animate(frame_idx):
        ax.clear()
        
        # Load and show segmentation mask for this frame
        seg_mask_2d = get_segmentation_mask(
            video_name, frame_idx, DAVIS_SEGMASKS_PATH, 
            img_dims=(frames_np.shape[1], frames_np.shape[2]), flatten=False
        )
        
        # Create RGB visualization of segmentation mask
        # Background (0) = black, Object (1) = white
        seg_rgb = np.zeros((seg_mask_2d.shape[0], seg_mask_2d.shape[1], 3))
        seg_rgb[seg_mask_2d == 1] = [1, 1, 1]  # White for object
        seg_rgb[seg_mask_2d == 0] = [0, 0, 0]  # Black for background
        
        ax.imshow(seg_rgb)
        
        # Get coordinates and visibility for this frame
        coords = pred_tracks_np[frame_idx]
        visibility = pred_visibility_np[frame_idx]
        classifications = all_classifications[frame_idx]
        
        # Compute metrics for this frame
        tp_count = np.sum(classifications == 1)
        tn_count = np.sum(classifications == 0)
        fp_count = np.sum(classifications == 2)
        fn_count = np.sum(classifications == 3)
        
        # Calculate metrics
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        tpr = recall  # TPR = Recall
        fpr = fp_count / (fp_count + tn_count) if (fp_count + tn_count) > 0 else 0.0
        jaccard = tp_count / (tp_count + fp_count + fn_count) if (tp_count + fp_count + fn_count) > 0 else 0.0
        
        # Create legend handles for all classes (always show all 4)
        legend_handles = []
        
        # Plot points by classification
        for class_idx in range(4):
            mask = classifications == class_idx
            
            # Always create a legend handle for this class
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=colors[class_idx], 
                                           markersize=8, label=labels[class_idx]))
            
            if not np.any(mask):
                continue
                
            class_coords = coords[mask]
            class_visibility = visibility[mask]
            
            # Visible points (solid circles)
            visible_mask = class_visibility > 0.5
            if np.any(visible_mask):
                visible_coords = class_coords[visible_mask]
                ax.scatter(visible_coords[:, 0], visible_coords[:, 1], 
                          c=colors[class_idx], s=25, alpha=0.9, 
                          edgecolors='black', linewidth=0.5)
            
            # Invisible points (hollow circles)
            invisible_mask = class_visibility <= 0.5
            if np.any(invisible_mask):
                invisible_coords = class_coords[invisible_mask]
                ax.scatter(invisible_coords[:, 0], invisible_coords[:, 1], 
                          facecolors='none', edgecolors=colors[class_idx], 
                          s=25, alpha=0.7, linewidth=2)
        
        ax.set_xlim(0, frames_np.shape[2])
        ax.set_ylim(frames_np.shape[1], 0)
        ax.set_title(f'{video_name} - Frame {frame_idx} - CoTracker on Segmentation Mask', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend with all 4 classes (always visible)
        ax.legend(handles=legend_handles, loc='upper left', fontsize=12, 
                 frameon=True, fancybox=True, shadow=True,
                 bbox_to_anchor=(0.02, 0.98), ncol=1)
        
        # Add metrics text on the left side outside the image
        metrics_text = (
            f"Frame {frame_idx}\n"
            f"TP: {tp_count:3d}  FP: {fp_count:3d}\n"
            f"TN: {tn_count:3d}  FN: {fn_count:3d}\n"
            f"TPR: {tpr:.3f}\n"
            f"FPR: {fpr:.3f}\n"
            f"Precision: {precision:.3f}\n"
            f"Recall: {recall:.3f}\n"
            f"Jaccard: {jaccard:.3f}"
        )
        
        ax.text(-0.15, 0.5, metrics_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                fontfamily='monospace')
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(frames_np), 
                                  interval=200, blit=False, repeat=True)
    
    # Save as MP4
    print(f"Saving tracking video to {output_path}...")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='CoTracker'), bitrate=2400)
    anim.save(output_path, writer=writer, dpi=120)
    plt.close()

# ============================================================================
# Process Each Video
# ============================================================================

all_results = {}

for video_name in tqdm(VIDEO_NAMES, desc="Processing videos"):
    print(f"\n{'='*80}")
    print(f"Processing: {video_name}")
    print(f"{'='*80}")

    try:
        # Reset random seed for each video for reproducibility
        np.random.seed(RANDOM_SEED)

        # Load video frames
        rgb_dir = os.path.join(DAVIS_RGB_PATH, video_name)
        rgb_files = sorted(glob(os.path.join(rgb_dir, "*.jpg")))

        if len(rgb_files) == 0:
            print(f"Warning: No RGB frames found for {video_name}, skipping...")
            continue

        # Get target dimensions from 3D motion data
        motion_data = np.load(os.path.join(MOTION_DATA_PATH, f"{video_name}_3d_motion.npz"))
        positions = motion_data['points_3d']
        T, img_height, img_width, _ = positions.shape
        img_dims = (img_height, img_width)

        # Load and resize frames
        frames_list = []
        for rgb_file in rgb_files[:T]:
            frame = cv2.imread(rgb_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            frames_list.append(frame)

        frames_np = np.stack(frames_list, axis=0)
        
        # Limit to 90 frames for jello_trim
        if video_name == "jello_trim":
            frames_np = frames_np[:90]
        
        print(f"Loaded {len(frames_np)} frames with shape {frames_np.shape}")

        # Load first frame segmentation mask
        first_frame_seg = get_segmentation_mask(
            video_name, 0, DAVIS_SEGMASKS_PATH, img_dims=img_dims, flatten=True
        )

        # ================================
        # Custom random point initialization
        # ================================
        if NUM_INIT_PARTICLES_ON_MASK is not None:
            # 1. Find object (mask==1) and background (mask==0) indices
            obj_pixel_indices = np.where(first_frame_seg == 1)[0]
            bg_pixel_indices  = np.where(first_frame_seg == 0)[0]

            # Guard for not enough pixels
            n_obj = len(obj_pixel_indices)
            n_bg  = len(bg_pixel_indices)
            n_obj_pts = min(NUM_INIT_PARTICLES_ON_MASK, n_obj)
            n_bg_pts  = NUM_RANDOM_POINTS - n_obj_pts
            if n_obj < 1 or n_bg < 1:
                raise RuntimeError(f"Mask has no object/background pixels for {video_name}")

            if n_bg_pts > n_bg:
                print(f"WARNING: More background points than available pixels, sampling with replacement for {video_name}")
                replace_bg = True
            else:
                replace_bg = False
            if n_obj_pts > n_obj:
                print(f"WARNING: More object points than available mask pixels, sampling with replacement for {video_name}")
                replace_obj = True
            else:
                replace_obj = False

            # 2. Sample spatial locations randomly from object and background
            obj_indices_chosen = np.random.choice(obj_pixel_indices, n_obj_pts, replace=replace_obj)
            bg_indices_chosen = np.random.choice(bg_pixel_indices, n_bg_pts, replace=replace_bg)
            all_indices = np.concatenate([obj_indices_chosen, bg_indices_chosen])

            y_coords = all_indices // img_width
            x_coords = all_indices % img_width

            # No subpixel noise, use exact pixel centers
            random_points = np.stack([x_coords, y_coords], axis=1)
        else:
            # Default behavior: random anywhere in image (integer pixel locations)
            random_x = np.random.randint(0, img_width, size=NUM_RANDOM_POINTS)
            random_y = np.random.randint(0, img_height, size=NUM_RANDOM_POINTS)
            random_points = np.stack([random_x, random_y], axis=1)

        # ================================

        # Prepare video tensor for CoTracker
        video = torch.tensor(frames_np).permute(0, 3, 1, 2)[None].float().to(device)

        # Prepare queries
        queries = torch.zeros((1, NUM_RANDOM_POINTS, 3), device=device)
        queries[0, :, 0] = 0
        queries[0, :, 1] = torch.tensor(random_points[:, 0], device=device)
        queries[0, :, 2] = torch.tensor(random_points[:, 1], device=device)

        # Run CoTracker
        print("Running CoTracker...")
        with torch.no_grad():
            pred_tracks, pred_visibility = cotracker(video, queries=queries)

        print(f"Finished running CoTracker on {video_name}")

        pred_tracks_np = pred_tracks[0].cpu().numpy()
        pred_visibility_np = pred_visibility[0].cpu().numpy()

        # Classify particles in frame 0
        frame0_coords = pred_tracks_np[0]
        x_coords = np.clip(frame0_coords[:, 0].astype(int), 0, img_width - 1)
        y_coords = np.clip(frame0_coords[:, 1].astype(int), 0, img_height - 1)
        pixel_indices_frame0 = y_coords * img_width + x_coords

        # If we did custom initialization, then:
        # - object_particles_frame0 = np.arange(n_obj_pts) (first points), 
        # - background_particles_frame0 = np.arange(n_obj_pts, NUM_RANDOM_POINTS) (remaining points)
        if NUM_INIT_PARTICLES_ON_MASK is not None:
            object_particles_frame0 = np.zeros(NUM_RANDOM_POINTS, dtype=bool)
            if n_obj_pts > 0:
                object_particles_frame0[:n_obj_pts] = True
            background_particles_frame0 = np.zeros(NUM_RANDOM_POINTS, dtype=bool)
            if n_bg_pts > 0:
                background_particles_frame0[n_obj_pts:] = True
        else:
            # Default: infer by prediction location
            object_particles_frame0 = first_frame_seg[pixel_indices_frame0] == 1
            background_particles_frame0 = first_frame_seg[pixel_indices_frame0] == 0

        n_object_particles = np.sum(object_particles_frame0)
        n_background_particles = np.sum(background_particles_frame0)

        # Batch compute classifications for all frames
        print("Computing classifications for all frames...")
        all_classifications = batch_compute_classifications(
            pred_tracks_np, pred_visibility_np, object_particles_frame0, 
            background_particles_frame0, video_name, img_dims, batch_size=32
        )

        # Compute accuracy metrics
        print("Computing accuracy metrics...")
        precision_scores = []
        recall_scores = []
        f1_scores = []
        jaccard_scores = []
        false_negative_rates = []
        false_positive_rates = []

        for frame_idx in tqdm(range(len(pred_tracks_np)), desc="Computing accuracy metrics"):
            classifications = all_classifications[frame_idx]
            
            # Count classifications
            tp_count = np.sum(classifications == 1)  # TP
            tn_count = np.sum(classifications == 0)  # TN
            fp_count = np.sum(classifications == 2)  # FP
            fn_count = np.sum(classifications == 3)  # FN
            
            # Count visible object particles (for FN% denominator)
            # Visible object particles = TP + FN (all visible object particles are either TP or FN)
            visible_object_particles = tp_count + fn_count

            # Calculate metrics
            # Precision = TP / (TP + FP)
            if tp_count + fp_count > 0:
                precision = tp_count / (tp_count + fp_count)
            else:
                precision = 0.0

            # Recall = TP / (TP + FN)
            if tp_count + fn_count > 0:
                recall = tp_count / (tp_count + fn_count)
            else:
                recall = 0.0

            # F1 = 2 * (precision * recall) / (precision + recall)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            # Jaccard = TP / (TP + FP + FN)
            if tp_count + fp_count + fn_count > 0:
                jaccard = tp_count / (tp_count + fp_count + fn_count)
            else:
                jaccard = 0.0

            # False Negative Rate = FN / (TP + FN) = FN / visible_object_particles
            # NOTE: Must use visible_object_particles (not n_object_particles) to match Jaccard calculation
            # which only counts visible particles. This ensures FN% is consistent with Precision/Recall/Jaccard.
            if visible_object_particles > 0:
                fn_rate = (fn_count / visible_object_particles) * 100
            else:
                fn_rate = 0.0

            # False Positive Rate = FP / (FP + TN)
            if fp_count + tn_count > 0:
                fp_rate = (fp_count / (fp_count + tn_count)) * 100
            else:
                fp_rate = 0.0

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            jaccard_scores.append(jaccard)
            false_negative_rates.append(fn_rate)
            false_positive_rates.append(fp_rate)

        # Calculate averages
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)
        mean_f1 = np.mean(f1_scores)
        mean_jaccard = np.mean(jaccard_scores)
        mean_fn_rate = np.mean(false_negative_rates)
        mean_fp_rate = np.mean(false_positive_rates)

        # Create tracking video
        if SAVE_COTRACKER_OVERLAY_ON_SEGMASK_VIDEO:
            print("Creating tracking visualization video...")
            video_output_path = os.path.join(OUTPUT_DIR, "videos", f"{video_name}_tracking.mp4")
            create_tracking_video(video_name, frames_np, pred_tracks_np, pred_visibility_np, 
                                all_classifications, video_output_path)

        # Store results
        all_results[video_name] = {
            'video_name': video_name,
            'num_points': NUM_RANDOM_POINTS,
            'num_frames': len(pred_tracks_np),
            'img_dims': img_dims,
            'n_object_particles': int(n_object_particles),
            'n_background_particles': int(n_background_particles),

            # Per-frame scores
            'precision_scores': [float(x) for x in precision_scores],
            'recall_scores': [float(x) for x in recall_scores],
            'f1_scores': [float(x) for x in f1_scores],
            'jaccard_scores': [float(x) for x in jaccard_scores],
            'false_negative_rates': [float(x) for x in false_negative_rates],
            'false_positive_rates': [float(x) for x in false_positive_rates],

            # Average metrics
            'mean_precision': float(mean_precision),
            'mean_recall': float(mean_recall),
            'mean_f1': float(mean_f1),
            'mean_jaccard': float(mean_jaccard),
            'mean_fn_rate': float(mean_fn_rate),
            'mean_fp_rate': float(mean_fp_rate),
        }

        # Save individual JSON
        json_path = os.path.join(OUTPUT_DIR, f"{video_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(all_results[video_name], f, indent=2)

        # Plot metrics over time
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top plot: Precision, Recall, F1, Jaccard
        ax1.plot(range(len(precision_scores)), precision_scores, linewidth=2,
                color='blue', label=f'Precision (mean: {mean_precision:.3f})', alpha=0.8)
        ax1.plot(range(len(recall_scores)), recall_scores, linewidth=2,
                color='red', label=f'Recall (mean: {mean_recall:.3f})', alpha=0.8)
        ax1.plot(range(len(f1_scores)), f1_scores, linewidth=2,
                color='green', label=f'F1 (mean: {mean_f1:.3f})', alpha=0.8)
        ax1.plot(range(len(jaccard_scores)), jaccard_scores, linewidth=2,
                color='orange', label=f'Jaccard (mean: {mean_jaccard:.3f})', alpha=0.8)
        ax1.set_xlabel('Frame', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title(f'{video_name} - CoTracker Random Init ({NUM_RANDOM_POINTS} points) - Tracking Metrics',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        ax1.set_ylim([0, 1.05])

        # Bottom plot: Error rates
        ax2.plot(range(len(false_negative_rates)), false_negative_rates, linewidth=2,
                color='red', label=f'False Negative Rate (mean: {mean_fn_rate:.2f}%)', alpha=0.8)
        ax2.plot(range(len(false_positive_rates)), false_positive_rates, linewidth=2,
                color='blue', label=f'False Positive Rate (mean: {mean_fp_rate:.2f}%)', alpha=0.8)
        ax2.set_xlabel('Frame', fontsize=12)
        ax2.set_ylabel('Rate (%)', fontsize=12)
        ax2.set_title('Error Rates', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        ax2.set_ylim([0, 105])

        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, "plots", f"{video_name}_metrics.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nResults:")
        print(f"  Precision:  {mean_precision:.3f}")
        print(f"  Recall:     {mean_recall:.3f}")
        print(f"  F1:         {mean_f1:.3f}")
        print(f"  Jaccard:    {mean_jaccard:.3f}")
        print(f"  FN Rate:    {mean_fn_rate:.2f}%")
        print(f"  FP Rate:    {mean_fp_rate:.2f}%")
        print(f"Saved: {json_path}")
        print(f"Saved: {plot_path}")
        if SAVE_COTRACKER_OVERLAY_ON_SEGMASK_VIDEO:
            print(f"Saved: {video_output_path}")

        # Clean up GPU memory
        del video, queries, pred_tracks, pred_visibility
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error processing {video_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# ============================================================================
# Save Summary Results
# ============================================================================

print(f"\n{'='*80}")
print("SAVING SUMMARY RESULTS")
print(f"{'='*80}")

summary_path = os.path.join(OUTPUT_DIR, "all_videos_summary.json")
with open(summary_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"Saved summary: {summary_path}")

# Print summary table
print(f"\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}")
print(f"{'Video':<20} {'Frames':>7} {'Obj Pts':>8} {'Bg Pts':>8} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Jaccard':>7} {'FN%':>7} {'FP%':>7}")
print("-" * 110)

for video_name in VIDEO_NAMES:
    if video_name in all_results:
        r = all_results[video_name]
        print(f"{video_name:<20} {r['num_frames']:>7} {r['n_object_particles']:>8} "
              f"{r['n_background_particles']:>8} {r['mean_precision']:>7.3f} "
              f"{r['mean_recall']:>7.3f} {r['mean_f1']:>7.3f} {r['mean_jaccard']:>7.3f} "
              f"{r['mean_fn_rate']:>7.2f} {r['mean_fp_rate']:>7.2f}")

# Compute overall statistics
all_precision = [r['mean_precision'] for r in all_results.values()]
all_recall = [r['mean_recall'] for r in all_results.values()]
all_f1 = [r['mean_f1'] for r in all_results.values()]
all_jaccard = [r['mean_jaccard'] for r in all_results.values()]
all_fn_rates = [r['mean_fn_rate'] for r in all_results.values()]
all_fp_rates = [r['mean_fp_rate'] for r in all_results.values()]

print("-" * 110)
print(f"{'MEAN':<20} {'':<7} {'':<8} {'':<8} {np.mean(all_precision):>7.3f} "
      f"{np.mean(all_recall):>7.3f} {np.mean(all_f1):>7.3f} {np.mean(all_jaccard):>7.3f} "
      f"{np.mean(all_fn_rates):>7.2f} {np.mean(all_fp_rates):>7.2f}")
print(f"{'MEDIAN':<20} {'':<7} {'':<8} {'':<8} {np.median(all_precision):>7.3f} "
      f"{np.median(all_recall):>7.3f} {np.median(all_f1):>7.3f} {np.median(all_jaccard):>7.3f} "
      f"{np.median(all_fn_rates):>7.2f} {np.median(all_fp_rates):>7.2f}")
print(f"{'STD':<20} {'':<7} {'':<8} {'':<8} {np.std(all_precision):>7.3f} "
      f"{np.std(all_recall):>7.3f} {np.std(all_f1):>7.3f} {np.std(all_jaccard):>7.3f} "
      f"{np.std(all_fn_rates):>7.2f} {np.std(all_fp_rates):>7.2f}")

if len(all_results) > 0:
    print(f"\n{'='*80}")
    print(f"AGGREGATE STATISTICS ACROSS ALL VIDEOS ({len(all_results)} videos)")
    print(
        f"  (95% CIs: percentile bootstrap on video means, B={BOOTSTRAP_N_SAMPLES}, seed={BOOTSTRAP_RANDOM_SEED})"
    )
    print(f"\n  Per-video mean metrics (same scale as table above):")
    for label, arr in [
        ("Precision", all_precision),
        ("Recall", all_recall),
        ("F1", all_f1),
        ("Jaccard", all_jaccard),
    ]:
        m, lo, hi = bootstrap_mean_ci_95(arr)
        print(f"    {label:18s} {m:.3f} [{lo:.3f}, {hi:.3f}]")
    for label, arr in [
        ("FN rate (%)", all_fn_rates),
        ("FP rate (%)", all_fp_rates),
    ]:
        m, lo, hi = bootstrap_mean_ci_95(arr)
        print(f"    {label:18s} {m:.2f} [{lo:.2f}, {hi:.2f}]")

print(f"\nTotal videos processed: {len(all_results)}/{len(VIDEO_NAMES)}")
print(f"Results saved to: {OUTPUT_DIR}")
print("\nDone!")

"""
Evaluation metrics for GenMatter tracking.

This module computes multiple metrics for evaluating particle-based tracking:

1. Particle-level metrics (recall, precision, FPR, IoU)
   - Treats particles as discrete units with fractional pixel contributions
   - Used for visualization and understanding per-particle behavior
   - LIMITATION: Treats all particles equally, sensitive to boundary particles

2. Matter-weighted metrics (**fixed frame-0 blob weights**) — reported DAVIS primary metrics:

   - ``avg_matter_weighted_recall_fixed``, ``avg_matter_weighted_precision_fixed``,
     ``avg_matter_weighted_jaccard_fixed`` (and optional accuracy).
   - Unweighted fractional scores may still be computed internally for ROC AUC.

Matter-weighted variants weight by (pixel_count × blob_weight) so larger spatial support dominates.
"""

import os
import numpy as np
from PIL import Image
import json
import os
import numpy as np

def get_segmentation_mask(davis_name, frame_idx, annotations_path, img_dims = None, flatten = True):
    seg_masks_path = os.path.join(annotations_path, davis_name)
    frame_path = os.path.join(seg_masks_path, f"{frame_idx:05d}.png")
    if img_dims is not None:
        new_height, new_width = img_dims
        frame_seg = np.array(Image.open(frame_path).resize((new_width, new_height), Image.NEAREST))
    else:
        frame_seg = np.array(Image.open(frame_path))
    
    # Handle both single-channel and multi-channel images
    if len(frame_seg.shape) == 3:  # Multi-channel image (H, W, C)
        # Consider a pixel as part of the mask if any channel has value >= 1
        frame_mask = np.any(frame_seg >= 1, axis=2)
    else:  # Single-channel image (H, W)
        frame_mask = frame_seg >= 1
        
    if flatten:
        frame_mask = frame_mask.reshape(-1)
    return frame_mask


def evaluate_tracking_results(
    davis_genmatter_dict, 
    annotations_path,
    counting_threshold=100, 
    img_dims=(520, 960), 
    save_data=False,
    output_dir=None,
    experiment_name="tracking_evaluation"
):
    """
    Evaluate tracking results for multiple GenMatter runs with different random trials across multiple DAVIS datasets.

    Args:
        davis_genmatter_dict: Dictionary where keys are davis_names and values are lists of lists.
            Each outer list is for a different random trial, each inner list is over all frames.
            Each element is a dict with 'n_blobs' and 'blob_assignments'.
        annotations_path: Path to the DAVIS annotations
        counting_threshold: Threshold for counting blobs
        img_dims: Image dimensions
        save_data: Whether to save results to a JSON file
        output_dir: Directory where to save results (required if save_data is True)
        experiment_name: Name of the experiment
    """


    # Validate inputs
    if (save_data) and output_dir is None:
        raise ValueError("output_dir must be provided if save_data is True")

    # Create output directories if needed
    if save_data:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize results dictionary
    results_data = {
        "experiment_name": experiment_name,
        "datasets": {}
    }

    # Store all dataset average results for summary table
    all_datasets_results = {}
    
    # Process each dataset
    for davis_name, multiple_genmatter_list in davis_genmatter_dict.items():
        # Get first frame segmentation mask
        first_frame_mask = get_segmentation_mask(davis_name, 0, annotations_path, img_dims, flatten=False)
        
        # Initialize arrays to store metrics for each trial
        all_trials_data = []
        
        # Process each trial
        for trial_idx, per_trial_list in enumerate(multiple_genmatter_list):
            # per_trial_list: list over frames, each is a dict with 'n_blobs' and 'blob_assignments'
            num_frames = len(per_trial_list)
            
            # Get first frame assignments
            first_frame_assignments = np.array(per_trial_list[0]['blob_assignments']).reshape(img_dims)
            first_frame_n_blobs = per_trial_list[0]['n_blobs']
            
            # Get unique assignment values from the first frame (reference blobs)
            values, counts = np.unique(first_frame_assignments[first_frame_mask], return_counts=True)
            # Filter values that appear at least counting_threshold times and exclude the outlier blob
            reference_blobs = values[(counts >= counting_threshold) & (values != first_frame_n_blobs)]
            total_reference_blobs = len(reference_blobs)
            
            # Initialize arrays to store metrics for each frame
            percentage_preserved_scores = np.zeros(num_frames)
            
            # Process each frame to calculate metrics
            for frame_idx in range(num_frames):
                # Load the ground truth segmentation mask for this frame
                frame_mask = get_segmentation_mask(davis_name, frame_idx, annotations_path, img_dims, flatten=False)
                
                # Get the blob assignments for this frame
                if frame_idx < len(per_trial_list):
                    frame_assignments = np.array(per_trial_list[frame_idx]['blob_assignments']).reshape(img_dims)
                    frame_n_blobs = per_trial_list[frame_idx]['n_blobs']
                else:
                    # If we don't have assignments for all frames, use the last available one
                    frame_assignments = np.array(per_trial_list[-1]['blob_assignments']).reshape(img_dims)
                    frame_n_blobs = per_trial_list[-1]['n_blobs']
                
                # Calculate percentage of preserved blobs
                frame_values = np.unique(frame_assignments[frame_mask])
                frame_values = frame_values[frame_values != frame_n_blobs]  # Exclude outlier blob
                common_blobs = np.intersect1d(reference_blobs, frame_values)
                percentage_preserved_scores[frame_idx] = (len(common_blobs) / total_reference_blobs) * 100 if total_reference_blobs > 0 else 0
            
            # Calculate average for this trial
            avg_preserved = float(np.mean(percentage_preserved_scores))
            
            # Store data for this trial
            trial_data = {
                'avg_preserved': avg_preserved
            }
            
            all_trials_data.append(trial_data)

        # Calculate average metrics across all trials
        avg_preserved_values = [data['avg_preserved'] for data in all_trials_data]
        
        avg_preserved = float(np.mean(avg_preserved_values))
        std_preserved = float(np.std(avg_preserved_values))
        
        # Store average results for summary table
        all_datasets_results[davis_name] = {
            'avg_preserved': avg_preserved,
            'std_preserved': std_preserved
        }
                
        # Store dataset results
        dataset_results = {
            "all_trials_data": all_trials_data,
            "avg_metrics": {
                'avg_preserved': avg_preserved,
                'std_preserved': std_preserved
            },
        }
        
        # Add to results data
        if save_data:
            results_data["datasets"][davis_name] = dataset_results
        
    # Save results to JSON if requested
    if save_data:
        json_path = os.path.join(output_dir, f"{experiment_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to: {json_path}")


def evaluate_single_davis_video(
    davis_name,
    multiple_genmatter_list,
    annotations_path,
    counting_threshold=0,
    img_dims=(520, 960),
    fps_list=None,
    render_results_video=True,
    experiment_save_dir=None,
    force_below_count_thresh_as_outlier=False,
    subsampled_indices=None
):
    """
    Evaluate tracking results for a single DAVIS video with multiple random trials.
    For visualization, shows the best trial in terms of recall for most plots,
    but overlays the mean tracking metrics across all trials as lines (by-frame),
    not static lines. All plots and legends are made explicit to clarify this
    distinction. Also displays particle counts before and after first frame
    Gibbs filtering, and improves legends, coloring, and layout.
    
    Args:
        force_below_count_thresh_as_outlier: If True, particles below counting_threshold
            in the first frame mask are treated as outliers and excluded from all
            TP/FP/TN/FN calculations. If False (default), they contribute to background
            classification (TN/FP).
        subsampled_indices: If provided, indices to subsample the flattened image arrays.
            Used for memory efficiency with large images.
    """
    # if render_results_video and experiment_save_dir is None:
    #     raise ValueError("experiment_save_dir must be provided if render_results_video is True")
    
    print(f"\n{'='*80}")
    print(f"📊 Processing dataset: {davis_name}")
    print(f"{'='*80}")
    
    # Get first frame segmentation mask (slice to subsampled indices if provided)
    first_frame_mask_full = get_segmentation_mask(davis_name, 0, annotations_path, img_dims, flatten=True)
    first_frame_mask = first_frame_mask_full if subsampled_indices is None else first_frame_mask_full[subsampled_indices]
    
    all_trials_data = []
    all_visualization_trial_data = []
    all_reference_particles_before_threshold = None
    first_frame_outlier_count = None

    # To record per-frame metrics for mean calculation across runs
    per_frame_recall_all_trials = []
    per_frame_precision_all_trials = []
    per_frame_fpr_all_trials = []
    per_frame_jaccard_all_trials = []
    per_frame_accuracy_all_trials = []
    per_frame_matter_weighted_recall_fixed_all_trials = []
    per_frame_matter_weighted_precision_fixed_all_trials = []
    per_frame_matter_weighted_jaccard_fixed_all_trials = []
    per_frame_matter_weighted_accuracy_fixed_all_trials = []

    for trial_idx, per_trial_list in enumerate(multiple_genmatter_list):
        num_frames = len(per_trial_list)
        first_frame_assignments = np.array(per_trial_list[0]['blob_assignments'])
        
        # Ensure assignments are always 1D (flattened) for consistent processing
        # Note: If subsampled_indices is provided, assignments are already subsampled
        # (they match the subsampled datapoints from tracking). subsampled_indices is only used for the mask.
        if first_frame_assignments.ndim > 1:
            first_frame_assignments = first_frame_assignments.flatten()
        
        first_frame_n_blobs = per_trial_list[0]['n_blobs']

        true_unique_particles = [bid for bid in np.unique(first_frame_assignments) if bid != first_frame_n_blobs]
        num_true_unique_particles = len(true_unique_particles)

        # Use first_frame_assignments directly (already 1D) for indexing with first_frame_mask (which is always 1D)
        values_all_particles, counts_all_particles = np.unique(first_frame_assignments[first_frame_mask], return_counts=True)
        all_reference_particles_before_threshold = values_all_particles[values_all_particles != first_frame_n_blobs]
        n_all_reference_particles_before_threshold = len(all_reference_particles_before_threshold)

        reference_particles = values_all_particles[(counts_all_particles >= counting_threshold) & (values_all_particles != first_frame_n_blobs)]
        total_reference_particles = len(reference_particles)
        n_total_particles_in_scene = len(np.unique(first_frame_assignments[first_frame_assignments != first_frame_n_blobs]))

        # Determine which particles should be treated as outliers
        if force_below_count_thresh_as_outlier:
            # Particles below threshold in first frame mask become outliers
            below_threshold_particles = values_all_particles[(counts_all_particles < counting_threshold) & (values_all_particles != first_frame_n_blobs)]
            outlier_particles = np.concatenate([np.array([first_frame_n_blobs]), below_threshold_particles])
        else:
            # Only the original outlier blob
            outlier_particles = np.array([first_frame_n_blobs])

        outlier_mask = (first_frame_assignments == first_frame_n_blobs)
        first_frame_outlier_count = np.unique(first_frame_assignments)[np.unique(first_frame_assignments) == first_frame_n_blobs]
        n_outlier_particles = int(len(first_frame_outlier_count > 0))
        actual_outlier_pixels = np.sum(outlier_mask)

        recall_scores = np.zeros(num_frames)
        precision_scores = np.zeros(num_frames)
        fpr_scores = np.zeros(num_frames)
        jaccard_scores = np.zeros(num_frames)
        accuracy_scores = np.zeros(num_frames)

        # Matter-weighted fixed (frame-0 blob weights): recall, precision, Jaccard, accuracy
        matter_weighted_recall_fixed_scores = np.zeros(num_frames)
        matter_weighted_precision_fixed_scores = np.zeros(num_frames)
        matter_weighted_jaccard_fixed_scores = np.zeros(num_frames)
        matter_weighted_accuracy_fixed_scores = np.zeros(num_frames)

        tpr_values = []
        fpr_values = []
        frame0_blob_weights = None  # Cache for fixed-weight metric

        for frame_idx in range(num_frames):
            frame_mask_full = get_segmentation_mask(davis_name, frame_idx, annotations_path, img_dims, flatten=True)
            frame_mask = frame_mask_full if subsampled_indices is None else frame_mask_full[subsampled_indices]
            
            if frame_idx < len(per_trial_list):
                frame_assignments = np.array(per_trial_list[frame_idx]['blob_assignments'])
                
                # Ensure assignments are always 1D (flattened) for consistent processing
                # Note: If subsampled_indices is provided, assignments are already subsampled
                # (they match the subsampled datapoints from tracking). subsampled_indices is only used for the mask.
                if frame_assignments.ndim > 1:
                    frame_assignments = frame_assignments.flatten()
                
                frame_n_blobs = per_trial_list[frame_idx]['n_blobs']
                frame_blob_weights = np.array(per_trial_list[frame_idx]['blob_weights'])
            else:
                raise ValueError(f"No assignments for frame {frame_idx} in trial {trial_idx}")

            all_current_particles = np.unique(frame_assignments)
            all_current_particles = all_current_particles[all_current_particles != frame_n_blobs]

            # Create mask for valid particles (excluding outliers)
            valid_mask = ~np.isin(frame_assignments, outlier_particles)
            
            # Both assignments and masks are already 1D, so we can index directly
            flat_assignments = frame_assignments[valid_mask]
            flat_gt_mask = frame_mask[valid_mask]

            unique_particles, particle_inverse, particle_counts = np.unique(flat_assignments, return_inverse=True, return_counts=True)
            ref_particle_mask = np.isin(unique_particles, reference_particles)

            pixels_inside_gt_per_particle = np.bincount(particle_inverse, weights=flat_gt_mask, minlength=len(unique_particles))
            total_particle_pixels = particle_counts
            pixels_outside_gt_per_particle = total_particle_pixels - pixels_inside_gt_per_particle

            with np.errstate(divide='ignore', invalid='ignore'):
                fraction_inside = np.where(total_particle_pixels > 0, pixels_inside_gt_per_particle / total_particle_pixels, 0)
                fraction_outside = np.where(total_particle_pixels > 0, pixels_outside_gt_per_particle / total_particle_pixels, 0)

            tp_count = np.sum(fraction_inside[ref_particle_mask])
            fn_count = np.sum(fraction_outside[ref_particle_mask])
            fp_count = np.sum(fraction_inside[~ref_particle_mask])
            tn_count = np.sum(fraction_outside[~ref_particle_mask])
            
            recall_scores[frame_idx] = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            precision_scores[frame_idx] = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            fpr_scores[frame_idx] = fp_count / (fp_count + tn_count) if (fp_count + tn_count) > 0 else 0
            accuracy_scores[frame_idx] = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count) if (tp_count + tn_count + fp_count + fn_count) > 0 else 0
            
            intersection = tp_count
            union = tp_count + fp_count + fn_count
            jaccard_scores[frame_idx] = intersection / union if union > 0 else 0
            tpr_values.append(recall_scores[frame_idx])
            fpr_values.append(fpr_scores[frame_idx])

            # Matter-weighted metrics with frame-0 blob weights (DAVIS benchmark suite).
            # matter_weight = pixel_count × blob_weight (frame 0).
            if frame_idx == 0:
                frame0_blob_weights = frame_blob_weights.copy()

            particle_blob_weights_fixed = np.array([
                frame0_blob_weights[p] if p < len(frame0_blob_weights) else 0.0
                for p in unique_particles
            ])
            matter_weight_fixed = total_particle_pixels * particle_blob_weights_fixed

            tp_matter_fixed = np.sum(fraction_inside[ref_particle_mask] * matter_weight_fixed[ref_particle_mask])
            fn_matter_fixed = np.sum(fraction_outside[ref_particle_mask] * matter_weight_fixed[ref_particle_mask])
            fp_matter_fixed = np.sum(fraction_inside[~ref_particle_mask] * matter_weight_fixed[~ref_particle_mask])
            tn_matter_fixed = np.sum(fraction_outside[~ref_particle_mask] * matter_weight_fixed[~ref_particle_mask])

            if (tp_matter_fixed + fn_matter_fixed) > 0:
                matter_weighted_recall_fixed_scores[frame_idx] = tp_matter_fixed / (tp_matter_fixed + fn_matter_fixed)
            else:
                matter_weighted_recall_fixed_scores[frame_idx] = 0.0

            if (tp_matter_fixed + fp_matter_fixed) > 0:
                matter_weighted_precision_fixed_scores[frame_idx] = tp_matter_fixed / (tp_matter_fixed + fp_matter_fixed)
            else:
                matter_weighted_precision_fixed_scores[frame_idx] = 0.0

            total_matter_fixed = tp_matter_fixed + tn_matter_fixed + fp_matter_fixed + fn_matter_fixed
            if total_matter_fixed > 0:
                matter_weighted_accuracy_fixed_scores[frame_idx] = (tp_matter_fixed + tn_matter_fixed) / total_matter_fixed
            else:
                matter_weighted_accuracy_fixed_scores[frame_idx] = 0.0

            union_matter_fixed = tp_matter_fixed + fp_matter_fixed + fn_matter_fixed
            if union_matter_fixed > 0:
                matter_weighted_jaccard_fixed_scores[frame_idx] = tp_matter_fixed / union_matter_fixed
            else:
                matter_weighted_jaccard_fixed_scores[frame_idx] = 0.0
        
        if len(tpr_values) > 1 and len(set(fpr_values)) > 1:
            sorted_indices = np.argsort(fpr_values)
            sorted_fpr = np.array(fpr_values)[sorted_indices]
            sorted_tpr = np.array(tpr_values)[sorted_indices]
            auc_roc = np.trapz(sorted_tpr, sorted_fpr)
        else:
            auc_roc = 0.0

        trial_data = {
            'avg_matter_weighted_recall_fixed': float(np.mean(matter_weighted_recall_fixed_scores)),
            'avg_matter_weighted_precision_fixed': float(np.mean(matter_weighted_precision_fixed_scores)),
            'avg_matter_weighted_jaccard_fixed': float(np.mean(matter_weighted_jaccard_fixed_scores)),
            'avg_matter_weighted_accuracy_fixed': float(np.mean(matter_weighted_accuracy_fixed_scores)),
            'auc_roc': float(auc_roc),
        }
        visualization_data_entry = {
            'per_trial_list': per_trial_list,
            'reference_particles': reference_particles,
            'total_reference_particles': total_reference_particles,
            'all_reference_particles_before_threshold': all_reference_particles_before_threshold,
            'n_all_reference_particles_before_threshold': n_all_reference_particles_before_threshold,
            'n_total_particles_in_scene': n_total_particles_in_scene,
            'num_true_unique_particles_first_frame': num_true_unique_particles,
            'first_frame_n_blobs': first_frame_n_blobs,
            'counting_threshold': counting_threshold,
            'recall_scores': recall_scores,
            'precision_scores': precision_scores,
            'fpr_scores': fpr_scores,
            'jaccard_scores': jaccard_scores,
            'iou_scores': jaccard_scores,
            'accuracy_scores': accuracy_scores,
            'matter_weighted_recall_fixed_scores': matter_weighted_recall_fixed_scores,
            'matter_weighted_precision_fixed_scores': matter_weighted_precision_fixed_scores,
            'matter_weighted_jaccard_fixed_scores': matter_weighted_jaccard_fixed_scores,
            'matter_weighted_accuracy_fixed_scores': matter_weighted_accuracy_fixed_scores,
            'tpr_values': tpr_values,
            'fpr_values': fpr_values,
            'num_frames': num_frames,
            'n_outlier_particles': n_outlier_particles,
            'actual_outlier_pixels': actual_outlier_pixels,
            'avg_recall': trial_data['avg_matter_weighted_jaccard_fixed'],
        }
        all_visualization_trial_data.append(visualization_data_entry)
        all_trials_data.append(trial_data)

        per_frame_recall_all_trials.append(recall_scores)
        per_frame_precision_all_trials.append(precision_scores)
        per_frame_fpr_all_trials.append(fpr_scores)
        per_frame_jaccard_all_trials.append(jaccard_scores)
        per_frame_accuracy_all_trials.append(accuracy_scores)
        per_frame_matter_weighted_recall_fixed_all_trials.append(matter_weighted_recall_fixed_scores)
        per_frame_matter_weighted_precision_fixed_all_trials.append(matter_weighted_precision_fixed_scores)
        per_frame_matter_weighted_jaccard_fixed_all_trials.append(matter_weighted_jaccard_fixed_scores)
        per_frame_matter_weighted_accuracy_fixed_all_trials.append(matter_weighted_accuracy_fixed_scores)
    
    print(f"\n📈 Trial Results for {davis_name} (matter-weighted, frame-0 blob weights):")
    print(
        f"{'Trial':<8} {'MW-R-F':<10} {'MW-P-F':<10} {'MW-J-F':<10} "
        f"{'MW-Acc-F':<10} {'AUC':<8}"
    )
    print(f"{'-'*60}")
    for trial_i, td in enumerate(all_trials_data):
        print(
            f"{trial_i+1:<8} {td['avg_matter_weighted_recall_fixed']:<10.3f} "
            f"{td['avg_matter_weighted_precision_fixed']:<10.3f} "
            f"{td['avg_matter_weighted_jaccard_fixed']:<10.3f} "
            f"{td['avg_matter_weighted_accuracy_fixed']:<10.3f} {td['auc_roc']:<8.3f}"
        )
    print(f"{'-'*60}")

    avg_matter_weighted_jaccard_fixed_values = [d['avg_matter_weighted_jaccard_fixed'] for d in all_trials_data]
    avg_matter_weighted_recall_fixed_values = [d['avg_matter_weighted_recall_fixed'] for d in all_trials_data]
    avg_matter_weighted_precision_fixed_values = [d['avg_matter_weighted_precision_fixed'] for d in all_trials_data]
    avg_matter_weighted_accuracy_fixed_values = [d['avg_matter_weighted_accuracy_fixed'] for d in all_trials_data]
    auc_roc_values = [d['auc_roc'] for d in all_trials_data]

    matter_weighted_jaccard_fixed_mean = float(np.mean(avg_matter_weighted_jaccard_fixed_values))
    matter_weighted_recall_fixed_mean = float(np.mean(avg_matter_weighted_recall_fixed_values))
    matter_weighted_precision_fixed_mean = float(np.mean(avg_matter_weighted_precision_fixed_values))
    matter_weighted_accuracy_fixed_mean = float(np.mean(avg_matter_weighted_accuracy_fixed_values))
    auc_roc_mean = float(np.mean(auc_roc_values))
    print(
        f"{'Mean':<8} {matter_weighted_recall_fixed_mean:<10.3f} "
        f"{matter_weighted_precision_fixed_mean:<10.3f} "
        f"{matter_weighted_jaccard_fixed_mean:<10.3f} "
        f"{matter_weighted_accuracy_fixed_mean:<10.3f} {auc_roc_mean:<8.3f}"
    )
    print("Note: MW-*-F = matter-weighted recall / precision / Jaccard / accuracy (frame-0 weights).")

    fps_mean = None
    fps_std = None
    if fps_list is not None and len(fps_list) > 0:
        fps_mean = float(np.mean(fps_list))
        fps_std = float(np.std(fps_list))
        print(f"{'FPS Mean':<8} {fps_mean:<18.2f} {fps_std:<12.2f}")

    # Compute per-frame mean metrics across all runs for visualizing as framewise average lines (not flat lines)
    per_frame_recall_all_trials = np.stack(per_frame_recall_all_trials, axis=0) if len(per_frame_recall_all_trials) > 0 else None
    per_frame_precision_all_trials = np.stack(per_frame_precision_all_trials, axis=0) if len(per_frame_precision_all_trials) > 0 else None
    per_frame_fpr_all_trials = np.stack(per_frame_fpr_all_trials, axis=0) if len(per_frame_fpr_all_trials) > 0 else None
    per_frame_jaccard_all_trials = np.stack(per_frame_jaccard_all_trials, axis=0) if len(per_frame_jaccard_all_trials) > 0 else None
    per_frame_accuracy_all_trials = np.stack(per_frame_accuracy_all_trials, axis=0) if len(per_frame_accuracy_all_trials) > 0 else None
    per_frame_matter_weighted_recall_fixed_all_trials = np.stack(per_frame_matter_weighted_recall_fixed_all_trials, axis=0) if len(per_frame_matter_weighted_recall_fixed_all_trials) > 0 else None
    per_frame_matter_weighted_precision_fixed_all_trials = np.stack(per_frame_matter_weighted_precision_fixed_all_trials, axis=0) if len(per_frame_matter_weighted_precision_fixed_all_trials) > 0 else None
    per_frame_matter_weighted_jaccard_fixed_all_trials = np.stack(per_frame_matter_weighted_jaccard_fixed_all_trials, axis=0) if len(per_frame_matter_weighted_jaccard_fixed_all_trials) > 0 else None
    per_frame_matter_weighted_accuracy_fixed_all_trials = np.stack(per_frame_matter_weighted_accuracy_fixed_all_trials, axis=0) if len(per_frame_matter_weighted_accuracy_fixed_all_trials) > 0 else None

    # Enhanced visualization: Use the trial with best average recall (not index 0)
    best_visualization_data = None
    if render_results_video:
        print(f"\n🎬 Creating results video (with particle overlay)...")
        try:
            # Best trial by max matter-weighted Jaccard (fixed frame-0 weights)
            best_trial_idx = int(np.argmax([v['matter_weighted_jaccard_fixed_scores'].mean() for v in all_visualization_trial_data]))
            best_visualization_data = all_visualization_trial_data[best_trial_idx].copy()
            best_visualization_data['per_frame_recall_mean'] = np.mean(per_frame_matter_weighted_recall_fixed_all_trials, axis=0) if per_frame_matter_weighted_recall_fixed_all_trials is not None else None
            best_visualization_data['per_frame_precision_mean'] = np.mean(per_frame_matter_weighted_precision_fixed_all_trials, axis=0) if per_frame_matter_weighted_precision_fixed_all_trials is not None else None
            best_visualization_data['per_frame_fpr_mean'] = None
            best_visualization_data['per_frame_jaccard_mean'] = np.mean(per_frame_matter_weighted_jaccard_fixed_all_trials, axis=0) if per_frame_matter_weighted_jaccard_fixed_all_trials is not None else None
            best_visualization_data['per_frame_iou_mean'] = best_visualization_data['per_frame_jaccard_mean']
            best_visualization_data['per_frame_accuracy_mean'] = np.mean(per_frame_matter_weighted_accuracy_fixed_all_trials, axis=0) if per_frame_matter_weighted_accuracy_fixed_all_trials is not None else None
            best_visualization_data['per_frame_matter_weighted_accuracy_fixed_mean'] = best_visualization_data['per_frame_accuracy_mean']
            best_visualization_data['runs_n_trials']            = len(all_trials_data)
            best_visualization_data['selected_trial_idx']       = best_trial_idx
            best_visualization_data['davis_name']               = davis_name
            # create_genmatter_results_video(
            #     best_visualization_data,
            #     annotations_path,
            #     img_dims,
            #     experiment_save_dir
            # )
        except Exception as e:
            print(f"⚠️  Could not plot results video due to error: {str(e)}")

    results = {
        'davis_name': davis_name,
        'all_trials_data': all_trials_data,
        'avg_matter_weighted_recall_fixed': matter_weighted_recall_fixed_mean,
        'avg_matter_weighted_precision_fixed': matter_weighted_precision_fixed_mean,
        'avg_matter_weighted_jaccard_fixed': matter_weighted_jaccard_fixed_mean,
        'avg_matter_weighted_accuracy_fixed': matter_weighted_accuracy_fixed_mean,
        'avg_auc_roc': auc_roc_mean,
        'fps_mean': fps_mean,
        'fps_std': fps_std,
    }
    return results, best_visualization_data


# Custom-video pseudo-GT evaluation (SAM segmasks, TAP-Vid layout).
CUSTOM_VIDEO_BLOB_COUNTING_THRESHOLD = 0


def evaluate_custom_instance_tracking(
    video_id: str,
    tracking_data: list,
    *,
    annotations_path,
    img_dims: tuple[int, int] = (520, 960),
    match_iou_threshold: float = 0.0,
    score_iou_threshold: float = 0.5,
) -> dict:
    """
    Multi-instance Jaccard (IoU) for **all** GT objects vs dense blob masks.

    Uses per-frame uint16 pseudo-GT label maps (every instance each frame, including
    objects that enter after frame 0). Use ``avg_mean_gt_iou`` as the main score.
    """
    from genmatter.instance_seg_metrics import evaluate_instance_segmentation_tracking

    return evaluate_instance_segmentation_tracking(
        video_id,
        tracking_data,
        annotations_path=annotations_path,
        img_dims=img_dims,
        match_iou_threshold=match_iou_threshold,
        score_iou_threshold=score_iou_threshold,
    )


def evaluate_custom_tracking(
    video_id: str,
    tracking_data: list,
    *,
    annotations_path,
    img_dims=(520, 960),
    render_results_video: bool = False,
) -> dict:
    """
    Dense tracking vs SAM pseudo-GT; maximize ``avg_matter_weighted_jaccard_fixed``.

    Legacy blob-level metric (binary foreground). For all-object mask IoU use
    ``evaluate_custom_instance_tracking`` instead.
    """
    metrics, _viz = evaluate_single_davis_video(
        davis_name=video_id,
        multiple_genmatter_list=[tracking_data],
        annotations_path=str(annotations_path),
        counting_threshold=CUSTOM_VIDEO_BLOB_COUNTING_THRESHOLD,
        img_dims=img_dims,
        fps_list=None,
        render_results_video=render_results_video,
        experiment_save_dir=None,
        force_below_count_thresh_as_outlier=True,
        subsampled_indices=None,
    )
    return metrics


def create_genmatter_results_video(viz_data, annotations_path, img_dims, experiment_save_dir = None):
    """
    Create an animated visualization where all mask/assignment/particle/GT plots
    show the best single run (highest mean recall), but the line plot subplot
    overlays the framewise AVERAGE over all runs for each metric as dynamic 
    colored lines (not static horizontal lines). Particles replace "blobs"
    nomenclature throughout. Legends and supertitle are explicit.
    Improved legend placement, layout, and clearer text info for particle
    counts before/after Gibbs, etc. Main title is "GenMatter Tracking Results -- {trial name}".
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    import matplotlib as mpl
    import numpy as np

    per_trial_list = viz_data['per_trial_list']
    reference_particles = viz_data['reference_particles']
    num_frames = viz_data['num_frames']
    n_total_particles_in_scene = viz_data.get('n_total_particles_in_scene', None)
    n_all_reference_particles_before_threshold = viz_data.get('n_all_reference_particles_before_threshold', None)
    counting_threshold = viz_data.get('counting_threshold', None)
    total_reference_particles = viz_data.get('total_reference_particles', None)
    n_outlier_particles = viz_data.get('n_outlier_particles', 0)
    actual_outlier_pixels = viz_data.get('actual_outlier_pixels', 0)
    first_frame_n_blobs = viz_data.get('first_frame_n_blobs', None)
    num_true_unique_particles_first_frame = viz_data.get('num_true_unique_particles_first_frame', None)
    davis_name = viz_data.get('davis_name', 'Unknown-Stimulus')

    # Per-trial best run: matter-weighted scores (fixed frame-0 blob weights)
    recall_scores      = viz_data['matter_weighted_recall_fixed_scores']
    precision_scores   = viz_data['matter_weighted_precision_fixed_scores']
    iou_scores         = viz_data['matter_weighted_jaccard_fixed_scores']
    fpr_scores         = None  # not plotted (primary metrics are MW recall / precision / Jaccard)
    # Framewise mean across all runs
    per_frame_recall_mean    = viz_data.get('per_frame_recall_mean', None)
    per_frame_precision_mean = viz_data.get('per_frame_precision_mean', None)
    per_frame_fpr_mean       = viz_data.get('per_frame_fpr_mean', None)
    per_frame_iou_mean       = viz_data.get('per_frame_iou_mean', None)
    runs_n_trials            = viz_data.get('runs_n_trials', 1)
    selected_trial_idx       = viz_data.get('selected_trial_idx', None)  # zero-based index

    # --- Color schemes ---
    gt_cmap = LinearSegmentedColormap.from_list('gt', ['#f0f0f0', '#3498db'])
    pred_cmap = LinearSegmentedColormap.from_list('pred', ['#f0f0f0', '#e74c3c'])
    tp_color = [46 / 255, 204 / 255, 113 / 255]  # Green
    fp_color = [231 / 255, 76 / 255, 60 / 255]   # Red
    fn_color = [52 / 255, 152 / 255, 219 / 255]  # Blue
    tn_color = [149 / 255, 165 / 255, 166 / 255] # Gray

    def make_particle_cmap(n):
        hsvs = np.linspace(0, 1, n, endpoint=False)
        cm = mpl.colormaps.get_cmap("hsv")
        colors = [cm(h) for h in hsvs]
        np.random.shuffle(colors)
        return ListedColormap(colors)
    particle_cmap = make_particle_cmap(max(len(reference_particles), 2))

    fig = plt.figure(figsize=(22, 12))
    trial_str = f"(best run, trial {1 + selected_trial_idx if selected_trial_idx is not None else 1})"

    fig.suptitle(f"GenMatter Tracking Results -- {davis_name}", fontsize=22, fontweight='bold')

    plt.subplots_adjust(top=0.94)  # Give room for suptitle and subtitle

    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.25)
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_particles = fig.add_subplot(gs[0, 2])
    ax_classification = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[1, 2])

    # --- Set up axes ---
    gt_im = ax_gt.imshow(np.zeros(img_dims), cmap=gt_cmap, vmin=0, vmax=1)
    ax_gt.set_title(f"Ground Truth", fontsize=13)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])

    pred_im = ax_pred.imshow(np.zeros(img_dims), cmap=pred_cmap, vmin=0, vmax=1)
    ax_pred.set_title(f"Model Prediction (best run)", fontsize=13)
    ax_pred.set_xticks([]); ax_pred.set_yticks([])

    particle_im = ax_particles.imshow(np.zeros(img_dims), cmap=particle_cmap, interpolation='nearest', vmin=0, vmax=max(len(reference_particles)-1, 1))
    ax_particles.set_title(f"Reference Particle Assignment\n(best run)", fontsize=13)
    ax_particles.set_xticks([]); ax_particles.set_yticks([])

    classification_im = ax_classification.imshow(np.zeros((*img_dims, 3)))
    ax_classification.set_title('Classification Analysis (best run)', fontsize=13)
    ax_classification.set_xticks([]); ax_classification.set_yticks([])

    legend_elements = [
        patches.Patch(color=tp_color, label='True Positive'),
        patches.Patch(color=fp_color, label='False Positive'),
        patches.Patch(color=fn_color, label='False Negative'),
        patches.Patch(color=tn_color, label='True Negative')
    ]
    # Move classification legend to the very top of the plot, above the title
    legend = ax_classification.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.35, 1.35),
        ncol=4,
        fontsize=12,
        frameon=True,
        title="Classification"
    )
    fig.add_artist(legend)  # Ensure it stays above the suptitle

    # METRICS plot: overlay per-frame mean lines (dashed), and best run (solid)
    metrics_title = f"Tracking Metrics per Frame"
    ax_metrics.set_title(metrics_title, fontsize=15)
    ax_metrics.set_xlabel('Frame', fontsize=12)
    ax_metrics.set_ylabel('Score', fontsize=12)
    ax_metrics.set_xlim(0, num_frames-1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.grid(True, alpha=0.3)
    x_frames = np.arange(num_frames)

    # Per-frame mean as dashed lines (across runs, if multiple)
    lines_mean = {}
    if per_frame_recall_mean is not None:
        lines_mean['recall'], = ax_metrics.plot(x_frames, per_frame_recall_mean, '--', color='#2ecc71', alpha=0.38, lw=2, label='MW recall (mean)')
    if per_frame_precision_mean is not None:
        lines_mean['precision'], = ax_metrics.plot(x_frames, per_frame_precision_mean, '--', color='#3498db', alpha=0.38, lw=2, label='MW precision (mean)')
    if per_frame_iou_mean is not None:
        lines_mean['iou'], = ax_metrics.plot(x_frames, per_frame_iou_mean, '--', color='#9b59b6', alpha=0.38, lw=2, label='MW Jaccard (mean)')

    # Best run (this trial) as colored solid lines
    recall_line, = ax_metrics.plot([], [], '-', color='#2ecc71', linewidth=2, label='MW recall (best run)')
    precision_line, = ax_metrics.plot([], [], '-', color='#3498db', linewidth=2, label='MW precision (best run)')
    fpr_line, = ax_metrics.plot([], [], '-', color='#e74c3c', linewidth=2, label='_nolegend_')
    iou_line, = ax_metrics.plot([], [], '-', color='#9b59b6', linewidth=2, label='MW Jaccard (best run)')

    # Move the legend to the right side of the plot
    ax_metrics.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.35),
        ncol=1,
        fontsize=11,
        frameon=True,
        title="Metric"
    )

    # ax_info (textbox panel) put text heavily right
    ax_info.axis("off")
    info_text_obj = ax_info.text(
        0.11, 0.99, "", va="top", ha="left", fontsize=14, fontweight="medium", family='monospace', color="#34495e",
        transform=ax_info.transAxes, linespacing=1.32
    )

    def info_lines_fmt(frame_idx):
        lines = []
        total_particles = None
        if n_total_particles_in_scene is not None and first_frame_n_blobs is not None and num_true_unique_particles_first_frame is not None:
            total_particles = int(n_total_particles_in_scene + n_outlier_particles)
            lines.append(
                f"Frame: {frame_idx+1} / {num_frames}"
            )
            lines.append(
                f"First frame: n_particles (initialization): {first_frame_n_blobs}"
            )
            lines.append(
                f"Particles in scene (post- first frame Gibbs): {n_total_particles_in_scene}"
            )
        if n_all_reference_particles_before_threshold is not None:
            lines.append(
                f"Reference particles pre-threshold: {n_all_reference_particles_before_threshold}"
            )
        if total_reference_particles is not None:
            lines.append(
                f"Reference particles post-threshold: {total_reference_particles}"
            )
        if counting_threshold is not None:
            lines.append(
                f"Counting threshold: {counting_threshold}"
            )
        return "\n".join(lines)

    def update_frame(frame_idx):
        gt_mask = get_segmentation_mask(davis_name, frame_idx, annotations_path, img_dims, flatten=False)
        frame_assignments = np.array(per_trial_list[frame_idx]['blob_assignments']).reshape(img_dims)
        frame_n_blobs = per_trial_list[frame_idx]['n_blobs']

        pred_mask = np.isin(frame_assignments, reference_particles) & (frame_assignments != frame_n_blobs)
        gt_im.set_data(gt_mask.astype(float))
        pred_im.set_data(pred_mask.astype(float))

        particle_display = np.full(img_dims, fill_value=np.nan)
        if len(reference_particles) > 0:
            for idx, pid in enumerate(reference_particles):
                particle_display[frame_assignments == pid] = idx
        particle_im.set_data(particle_display)

        # 4-way classification
        classification_viz = np.zeros((*img_dims, 3))
        is_inside_gt = gt_mask.astype(bool)
        is_reference_particle = np.isin(frame_assignments, reference_particles) & (frame_assignments != frame_n_blobs)
        tp_mask = is_inside_gt & is_reference_particle
        fp_mask = is_inside_gt & ~is_reference_particle
        fn_mask = ~is_inside_gt & is_reference_particle
        tn_mask = ~is_inside_gt & ~is_reference_particle
        classification_viz[tp_mask] = tp_color
        classification_viz[fp_mask] = fp_color
        classification_viz[fn_mask] = fn_color
        classification_viz[tn_mask] = tn_color
        classification_im.set_data(classification_viz)

        frames_range = np.arange(frame_idx + 1)
        recall_line.set_data(frames_range, recall_scores[:frame_idx+1])
        precision_line.set_data(frames_range, precision_scores[:frame_idx+1])
        fpr_line.set_data([], [])
        iou_line.set_data(frames_range, iou_scores[:frame_idx+1])

        # Subplot titles
        ax_gt.set_title(f'Ground Truth (best run, Frame {frame_idx+1})', fontsize=13)
        ax_pred.set_title(f'Model Prediction (best run, Frame {frame_idx+1})', fontsize=13)
        ax_classification.set_title(f'Classification Analysis (best run, Frame {frame_idx+1})', fontsize=13)
        ax_particles.set_title(f"Reference Particle Assignment\n(best run, Frame {frame_idx+1})", fontsize=13)

        info_text_obj.set_text(info_lines_fmt(frame_idx))
        info_text_obj.set_color('#34495e')
        info_text_obj.set_fontsize(14)
        return [
            gt_im, pred_im, particle_im, classification_im, recall_line, precision_line, fpr_line, iou_line, info_text_obj
        ]

    from IPython.display import HTML, display

    anim = FuncAnimation(fig, update_frame, frames=num_frames, interval=200, blit=True, repeat=True)

    if experiment_save_dir is None:
        # In notebook/in-memory mode: display as HTML animation and return it
        plt.close(fig)
        print(f"💡 experiment_save_dir is None, returning inline HTML animation (not saving to disk)")
        return HTML(anim.to_html5_video())
    else:
        output_path = os.path.join(experiment_save_dir, f"{davis_name}_results_video.mp4")
        print(f"💾 Saving visualization to: {output_path}")
        video_saved = False

        try:
            anim.save(output_path, writer='ffmpeg', fps=5)
            video_saved = True
        except Exception as e1:
            print(f"⚠️  ffmpeg writer failed: {e1}")
            try:
                anim.save(output_path, writer='pillow', fps=5)
                video_saved = True
            except Exception as e2:
                print(f"⚠️  pillow (as mp4) writer failed: {e2}")
                try:
                    print(f"⚠️  Could not save video. Saving as GIF instead...")
                    gif_path = os.path.join(experiment_save_dir, f"{davis_name}_results_video.gif")
                    anim.save(gif_path, writer='pillow', fps=2)
                    print(f"💾 GIF saved to: {gif_path}")
                except Exception as e3:
                    print(f"❌ Failed to save GIF as well: {e3}")

        plt.close(fig)
        print(f"✅ Visualization complete!")
    

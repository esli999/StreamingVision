import numpy as np
import os
from threed_motion_vector_extraction import process_all_frames

# Define paths
videos_dir = 'DAVIS_VIDEO_DIRECTORY'
depths_dir = 'DAVIS_DEPTH_DIRECTORY'
motion_vectors_dir = 'DAVIS_3D_MOTIONS_DIRECTORY'

# Ensure output directory exists
os.makedirs(motion_vectors_dir, exist_ok=True)

# List of videos to process
videos = [
    "blackswan", "boat", "breakdance", "breakdance-flare", "bus", "car-roundabout",
    "car-shadow", "car-turn", "dance-jump", "dance-twirl", "drift-chicane", "drift-straight",
    "drift-turn", "elephant", "flamingo", "goat", "hike", "libby", "lucia", "mallard-water", "parkour",
    "rhino", "rollerblade"
]

# Process each video
for video in videos:
    print(f"Processing video: {video}")
    video_path = os.path.join(videos_dir, f"{video}.mp4")
    depth_path = os.path.join(depths_dir, f"{video}_depths.npz")
    
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        continue
    if not os.path.exists(depth_path):
        print(f"Warning: Depth file not found: {depth_path}")
        continue
        
    all_points_3d, all_motion_vectors_3d = process_all_frames(
        video_path, 
        depth_path, 
        save_dir=motion_vectors_dir
    )
    print(f"Successfully processed {video}")


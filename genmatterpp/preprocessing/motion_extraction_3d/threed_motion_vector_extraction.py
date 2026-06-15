import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import os
from torchvision.io import read_video
from torchvision.models.optical_flow import raft_large


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 960)),
        ]
    )
    batch = transforms(batch)
    return batch


# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"


model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()


# Function to convert 2D points with depth to 3D points
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


# Process all frames to get 3D points and motion vectors
def process_all_frames(video_name, depth_file, save_dir = None):
    """
    Process all frames to get 3D points and 3D motion vectors
    """
    # Extract the base name without extension
    base_name = os.path.splitext(os.path.basename(video_name))[0]
    
    # Load video frames
    frames, _, _ = read_video(video_name)
    frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    print(f"Total frames in video: {len(frames)}")
    
    #########################################################
    # Load depth data

    inverse_depth_data = np.load(depth_file)['depths']
    print(f"Loaded inverse depths with shape: {inverse_depth_data.shape}")
    
    # Clip the inverse depth to a reasonable range for processing
    clipped_inverse_depth = np.clip(inverse_depth_data, 200, 1200)
    
    # Convert inverse depth to actual depth (add small epsilon to avoid division by zero)
    epsilon = 1e-6
    depths = 3000.0 / (clipped_inverse_depth + epsilon)  # Adjusted scale to match target statistics
    
    # Clip to depth range that produces target coordinate scales
    depths = np.clip(depths, 3.0, 12.0)
    
    #########################################################

    
    # Determine number of frames to process (up to second-to-last frame)
    num_frames = min(len(frames) - 1, len(depths) - 1)
    print(f"Processing {num_frames} frames")
    
    all_points_3d = []
    all_motion_vectors_3d = []
    all_colors = []
    
    # Process frames in batches for efficiency
    batch_size = 16  # Adjust based on your GPU memory
    
    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        print(f"Processing frames {batch_start} to {batch_end-1}/{num_frames}")
        
        # Prepare batches of current and next frames
        img1_batch = preprocess(frames[batch_start:batch_end]).to(device)
        img2_batch = preprocess(frames[batch_start+1:batch_end+1]).to(device)
        
        # Get optical flow between current and next frames (batched)
        with torch.no_grad():
            flow_batch = model(img1_batch, img2_batch)
        
        # Process each frame in the batch
        for i in range(batch_end - batch_start):
            frame_idx = batch_start + i
            
            # Get the flow for this frame
            flow = flow_batch[-1][i].detach().cpu().numpy()  # Get the final flow prediction
            
            # Get depth for current frame
            depth_data = depths[frame_idx]
                
            # Resize depth to match flow dimensions if needed
            if depth_data.shape != (flow.shape[1], flow.shape[2]):
                from skimage.transform import resize
                depth_data = resize(depth_data, (flow.shape[1], flow.shape[2]), preserve_range=True)
            
            # Create coordinate grids
            h, w = flow.shape[1], flow.shape[2]
            y_grid, x_grid = np.mgrid[0:h, 0:w]
            
            # Calculate principal points from image dimensions
            cx = w / 2
            cy = h / 2
            
            # Get 3D points for current frame
            points_3d = unproject_points(x_grid, y_grid, depth_data, cx=cx, cy=cy)
            
            # Extract color information from the current frame
            current_frame = frames[frame_idx].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
            
            # Resize color frame to match flow dimensions if needed
            if current_frame.shape[:2] != (h, w):
                from skimage.transform import resize
                current_frame = resize(current_frame, (h, w, 3), preserve_range=True)
            
            # Get color values for each point
            colors = current_frame.astype(np.uint8)
            
            # Calculate destination coordinates using optical flow
            x_dest = x_grid + flow[0]
            y_dest = y_grid + flow[1]
            
            # Sample depth at destination points (need to handle out-of-bounds)
            x_dest_clipped = np.clip(x_dest, 0, w-1).astype(int)
            y_dest_clipped = np.clip(y_dest, 0, h-1).astype(int)
            
            # Get depth for next frame
            next_depth = depths[frame_idx+1]
            if next_depth.shape != (h, w):
                from skimage.transform import resize
                next_depth = resize(next_depth, (h, w), preserve_range=True)
            
            # Sample depth at destination points
            dest_depth = next_depth[y_dest_clipped, x_dest_clipped]
            
            # Get 3D points for destination
            dest_points_3d = unproject_points(x_dest, y_dest, dest_depth, cx=cx, cy=cy)
            
            # Calculate 3D motion vectors
            motion_vectors_3d = dest_points_3d - points_3d
            
            all_points_3d.append(points_3d)
            all_motion_vectors_3d.append(motion_vectors_3d)
            all_colors.append(colors)
    
    # Store camera intrinsics for reprojection
    fx, fy = 520, 520  # Focal lengths used in unproject_points
    intrinsics = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'width': w,
        'height': h
    }
    
    if save_dir is not None:
        # Save results with the same prefix as the input video
        output_file = f"{save_dir}/{base_name}_3d_motion.npz"
    else:
        output_file = f"{base_name}_3d_motion.npz"
    np.savez(output_file, 
             points_3d=np.array(all_points_3d), 
             motion_vectors_3d=np.array(all_motion_vectors_3d),
             colors=np.array(all_colors),
             intrinsics=intrinsics)
    print(f"Results saved to {output_file}")
    
    return all_points_3d, all_motion_vectors_3d

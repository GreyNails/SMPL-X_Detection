import json
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from detrsmpl.core.visualization.visualize_smpl import render_smpl
from detrsmpl.models.body_models.builder import build_body_model


def parse_pose_parameters(pose_array):
    """
    Parse the 156-dimensional pose array into SMPLX components.
    pose_array: [body_pose(63) + lhand_pose(45) + rhand_pose(45) + root_pose(3)]
    """
    pose_array = np.array(pose_array)
    
    body_pose = pose_array[:63]      # 63 dims
    lhand_pose = pose_array[63:108]  # 45 dims
    rhand_pose = pose_array[108:153] # 45 dims
    root_pose = pose_array[153:156]  # 3 dims
    
    return {
        'body_pose': body_pose,
        'left_hand_pose': lhand_pose,
        'right_hand_pose': rhand_pose,
        'global_orient': root_pose
    }


def get_image_resolution(sample):
    """
    Calculate image resolution from type=5 element.
    resolution = (type=5 element's width * canvas_width, 
                  type=5 element's height * canvas_height)
    """
    type_list = sample['type']
    canvas_width = sample['canvas_width']
    canvas_height = sample['canvas_height']
    
    # Find index where type == 5 (human element)
    human_idx = None
    for idx, t in enumerate(type_list):
        if t == 5:
            human_idx = idx
            break
    
    if human_idx is None:
        # Default resolution if no human element found
        return (1024, 1024)
    
    # Get width and height at human element index
    element_width = sample['width'][human_idx]
    element_height = sample['height'][human_idx]
    
    # Calculate resolution
    img_width = int(element_width * canvas_width)
    img_height = int(element_height * canvas_height)
    
    return (img_height, img_width)


def render_sample(sample, body_model, output_dir, sample_idx, device='cpu'):
    """
    Render a single sample from the JSON data.
    """
    # Extract pose and camera
    pose_list = sample['pose'][0]  # First pose in the list
    camera_list = sample['camera'][0]  # First camera in the list
    
    # Parse pose parameters
    pose_dict = parse_pose_parameters(pose_list)
    
    # Convert to torch tensors and reshape properly
    body_pose = torch.tensor(pose_dict['body_pose'], dtype=torch.float32, device=device).view(1, 21, 3)
    lhand_pose = torch.tensor(pose_dict['left_hand_pose'], dtype=torch.float32, device=device).view(1, 15, 3)
    rhand_pose = torch.tensor(pose_dict['right_hand_pose'], dtype=torch.float32, device=device).view(1, 15, 3)
    root_pose = torch.tensor(pose_dict['global_orient'], dtype=torch.float32, device=device).view(1, 1, 3)
    
    # Fixed parameters - ensure they're on the correct device
    jaw_pose = torch.zeros(1, 1, 3, dtype=torch.float32, device=device)
    leye_pose = torch.zeros(1, 1, 3, dtype=torch.float32, device=device)
    reye_pose = torch.zeros(1, 1, 3, dtype=torch.float32, device=device)
    betas = torch.zeros(1, 10, dtype=torch.float32, device=device)
    expression = torch.zeros(1, 10, dtype=torch.float32, device=device)
    
    # Camera translation
    cam_trans = torch.tensor(camera_list, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Concatenate all poses: global_orient + body_pose + jaw + leye + reye + lhand + rhand
    # Total: 1*3 + 21*3 + 1*3 + 1*3 + 1*3 + 15*3 + 15*3 = 165
    full_pose = torch.cat([
        root_pose.reshape(1, -1),    # 3
        body_pose.reshape(1, -1),    # 63
        jaw_pose.reshape(1, -1),     # 3
        leye_pose.reshape(1, -1),    # 3
        reye_pose.reshape(1, -1),    # 3
        lhand_pose.reshape(1, -1),   # 45
        rhand_pose.reshape(1, -1),   # 45
    ], dim=1)  # Total: 165
    
    # Use tensor format instead of dict
    poses = full_pose
    
    # Get image resolution
    resolution = get_image_resolution(sample)
    
    # Output path
    sample_id = sample.get('id', f'sample_{sample_idx}')
    output_path = os.path.join(output_dir, f'{sample_id}_rendered.png')
    
    # Debug info
    print(f"  - Resolution: {resolution}")
    print(f"  - Poses shape: {poses.shape}")
    print(f"  - Betas shape: {betas.shape}")
    print(f"  - Cam trans shape: {cam_trans.shape}")
    print(f"  - Device: {device}")
    
    # Create white background image (RGB, 3 channels)
    # Shape: (1, height, width, 3) for RGB
    white_bg = torch.ones(1, resolution[0], resolution[1], 3, 
                          dtype=torch.float32, device=device)
    
    # Render parameters
    K = np.array([
        [5000, 0, resolution[1]/2],
        [0, 5000, resolution[0]/2],
        [0, 0, 1]
    ])
    
    try:
        # Render the SMPL model with white background
        render_smpl(
            poses=poses,
            betas=betas,
            transl=cam_trans,
            body_model=body_model,
            K=K,
            R=None,
            T=None,
            projection='perspective',
            convention='opencv',
            in_ndc=False,
            render_choice='hq',  # hq outputs (frame, h, w, 4) with alpha channel
            palette='white',
            resolution=resolution,
            alpha=1.0,  # Mesh fully opaque
            output_path=output_path,
            overwrite=True,
            no_grad=True,
            device=device,
            verts=None,
            origin_frames=None,
            frame_list=None,
            image_array=white_bg,  # Use white background
            return_tensor=False,
        )
        
        # Post-process to make white background transparent
        try:
            from PIL import Image
            img = Image.open(output_path)
            
            # Convert to RGBA if not already
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Convert to numpy for processing
            img_array = np.array(img)
            
            # Make white/near-white pixels transparent
            # Pixels with RGB values close to (255, 255, 255) become transparent
            white_threshold = 250
            white_mask = (img_array[:, :, 0] > white_threshold) & \
                        (img_array[:, :, 1] > white_threshold) & \
                        (img_array[:, :, 2] > white_threshold)
            
            img_array[white_mask, 3] = 0  # Set alpha to 0 for white pixels
            
            # Save with transparency
            img_transparent = Image.fromarray(img_array, 'RGBA')
            img_transparent.save(output_path, 'PNG')
            
            print(f"Rendered sample {sample_id} with transparent background")
            
        except Exception as e:
            print(f"  Warning: Could not add transparency: {e}")
            print(f"  Rendered sample {sample_id} with white background")
        
    except Exception as e:
        print(f"Error rendering sample {sample_id}: {str(e)}")
        import traceback
        traceback.print_exc()


def main(json_path, output_dir, body_model_path='/home/dell/DataTool-HumanCentric/detection_aios/AiOS/data/body_models/smplx', device='cpu'):
    """
    Main function to render all samples from JSON file.
    
    Args:
        json_path: Path to the input JSON file
        output_dir: Directory to save rendered images
        body_model_path: Path to SMPLX body model files
        device: Device to use ('cpu' or 'cuda')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build SMPLX body model
    body_model_cfg = dict(
        type='smplx',
        keypoint_src='smplx',
        num_expression_coeffs=10,
        num_betas=10,
        gender='neutral',
        keypoint_dst='smplx_137',
        model_path=body_model_path,
        use_pca=False,
        use_face_contour=True,
    )
    
    print(f"Building SMPLX body model on {device}...")
    body_model = build_body_model(body_model_cfg).to(device)
    
    # Load JSON data
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Process each sample
    print(f"Processing {len(data)} samples...")
    for idx, sample in enumerate(data):
        print(f"\nProcessing sample {idx+1}/{len(data)}")
        render_sample(sample, body_model, output_dir, idx, device)
    
    print(f"\nDone! All rendered images saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    json_path = "/storage/HCL_data/crello/detections/dataset_test_v2_with_pose.json"  # Path to your JSON file
    output_dir = "rendered_outputs"  # Output directory for rendered images
    
    # Use CPU instead of CUDA
    main(json_path, output_dir, device='cpu')
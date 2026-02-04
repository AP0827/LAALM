
import os
import sys
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add paths
sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'auto_avsr'))

from video_utils import SafeVideoPreprocessor
from auto_avsr.inference_wrapper import InferencePipelineWithConfidence

def generate_figure(video_path, output_path):
    print(f"Processing video: {video_path}")
    
    # 1. Run Safe Preprocessing first (Standardization)
    print("Running SafeVideoPreprocessor...")
    preprocessor = SafeVideoPreprocessor(output_dir=Path("temp_vis"))
    processed_path, success = preprocessor.process(
        video_path, 
        "sample_vis", 
        apply_enhancement=True, # Show off the enhancement
        denoise_strength=5      # Moderate denoise
    )
    
    if not success:
        print("Preprocessing failed")
        return

    # 2. Initialize VSR Pipeline components to get crops
    print("Initializing VSR components...")
    model_path = "auto_avsr/pretrained_models/vsr_trlrs2lrs3vox2avsp_base.pth"
    vsr_pipeline = InferencePipelineWithConfidence(model_path, detector="retinaface")
    
    # 3. Extract Frames
    print("Extracting frames...")
    
    # A. Original Frame
    cap_orig = cv2.VideoCapture(video_path)
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_idx = total_frames // 2
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, frame_orig = cap_orig.read()
    frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    
    # B. Standardized Frame
    cap_std = cv2.VideoCapture(processed_path)
    cap_std.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx) # Approx same frame
    ret, frame_std = cap_std.read()
    frame_std = cv2.cvtColor(frame_std, cv2.COLOR_BGR2RGB)
    
    # C. Mouth Crop (Using pipeline internals)
    # Load video using pipeline's loader
    video_tensor = vsr_pipeline.load_video(processed_path)
    landmarks = vsr_pipeline.landmarks_detector(video_tensor)
    video_crops = vsr_pipeline.video_process(video_tensor, landmarks)
    
    # video_crops is (T, 88, 88) usually, grayscale
    # Let's pick the middle crop
    mid_crop = video_crops[len(video_crops)//2]
    
    # 4. Create Plot
    print("Creating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Original
    axes[0].imshow(frame_orig)
    axes[0].set_title("Original Input (Raw)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Plot Standardized
    axes[1].imshow(frame_std)
    axes[1].set_title("Safe Preprocessed\n(25fps, 720p, Enhanced)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Plot Crop
    axes[2].imshow(mid_crop, cmap='gray')
    axes[2].set_title("VSR Input Crop\n(88x88 ROI)", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")

    # Cleanup
    if os.path.exists(processed_path):
        os.remove(processed_path)

if __name__ == "__main__":
    # Choose a specific GT video from LRS3
    # 0ZfSOArXbGQ_00003_gt.mp4 was found earlier
    video_source = "/home/asish/Downloads/LRS3/LRS3/good/0ZfSOArXbGQ_00003_gt.mp4"
    output_target = "paper/speech_sample.png"
    
    generate_figure(video_source, output_target)

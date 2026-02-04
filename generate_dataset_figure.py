
import os
import sys
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np
import librosa
from pathlib import Path

# Add paths
sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'auto_avsr'))

from video_utils import SafeVideoPreprocessor
from auto_avsr.inference_wrapper import InferencePipelineWithConfidence

def extract_data(video_path, vsr_pipeline):
    # 1. Audio Waveform
    y, sr = librosa.load(video_path, sr=16000)
    
    # 2. Visual Crops using pipeline
    # Load video tensor
    video_tensor = vsr_pipeline.load_video(video_path)
    landmarks = vsr_pipeline.landmarks_detector(video_tensor)
    video_crops = vsr_pipeline.video_process(video_tensor, landmarks)
    
    # Select 5 evenly spaced frames
    num_frames = len(video_crops)
    indices = np.linspace(0, num_frames-1, 5, dtype=int)
    selected_frames = [video_crops[i] for i in indices]
    
    return y, sr, selected_frames

def generate_figure(video_paths, output_path):
    print("Initializing pipeline...")
    model_path = "auto_avsr/pretrained_models/vsr_trlrs2lrs3vox2avsp_base.pth"
    vsr_pipeline = InferencePipelineWithConfidence(model_path, detector="retinaface")
    
    num_samples = len(video_paths)
    fig = plt.figure(figsize=(15, 4 * num_samples))
    
    # Layout: GridSpec
    # For each sample: Top row = 5 images, Bottom row = Waveform
    gs = fig.add_gridspec(num_samples * 2, 5, height_ratios=[1, 0.5]*num_samples)
    
    for i, video_path in enumerate(video_paths):
        print(f"Processing sample {i+1}: {video_path}")
        
        try:
            waveform, sr, frames = extract_data(video_path, vsr_pipeline)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
            
        row_img_idx = i * 2
        row_wave_idx = i * 2 + 1
        
        # Plot 5 Frames
        for k in range(5):
            ax = fig.add_subplot(gs[row_img_idx, k])
            ax.imshow(frames[k], cmap='gray')
            ax.axis('off')
            if k == 0:
                ax.set_ylabel(f"Sample {i+1}", fontsize=12, fontweight='bold', labelpad=10)

        # Plot Waveform (spanning all columns)
        ax_wave = fig.add_subplot(gs[row_wave_idx, :])
        time_axis = np.linspace(0, len(waveform)/sr, len(waveform))
        ax_wave.plot(time_axis, waveform, color='steelblue', alpha=0.8, linewidth=0.5)
        ax_wave.set_xlim(0, len(waveform)/sr)
        ax_wave.set_ylim(-1, 1)
        ax_wave.axis('off')
        # Add a subtle grid
        ax_wave.grid(True, alpha=0.3)

    plt.suptitle("Multimodal Input Samples (Mouth ROI + Audio)", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    # Select 3 distinct videos from the good set
    base_dir = "/home/asish/Downloads/LRS3/LRS3/good"
    videos = [
        os.path.join(base_dir, "0ZfSOArXbGQ_00003_gt.mp4"),
        os.path.join(base_dir, "NqOjj1FCcVY_00007_gt.mp4"),
        os.path.join(base_dir, "SSzRfSJTNW4_00001_gt.mp4")
    ]
    
    output_target = "paper/dataset_samples_generated.png"
    generate_figure(videos, output_target)

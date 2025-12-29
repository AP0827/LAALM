#!/usr/bin/env python3
"""
Generate Figure 2: Representative visual speech samples
Shows mouth-region crops with corresponding audio waveforms
"""

import sys
sys.path.insert(0, 'auto_avsr')

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import librosa
import librosa.display

from preparation.detectors.retinaface.detector import LandmarksDetector
from preparation.detectors.retinaface.video_process import VideoProcess


def load_and_process_video(video_path, num_frames=5):
    """Load video and extract mouth crops."""
    # Load video
    video = torchvision.io.read_video(video_path, pts_unit="sec")[0].numpy()
    
    # Initialize detector
    landmarks_detector = LandmarksDetector(device="cuda:0" if torch.cuda.is_available() else "cpu")
    video_process = VideoProcess(convert_gray=False)
    
    # Process video
    landmarks = landmarks_detector(video)
    cropped_video = video_process(video, landmarks)
    
    # Sample frames evenly
    total_frames = len(cropped_video)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    return cropped_video[indices]


def load_audio_waveform(audio_path, duration=2.0):
    """Load audio and return waveform."""
    y, sr = librosa.load(audio_path, duration=duration)
    return y, sr


def create_figure2(video_files, audio_files, output_path="figure2.png"):
    """
    Create Figure 2 with mouth crops and audio waveforms.
    
    Args:
        video_files: List of video file paths
        audio_files: List of corresponding audio file paths
        output_path: Where to save the figure
    """
    n_samples = len(video_files)
    frames_per_sample = 5
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(n_samples * 2, frames_per_sample, figure=fig, hspace=0.4, wspace=0.2)
    
    fig.suptitle('Example samples from visual speech datasets\nillustrating mouth-region crops and corresponding\nutterance-level audio segments.', 
                 fontsize=12, fontweight='normal', style='italic', y=0.98)
    
    for i, (video_path, audio_path) in enumerate(zip(video_files, audio_files)):
        print(f"Processing sample {i+1}/{n_samples}: {os.path.basename(video_path)}")
        
        try:
            # Extract mouth crops
            mouth_crops = load_and_process_video(video_path, num_frames=frames_per_sample)
            
            # Display mouth crops
            for j in range(frames_per_sample):
                ax = fig.add_subplot(gs[i*2, j])
                frame = mouth_crops[j]
                
                # Convert from tensor to displayable format
                if isinstance(frame, torch.Tensor):
                    frame = frame.numpy()
                
                # If grayscale, squeeze
                if frame.shape[0] == 1:
                    frame = frame.squeeze(0)
                    ax.imshow(frame, cmap='gray')
                else:
                    # RGB image
                    if frame.shape[0] == 3:
                        frame = np.transpose(frame, (1, 2, 0))
                    ax.imshow(frame)
                
                ax.axis('off')
                
                # Add frame label on first frame
                if j == 0:
                    ax.text(-0.1, 0.5, f'Sample {i+1}', 
                           transform=ax.transAxes, rotation=90,
                           va='center', ha='right', fontsize=10, fontweight='bold')
            
            # Load and display audio waveform
            y, sr = load_audio_waveform(audio_path, duration=2.0)
            
            ax_audio = fig.add_subplot(gs[i*2+1, :])
            librosa.display.waveshow(y, sr=sr, ax=ax_audio, color='steelblue', alpha=0.7)
            ax_audio.set_xlabel('Time (s)', fontsize=9)
            ax_audio.set_ylabel('Amplitude', fontsize=9)
            ax_audio.set_ylim([-1, 1])
            ax_audio.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    # Add figure caption at bottom
    fig.text(0.5, 0.02, 'Fig. 2: Representative visual speech samples used for training\nand evaluation, demonstrating variability in pose, articulation,\nand recording conditions.', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Select diverse samples from your dataset
    base_video = "samples/video"
    base_audio = "samples/audio"
    
    # Choose 3-4 representative samples with variety
    samples = [
        ("lwwz9s.mpg", "lwwz9s.wav"),  # Different phonemes
        ("bbaf2n.mpg", "bbaf2n.wav"),  # Different speaker
        ("bgaa6n.mpg", "bgaa6n.wav"),  # Different conditions
    ]
    
    video_files = [os.path.join(base_video, v) for v, _ in samples]
    audio_files = [os.path.join(base_audio, a) for _, a in samples]
    
    # Filter to existing files
    valid_pairs = [(v, a) for v, a in zip(video_files, audio_files) 
                   if os.path.exists(v) and os.path.exists(a)]
    
    if not valid_pairs:
        print("No valid video-audio pairs found!")
        print("Available videos:")
        for f in os.listdir(base_video)[:10]:
            print(f"  - {f}")
        sys.exit(1)
    
    video_files, audio_files = zip(*valid_pairs)
    
    print(f"Creating Figure 2 with {len(video_files)} samples...")
    create_figure2(video_files, audio_files, output_path="paper_figure2.png")

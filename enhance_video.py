#!/usr/bin/env python3
"""
Video enhancement script for auto_avsr preprocessing.
Improves video quality for better VSR accuracy.
"""

import cv2
import numpy as np
import sys
from pathlib import Path


def enhance_video(input_path: str, output_path: str, target_fps: int = 25):
    """
    Enhance video for VSR:
    - Normalize frame rate to 25fps
    - Increase contrast/brightness
    - Denoise
    - Sharpen mouth region
    """
    cap = cv2.VideoCapture(input_path)
    
    # Get original properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original: {width}x{height} @ {orig_fps}fps, {total_frames} frames")
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    # Frame sampling ratio
    frame_ratio = orig_fps / target_fps
    frame_idx = 0
    processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames to match target fps
        if frame_idx % int(frame_ratio) == 0:
            # Enhancement pipeline
            enhanced = enhance_frame(frame)
            out.write(enhanced)
            processed += 1
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"Enhanced: {width}x{height} @ {target_fps}fps, {processed} frames")
    print(f"Saved to: {output_path}")


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Apply enhancement to single frame."""
    
    # 1. Convert to LAB color space for better processing
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # 3. Merge and convert back
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 4. Denoise while preserving edges
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    # 5. Sharpen
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python enhance_video.py <input_video> [output_video]")
        print("Example: python enhance_video.py samples/video/swwp4p.mpg samples/video/swwp4p_enhanced.mp4")
        sys.exit(1)
    
    input_video = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_video = sys.argv[2]
    else:
        # Auto-generate output name
        input_path = Path(input_video)
        output_video = str(input_path.parent / f"{input_path.stem}_enhanced.mp4")
    
    enhance_video(input_video, output_video)

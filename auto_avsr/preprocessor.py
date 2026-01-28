"""
Video Preprocessing for auto_avsr

This module enhances video quality before sending to auto_avsr for lip-reading.
It applies frame-level enhancements to improve VSR accuracy:
- Brightness/Contrast normalization (CLAHE)
- Sharpening to improve facial feature clarity
- Frame stabilization (future enhancement)

Author: LAALM Project
Date: January 2026
"""

import cv2
import numpy as np
import tempfile
import os
from typing import Optional


class VideoPreprocessor:
    """
    Preprocesses video files to enhance quality for visual speech recognition.
    """
    
    def __init__(
        self,
        apply_clahe: bool = True,
        apply_sharpen: bool = True,
        target_fps: Optional[int] = None
    ):
        """
        Initialize video preprocessor.
        
        Args:
            apply_clahe: Whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            apply_sharpen: Whether to apply sharpening filter
            target_fps: Target frame rate (None = keep original)
        """
        self.apply_clahe = apply_clahe
        self.apply_sharpen = apply_sharpen
        self.target_fps = target_fps
        
        # CLAHE settings
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Sharpening kernel
        self.sharpen_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance a single frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Enhanced frame (BGR)
        """
        # Convert to LAB color space for better CLAHE
        if self.apply_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l = self.clahe.apply(l)
            
            # Merge back
            lab = cv2.merge([l, a, b])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply sharpening
        if self.apply_sharpen:
            frame = cv2.filter2D(frame, -1, self.sharpen_kernel)
        
        return frame
    
    def process(self, input_path: str) -> str:
        """
        Process video file and save to temporary file.
        
        Args:
            input_path: Path to input video file
            
        Returns:
            Path to processed temporary video file
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")
        
        # Create temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)
        
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {input_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Use target FPS if specified
            output_fps = self.target_fps if self.target_fps else fps
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, output_fps, (width, height))
            
            if not out.isOpened():
                raise ValueError(f"Failed to create output video writer")
            
            frame_count = 0
            
            # Process each frame
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Enhance frame
                enhanced_frame = self._enhance_frame(frame)
                
                # Write frame
                out.write(enhanced_frame)
                frame_count += 1
            
            # Release resources
            cap.release()
            out.release()
            
            print(f"   ✓ Processed {frame_count} frames")
            print(f"   ✓ Original: {width}x{height} @ {fps:.1f} FPS")
            print(f"   ✓ Enhanced: {width}x{height} @ {output_fps:.1f} FPS")
            
            return temp_path
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocessor.py <input_video>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    
    preprocessor = VideoPreprocessor()
    output_path = preprocessor.process(input_video)
    
    print(f"\nProcessed video saved to: {output_path}")

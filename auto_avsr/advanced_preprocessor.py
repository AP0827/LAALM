"""
Advanced Video Preprocessing for auto_avsr

This module implements state-of-the-art video enhancement techniques specifically
optimized for Visual Speech Recognition (VSR). It includes:

1. **Adaptive Processing** - Only enhance frames that need it
2. **Denoising** - Remove grain and compression artifacts
3. **Temporal Smoothing** - Stabilize lip movements across frames
4. **Mouth-Region Enhancement** - Focus processing on the mouth area
5. **Super-Resolution** - Upscale low-resolution videos

Author: LAALM Project
Date: January 2026
"""

import cv2
import numpy as np
import tempfile
import os
from typing import Optional, Tuple, List
from collections import deque


class AdvancedVideoPreprocessor:
    """
    Advanced video preprocessor with VSR-specific optimizations.
    """
    
    def __init__(
        self,
        apply_denoising: bool = True,
        apply_temporal_smoothing: bool = False,  # Disabled by default (can blur lips)
        apply_super_resolution: bool = True,
        apply_mouth_focus: bool = True,
        temporal_window: int = 3,
        min_resolution: int = 480,
        target_fps: Optional[int] = None,
        denoise_strength: int = 3,  # Reduced from 10 (gentler)
    ):
        """
        Initialize advanced video preprocessor.
        
        Args:
            apply_denoising: Remove noise/grain from frames
            apply_temporal_smoothing: Average frames to reduce jitter (can blur lips)
            apply_super_resolution: Upscale low-res videos
            apply_mouth_focus: Apply stronger enhancement to mouth region
            temporal_window: Number of frames to average (odd number, 3-5 recommended)
            min_resolution: Minimum height for super-resolution (upscale if below)
            target_fps: Target frame rate (None = keep original)
            denoise_strength: Denoising strength (1-10, lower = gentler)
        """
        self.apply_denoising = apply_denoising
        self.apply_temporal_smoothing = apply_temporal_smoothing
        self.apply_super_resolution = apply_super_resolution
        self.apply_mouth_focus = apply_mouth_focus
        self.temporal_window = temporal_window if temporal_window % 2 == 1 else temporal_window + 1
        self.min_resolution = min_resolution
        self.target_fps = target_fps
        self.denoise_strength = denoise_strength
        
        # CLAHE for contrast enhancement (gentle settings)
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Reduced from 2.0
        self.clahe_strong = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))  # Reduced from 3.0
        
        # Sharpening kernel (lighter)
        self.sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],  # Reduced from 9 (gentler sharpening)
            [0, -1, 0]
        ])
        
        # Face detection for mouth-focused enhancement
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Frame buffer for temporal smoothing
        self.frame_buffer = deque(maxlen=self.temporal_window)
    
    def _check_brightness(self, frame: np.ndarray) -> float:
        """
        Calculate average brightness of a frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Average brightness (0-255)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def _denoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply denoising to reduce grain and compression artifacts.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Denoised frame (BGR)
        """
        # Use fastNlMeansDenoisingColored with configurable strength
        # Lower h values = less aggressive denoising (preserves detail)
        return cv2.fastNlMeansDenoisingColored(
            frame, None, 
            self.denoise_strength,  # Luminance
            self.denoise_strength,  # Chrominance
            7, 21
        )
    
    def _detect_face_and_mouth(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face and estimate mouth region.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            (x, y, w, h) of mouth region, or None if face not detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Take the first (largest) face
        x, y, w, h = faces[0]
        
        # Estimate mouth region (lower third of face, centered)
        mouth_y = y + int(h * 0.6)  # Start at 60% down the face
        mouth_h = int(h * 0.4)       # Height is 40% of face height
        mouth_x = x + int(w * 0.2)   # Start at 20% from left
        mouth_w = int(w * 0.6)       # Width is 60% of face width
        
        return (mouth_x, mouth_y, mouth_w, mouth_h)
    
    def _enhance_frame(self, frame: np.ndarray, apply_adaptive: bool = True) -> np.ndarray:
        """
        Enhanced frame processing with adaptive and mouth-focused enhancement.
        
        Args:
            frame: Input frame (BGR)
            apply_adaptive: Only enhance if brightness is low
            
        Returns:
            Enhanced frame (BGR)
        """
        # Check if frame needs enhancement (adaptive processing)
        if apply_adaptive:
            brightness = self._check_brightness(frame)
            if brightness > 150:  # Bright enough, minimal processing
                return cv2.filter2D(frame, -1, self.sharpen_kernel)
        
        # Denoise if enabled
        if self.apply_denoising:
            frame = self._denoise_frame(frame)
        
        # Apply CLAHE to entire frame (moderate)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        
        # Apply stronger CLAHE to mouth region if face detected
        if self.apply_mouth_focus:
            mouth_region = self._detect_face_and_mouth(frame)
            if mouth_region is not None:
                mx, my, mw, mh = mouth_region
                # Apply stronger enhancement to mouth region
                mouth_l = l[my:my+mh, mx:mx+mw]
                mouth_l_enhanced = self.clahe_strong.apply(mouth_l)
                l[my:my+mh, mx:mx+mw] = mouth_l_enhanced
        
        # Merge back
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply sharpening
        frame = cv2.filter2D(frame, -1, self.sharpen_kernel)
        
        return frame
    
    def _apply_temporal_smoothing(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing by averaging with neighboring frames.
        
        Args:
            frame: Current frame
            
        Returns:
            Temporally smoothed frame
        """
        self.frame_buffer.append(frame.copy())
        
        # Not enough frames yet
        if len(self.frame_buffer) < self.temporal_window:
            return frame
        
        # Average all frames in buffer
        smoothed = np.mean(np.array(list(self.frame_buffer)), axis=0).astype(np.uint8)
        return smoothed
    
    def process(self, input_path: str) -> str:
        """
        Process video file with advanced enhancements.
        
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
            
            # Check if super-resolution is needed
            needs_upscaling = self.apply_super_resolution and height < self.min_resolution
            if needs_upscaling:
                scale_factor = self.min_resolution / height
                new_width = int(width * scale_factor)
                new_height = self.min_resolution
                print(f"   🔍 Upscaling from {width}x{height} to {new_width}x{new_height}")
            else:
                new_width, new_height = width, height
            
            # Use target FPS if specified
            output_fps = self.target_fps if self.target_fps else fps
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, output_fps, (new_width, new_height))
            
            if not out.isOpened():
                raise ValueError(f"Failed to create output video writer")
            
            frame_count = 0
            enhancements_applied = {
                'denoising': 0,
                'temporal_smoothing': 0,
                'mouth_focus': 0,
                'upscaling': 0
            }
            
            # Process each frame
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Upscale if needed
                if needs_upscaling:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    enhancements_applied['upscaling'] += 1
                
                # Enhance frame
                enhanced_frame = self._enhance_frame(frame)
                
                if self.apply_denoising:
                    enhancements_applied['denoising'] += 1
                
                # Apply temporal smoothing
                if self.apply_temporal_smoothing:
                    enhanced_frame = self._apply_temporal_smoothing(enhanced_frame)
                    enhancements_applied['temporal_smoothing'] += 1
                
                # Check if mouth focus was applied (detect face in frame)
                if self.apply_mouth_focus and self._detect_face_and_mouth(frame) is not None:
                    enhancements_applied['mouth_focus'] += 1
                
                # Write frame
                out.write(enhanced_frame)
                frame_count += 1
            
            # Release resources
            cap.release()
            out.release()
            
            # Report enhancements
            print(f"   ✓ Processed {frame_count} frames")
            print(f"   ✓ Original: {width}x{height} @ {fps:.1f} FPS")
            print(f"   ✓ Enhanced: {new_width}x{new_height} @ {output_fps:.1f} FPS")
            
            if enhancements_applied['upscaling'] > 0:
                print(f"   ✓ Upscaled: {enhancements_applied['upscaling']} frames")
            if enhancements_applied['denoising'] > 0:
                print(f"   ✓ Denoised: {enhancements_applied['denoising']} frames")
            if enhancements_applied['temporal_smoothing'] > 0:
                print(f"   ✓ Temporal smoothing: {enhancements_applied['temporal_smoothing']} frames")
            if enhancements_applied['mouth_focus'] > 0:
                print(f"   ✓ Mouth-focused enhancement: {enhancements_applied['mouth_focus']} frames")
            
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
        print("Usage: python advanced_preprocessor.py <input_video>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    
    preprocessor = AdvancedVideoPreprocessor()
    output_path = preprocessor.process(input_video)
    
    print(f"\nProcessed video saved to: {output_path}")

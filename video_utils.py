import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

class SafeVideoPreprocessor:
    """
    Handles "Safe" video preprocessing for VSR.
    
    Goals:
    1. Standardize FPS to 25 (Required by many VSR models).
    2. Standardize Resolution to 720p (Good balance of detail/speed).
    3. Apply MILD enhancement (Unsharp Mask) to define lip boundaries.
    4. Ensure Audio is preserved/transcoded correctly.
    
    Avoiding:
    - Heavy denoising (removes texture).
    - Aggressive super-resolution (adds artifacts).
    """
    
    def __init__(self, output_dir: Path = Path("uploads")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
    def process(self, input_path: str, filename_stem: str, apply_enhancement: bool = True, denoise_strength: int = 0, temporal_smoothing: bool = False) -> Tuple[str, bool]:
        """
        Process the video.
        
        Args:
            input_path: Path to input video.
            filename_stem: Base filename for output.
            apply_enhancement: Whether to apply unsharp mask and denoising.
            denoise_strength: Strength of denoising (0-10).
            temporal_smoothing: Whether to apply extra temporal smoothing.
            
        Returns:
            (output_path, success_flag)
        """
        # Create unique filename based on params to avoid cache collisions logic if needed
        # For now, simplistic naming (or overwrite)
        output_filename = f"processed_{filename_stem}_s{denoise_strength}{'_t' if temporal_smoothing else ''}.mp4"
        output_path = self.output_dir / output_filename
        
        # If output already exists, return it (caching)
        if output_path.exists():
            print(f"   ♻ Using cached processed video: {output_path}")
            return str(output_path), True
            
        print(f"   🎬 details: input={input_path}")
        
        # Build filters dynamically
        # MANDATORY: Fix FPS and Resolution for VSR
        filters = [
            "fps=25", 
            "scale=-2:720", 
        ]
        
        # OPTIONAL: Enhancement Layers
        if apply_enhancement:
            # Mild lip enhancement
            filters.append("unsharp=3:3:0.5:3:3:0.0")
            
            # Apply Denoising if requested
            if denoise_strength > 0:
                # Map 1-10 to hqdn3d values (Luma/Chroma spatial). 
                # Safe range: 1.0 to 10.0
                spatial = float(denoise_strength)
                # Temporal strength
                temporal = spatial * 1.5 if temporal_smoothing else spatial * 0.5
                
                filters.append(f"hqdn3d={spatial}:{spatial}:{temporal}:{temporal}")
                print(f"   ✨ Applying Denoise (S={spatial}, T={temporal})")

        command = [
            "ffmpeg",
            "-y", # Overwrite
            "-i", input_path,
            "-vf", ",".join(filters),
            "-c:v", "libx264", # Standard H.264
            "-preset", "fast",
            "-crf", "23", # Good quality
            "-c:a", "aac", # Standard Audio
            "-b:a", "128k",
            "-ar", "16000", # 16kHz for VSR/DeepGram compatibility
            "-ac", "1", # Mono
            str(output_path)
        ]
        
        print(f"   ⚙️ Running safe preprocessing...")
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"   ✓ Preprocessing complete: {output_path}")
            return str(output_path), True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Preprocessing failed: {e.stderr.decode()}")
            return input_path, False # Fallback to original
        except Exception as e:
            print(f"   ❌ Preprocessing error: {e}")
            return input_path, False

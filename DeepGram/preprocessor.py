"""
Audio Preprocessing Module for LAALM

This module provides tools to clean and normalize audio before it is sent to
the DeepGram API. Cleaner audio results in significantly better WER.

Features:
- Volume Normalization: Boosts quiet audio to continuous levels.
- Bandpass Filtering: Removes low-end rumble (<80Hz) and high-end hiss (>8kHz).
- Format Conversion: Ensures audio is in RIFF WAV format.

Dependencies:
- scipy
- numpy
- pydub (optional, but good for format handling)
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import tempfile
import shutil

class AudioPreprocessor:
    """Preprocesses audio files for better ASR accuracy."""
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        
    def _butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _apply_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """Apply a digital bandpass filter to the audio data."""
        # Check if we have enough data points for the filter
        if len(data) < 3 * order:
            return data
            
        b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y.astype(data.dtype)
    
    def process(self, input_path: str, apply_filter: bool = False) -> str:
        """
        Clean and normalize the audio file.
        
        Args:
            input_path: Path to the input audio file.
            apply_filter: Whether to apply bandpass filtering (default: False).
            
        Returns:
            Path to the processed temporary file.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Audio file not found: {input_path}")
            
        try:
            # Create temp file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            
            # Read audio
            try:
                fs, data = wavfile.read(input_path)
            except ValueError:
                print(f"   ⚠ WAV format not supported by scipy, skipping preprocessing for: {input_path}")
                shutil.copy2(input_path, temp_path)
                return temp_path
            
            # 1. Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1).astype(data.dtype)
                
            # 2. Bandpass Filter (Optional)
            if apply_filter:
                if fs > 16000:
                    data = self._apply_bandpass_filter(data, 80, 8000, fs)
                elif fs > 8000:
                    data = self._apply_bandpass_filter(data, 80, fs/2 - 100, fs)
                
            # 3. Normalization
            # Normalize to -1.0 dB peak
            max_val = np.max(np.abs(data))
            if max_val > 0:
                # Determine type max (int16 vs float)
                if np.issubdtype(data.dtype, np.integer):
                    type_max = np.iinfo(data.dtype).max
                    normalization_factor = (type_max * 0.9) / max_val
                else:
                    normalization_factor = 0.9 / max_val
                    
                data = data * normalization_factor
                data = data.astype(data.dtype) # Cast back to original type
            
            # Write to temp file
            wavfile.write(temp_path, fs, data)
            
            return temp_path
            
        except Exception as e:
            print(f"   ⚠ Preprocessing failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Return original if processing fails
            return input_path

if __name__ == "__main__":
    # Test stub
    import sys
    if len(sys.argv) > 1:
        p = AudioPreprocessor()
        out = p.process(sys.argv[1])
        print(f"Processed: {out}")

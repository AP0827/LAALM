
import sys
import os
import numpy as np
from scipy.io import wavfile

# Add project root to path
sys.path.insert(0, ".")
sys.path.insert(0, "DeepGram")

from DeepGram.preprocessor import AudioPreprocessor

def create_noisy_sine_wave(filename, duration=1.0, freq=500, noise_level=0.5):
    """Create a sine wave with noise."""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Clean signal (500Hz)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    
    # Add low frequency rumble (50Hz)
    rumble = 0.3 * np.sin(2 * np.pi * 50 * t)
    
    # Add random noise
    noise = noise_level * np.random.normal(size=len(t))
    
    # Combine
    audio = signal + rumble + noise
    
    # Normalize to 16-bit range
    audio = audio / np.max(np.abs(audio)) * 32767
    wavfile.write(filename, sample_rate, audio.astype(np.int16))
    print(f"Created noisy test file: {filename}")

def test_preprocessing():
    print("=" * 60)
    print("TESTING AUDIO PREPROCESSING")
    print("=" * 60)
    
    input_file = "test_noise.wav"
    create_noisy_sine_wave(input_file)
    
    try:
        preprocessor = AudioPreprocessor()
        output_file = preprocessor.process(input_file)
        
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        
        if os.path.exists(output_file):
            print("✅ Preprocessing completed successfully")
            
            # Verify file is different (filtered)
            rate_in, data_in = wavfile.read(input_file)
            rate_out, data_out = wavfile.read(output_file)
            
            if not np.array_equal(data_in, data_out):
                print("✅ File was modified (filtering applied)")
            else:
                print("⚠ File was NOT modified (check filter types)")
                
            # Verify normalization (should be close to max range)
            max_val = np.max(np.abs(data_out))
            print(f"Max amplitude after normalization: {max_val} (Target ~29000-32767)")
            
            if max_val > 25000:
                print("✅ Normalization verified")
            else:
                print("⚠ Normalization might be too low")
                
            # Cleanup output
            os.remove(output_file)
        else:
            print("❌ Output file not created")
            
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)

if __name__ == "__main__":
    test_preprocessing()

# LAALM Setup Guide

Complete setup instructions for the LAALM multi-modal transcription system.

## Prerequisites

- **Python 3.11+** (required for TensorFlow 2.12/Keras 2.12)
- **pip** package manager
- **git** for cloning
- **CUDA** (optional, for GPU acceleration)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/AP0827/LAALM.git
cd LAALM
```

### 2. Python 3.11 Setup

**Ubuntu/Debian:**
```bash
# Install Python 3.11
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Verify installation
python3.11 --version
```

**Other Systems:**
- Windows: Download from [python.org](https://www.python.org/downloads/)
- macOS: `brew install python@3.11`

### 3. Create Virtual Environment

```bash
# Create venv with Python 3.11
python3.11 -m venv .venv

# Activate
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies:**
- TensorFlow 2.12.0
- Keras 2.12.0
- numpy < 1.24 (required for TF 2.12)
- groq, openai, deepgram-sdk
- opencv-python, scipy, dlib

### 5. Install LipNet Package

```bash
cd LipNet
pip install -e .
cd ..
```

### 6. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit with your API keys
nano .env  # or use your preferred editor
```

**Required API Keys:**
```
GROQ_API_KEY=gsk_your_groq_key_here
DEEPGRAM_API_KEY=your_deepgram_key_here
OPENAI_API_KEY=sk_your_openai_key_here  # Optional
```

**Get API Keys:**
- Groq: [console.groq.com](https://console.groq.com)
- DeepGram: [deepgram.com](https://deepgram.com)
- OpenAI: [platform.openai.com](https://platform.openai.com)

### 7. Verify Installation

```bash
# Test with mock mode (no API keys needed)
python test.py
```

## Troubleshooting

### Python Version Issues

**Problem:** TensorFlow 2.12 not available for Python 3.12+

**Solution:** Use Python 3.11
```bash
# Deadsnakes PPA (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

### Keras Compatibility

**Problem:** `AttributeError: module 'keras.backend' has no attribute 'ctc_decode'`

**Solution:** Ensure you're using Keras 2.12 (NOT Keras 3.x)
```bash
pip install tensorflow==2.12.0 keras==2.12.0
```

### NumPy Version

**Problem:** `numpy.dtype size changed` errors

**Solution:** Use numpy < 1.24
```bash
pip install "numpy>=1.23.0,<1.24.0"
```

### LipNet Import Errors

**Problem:** `ModuleNotFoundError: No module named 'lipnet'`

**Solution:** Install LipNet in editable mode
```bash
cd LipNet
pip install -e .
```

### Missing Models

**Problem:** Pre-trained model files not found

**Solution:** Models should be in `LipNet/evaluation/models/`:
- `unseen-weights178.h5` (17.49 MB)
- `overlapped-weights368.h5` (17.49 MB)

If missing, check the LipNet repository or contact maintainers.

### API Rate Limits

**Problem:** API calls failing or slow

**Solution:** 
- Check API key validity
- Verify rate limits on provider dashboards
- Use mock mode for testing: set `MOCK_MODE=true` in test.py

### GPU/CUDA Issues

**Problem:** TensorFlow not detecting GPU

**Solution:**
```bash
# Check CUDA availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install CUDA toolkit if needed (Ubuntu)
sudo apt install nvidia-cuda-toolkit
```

## Running the System

### Basic Usage

```bash
# Activate environment
source .venv/bin/activate

# Run main pipeline
python main.py
```

### Custom Files

Edit `main.py` to use your own audio/video files:

```python
result = run_mvp(
    video_file="samples/video/your_video.mpg",
    audio_file="samples/audio/your_audio.wav",
    lipnet_weights="LipNet/evaluation/models/unseen-weights178.h5"
)
```

### Testing Without APIs

```bash
# Mock mode (no API keys required)
python test.py
```

## System Requirements

**Minimum:**
- CPU: Dual-core 2.0 GHz
- RAM: 4 GB
- Storage: 2 GB
- Python: 3.11+

**Recommended:**
- CPU: Quad-core 3.0+ GHz
- RAM: 8 GB+
- GPU: NVIDIA with CUDA support
- Storage: 5 GB+

## Next Steps

1. Place your audio files in `samples/audio/`
2. Place your video files in `samples/video/`
3. Update `main.py` with your file paths
4. Run: `python main.py`

## Support

For issues or questions:
- Check module READMEs: `DeepGram/README.md`, `LipNet/README.md`
- Review error messages in console output
- Ensure all API keys are valid and have sufficient credits

## Version Info

- **TensorFlow**: 2.12.0
- **Keras**: 2.12.0
- **Python**: 3.11.x
- **LipNet**: Custom implementation
- **Last Updated**: December 2025

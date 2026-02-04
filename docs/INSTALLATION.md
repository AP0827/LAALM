# Installation Guide

Comprehensive installation instructions for the LAALM multi-modal transcription system.

## Table of Contents

- [Prerequisites](#prerequisites)
- [System-Specific Setup](#system-specific-setup)
- [Python Environment](#python-environment)
- [Dependencies](#dependencies)
- [API Configuration](#api-configuration)
- [Model Downloads](#model-downloads)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- **Python 3.11+** (required for TensorFlow 2.12/Keras 2.12 compatibility)
- **pip** package manager (usually comes with Python)
- **git** for cloning the repository
- **Node.js 20.x+** (for web interface only)

### Optional Software

- **CUDA Toolkit 11.8** (for GPU acceleration)
- **FFmpeg** (for video processing, usually auto-installed)

### System Requirements

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

## System-Specific Setup

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install build essentials
sudo apt install build-essential

# Optional: Install CUDA (for GPU support)
sudo apt install nvidia-cuda-toolkit

# Verify Python installation
python3.11 --version
```

### Windows

1. Download Python 3.11 from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Install Visual C++ Build Tools from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
4. Optional: Install CUDA from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Verify installation
python3.11 --version
```

## Python Environment

### 1. Clone Repository

```bash
git clone https://github.com/AP0827/LAALM.git
cd LAALM
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

## Dependencies

### Core Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

**Key packages installed:**
- `tensorflow==2.12.0` - Deep learning framework
- `keras==2.12.0` - Neural network API
- `torch>=2.0.0` - PyTorch for Auto-AVSR
- `pytorch-lightning>=2.0.0` - Training framework
- `groq` - Groq API client
- `deepgram-sdk` - DeepGram API client
- `fastapi` - Web API framework
- `opencv-python` - Computer vision
- `scipy` - Scientific computing
- `numpy<1.24` - Numerical computing (version constraint for TF 2.12)

### Frontend Dependencies (Web Interface Only)

```bash
cd frontend
npm install
cd ..
```

**Key packages:**
- `react` - UI framework
- `vite` - Build tool
- `tailwindcss` - CSS framework
- `axios` - HTTP client

## API Configuration

### 1. Copy Environment Template

```bash
cp .env.example .env
```

### 2. Get API Keys

#### Groq API (Required)
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `gsk_`)

#### DeepGram API (Required)
1. Visit [deepgram.com](https://deepgram.com)
2. Sign up for a free account
3. Navigate to API Keys in dashboard
4. Create a new API key
5. Copy the key

#### OpenAI API (Optional)
1. Visit [platform.openai.com](https://platform.openai.com)
2. Sign up and add payment method
3. Navigate to API Keys
4. Create a new API key
5. Copy the key (starts with `sk-`)

### 3. Configure .env File

Edit `.env` with your favorite editor:

```bash
nano .env  # or vim, code, etc.
```

Add your API keys:
```bash
# Required
GROQ_API_KEY=gsk_your_groq_key_here
DEEPGRAM_API_KEY=your_deepgram_key_here

# Optional
OPENAI_API_KEY=sk_your_openai_key_here

# Optional configuration
LOG_LEVEL=INFO
MAX_WORKERS=4
```

## Model Downloads

### Auto-AVSR Models

The Auto-AVSR models should be automatically downloaded when you first run the system. If you need to download them manually:

```bash
cd auto_avsr
python -c "
from preparation.detectors.retinaface.detector import LandmarksDetector
detector = LandmarksDetector()
"
cd ..
```

**Expected model locations:**
- `auto_avsr/pretrained_models/*.pth` - Main AVSR models
- `auto_avsr/preparation/detectors/retinaface/weights/*.pth` - Face detection models
- `ibug_face_detection/ibug/face_detection/retina_face/weights/*.pth` - Additional face detection

### Verify Model Files

```bash
# Check Auto-AVSR models
ls -lh auto_avsr/pretrained_models/

# Check face detection models
find . -name "*.pth" -type f
```

## Verification

### 1. Test Python Environment

```bash
# Activate environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Check Python version
python --version  # Should be 3.11.x

# Check installed packages
pip list | grep -E "(tensorflow|torch|groq|deepgram)"
```

### 2. Test CUDA (GPU Support)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import tensorflow as tf; print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"
```

### 3. Test API Keys

```bash
python -c "
from load_env import load_environment_variables
env = load_environment_variables()
print(f'Groq API: {\"✓\" if env.get(\"GROQ_API_KEY\") else \"✗\"}')
print(f'DeepGram API: {\"✓\" if env.get(\"DEEPGRAM_API_KEY\") else \"✗\"}')
"
```

### 4. Run Test Pipeline

```bash
# Test with sample data
python main.py
```

Expected output:
```
Loading models...
Processing video...
Processing audio...
Running LLM correction...
Final Transcript: [transcribed text]
```

### 5. Test Web Interface (Optional)

```bash
# Start backend
python api.py &

# Start frontend (in new terminal)
cd frontend
npm run dev

# Open browser to http://localhost:5173
```

## Troubleshooting

### Python Version Issues

**Problem:** `TensorFlow 2.12 not available for Python 3.12+`

**Solution:** Install Python 3.11
```bash
# Ubuntu/Debian
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Recreate virtual environment
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Keras Compatibility

**Problem:** `AttributeError: module 'keras.backend' has no attribute 'ctc_decode'`

**Solution:** Ensure Keras 2.12 (NOT Keras 3.x)
```bash
pip uninstall keras tensorflow
pip install tensorflow==2.12.0 keras==2.12.0
```

### NumPy Version

**Problem:** `numpy.dtype size changed` errors

**Solution:** Use numpy < 1.24
```bash
pip install "numpy>=1.23.0,<1.24.0"
```

### Missing Models

**Problem:** Pre-trained model files not found

**Solution:**
```bash
# Check if models exist
ls -lh auto_avsr/pretrained_models/

# If missing, try manual download
cd auto_avsr
python verify_setup.py
cd ..
```

### API Connection Issues

**Problem:** API calls failing

**Solution:**
1. Verify API keys are correct in `.env`
2. Check internet connection
3. Verify API key validity on provider dashboards
4. Check rate limits

**Test API connectivity:**
```bash
# Test Groq
curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models

# Test DeepGram
curl -H "Authorization: Token $DEEPGRAM_API_KEY" https://api.deepgram.com/v1/projects
```

### GPU/CUDA Issues

**Problem:** TensorFlow/PyTorch not detecting GPU

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Install CUDA toolkit (Ubuntu)
sudo apt install nvidia-cuda-toolkit

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors

**Problem:** `ModuleNotFoundError` for various packages

**Solution:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# For specific module issues
pip install --force-reinstall <package-name>
```

### Memory Issues

**Problem:** Out of memory errors during processing

**Solution:**
- Close other applications
- Process smaller videos
- Use CPU instead of GPU (if GPU memory is limited)
- Reduce batch size in configuration

### Permission Issues (Linux/macOS)

**Problem:** Permission denied errors

**Solution:**
```bash
# Make scripts executable
chmod +x demo.sh start_web.sh

# Fix ownership (if needed)
sudo chown -R $USER:$USER .
```

## Next Steps

After successful installation:

1. **Test the system**: Run `python main.py` with sample data
2. **Explore documentation**: Read [API.md](API.md) and [WEB_INTERFACE.md](WEB_INTERFACE.md)
3. **Try the web interface**: Run `./start_web.sh`
4. **Process your own files**: Place files in `samples/` and update `main.py`

## Getting Help

If you encounter issues not covered here:

1. Check the main [README.md](../README.md)
2. Review [DEVELOPMENT.md](DEVELOPMENT.md) for advanced topics
3. Search existing GitHub issues
4. Create a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce

---

**Installation guide last updated:** February 2026

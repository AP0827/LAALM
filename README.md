# LAALM - Lip-reading and Audio Analysis with LLM

Multi-modal speech transcription system combining audio analysis, visual lip-reading, and LLM-based correction for improved accuracy in challenging audio environments.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [Web Interface](#web-interface)
  - [Python API](#python-api)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Common Commands](#common-commands)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

LAALM (Lip-reading and Audio Analysis with LLM) is a state-of-the-art multi-modal speech transcription system that combines:

- **Audio Transcription**: DeepGram API for high-accuracy speech-to-text
- **Visual Lip-reading**: Auto-AVSR neural network for video-based transcription
- **LLM Fusion**: Groq/OpenAI for intelligent multi-modal transcript correction
- **Confidence Scoring**: Word-level confidence for both audio and visual inputs
- **Automatic Fallback**: Graceful degradation when APIs are unavailable

This approach is particularly effective in noisy environments, with accented speech, or when audio quality is compromised.

## Features

✨ **Multi-Modal Processing**
- Parallel audio and visual speech recognition
- Intelligent fusion using large language models
- Word-level confidence scoring and alignment

🎯 **High Accuracy**
- DeepGram: Industry-leading audio transcription
- Auto-AVSR: State-of-the-art visual speech recognition (20.3% WER on LRS3)
- LLM correction: Context-aware error correction

🌐 **Web Interface**
- Beautiful, modern UI with glassmorphism design
- Drag-and-drop file upload
- Real-time progress tracking
- Downloadable results in multiple formats

🔧 **Developer Friendly**
- Clean Python API
- FastAPI backend with OpenAPI documentation
- Comprehensive logging and metrics
- Mock mode for testing without API keys

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/AP0827/LAALM.git
cd LAALM

# 2. Set up Python environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 5. Run the pipeline
python main.py
```

## Installation

### Prerequisites

- **Python 3.11+** (required for TensorFlow 2.12/Keras 2.12 compatibility)
- **pip** package manager
- **git** for cloning
- **CUDA** (optional, for GPU acceleration)
- **Node.js 20.x+** (for web interface)

### Detailed Installation

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for comprehensive installation instructions including:
- System-specific setup (Ubuntu, Windows, macOS)
- GPU/CUDA configuration
- Troubleshooting common issues
- Model downloads

### API Keys

You'll need API keys from:
- **Groq**: [console.groq.com](https://console.groq.com) (free tier available)
- **DeepGram**: [deepgram.com](https://deepgram.com) (free tier available)
- **OpenAI**: [platform.openai.com](https://platform.openai.com) (optional fallback)

Add them to your `.env` file:
```bash
GROQ_API_KEY=gsk_your_groq_key_here
DEEPGRAM_API_KEY=your_deepgram_key_here
OPENAI_API_KEY=sk_your_openai_key_here  # Optional
```

## Usage

### Command Line

**Basic usage:**
```bash
python main.py
```

**Custom files:**
```python
from pipeline import run_mvp

result = run_mvp(
    video_file="samples/video/your_video.mpg",
    audio_file="samples/audio/your_audio.wav"
)

print(f"Final Transcript: {result['final_transcript']}")
print(f"Confidence: {result['groq']['confidence']:.2%}")
```

**View results:**
```bash
# Latest transcripts
cat logs/transcripts_*.log | tail -20

# Detailed metrics
cat logs/metrics_*.log | tail -30

# JSON results
python -m json.tool logs/results_*.json | less
```

### Web Interface

**Start the web interface:**
```bash
# Option 1: Use startup script (recommended)
./start_web.sh

# Option 2: Start manually
# Terminal 1 - Backend
source .venv/bin/activate
python api.py

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
```

Access at:
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

See [docs/WEB_INTERFACE.md](docs/WEB_INTERFACE.md) for detailed web interface documentation.

### Python API

```python
from pipeline import run_mvp

# Process video with audio
result = run_mvp(
    video_file="path/to/video.mpg",
    audio_file="path/to/audio.wav"
)

# Access results
print(f"Audio: {result['deepgram']['transcript']}")
print(f"Video: {result['avsr']['transcript']}")
print(f"Final: {result['final_transcript']}")

# Word-level details
for word in result['word_details']:
    print(f"{word['word']}: audio={word['audio_conf']:.2f}, video={word['video_conf']:.2f}")
```

See [docs/API.md](docs/API.md) for complete API documentation.

## Project Structure

```
LAALM/
├── main.py                 # Entry point for CLI
├── pipeline.py             # Core multi-modal pipeline
├── api.py                  # FastAPI backend server
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
│
├── DeepGram/               # Audio transcription module
│   ├── transcriber.py
│   └── word_confidence.py
│
├── auto_avsr/              # Visual speech recognition
│   ├── inference_wrapper.py
│   ├── pretrained_models/
│   └── preparation/
│
├── Transformer/            # LLM correction module
│   ├── llm_corrector.py
│   ├── fusion.py
│   └── alignment.py
│
├── frontend/               # React web interface
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── samples/                # Test data
│   ├── audio/*.wav
│   └── video/*.mpg
│
├── docs/                   # Documentation
│   ├── INSTALLATION.md
│   ├── API.md
│   ├── WEB_INTERFACE.md
│   ├── ARCHITECTURE.md
│   └── DEVELOPMENT.md
│
├── logs/                   # Runtime logs (auto-generated)
├── uploads/                # Temporary uploads (auto-generated)
└── captions/               # Generated captions (auto-generated)
```

## Configuration

### Environment Variables

Configure in `.env` file:

```bash
# Required
GROQ_API_KEY=your_groq_key
DEEPGRAM_API_KEY=your_deepgram_key

# Optional
OPENAI_API_KEY=your_openai_key
LOG_LEVEL=INFO
MAX_WORKERS=4
```

### Model Configuration

Edit `pipeline.py` to customize:
- Face detector: `detector="retinaface"` or `detector="mediapipe"`
- LLM model: `model="llama-3.3-70b-versatile"` (Groq) or `model="gpt-4"` (OpenAI)
- Confidence thresholds

## Common Commands

### Running Demos
```bash
# Full interactive demo
./demo.sh

# Quick run
python main.py
```

### Generate Figures
```bash
# Dataset samples visualization
python generate_dataset_figure.py

# System architecture diagram
python generate_figure2.py
```

### Calculate Metrics
```bash
# Word Error Rate (WER) calculation
python calculate_metrics.py

# View agreement rates
grep "AGREEMENT METRICS" logs/metrics_*.log -A 5
```

### Batch Processing
```bash
# Process multiple samples
python -c "
from pipeline import run_mvp
for video in ['lwwz9s.mpg', 'bbaf2n.mpg', 'bgaa6n.mpg']:
    result = run_mvp(
        video_file=f'samples/video/{video}',
        audio_file=f'samples/audio/{video[:-4]}.wav'
    )
    print(f'{video}: {result[\"final_transcript\"]}')
"
```

### Cleanup
```bash
# Remove old logs
rm -rf logs/*

# Remove uploaded files
rm -rf uploads/*

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

## Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions
- **[API Documentation](docs/API.md)** - REST API reference
- **[Web Interface](docs/WEB_INTERFACE.md)** - Web UI guide
- **[Architecture](docs/ARCHITECTURE.md)** - System design and components
- **[Development](docs/DEVELOPMENT.md)** - Contributing and development guide

## Troubleshooting

### Python Version Issues
**Problem:** TensorFlow 2.12 not available for Python 3.12+

**Solution:** Use Python 3.11
```bash
sudo apt install python3.11 python3.11-venv
python3.11 -m venv .venv
```

### API Rate Limits
**Problem:** API calls failing or slow

**Solution:**
- Check API key validity in provider dashboards
- Verify rate limits
- Use mock mode for testing

### Model Not Found
**Problem:** Pre-trained model files not found

**Solution:**
```bash
# Check model files exist
ls -lh auto_avsr/pretrained_models/*.pth

# Download if missing (see docs/INSTALLATION.md)
```

### CUDA Issues
**Problem:** TensorFlow not detecting GPU

**Solution:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA toolkit if needed
sudo apt install nvidia-cuda-toolkit
```

For more troubleshooting, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and conventions
- Pull request process
- Issue reporting
- Development setup

## Citation

If you use LAALM in your research, please cite:

```bibtex
@article{laalm2026,
  title={LAALM: Lip-reading and Audio Analysis with Large Language Models},
  author={Yeleti, Asish Kumar and Pandey, Aayush},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see individual module directories for specific licenses.

## Contact

- **Asish Kumar Yeleti**: asishkumary.is23@rvce.edu.in
- **Aayush Pandey**: aayushpandey.is23@rvce.edu.in
- **Institution**: R V College of Engineering

---

**Built with ❤️ by the LAALM Team**

# Development Guide

Guide for developers contributing to or extending the LAALM project.

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Code Organization](#code-organization)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style](#code-style)
- [Adding Features](#adding-features)
- [Debugging](#debugging)
- [Performance Profiling](#performance-profiling)

## Getting Started

### Development Setup

```bash
# Clone repository
git clone https://github.com/AP0827/LAALM.git
cd LAALM

# Create development environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies with dev tools
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### IDE Setup

**VS Code:**
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

**PyCharm:**
- Set Python interpreter to `.venv/bin/python`
- Enable Black formatter
- Enable Flake8 linter

## Project Structure

```
LAALM/
├── main.py                    # CLI entry point
├── pipeline.py                # Core orchestration logic
├── api.py                     # FastAPI web server
├── load_env.py               # Environment configuration
├── logger.py                 # Logging utilities
├── video_utils.py            # Video processing utilities
├── calculate_metrics.py      # Metrics calculation
│
├── DeepGram/                 # Audio transcription module
│   ├── __init__.py
│   ├── transcriber.py        # DeepGram API client
│   ├── word_confidence.py    # Confidence extraction
│   ├── preprocessor.py       # Audio preprocessing
│   └── config.py             # Module configuration
│
├── auto_avsr/                # Visual speech recognition
│   ├── inference_wrapper.py  # Main inference interface
│   ├── preprocessor.py       # Video preprocessing
│   ├── lightning.py          # PyTorch Lightning module
│   ├── datamodule/           # Data loading
│   ├── espnet/               # ESPnet integration
│   ├── preparation/          # Preprocessing tools
│   │   ├── detectors/        # Face detection
│   │   │   ├── retinaface/
│   │   │   └── mediapipe/
│   │   └── transforms.py     # Data transforms
│   └── pretrained_models/    # Model weights
│
├── Transformer/              # LLM fusion module
│   ├── __init__.py
│   ├── llm_corrector.py      # LLM API integration
│   ├── fusion.py             # Multi-modal fusion
│   ├── alignment.py          # Word alignment
│   └── reliability.py        # Confidence scoring
│
├── frontend/                 # React web interface
│   ├── src/
│   │   ├── App.jsx           # Main component
│   │   ├── App.css           # Styles
│   │   └── main.jsx          # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── tests/                    # Test suite
│   ├── test_pipeline.py
│   ├── test_deepgram.py
│   ├── test_avsr.py
│   └── test_fusion.py
│
├── docs/                     # Documentation
├── samples/                  # Test data
├── logs/                     # Runtime logs
└── uploads/                  # Temporary uploads
```

## Code Organization

### Module Responsibilities

**pipeline.py:**
- Orchestrates the entire transcription pipeline
- Coordinates DeepGram, Auto-AVSR, and LLM components
- Handles error recovery and fallbacks

**DeepGram module:**
- Audio transcription via DeepGram API
- Word-level confidence extraction
- Audio preprocessing

**auto_avsr module:**
- Visual speech recognition
- Face detection and mouth cropping
- Video preprocessing

**Transformer module:**
- Multi-modal fusion logic
- LLM-based correction
- Word alignment and confidence scoring

**api.py:**
- REST API endpoints
- File upload handling
- Request/response formatting

### Design Patterns

**Factory Pattern:**
```python
# detector_factory.py
def get_detector(detector_type: str):
    if detector_type == "retinaface":
        return RetinaFaceDetector()
    elif detector_type == "mediapipe":
        return MediaPipeDetector()
    else:
        raise ValueError(f"Unknown detector: {detector_type}")
```

**Strategy Pattern:**
```python
# llm_strategy.py
class LLMStrategy(ABC):
    @abstractmethod
    def correct(self, audio_text: str, video_text: str) -> str:
        pass

class GroqStrategy(LLMStrategy):
    def correct(self, audio_text: str, video_text: str) -> str:
        # Groq-specific implementation
        pass

class OpenAIStrategy(LLMStrategy):
    def correct(self, audio_text: str, video_text: str) -> str:
        # OpenAI-specific implementation
        pass
```

**Dependency Injection:**
```python
# pipeline.py
def run_mvp(
    video_file: str,
    audio_file: Optional[str] = None,
    detector: str = "retinaface",
    llm_strategy: Optional[LLMStrategy] = None
):
    if llm_strategy is None:
        llm_strategy = GroqStrategy()
    
    # Use injected strategy
    result = llm_strategy.correct(audio_text, video_text)
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the code style guidelines and write tests.

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pipeline.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### 4. Format Code

```bash
# Format with Black
black .

# Check with Flake8
flake8 .

# Type checking with mypy
mypy pipeline.py
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

**Commit Message Format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Build/tooling changes

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Testing

### Unit Tests

**Example test:**
```python
# tests/test_pipeline.py
import pytest
from pipeline import run_mvp

def test_pipeline_with_valid_inputs():
    """Test pipeline with valid video and audio files."""
    result = run_mvp(
        video_file="samples/video/lwwz9s.mpg",
        audio_file="samples/audio/lwwz9s.wav"
    )
    
    assert "final_transcript" in result
    assert result["final_confidence"] > 0
    assert len(result["word_details"]) > 0

def test_pipeline_video_only():
    """Test pipeline with video only."""
    result = run_mvp(
        video_file="samples/video/lwwz9s.mpg"
    )
    
    assert result["audio_transcript"] is None
    assert result["video_transcript"] is not None
```

### Integration Tests

```python
# tests/test_integration.py
def test_end_to_end_transcription():
    """Test complete transcription pipeline."""
    # Upload files via API
    response = client.post(
        "/transcribe",
        files={
            "video": open("samples/video/lwwz9s.mpg", "rb"),
            "audio": open("samples/audio/lwwz9s.wav", "rb")
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "final_transcript" in data
```

### Mock Testing

```python
# tests/test_deepgram.py
from unittest.mock import Mock, patch

@patch('DeepGram.transcriber.DeepgramClient')
def test_deepgram_transcription(mock_client):
    """Test DeepGram transcription with mocked API."""
    # Setup mock
    mock_response = {
        "results": {
            "channels": [{
                "alternatives": [{
                    "transcript": "test transcript",
                    "confidence": 0.95
                }]
            }]
        }
    }
    mock_client.return_value.transcribe.return_value = mock_response
    
    # Test
    result = get_deepgram_confidence("test.wav")
    assert result["transcript"] == "test transcript"
```

## Code Style

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

**Line Length:** 88 characters (Black default)

**Imports:**
```python
# Standard library
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import numpy as np
import torch
from fastapi import FastAPI

# Local
from DeepGram.transcriber import get_deepgram_confidence
from auto_avsr.inference_wrapper import get_avsr_confidence
```

**Naming Conventions:**
```python
# Classes: PascalCase
class TranscriptionPipeline:
    pass

# Functions: snake_case
def process_video_file():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_FILE_SIZE = 100 * 1024 * 1024

# Private: _leading_underscore
def _internal_helper():
    pass
```

**Type Hints:**
```python
def transcribe(
    video_path: str,
    audio_path: Optional[str] = None,
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """Transcribe video with optional audio.
    
    Args:
        video_path: Path to video file
        audio_path: Optional path to audio file
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Dictionary containing transcription results
    """
    pass
```

**Docstrings:**
```python
def complex_function(param1: str, param2: int) -> bool:
    """One-line summary.
    
    Longer description if needed. Explain what the function does,
    any important details, edge cases, etc.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        
    Example:
        >>> complex_function("test", 42)
        True
    """
    pass
```

### JavaScript/React Style

**Component Structure:**
```javascript
// Imports
import React, { useState, useEffect } from 'react';
import axios from 'axios';

// Component
export default function TranscriptionApp() {
  // State
  const [results, setResults] = useState(null);
  
  // Effects
  useEffect(() => {
    // Effect logic
  }, []);
  
  // Handlers
  const handleSubmit = async () => {
    // Handler logic
  };
  
  // Render
  return (
    <div className="container">
      {/* JSX */}
    </div>
  );
}
```

## Adding Features

### Adding a New Detector

1. **Create detector class:**
```python
# auto_avsr/preparation/detectors/new_detector/detector.py
class NewDetector:
    def __init__(self):
        # Initialize detector
        pass
    
    def detect_faces(self, frame):
        # Detect faces in frame
        pass
    
    def extract_mouth(self, frame, landmarks):
        # Extract mouth region
        pass
```

2. **Update factory:**
```python
# auto_avsr/preparation/detectors/factory.py
def get_detector(detector_type: str):
    if detector_type == "new_detector":
        from .new_detector.detector import NewDetector
        return NewDetector()
    # ... existing detectors
```

3. **Add tests:**
```python
# tests/test_new_detector.py
def test_new_detector():
    detector = NewDetector()
    # Test detector functionality
```

### Adding a New LLM Provider

1. **Create strategy class:**
```python
# Transformer/strategies/new_llm.py
class NewLLMStrategy(LLMStrategy):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def correct(self, audio_text: str, video_text: str) -> str:
        # Implementation
        pass
```

2. **Update configuration:**
```python
# Transformer/llm_corrector.py
def get_llm_strategy(provider: str):
    if provider == "new_llm":
        return NewLLMStrategy(api_key=os.getenv("NEW_LLM_API_KEY"))
    # ... existing providers
```

### Adding a New API Endpoint

1. **Define endpoint:**
```python
# api.py
@app.post("/new-endpoint")
async def new_endpoint(data: NewDataModel):
    """Endpoint description."""
    try:
        result = process_new_feature(data)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

2. **Add data model:**
```python
from pydantic import BaseModel

class NewDataModel(BaseModel):
    field1: str
    field2: int
    field3: Optional[float] = None
```

3. **Update frontend:**
```javascript
// frontend/src/App.jsx
const callNewEndpoint = async (data) => {
  const response = await axios.post('/new-endpoint', data);
  return response.data;
};
```

## Debugging

### Enable Debug Logging

```python
# Set in .env
LOG_LEVEL=DEBUG

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Use Python Debugger

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use built-in breakpoint (Python 3.7+)
breakpoint()
```

### Debug API Requests

```bash
# Use curl with verbose output
curl -v -X POST http://localhost:8000/transcribe \
  -F "video=@samples/video/lwwz9s.mpg"

# Or use httpie
http -v POST localhost:8000/transcribe video@samples/video/lwwz9s.mpg
```

### Frontend Debugging

```javascript
// Console logging
console.log('Debug:', variable);
console.table(arrayData);

// React DevTools
// Install browser extension for component inspection
```

## Performance Profiling

### Profile Python Code

```python
import cProfile
import pstats

# Profile function
profiler = cProfile.Profile()
profiler.enable()

result = run_mvp("video.mpg", "audio.wav")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function code
    pass
```

### Benchmark Performance

```python
import time

start = time.time()
result = run_mvp("video.mpg", "audio.wav")
end = time.time()

print(f"Processing time: {end - start:.2f}s")
```

---

**For more information, see:**
- [Architecture Documentation](ARCHITECTURE.md)
- [API Documentation](API.md)
- [Contributing Guidelines](../CONTRIBUTING.md)

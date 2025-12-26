# MVP Integration Guide: Multi-Modal Transcription with Groq

This guide explains how to use the new MVP system that combines DeepGram audio, LipNet visual, and Groq semantic correction with word-level confidence scores.

## 📋 Quick Overview

The MVP pipeline:
1. **DeepGram** transcribes audio and extracts word-level confidence scores
2. **LipNet** analyzes video lip movements and extracts word-level confidence scores
3. **Combined Analysis** merges word-level confidence from both modalities
4. **Groq LLM** refines the transcript using semantic understanding and confidence data

## 🚀 Setup (5 minutes)

### 1. Install Dependencies

```bash
# Core dependencies
pip install groq deepgram-sdk

# For LipNet (if using video)
pip install keras tensorflow opencv-python numpy

# Optional but recommended
pip install python-dotenv
```

### 2. Set Environment Variables

Create a `.env` file in the project root:

```bash
# Required: Groq API key (https://console.groq.com)
GROQ_API_KEY=gsk_your_groq_key_here

# Optional: DeepGram API key (https://console.deepgram.com)
DEEPGRAM_API_KEY=your_deepgram_key

# Optional: Ollama base URL (if using local LLMs)
OLLAMA_BASE_URL=http://localhost:11434
```

Or set them directly:

```bash
export GROQ_API_KEY="gsk_..."
export DEEPGRAM_API_KEY="..."
```

### 3. Get LipNet Weights

LipNet weights should be at: `LipNet/evaluation/models/unseen-weights178.h5`

If not present, download from the original LipNet repository or use the overlapped model.

## 💻 Usage

### Option A: Using the MVP Script Directly

```bash
# Run with mock data (no API keys needed for demo)
python mvp_multi_modal_groq.py

# Run with real audio file
python mvp_multi_modal_groq.py --audio audio.wav

# Run with video file
python mvp_multi_modal_groq.py --video video.mp4

# Run with both
python mvp_multi_modal_groq.py --audio audio.wav --video video.mp4
```

### Option B: Import in Your Python Code

```python
from mvp_multi_modal_groq import run_mvp

# Run with mock data
results = run_mvp()

# Run with real files
results = run_mvp(
    audio_file="path/to/audio.wav",
    video_file="path/to/video.mp4",
    deepgram_api_key="your_key",
    groq_api_key="your_key"
)

# Access results
print(f"Final transcript: {results['final_transcript']}")
print(f"Confidence: {results['groq_correction']['confidence']:.3f}")
```

### Option C: Use the Updated Transformer Pipeline

```python
from Transformer import TransformerPipeline, LLMProvider
import os

# Create pipeline with Groq (now default)
pipeline = TransformerPipeline(
    llm_provider=LLMProvider.GROQ,
    llm_model="mixtral-8x7b-32768",
    llm_api_key=os.getenv("GROQ_API_KEY")
)

# Get transcriptions from DeepGram and LipNet
deepgram_result = {...}  # Your DeepGram result
lipnet_result = {...}    # Your LipNet result

# Process through pipeline
result = pipeline.process(
    deepgram_transcript=deepgram_result["transcript"],
    deepgram_confidence=deepgram_result["overall_confidence"],
    deepgram_word_confidences=deepgram_result["word_confidences"],
    lipnet_transcript=lipnet_result["transcript"],
    lipnet_confidence=lipnet_result["overall_confidence"],
    lipnet_word_confidences=lipnet_result["word_confidences"],
    domain_context="medical"  # Optional
)

print(f"Final: {result['final_transcript']}")
```

## 📊 Understanding the Output

### Word-Level Confidence Example

```
✓ the          [DG: 0.95] [LipNet: 0.92] [Avg: 0.94] ✓
✓ quick        [DG: 0.89] [LipNet: 0.85] [Avg: 0.87] ✓
✗ brown        [DG: 0.91] [LipNet: 0.72] [Avg: 0.82] ⚠
✓ fox          [DG: 0.93] [LipNet: 0.90] [Avg: 0.92] ✓
```

**Symbols:**
- `✓` = Both models agree on the word
- `✗` = Models disagree (different word)
- `⚠` = Low confidence (<0.7 in either model)
- `✓` (right) = High average confidence (>0.8)

### Full Results Structure

```python
{
    "deepgram": {
        "transcript": "the quick brown fox...",
        "overall_confidence": 0.92,
        "word_confidences": [("the", 0.95), ("quick", 0.89), ...]
    },
    "lipnet": {
        "transcript": "the quick brown fox...",
        "overall_confidence": 0.88,
        "word_confidences": [("the", 0.92), ("quick", 0.85), ...]
    },
    "combined_words": [
        {
            "position": 0,
            "word": "the",
            "deepgram": {"word": "the", "confidence": 0.95},
            "lipnet": {"word": "the", "confidence": 0.92},
            "average_confidence": 0.94,
            "agreement": true,
            "low_confidence": false
        },
        # ... more words
    ],
    "groq_correction": {
        "corrected_transcript": "the quick brown fox jumps over the lazy dog",
        "corrections": [
            {
                "original_phrase": "...",
                "corrected_phrase": "...",
                "reason": "..."
            }
        ],
        "confidence": 0.92,
        "status": "success"
    },
    "final_transcript": "the quick brown fox jumps over the lazy dog"
}
```

## 🎯 Key Features

### 1. Word-Level Confidence Fusion

Instead of just averaging scores, the system:
- Identifies words where both models agree (high agreement = high confidence)
- Flags discrepancies (different words between models)
- Marks low-confidence words (either model < 0.7)
- Provides per-position analysis

### 2. Groq for Fast Semantic Correction

Groq is used because it's:
- **Fast**: Mixtral model responses in seconds
- **Free tier available**: Good for development
- **JSON-compatible**: Easy to parse structured responses
- **No LLM lock-in**: Easy to switch to other providers

### 3. Fallback Strategy

If any step fails:
- No audio? → Use mock DeepGram data
- No video? → Use mock LipNet data
- Groq fails? → Return the higher-confidence modality unchanged

## 📝 Example: Medical Transcription

```python
from mvp_multi_modal_groq import run_mvp

# Medical context - more accurate transcription needed
results = run_mvp(
    audio_file="medical_recording.wav",
    video_file="doctor_visit.mp4",
    deepgram_api_key="your_key",
    groq_api_key="your_key"
)

# Check high-risk low-confidence words
low_conf = [w for w in results["combined_words"] if w["low_confidence"]]
for word in low_conf:
    print(f"⚠️  Low confidence word at position {word['position']}: {word['word']}")
    print(f"   DeepGram: {word['deepgram']['confidence']:.2f}")
    print(f"   LipNet: {word['lipnet']['confidence']:.2f}")
```

## 🔧 Advanced Usage

### Use Different Groq Model

```python
from Transformer import TransformerPipeline, LLMProvider

pipeline = TransformerPipeline(
    llm_provider=LLMProvider.GROQ,
    llm_model="llama-3-70b-versatile",  # Different model
    llm_api_key="your_groq_key"
)
```

### Use Different LLM Provider

```python
# Switch to OpenAI
pipeline = TransformerPipeline(
    llm_provider=LLMProvider.OPENAI,
    llm_model="gpt-4",
    llm_api_key="your_openai_key"
)

# Switch to Claude
pipeline = TransformerPipeline(
    llm_provider=LLMProvider.ANTHROPIC,
    llm_model="claude-3-opus-20240229",
    llm_api_key="your_anthropic_key"
)

# Use local Ollama
pipeline = TransformerPipeline(
    llm_provider=LLMProvider.OLLAMA,
    llm_model="llama2",
)
```

### Disable LLM Correction

```python
pipeline = TransformerPipeline(
    llm_provider=LLMProvider.GROQ,
    llm_api_key="your_key",
    llm_enabled=False  # Just fuse, don't correct
)
```

### Get Detailed Report

```python
result = pipeline.process(...)

# Print comprehensive report
report = pipeline.get_full_report(result)
print(report)
```

## 📊 Supported Groq Models

```python
# Available models on Groq (as of Dec 2024)
models = [
    "mixtral-8x7b-32768",           # Default - fast and capable
    "llama-3-70b-versatile",        # Larger, more capable
    "llama-3-8b-instant",           # Lightweight
    "gemma-7b-it",                  # Lightweight alternative
]
```

Check [Groq API docs](https://console.groq.com/docs/models) for latest models.

## 🐛 Troubleshooting

### "GROQ_API_KEY not set"
**Solution**: Set environment variable
```bash
export GROQ_API_KEY="gsk_your_key"
```

### "Groq API Error: Rate limit exceeded"
**Solution**: You've hit the free tier rate limit. Options:
- Wait a few seconds and retry
- Upgrade Groq account
- Switch to different LLM provider (OpenAI, Anthropic, etc.)

### "JSON decode error from Groq"
**Solution**: Groq returned non-JSON response. Usually due to:
- Token limit exceeded (try shorter input)
- Model overloaded (retry after a minute)
- Prompt too complex (simplify the prompt)

### "LipNet weights not found"
**Solution**: Download weights or provide correct path
```bash
# Download from GitHub
wget https://github.com/rizkiarm/LipNet/releases/download/weights/unseen-weights178.h5

# Or specify path
results = run_mvp(
    video_file="video.mp4",
    lipnet_weights="/path/to/weights.h5"
)
```

### "DeepGram transcription failed"
**Solution**: 
- Check API key: `echo $DEEPGRAM_API_KEY`
- Verify audio file is valid
- Check internet connection
- Try mock mode: don't provide audio_file parameter

## 📚 Next Steps

1. **Test MVP**: Run `python mvp_multi_modal_groq.py` to see it working
2. **Real Files**: Try with your actual audio/video files
3. **Integration**: Import into your application
4. **Customization**: Adjust confidence thresholds, models, domain context

## 🔗 Related Documentation

- [DeepGram Word Confidence Guide](./LipNet/WORD_CONFIDENCE_GUIDE.md)
- [Transformer Pipeline API](./Transformer/README.md)
- [Groq API Reference](https://console.groq.com/docs)
- [LipNet Evaluation](./LipNet/evaluation/predict_with_confidence.py)

## ✅ Checklist: Getting Started

- [ ] Install dependencies: `pip install groq deepgram-sdk`
- [ ] Set GROQ_API_KEY: `export GROQ_API_KEY=...`
- [ ] Optional: Set DEEPGRAM_API_KEY: `export DEEPGRAM_API_KEY=...`
- [ ] Run MVP: `python mvp_multi_modal_groq.py`
- [ ] Test with your files
- [ ] Integrate into your project

---

**Status**: Ready for MVP testing ✅  
**Last Updated**: December 2024  
**Maintained**: LAALM Project

# Complete Integration Guide: DeepGram + LipNet + LLM

## Quick Start

### 1. Installation

```bash
# Navigate to project
cd /path/to/LAALM

# Install dependencies
pip install deepgram-sdk openai transformers torch numpy

# For other LLM providers (optional):
pip install anthropic google-generativeai ollama
```

### 2. Set Environment Variables

```bash
# DeepGram API
export DEEPGRAM_API_KEY="your_deepgram_key"

# LLM Provider (example: OpenAI)
export OPENAI_API_KEY="sk-your_openai_key"

# Or Anthropic
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### 3. Run Example Pipeline

```bash
# Test with mock data (no API keys required)
python Transformer/example_usage.py --mock-mode

# With real APIs
python Transformer/example_usage.py \
  --audio-file audio.wav \
  --llm-key $OPENAI_API_KEY \
  --domain medical \
  --output-json result.json
```

## Integration Workflow

### Step 1: Extract Word-Level Confidence from DeepGram

```python
from DeepGram.enhanced_transcriber import DeepGramWithConfidence

# Initialize
dg = DeepGramWithConfidence(api_key="your_key")

# Get transcription with confidence
result = dg.transcribe_file_with_confidence("audio.wav")

# Result contains:
# - transcript: "the quick brown fox..."
# - overall_confidence: 0.92
# - word_confidences: [("the", 0.95), ("quick", 0.89), ...]
# - metrics: detailed statistics
# - low_confidence_words: problematic words
```

### Step 2: Get LipNet Output

```python
from lipnet.model import LipNet
import numpy as np

# Load model
lipnet = LipNet()
# ... load weights and video data ...

# Get predictions
y_pred = lipnet.predict(video_frames)  # Shape: (time_steps, num_chars)

# Convert to word-level confidence
# Extract character probabilities from softmax
char_probs = np.max(y_pred, axis=1)  # Per-timestep max probability

# For this example, assume transcript and word-level confidences are available:
lipnet_transcript = "the quick brown fox..."
lipnet_word_confidences = [("the", 0.91), ("quick", 0.82), ...]
```

### Step 3: Fuse Both Modalities

```python
from Transformer.fusion import ModalityFuser, ModalityOutput

# Create modality outputs
dg_output = ModalityOutput(
    modality="deepgram",
    transcript=dg_result["transcript"],
    word_confidences=dg_result["word_confidences"],
    overall_confidence=dg_result["overall_confidence"]
)

lipnet_output = ModalityOutput(
    modality="lipnet",
    transcript=lipnet_transcript,
    word_confidences=lipnet_word_confidences,
    overall_confidence=0.85  # LipNet overall confidence
)

# Fuse with confidence weighting
fuser = ModalityFuser(confidence_weighted=True)
fusion_result = fuser.fuse(dg_output, lipnet_output)

print(f"Alignment: {fusion_result.alignment_score:.3f}")
print(f"Fused Transcript: {fusion_result.fused_transcript}")
print(f"Weights: {fusion_result.fusion_weights}")
```

### Step 4: Apply LLM Refinement

```python
from Transformer.llm_corrector import (
    LLMSemanticCorrector,
    LLMProvider,
    CorrectionContext
)

# Initialize corrector
corrector = LLMSemanticCorrector(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    api_key="sk-...",
    temperature=0.3
)

# Prepare context
context = CorrectionContext(
    deepgram_transcript=dg_result["transcript"],
    deepgram_confidence=dg_result["overall_confidence"],
    deepgram_word_confidences=dg_result["word_confidences"],
    lipnet_transcript=lipnet_transcript,
    lipnet_confidence=0.85,
    lipnet_word_confidences=lipnet_word_confidences,
    alignment_score=fusion_result.alignment_score,
    flagged_discrepancies=fusion_result.flagged_discrepancies,
    domain_context="medical",  # Optional
    audio_metadata={"duration": 2.6}
)

# Apply correction
correction_result = corrector.correct(context)
print(f"Corrected: {correction_result.corrected_transcript}")
print(f"Confidence: {correction_result.confidence_in_corrections:.3f}")
```

### Step 5: Use Complete Pipeline

```python
from Transformer import TransformerPipeline, LLMProvider
import os

# Initialize full pipeline
pipeline = TransformerPipeline(
    llm_provider=LLMProvider.OPENAI,
    llm_api_key=os.getenv("OPENAI_API_KEY"),
    use_confidence_weighting=True,
    llm_enabled=True
)

# Process both modalities
result = pipeline.process(
    deepgram_transcript=dg_result["transcript"],
    deepgram_confidence=dg_result["overall_confidence"],
    deepgram_word_confidences=dg_result["word_confidences"],
    lipnet_transcript=lipnet_transcript,
    lipnet_confidence=0.85,
    lipnet_word_confidences=lipnet_word_confidences,
    domain_context="medical"
)

# Generate report
print(pipeline.get_full_report(result))

# Access final transcript
print(f"\nFinal: {result['final_transcript']}")
```

## Module Integration Points

### In DeepGram Pipeline

Update `DeepGram/pipeline.py` to include confidence extraction:

```python
from .enhanced_transcriber import DeepGramWithConfidence

class TranscriptionPipeline:
    def __init__(self, api_key: Optional[str] = None):
        # Use enhanced transcriber
        config = DeepGramConfig(api_key=api_key)
        self.transcriber = DeepGramWithConfidence(api_key=config.api_key)
        self.formatter = CaptionFormatter()
    
    def transcribe_and_caption_file_with_confidence(
        self,
        audio_file_path: str,
        output_dir: str = "./output",
        caption_format: str = "vtt",
        save_transcript: bool = True,
    ) -> dict:
        """Transcribe and get word-level confidence."""
        result = self.transcriber.transcribe_file_with_confidence(
            audio_file_path,
            include_utterances=True,
        )
        
        return {
            "transcript": result["transcript"],
            "overall_confidence": result["overall_confidence"],
            "word_confidences": result["word_confidences"],
            "metrics": result["metrics"],
            "low_confidence_words": result["low_confidence_words"],
            # ... existing caption generation ...
        }
```

### In LipNet Evaluation

Add confidence extraction to evaluation scripts:

```python
# evaluation/predict.py
from Transformer import TransformerPipeline, LLMProvider

def predict_with_confidence(video_file, lipnet_model, deepgram_result):
    """Get LipNet prediction and fuse with DeepGram."""
    
    # Get LipNet prediction
    y_pred = lipnet_model.predict(video_data)
    lipnet_transcript = decode_prediction(y_pred)
    
    # Extract word-level probabilities
    word_probs = extract_word_probabilities(y_pred)
    
    # Fuse with DeepGram
    pipeline = TransformerPipeline(llm_enabled=True)
    result = pipeline.process(
        deepgram_transcript=deepgram_result["transcript"],
        deepgram_confidence=deepgram_result["overall_confidence"],
        deepgram_word_confidences=deepgram_result["word_confidences"],
        lipnet_transcript=lipnet_transcript,
        lipnet_confidence=lipnet_confidence,
        lipnet_word_confidences=word_probs
    )
    
    return result
```

## Data Flow Diagram

```
Audio File ─────┬──────────> DeepGram API
                │            ├─> transcript
                │            ├─> overall_confidence
                │            └─> word_confidences [w1, c1), ...]
                │
Video File ─────┼──────────> LipNet Model
                │            ├─> transcript
                │            ├─> overall_confidence  
                │            └─> word_confidences [(w2, c2), ...]
                │
                ▼
        ┌───────────────────┐
        │ ModalityFuser     │
        └─────────┬─────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼
  alignment   weights      fused_words
  score       computed     (confidence-weighted)
                  │
                  ▼
         ┌──────────────────┐
         │ LLMCorrector     │
         │  (optional)      │
         └─────────┬────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
  corrections  confidence   explanation
  made         in result
      │
      ▼
  ┌─────────────────────┐
  │ FINAL TRANSCRIPT    │
  └─────────────────────┘
```

## Testing

### Mock Mode (No API Keys)

```python
# Direct test of fusion
from Transformer.fusion import ModalityFuser, ModalityOutput

dg = ModalityOutput(
    modality="deepgram",
    transcript="the quick brown fox",
    word_confidences=[("the", 0.95), ("quick", 0.89), ("brown", 0.91), ("fox", 0.94)],
    overall_confidence=0.92
)

lipnet = ModalityOutput(
    modality="lipnet",
    transcript="the quick brown fox",
    word_confidences=[("the", 0.91), ("quick", 0.82), ("brown", 0.88), ("fox", 0.90)],
    overall_confidence=0.85
)

fuser = ModalityFuser()
result = fuser.fuse(dg, lipnet)
assert result.alignment_score > 0.8
assert "deepgram" in result.fusion_weights
print("✓ Fusion test passed")
```

### Full Pipeline Test (Requires API Key)

```bash
python Transformer/example_usage.py --mock-mode --output-json test_result.json
cat test_result.json  # Verify output structure
```

## Performance Optimization

### Batch Processing

```python
from Transformer import TransformerPipeline

pipeline = TransformerPipeline(llm_enabled=True)

results = []
for audio_file, video_file in files:
    dg_result = get_deepgram_result(audio_file)
    lipnet_result = get_lipnet_result(video_file)
    
    result = pipeline.process(
        deepgram_transcript=dg_result["transcript"],
        deepgram_confidence=dg_result["overall_confidence"],
        deepgram_word_confidences=dg_result["word_confidences"],
        lipnet_transcript=lipnet_result["transcript"],
        lipnet_confidence=lipnet_result["confidence"],
        lipnet_word_confidences=lipnet_result["word_confidences"]
    )
    results.append(result)
```

### Caching

```python
import hashlib
import json

def cache_key(dg_transcript, lipnet_transcript):
    data = f"{dg_transcript}:{lipnet_transcript}"
    return hashlib.md5(data.encode()).hexdigest()

cache = {}

def get_correction(dg_result, lipnet_result):
    key = cache_key(dg_result["transcript"], lipnet_result["transcript"])
    
    if key in cache:
        return cache[key]
    
    result = pipeline.process(...)
    cache[key] = result
    return result
```

## Troubleshooting

### Issue: "No words found in DeepGram response"

**Solution**: Ensure `include_utterances=True` when calling DeepGram API

```python
response = transcriber.transcribe_file(
    file_path,
    include_utterances=True  # IMPORTANT
)
```

### Issue: LLM returns invalid JSON

**Solution**: Implement JSON parsing with fallback

```python
try:
    result_json = json.loads(response_text)
except json.JSONDecodeError:
    # Use higher-confidence modality as fallback
    result = fallback_correction(dg_result, lipnet_result)
```

### Issue: Slow LLM API calls

**Solution**: Use faster model or local inference

```python
# Use faster model
corrector = LLMSemanticCorrector(
    model="gpt-3.5-turbo",  # Faster than GPT-4
    temperature=0.3
)

# Or use local Ollama
corrector = LLMSemanticCorrector(
    provider=LLMProvider.OLLAMA,
    model="llama2"
)
```

## Next Steps

1. **Integrate with your training pipeline**: Add confidence metrics to training evaluation
2. **Fine-tune on your domain**: Train LipNet on domain-specific data
3. **Experiment with LLM providers**: Test different models for your use case
4. **Add streaming support**: Implement real-time processing
5. **Deploy as API**: Wrap in FastAPI/Flask for production use

## References

- [DeepGram API Docs](https://developers.deepgram.com/reference/pre-recorded)
- [LipNet Paper](https://arxiv.org/abs/1611.08460)
- [OpenAI API Docs](https://platform.openai.com/docs/api-reference)
- [Anthropic Claude Docs](https://docs.anthropic.com)

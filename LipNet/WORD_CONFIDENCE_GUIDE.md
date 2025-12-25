# LipNet Word-Level Confidence Extraction

## Overview

Your friend added a complete multi-modal pipeline that combines:
- **DeepGram** - Audio transcription with word-level confidence
- **LipNet** - Visual lip reading with character-level confidence  
- **Transformer** - Fusion of both modalities with LLM correction

## Quick Start: Get Word Confidence from LipNet

### Basic Usage

```powershell
cd D:\LipNet\LipNet\evaluation
D:\LipNet\.venv\Scripts\python.exe predict_with_confidence.py models\unseen-weights178.h5 samples\GRID\bbaf2n.mpg
```

### Output Example

```
================================================================================
 LipNet Prediction with Confidence Scores
================================================================================

Transcript: lay green at d six soon
Overall Confidence: 0.972

Word-Level Confidence:
----------------------------------------
  lay             0.970 ███████████████████
  green           0.959 ███████████████████
  at              1.000 ████████████████████
  d               1.000 ████████████████████
  six             0.989 ███████████████████
  soon            0.915 ██████████████████

Statistics:
  Mean:   0.972
  Median: 0.980
  Min:    0.915
  Max:    1.000
  Std:    0.030
```

## Understanding the Confidence Scores

### How It Works

1. **LipNet Prediction**: Model outputs softmax probabilities for each character at each timestep
2. **Character Confidence**: Max probability per timestep (how confident in that character)
3. **Word Confidence**: Average character confidence over word's timespan

### Confidence Scale

- **0.9+** (Excellent): High confidence, very reliable
- **0.7-0.9** (Good): Decent confidence, generally trustworthy
- **<0.7** (Low): Uncertain, may need correction

### Key Metrics

- **overall_confidence**: Mean across all frames (overall prediction quality)
- **word_confidences**: Per-word scores for identifying weak spots
- **char_confidences**: Raw per-frame probabilities (for detailed analysis)

## Integration with Your Friend's Pipeline

### Step 1: Get LipNet Word Confidence

```python
from lipnet.evaluation.predict_with_confidence import predict_with_confidence

result = predict_with_confidence(
    weight_path="models/unseen-weights178.h5",
    video_path="samples/GRID/bbaf2n.mpg"
)

# Result contains:
# - transcript: "lay green at d six soon"
# - overall_confidence: 0.972
# - word_confidences: [("lay", 0.970), ("green", 0.959), ...]
# - char_confidences: [0.99, 0.98, 0.97, ...]
```

### Step 2: Combine with DeepGram (from your friend's code)

```python
from DeepGram.enhanced_transcriber import DeepGramWithConfidence

# Get audio transcription
dg = DeepGramWithConfidence(api_key="your_key")
dg_result = dg.transcribe_file_with_confidence("audio.wav")

# Get visual transcription
lipnet_result = predict_with_confidence(
    weight_path="models/unseen-weights178.h5",
    video_path="video.mp4"
)
```

### Step 3: Fuse Both Modalities

```python
from Transformer import TransformerPipeline, LLMProvider

pipeline = TransformerPipeline(
    llm_provider=LLMProvider.OPENAI,
    llm_api_key="sk-your_key"
)

result = pipeline.process(
    # Audio modality
    deepgram_transcript=dg_result["transcript"],
    deepgram_confidence=dg_result["overall_confidence"],
    deepgram_word_confidences=dg_result["word_confidences"],
    
    # Visual modality
    lipnet_transcript=lipnet_result["transcript"],
    lipnet_confidence=lipnet_result["overall_confidence"],
    lipnet_word_confidences=lipnet_result["word_confidences"],
    
    # Optional
    domain_context="medical"  # or "legal", "general", etc.
)

print(f"Final transcript: {result['final_transcript']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Programmatic Access

If you need to use this in your own code:

```python
# Import the function
import sys
sys.path.append('D:/LipNet/LipNet')
from evaluation.predict_with_confidence import predict_with_confidence

# Get predictions with confidence
result = predict_with_confidence(
    weight_path="D:/LipNet/LipNet/evaluation/models/unseen-weights178.h5",
    video_path="D:/LipNet/LipNet/evaluation/samples/GRID/bbaf2n.mpg"
)

# Access the data
transcript = result['transcript']
overall_conf = result['overall_confidence']
word_confs = result['word_confidences']  # List of (word, confidence) tuples

# Process words
for word, conf in word_confs:
    if conf < 0.7:
        print(f"Low confidence word: {word} ({conf:.3f})")
```

## Files Overview

### Core LipNet Files (LipNet/)
- **evaluation/predict_with_confidence.py** ← **NEW!** Extract word confidence from LipNet
- evaluation/predict.py - Basic prediction (text only)
- lipnet/model2.py - LipNet architecture
- lipnet/core/decoders.py - CTC decoding

### Multi-Modal Pipeline (Your Friend Added)
- **Transformer/** - Fusion and LLM correction
  - fusion.py - Combines DeepGram + LipNet
  - llm_corrector.py - Semantic correction with GPT/Claude
  - example_usage.py - Complete pipeline example
  
- **DeepGram/** - Audio transcription
  - enhanced_transcriber.py - DeepGram with confidence
  - word_confidence.py - Extract word-level metrics
  
- **Documentation**
  - INTEGRATION_GUIDE.md - Complete pipeline guide
  - QUICKSTART.md - Fast setup
  - IMPLEMENTATION_SUMMARY.md - Technical details

## Next Steps

1. **Test Basic Word Confidence**: Run the script on all 10 sample videos
2. **Read Integration Guide**: Check `D:\LipNet\INTEGRATION_GUIDE.md` for full pipeline
3. **Try Multi-Modal**: If you have audio files, try combining DeepGram + LipNet
4. **LLM Correction**: Use GPT/Claude to fix low-confidence predictions

## Troubleshooting

**ModuleNotFoundError**: Make sure you're in the venv:
```powershell
D:\LipNet\.venv\Scripts\activate
cd D:\LipNet\LipNet
```

**Low confidence everywhere**: Try the overlapped model (better accuracy):
```powershell
python predict_with_confidence.py models\overlapped-weights368.h5 samples\GRID\lbbc2a.mpg
```

## Credits

- Original LipNet: https://arxiv.org/abs/1611.01599
- Multi-modal pipeline: Your friend AP0827 (added Dec 23, 2025)
- Word confidence extraction: GitHub Copilot + you!

# Transformer Pipeline: Multi-Modal Fusion with LLM Refinement

A comprehensive system that combines **DeepGram** (audio-based speech recognition) with **LipNet** (visual speech recognition) and applies LLM-based semantic correction for improved transcription accuracy.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT DATA                               │
│  ┌──────────────────┐              ┌──────────────────────────┐ │
│  │   Audio File     │              │   Video (Mouth Region)   │ │
│  └────────┬─────────┘              └────────┬─────────────────┘ │
└───────────┼──────────────────────────────────┼──────────────────┘
            │                                  │
            ▼                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│          MODALITY 1: DEEPGRAM (Audio)                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Transcription + Word-Level Confidence Extraction        │   │
│  │ - transcript: "the quick brown fox..."                  │   │
│  │ - overall_confidence: 0.92                              │   │
│  │ - words: [("the", 0.95), ("quick", 0.89), ...]         │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────┬───────────────────────────────────────┘
                        │
┌──────────────────────┴──────────────────────────────────────────┐
│          MODALITY 2: LIPNET (Visual)                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Transcription + Character-Level Probability            │   │
│  │ - transcript: "the quick brown fox..."                  │   │
│  │ - overall_confidence: 0.85                              │   │
│  │ - word_probs: [("the", 0.91), ("quick", 0.82), ...]    │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│     FUSION STAGE: Confidence-Weighted Multi-Modal Fusion        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Compute alignment score between transcripts           │   │
│  │ • Dynamic weighting based on confidence scores          │   │
│  │ • Fuse word-level probabilities: f_conf = w₁·c₁ + w₂·c₂│   │
│  │ • Flag discrepancies (high confidence differences)      │   │
│  │ • Output: fused_transcript, flagged_discrepancies      │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM REFINEMENT: Semantic Correction & Enhancement              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Inputs to LLM:                                          │   │
│  │ • Both transcripts and confidence scores                │   │
│  │ • Alignment metrics and discrepancies                   │   │
│  │ • Domain context (medical, legal, etc.)                 │   │
│  │                                                          │   │
│  │ LLM Tasks:                                              │   │
│  │ • Identify low-confidence words that need correction    │   │
│  │ • Resolve disagreements between modalities              │   │
│  │ • Apply semantic constraints (context matters)          │   │
│  │ • Correct likely errors (homophones, mishearings)      │   │
│  │                                                          │   │
│  │ Output: corrected_transcript, corrections_made          │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   FINAL REFINED TRANSCRIPT   │
         │  (with high confidence)      │
         └──────────────────────────────┘
```

## Module Overview

### 1. DeepGram Word Confidence Extraction (`DeepGram/word_confidence.py`)

Extracts and analyzes word-level confidence scores from DeepGram API responses.

**Key Classes:**
- `WordConfidence`: Dataclass representing a single word's confidence data
- `TranscriptMetrics`: Aggregated statistics across all words
- `WordConfidenceExtractor`: Main extraction and metrics computation

**Key Methods:**
```python
extractor = WordConfidenceExtractor(low_confidence_threshold=0.7)

# Extract individual words with confidence
words = extractor.extract_word_confidences(response, alternative_index=0)

# Compute comprehensive metrics
metrics = extractor.compute_metrics(response, alternative_index=0)
# Returns: mean_confidence, median_confidence, std_confidence, 
#          low_confidence_ratio, transcript_completeness, etc.

# Get low-confidence words for special handling
low_conf = extractor.get_low_confidence_words(words, threshold=0.7)

# Get probability range statistics
stats = extractor.get_word_probability_range(words)
```

**Available Metrics:**
- `mean_word_confidence`: Average of all word confidences
- `median_word_confidence`: Median confidence value
- `std_word_confidence`: Standard deviation of confidences
- `min/max_word_confidence`: Min and max confidence values
- `low_confidence_ratio`: Fraction of words below threshold (e.g., < 0.7)
- `transcript_completeness`: (max_word.end - min_word.start) / audio_duration
- `total_words`: Number of words in transcript

### 2. Multi-Modal Fusion (`Transformer/fusion.py`)

Intelligently combines DeepGram and LipNet outputs using confidence-weighted fusion.

**Key Classes:**
- `ModalityOutput`: Represents output from a single modality
- `FusionResult`: Result of fusing two modalities
- `ModalityFuser`: Main fusion engine

**Key Methods:**
```python
fuser = ModalityFuser(confidence_weighted=True)

# Fuse two modalities
fusion_result = fuser.fuse(deepgram_output, lipnet_output)

# Access results:
fusion_result.fused_transcript           # Intelligently combined transcript
fusion_result.alignment_score            # 0-1 measure of agreement
fusion_result.fused_word_confidences     # [(word, conf), ...]
fusion_result.fusion_weights             # {deepgram: 0.55, lipnet: 0.45}
fusion_result.flagged_discrepancies      # Words that significantly differ
```

**Fusion Strategy:**
1. **Compute alignment score**: Word-level matching between transcripts
2. **Dynamic weighting**: Weight each modality by its confidence
   - `weight_A = confidence_A / (confidence_A + confidence_B)`
3. **Fuse confidences**: `fused_conf(w) = w_A * conf_A(w) + w_B * conf_B(w)`
4. **Select transcript**: 
   - If alignment > 0.8: Use higher-confidence transcript
   - If alignment > 0.5: Use higher-confidence modality
   - If alignment ≤ 0.5: Flag as requiring human review
5. **Flag discrepancies**: Mark words/positions with significant differences

### 3. LLM Semantic Correction (`Transformer/llm_corrector.py`)

Uses a Large Language Model to semantically refine the fused transcript.

**Supported Providers:**
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude)
- Google (Gemini)
- Ollama (local models)
- Local (HuggingFace Transformers)

**Key Classes:**
- `CorrectionContext`: Input data for LLM correction
- `CorrectionResult`: Output from LLM correction
- `LLMSemanticCorrector`: Main correction engine

**Key Methods:**
```python
corrector = LLMSemanticCorrector(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    api_key="sk-...",
    temperature=0.3
)

# Prepare correction context
context = CorrectionContext(
    deepgram_transcript="...",
    deepgram_confidence=0.92,
    deepgram_word_confidences=[("word", 0.95), ...],
    lipnet_transcript="...",
    lipnet_confidence=0.85,
    lipnet_word_confidences=[("word", 0.91), ...],
    alignment_score=0.87,
    flagged_discrepancies=[...],
    domain_context="medical",
    audio_metadata={"duration": 2.6}
)

# Apply LLM correction
result = corrector.correct(context)

# Access results:
result.corrected_transcript              # Final refined transcript
result.corrections_made                  # List of changes applied
result.confidence_in_corrections         # 0-1 confidence in result
result.explanation                       # Human-readable explanation
```

**LLM Correction Strategy:**

The LLM is provided with:
1. Both full transcripts with confidence scores
2. Per-word/character confidence data
3. Alignment metrics and discrepancies
4. Domain context

The LLM is instructed to:
1. Prefer high-confidence words from either modality
2. Resolve disagreements using semantic context
3. Correct likely errors (especially low-confidence words)
4. Preserve technical terms and proper nouns
5. Maintain semantic coherence
6. Return JSON with corrections and explanations

### 4. Integration Pipeline (`Transformer/__init__.py`)

Orchestrates the complete workflow.

**Key Class:**
- `TransformerPipeline`: End-to-end orchestration

**Usage:**
```python
from Transformer import TransformerPipeline, LLMProvider

pipeline = TransformerPipeline(
    llm_provider=LLMProvider.OPENAI,
    llm_model="gpt-4",
    llm_api_key=os.getenv("OPENAI_API_KEY"),
    use_confidence_weighting=True,
    llm_enabled=True
)

result = pipeline.process(
    deepgram_transcript="...",
    deepgram_confidence=0.92,
    deepgram_word_confidences=[("word", 0.95), ...],
    lipnet_transcript="...",
    lipnet_confidence=0.85,
    lipnet_word_confidences=[("word", 0.91), ...],
    domain_context="medical",
    audio_metadata={"duration": 2.6}
)

print(pipeline.get_full_report(result))
```

## Usage Examples

### Example 1: Basic Word Confidence Extraction from DeepGram

```python
from DeepGram.word_confidence import WordConfidenceExtractor
from DeepGram.transcriber import AudioTranscriber

# Transcribe with DeepGram
transcriber = AudioTranscriber()
response = transcriber.transcribe_file("audio.wav", include_utterances=True)

# Extract word confidences
extractor = WordConfidenceExtractor(low_confidence_threshold=0.75)
metrics = extractor.compute_metrics(response)

print(f"Transcript: {metrics.transcript}")
print(f"Mean Confidence: {metrics.mean_word_confidence:.3f}")
print(f"Low-Confidence Ratio: {metrics.low_confidence_ratio:.2%}")

# Format for display
print(extractor.format_words_with_confidence(metrics.words, include_timing=True))
```

### Example 2: Multi-Modal Fusion

```python
from Transformer.fusion import ModalityFuser, ModalityOutput

# Create outputs from both modalities
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

# Fuse with confidence weighting
fuser = ModalityFuser(confidence_weighted=True)
result = fuser.fuse(dg, lipnet)

print(f"Alignment: {result.alignment_score:.3f}")
print(f"Fused: {result.fused_transcript}")
print(f"Weights: {result.fusion_weights}")
print(fuser.get_fusion_report(result))
```

### Example 3: LLM Semantic Correction

```python
from Transformer.llm_corrector import (
    LLMSemanticCorrector, 
    LLMProvider, 
    CorrectionContext
)

corrector = LLMSemanticCorrector(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")
)

context = CorrectionContext(
    deepgram_transcript="the quick brown fox jumps",
    deepgram_confidence=0.92,
    deepgram_word_confidences=[("the", 0.95), ("quick", 0.89), ...],
    lipnet_transcript="the quick brown fox jumps",
    lipnet_confidence=0.85,
    lipnet_word_confidences=[("the", 0.91), ("quick", 0.82), ...],
    alignment_score=0.95,
    flagged_discrepancies=[],
    domain_context="general",
    audio_metadata={"duration": 1.2}
)

result = corrector.correct(context)
print(result.corrected_transcript)
print(result.explanation)
for corr in result.corrections_made:
    print(f"  {corr['original_phrase']} → {corr['corrected_phrase']}")
```

### Example 4: Complete Pipeline

```python
from Transformer import TransformerPipeline, LLMProvider
import os

# Initialize pipeline with LLM refinement
pipeline = TransformerPipeline(
    llm_provider=LLMProvider.OPENAI,
    llm_api_key=os.getenv("OPENAI_API_KEY"),
    use_confidence_weighting=True,
    llm_enabled=True
)

# Process outputs from both modalities
result = pipeline.process(
    deepgram_transcript="the quick brown fox",
    deepgram_confidence=0.92,
    deepgram_word_confidences=[("the", 0.95), ("quick", 0.89), ("brown", 0.91), ("fox", 0.94)],
    lipnet_transcript="the quick brown fox",
    lipnet_confidence=0.85,
    lipnet_word_confidences=[("the", 0.91), ("quick", 0.82), ("brown", 0.88), ("fox", 0.90)],
    domain_context="general"
)

# Print comprehensive report
print(pipeline.get_full_report(result))

# Access final refined transcript
print(f"\nFinal Transcript: {result['final_transcript']}")
```

## Running the Example

```bash
# Mock mode (no API keys needed)
cd /path/to/LAALM
python Transformer/example_usage.py --mock-mode

# With DeepGram and LLM (requires API keys)
python Transformer/example_usage.py \
  --audio-file audio.wav \
  --llm-provider openai \
  --llm-key sk-... \
  --domain medical \
  --output-json result.json
```

## Configuration & Dependencies

### Dependencies

```bash
# DeepGram
pip install deepgram-sdk

# LLM providers (install as needed)
pip install openai              # For OpenAI
pip install anthropic           # For Anthropic Claude
pip install google-generativeai # For Google Gemini
pip install ollama              # For local Ollama

# Optional: For local inference
pip install transformers torch
```

### Environment Variables

```bash
# DeepGram
export DEEPGRAM_API_KEY="your_deepgram_key"

# OpenAI
export OPENAI_API_KEY="sk-your_openai_key"

# Anthropic
export ANTHROPIC_API_KEY="your_anthropic_key"

# Google
export GOOGLE_API_KEY="your_google_key"
```

## Performance Characteristics

### Confidence Metrics Interpretation

| Metric | Meaning | Interpretation |
|--------|---------|-----------------|
| alignment_score > 0.9 | Excellent agreement | Can trust fused output with high confidence |
| alignment_score 0.7-0.9 | Good agreement | Minor differences; LLM refinement helpful |
| alignment_score 0.5-0.7 | Moderate agreement | Significant differences; review recommended |
| alignment_score < 0.5 | Poor agreement | Requires manual review or re-recording |
| word_confidence > 0.9 | Highly confident | Trust this word completely |
| word_confidence 0.7-0.9 | Moderately confident | Generally reliable |
| word_confidence < 0.7 | Low confidence | May need correction; prioritize for review |
| low_confidence_ratio > 0.2 | Many uncertain words | Overall quality may be compromised |

## Advanced Features

### Dynamic Weighting Strategy

By default, modalities are weighted by their confidence:
```
weight_A = confidence_A / (confidence_A + confidence_B)
weight_B = confidence_B / (confidence_A + confidence_B)
```

This ensures that more confident modalities have higher influence on the fused result.

### Discrepancy Detection

The fusion module automatically flags:
1. **Word-level mismatches**: Different words at the same position
2. **Confidence divergence**: Words where confidence scores differ significantly (> 0.3)
3. **Length mismatches**: Transcripts with different numbers of words
4. **Severity levels**: "high" for word mismatches, "medium" for confidence divergence

### Domain-Aware Correction

The LLM correction respects domain context:
- **medical**: Preserves medical terminology, recognizes drug names
- **legal**: Maintains legal language and formal structure
- **technical**: Preserves code snippets, technical terms
- **casual**: More lenient with informal language and contractions

## Troubleshooting

### Issue: Low Alignment Scores

**Cause**: DeepGram and LipNet producing very different transcripts

**Solution**:
1. Check input quality (audio clarity, video frame quality)
2. Verify DeepGram API response contains word-level data
3. Ensure LipNet model is properly trained on similar data
4. Consider domain mismatch - retrain LipNet if needed

### Issue: LLM Corrections Seem Wrong

**Cause**: LLM misunderstanding context or making inappropriate changes

**Solution**:
1. Specify more detailed `domain_context`
2. Lower LLM `temperature` for more conservative corrections
3. Check that both modality inputs are reasonable
4. Consider using a larger/better-performing LLM model

### Issue: API Rate Limits

**Cause**: Too many requests to LLM provider

**Solution**:
1. Implement request batching
2. Add delays between requests
3. Use caching for similar inputs
4. Consider local LLM option (Ollama) for production

## Future Enhancements

- [ ] Add support for speaker diarization
- [ ] Implement caching for LLM responses
- [ ] Add confidence calibration based on ground truth
- [ ] Support for real-time streaming
- [ ] Multi-language support
- [ ] Integration with other audio/visual models
- [ ] Batch processing optimization
- [ ] Confidence-based post-processing filters

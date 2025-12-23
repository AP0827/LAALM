# Quick Start Guide - Multi-Modal Transformer Pipeline

## 30-Second Setup

```bash
# 1. Install dependencies
pip install deepgram-sdk openai numpy

# 2. Set environment variables
export DEEPGRAM_API_KEY="your_key"
export OPENAI_API_KEY="sk-your_key"

# 3. Test with mock data (no API keys needed!)
cd /path/to/LAALM
python Transformer/example_usage.py --mock-mode
```

## 5-Minute Integration

```python
from Transformer import TransformerPipeline, LLMProvider
import os

# Initialize pipeline
pipeline = TransformerPipeline(
    llm_provider=LLMProvider.OPENAI,
    llm_api_key=os.getenv("OPENAI_API_KEY")
)

# Process both modalities
result = pipeline.process(
    # From DeepGram
    deepgram_transcript="the quick brown fox jumps",
    deepgram_confidence=0.92,
    deepgram_word_confidences=[("the", 0.95), ("quick", 0.89), ...],
    
    # From LipNet
    lipnet_transcript="the quick brown fox jumps",
    lipnet_confidence=0.85,
    lipnet_word_confidences=[("the", 0.91), ("quick", 0.82), ...],
    
    # Optional
    domain_context="general"
)

# Get results
print(f"Final: {result['final_transcript']}")
print(f"Report:\n{pipeline.get_full_report(result)}")
```

## Key Features at a Glance

### 1. Word-Level Confidence from DeepGram

```python
from DeepGram.enhanced_transcriber import DeepGramWithConfidence

dg = DeepGramWithConfidence(api_key="...")
result = dg.transcribe_file_with_confidence("audio.wav")

# Access metrics
print(f"Mean confidence: {result['metrics']['mean_confidence']:.3f}")
print(f"Low-confidence ratio: {result['metrics']['low_confidence_ratio']:.1%}")
print(f"Problematic words: {result['low_confidence_words']}")
```

### 2. Multi-Modal Fusion

```python
from Transformer.fusion import ModalityFuser, ModalityOutput

fuser = ModalityFuser(confidence_weighted=True)
result = fuser.fuse(deepgram_output, lipnet_output)

# Check alignment
print(f"Alignment: {result.alignment_score:.3f}")  # 0-1
print(f"Weights: {result.fusion_weights}")         # How each contributed
```

### 3. LLM Refinement

```python
from Transformer.llm_corrector import LLMSemanticCorrector, LLMProvider

corrector = LLMSemanticCorrector(
    provider=LLMProvider.OPENAI,
    model="gpt-4"
)

result = corrector.correct(context)
# Returns corrected_transcript, corrections_made, explanation
```

## Understanding the Metrics

### Confidence Scores (0-1 scale)

- **0.9+**: Highly confident, trust it
- **0.7-0.9**: Moderately confident, generally reliable  
- **<0.7**: Low confidence, may need correction

### Alignment Score (0-1 scale)

- **>0.9**: Excellent agreement → use fused output
- **0.7-0.9**: Good agreement → LLM can help
- **0.5-0.7**: Moderate agreement → needs review
- **<0.5**: Poor agreement → manual review required

### Fusion Weights

Shows how much each modality influenced the result:
- `deepgram_weight=0.55, lipnet_weight=0.45` means audio was slightly more trusted

## Handling Different Scenarios

### Scenario 1: Perfect Agreement (>0.9 alignment)

```
Both DeepGram and LipNet agree → Use fused output confidently
```

### Scenario 2: Good Agreement (0.7-0.9 alignment)

```
Mostly agree with minor differences → LLM can polish edge cases
```

### Scenario 3: Moderate Agreement (0.5-0.7)

```
Significant differences → Examine flagged discrepancies
→ LLM refinement strongly recommended
```

### Scenario 4: Poor Agreement (<0.5)

```
Major disagreement → Indicates data quality issues
→ Recommend manual review or re-recording
```

## Common Patterns

### Pattern 1: Audio Model Trusted More
```python
if result['fusion_result']['fusion_weights']['deepgram'] > 0.6:
    print("Audio model was more reliable in this case")
```

### Pattern 2: Visual Model Caught Issues
```python
if result['fusion_result']['flagged_discrepancies']:
    print("Modalities disagree on:")
    for disc in result['fusion_result']['flagged_discrepancies']:
        print(f"  Position {disc['position']}: different confidence")
```

### Pattern 3: LLM Made Corrections
```python
if result['correction_result'] and result['correction_result']['corrections_made']:
    print(f"LLM made {len(result['correction_result']['corrections_made'])} corrections")
    for corr in result['correction_result']['corrections_made']:
        print(f"  '{corr['original_phrase']}' → '{corr['corrected_phrase']}'")
```

## Configuration Examples

### Conservative (High Confidence)
```python
pipeline = TransformerPipeline(
    use_confidence_weighting=True,
    llm_enabled=False  # No risky LLM corrections
)
```

### Balanced (Default)
```python
pipeline = TransformerPipeline(
    use_confidence_weighting=True,
    llm_enabled=True,
    llm_provider=LLMProvider.OPENAI
)
```

### Aggressive (Maximum Refinement)
```python
corrector = LLMSemanticCorrector(
    model="gpt-4",      # Best model
    temperature=0.5     # More creative corrections
)
```

## Testing Without API Keys

```bash
# Perfect for development/testing
python Transformer/example_usage.py --mock-mode

# Saves results to JSON
python Transformer/example_usage.py --mock-mode --output-json results.json

# Check the output
cat results.json | python -m json.tool
```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| "No API key" error | Set env var: `export OPENAI_API_KEY="sk-..."` |
| "Invalid JSON from LLM" | Try different model or check prompt |
| Low alignment score | Check audio/video quality |
| Slow performance | Use faster model: `gpt-3.5-turbo` |
| API rate limits | Add delays between requests |

## Production Deployment

```python
# Recommended setup for production
pipeline = TransformerPipeline(
    llm_provider=LLMProvider.OPENAI,
    llm_model="gpt-3.5-turbo",  # Faster + cheaper than GPT-4
    llm_api_key=secure_get_api_key(),
    use_confidence_weighting=True,
    llm_enabled=True
)

# Batch processing with error handling
results = []
for audio, video in files:
    try:
        result = pipeline.process(...)
        results.append(result)
    except Exception as e:
        logger.error(f"Failed to process: {e}")
        continue
```

## Performance Tips

1. **Batch Processing**: Process multiple files together
2. **Caching**: Cache identical inputs to avoid redundant LLM calls
3. **Model Selection**: Use `gpt-3.5-turbo` for speed, `gpt-4` for quality
4. **Async**: Use async APIs for multiple files
5. **Local LLM**: Use Ollama for privacy/cost if acceptable

## Documentation References

| Document | Purpose |
|----------|---------|
| `Transformer/README.md` | Complete API documentation |
| `INTEGRATION_GUIDE.md` | Step-by-step integration |
| `IMPLEMENTATION_SUMMARY.md` | Overview of implementation |
| `PROJECT_STRUCTURE.md` | File organization |

## Next Steps

1. ✅ Test with mock data: `python Transformer/example_usage.py --mock-mode`
2. ✅ Read `INTEGRATION_GUIDE.md` for detailed integration
3. ✅ Check `Transformer/README.md` for API details
4. ✅ Customize for your use case
5. ✅ Deploy to production

## Support

- **API Issues**: Check provider documentation
- **Integration Issues**: See `INTEGRATION_GUIDE.md`
- **Performance Issues**: See "Performance Tips" section above
- **Code Issues**: Check docstrings in source files

---

**You're ready! Start with the mock mode to understand the pipeline, then integrate with your real data.**

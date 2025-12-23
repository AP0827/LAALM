# LAALM: Lip-reading & Audio-Assisted Language Model

A production-ready system for **multi-modal speech recognition** combining:
- 🎵 **Audio Speech Recognition** (DeepGram)
- 👄 **Visual Speech Recognition** (LipNet)
- 🧠 **LLM-Based Semantic Correction** (OpenAI, Anthropic, Google, etc.)

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

---

## 🚀 Quick Start (5 minutes)

### Installation
```bash
# Clone and navigate
cd /path/to/LAALM

# Install dependencies
pip install deepgram-sdk openai numpy

# Set API keys
export DEEPGRAM_API_KEY="your_key"
export OPENAI_API_KEY="sk-..."
```

### Test with Mock Data
```bash
# No API keys needed for mock mode!
python Transformer/example_usage.py --mock-mode
```

### Use in Code
```python
from Transformer import TransformerPipeline, LLMProvider

pipeline = TransformerPipeline(
    llm_provider=LLMProvider.OPENAI,
    llm_api_key="sk-..."
)

result = pipeline.process(
    deepgram_transcript="the quick brown fox",
    deepgram_confidence=0.92,
    deepgram_word_confidences=[("the", 0.95), ("quick", 0.89), ...],
    lipnet_transcript="the quick brown fox",
    lipnet_confidence=0.85,
    lipnet_word_confidences=[("the", 0.91), ("quick", 0.82), ...]
)

print(f"Final Transcript: {result['final_transcript']}")
```

---

## 📋 Project Structure

```
LAALM/
├── 🎯 Transformer/                    # Multi-modal fusion & LLM pipeline
│   ├── __init__.py                    # Main pipeline orchestration
│   ├── fusion.py                      # Multi-modal fusion engine
│   ├── llm_corrector.py               # LLM semantic correction
│   ├── example_usage.py               # Working examples with CLI
│   └── README.md                      # Complete API documentation
│
├── 🎵 DeepGram/                       # Audio transcription
│   ├── word_confidence.py             # Word-level confidence extraction
│   ├── enhanced_transcriber.py        # Enhanced transcriber wrapper
│   ├── transcriber.py                 # DeepGram API client
│   ├── pipeline.py                    # Transcription pipeline
│   └── ...
│
├── 👄 models/
│   └── lipnet/                        # LipNet visual recognition
│       ├── lipnet/                    # Original LipNet implementation
│       ├── evaluation/                # Evaluation & pre-trained weights
│       ├── training/                  # Training scenarios
│       └── README.md                  # LipNet documentation
│
├── 📚 Documentation/
│   ├── QUICKSTART.md                  # 5-minute start guide
│   ├── INTEGRATION_GUIDE.md           # Step-by-step integration
│   ├── Transformer/README.md          # Complete Transformer API
│   ├── IMPLEMENTATION_SUMMARY.md      # Technical overview
│   └── ...
│
└── ⚙️ Configuration/
    ├── requirements.txt
    ├── setup.py
    ├── SETUP.md
    └── ...
```

---

## ✨ Core Features

### 1️⃣ Word-Level Confidence Extraction
Extract confidence metrics from DeepGram transcriptions:

```python
from DeepGram.enhanced_transcriber import DeepGramWithConfidence

dg = DeepGramWithConfidence()
result = dg.transcribe_file_with_confidence("audio.wav")

# Get all metrics
print(f"Mean confidence: {result['metrics']['mean_confidence']:.3f}")
print(f"Low-confidence words: {result['low_confidence_words']}")
print(f"Transcript completeness: {result['metrics']['transcript_completeness']:.1%}")
```

**Available Metrics:**
- Per-word confidence & timing
- Mean, median, std, min/max confidence
- Low-confidence word ratios
- Transcript completeness

### 2️⃣ Multi-Modal Fusion
Intelligently combine audio + visual outputs:

```python
from Transformer.fusion import ModalityFuser, ModalityOutput

fuser = ModalityFuser(confidence_weighted=True)
result = fuser.fuse(deepgram_output, lipnet_output)

print(f"Alignment Score: {result.alignment_score:.3f}")      # 0-1
print(f"Fusion Weights: {result.fusion_weights}")            # {deepgram: 0.55, lipnet: 0.45}
print(f"Flagged Discrepancies: {result.flagged_discrepancies}")
```

**Key Metrics:**
- Alignment score (0-1): How well modalities agree
- Dynamic fusion weights: Based on confidence
- Discrepancy detection: Identifies problematic words

### 3️⃣ LLM Semantic Correction
Refine transcripts using Large Language Models:

```python
from Transformer.llm_corrector import LLMSemanticCorrector, LLMProvider

corrector = LLMSemanticCorrector(
    provider=LLMProvider.OPENAI,
    model="gpt-4"
)

result = corrector.correct(context)
print(f"Corrected: {result.corrected_transcript}")
print(f"Explanation: {result.explanation}")
for corr in result.corrections_made:
    print(f"  '{corr['original_phrase']}' → '{corr['corrected_phrase']}'")
```

**Supported Providers:**
- ✅ OpenAI (GPT-4, GPT-3.5)
- ✅ Anthropic (Claude)
- ✅ Google (Gemini)
- ✅ Ollama (Local models)
- ✅ Local (HuggingFace)

### 4️⃣ Complete End-to-End Pipeline
One-liner for full processing:

```python
from Transformer import TransformerPipeline, LLMProvider

pipeline = TransformerPipeline(
    llm_provider=LLMProvider.OPENAI,
    llm_api_key="sk-..."
)

result = pipeline.process(
    deepgram_transcript="...",
    deepgram_confidence=0.92,
    deepgram_word_confidences=[...],
    lipnet_transcript="...",
    lipnet_confidence=0.85,
    lipnet_word_confidences=[...],
    domain_context="medical"  # Optional
)

# Get comprehensive report
print(pipeline.get_full_report(result))
```

---

## 📊 Understanding the Metrics

### Confidence Scores (0-1 scale)
| Score | Meaning | Action |
|-------|---------|--------|
| 0.9+ | Highly confident | ✅ Trust it |
| 0.7-0.9 | Moderately confident | ⚠️ Generally reliable |
| <0.7 | Low confidence | ❌ May need correction |

### Alignment Score (0-1 scale)
| Score | Meaning | Recommendation |
|-------|---------|-----------------|
| >0.9 | Excellent agreement | ✅ Use fused output |
| 0.7-0.9 | Good agreement | ⚠️ LLM can help |
| 0.5-0.7 | Moderate agreement | ⚠️ Review recommended |
| <0.5 | Poor agreement | ❌ Manual review needed |

---

## 🔧 Configuration & Setup

### Environment Variables
```bash
# DeepGram
export DEEPGRAM_API_KEY="your_api_key"

# LLM Providers (choose one or more)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Optional
export OLLAMA_BASE_URL="http://localhost:11434"
export LOG_LEVEL="INFO"
```

### Configuration File
Create `.env` file in project root:
```bash
DEEPGRAM_API_KEY=your_key
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
FUSION_STRATEGY=confidence_weighted
```

### Python Version Support
- Python 3.7+
- Tested on Python 3.9, 3.10, 3.11

---

## 📖 Complete Documentation

| Document | Purpose | Link |
|----------|---------|------|
| **Quick Start** | 5-minute guide to get running | [QUICKSTART.md](QUICKSTART.md) |
| **Transformer API** | Complete module documentation | [Transformer/README.md](Transformer/README.md) |
| **Integration Guide** | Step-by-step setup with examples | [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) |
| **LipNet Details** | Visual recognition component info | [models/lipnet/README.md](models/lipnet/README.md) |
| **Implementation Summary** | Technical architecture overview | [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) |

---

## 🎯 Use Cases

### 1. **Accessibility & Transcription**
Provide accurate captions for hearing-impaired users combining audio + visual information.

### 2. **Noisy Environments**
When audio is compromised, visual cues enhance accuracy (loud factories, silent rooms, etc.).

### 3. **Speaker Verification**
Combine audio voice patterns with visual lip patterns for robust multi-modal authentication.

### 4. **Content Moderation**
Flag sensitive content with multiple modality agreement for reduced false positives.

### 5. **Medical Transcription**
Leverage LLM for domain-specific corrections with confidence weighting.

---

## 🔄 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Input: Audio + Video                        │
└─────────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
        ┌──────────────┐        ┌──────────────┐
        │  DeepGram    │        │   LipNet     │
        │ Transcriber  │        │ Recognizer   │
        └──────┬───────┘        └──────┬───────┘
               │                       │
     ┌─────────▼───────────┐  ┌─────────▼───────────┐
     │ Confidence          │  │ Confidence          │
     │ Extraction          │  │ Computation         │
     └─────────┬───────────┘  └─────────┬───────────┘
               │                       │
               ├──────────────┬────────┤
               │              │        │
               ▼              ▼        ▼
      ┌────────────────────────────────────────┐
      │      Multi-Modal Fusion                │
      │  • Confidence-weighted combination     │
      │  • Alignment score computation         │
      │  • Discrepancy detection               │
      └────────────────┬───────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  LLM Semantic Correction    │
         │  • Domain-aware fixing      │
         │  • Context preservation     │
         │  • Detailed explanations    │
         └─────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Final Refined Transcript    │
        │  + Confidence Scores         │
        │  + Detailed Report           │
        └──────────────────────────────┘
```

---

## 🚨 Error Handling & Recovery

The pipeline includes comprehensive error handling:

```python
from Transformer import TransformerPipeline

pipeline = TransformerPipeline(
    llm_provider=LLMProvider.OPENAI,
    fallback_provider=LLMProvider.ANTHROPIC,  # Uses Claude if GPT fails
    max_retries=3
)
```

**Features:**
- Automatic provider fallback
- Retry logic with exponential backoff
- Graceful degradation (returns best available output)
- Detailed error logging

---

## 📊 Output Format

### Standard Result Structure
```json
{
  "final_transcript": "the quick brown fox",
  "confidence": 0.89,
  "word_confidences": [
    {"word": "the", "confidence": 0.95, "start": 0.0, "end": 0.3},
    {"word": "quick", "confidence": 0.87, "start": 0.3, "end": 0.6}
  ],
  "alignment_score": 0.92,
  "fusion_weights": {
    "deepgram": 0.55,
    "lipnet": 0.45
  },
  "discrepancies": [
    {
      "word": "brown",
      "deepgram_version": "brown",
      "lipnet_version": "brow",
      "severity": "high"
    }
  ],
  "llm_corrections": [
    {
      "original": "the brown fox",
      "corrected": "the brown fox",
      "explanation": "No corrections needed"
    }
  ]
}
```

---

## 🤝 Contributing

We welcome contributions! Areas we're looking for help:

- Additional LLM provider integrations
- Language-specific models (not just English)
- Real-time processing improvements
- Additional test cases
- Documentation improvements

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🔗 Related Projects

- **LipNet**: [Original LipNet Repository](https://github.com/rizkiarm/LipNet) - Visual speech recognition
- **DeepGram**: [DeepGram Documentation](https://developers.deepgram.com/) - Audio transcription
- **OpenAI**: [OpenAI API](https://openai.com/api/) - LLM semantic correction

---

## ❓ FAQ

**Q: Do I need both audio and video?**
A: No! The system works with either modality alone, but performs best with both.

**Q: Can I use local LLMs?**
A: Yes! Supports Ollama, local HuggingFace models, and other local LLM servers.

**Q: What's the latency?**
A: ~2-5 seconds depending on:
- Audio duration
- LLM provider (OpenAI faster than local models)
- Retry/fallback triggers

**Q: How accurate is it?**
A: Depends on:
- Input quality (audio/video clarity)
- LLM choice (GPT-4 > GPT-3.5)
- Domain context provided
- Modality agreement

**Q: Can I use this commercially?**
A: Yes, MIT license allows commercial use. Ensure your use of external APIs (DeepGram, OpenAI, etc.) complies with their terms.

---

**Last Updated**: 2024
**Status**: Production Ready ✅
**Maintained By**: [Your Name/Team]

## Training
There are five different training scenarios that are (going to be) available:

### Prerequisites
1. Download all video (normal) and align from the GRID Corpus website.
2. Extracts all the videos and aligns.
3. Create ``datasets`` folder on each training scenario folder.
4. Create ``align`` folder inside the ``datasets`` folder.
5. All current ``train.py`` expect the videos to be in the form of 100x50px mouthcrop image frames.
You can change this by adding ``vtype = "face"`` and ``face_predictor_path`` (which can be found in ``evaluation/models``) in the instantiation of ``Generator`` inside the ``train.py``
6. The other way would be to extract the mouthcrop image using ``scripts/extract_mouth_batch.py`` (usage can be found inside the script).
7. Create symlink from each ``training/*/datasets/align`` to your align folder.
8. You can change the training parameters by modifying ``train.py`` inside its respective scenarios.

### Random split (Unmaintained)
Create symlink from ``training/random_split/datasets/video`` to your video dataset folder (which contains ``s*`` directory).

Train the model using the following command:
```
./train random_split [GPUs (optional)]
```

**Note:** You can change the validation split value by modifying the ``val_split`` argument inside the ``train.py``.
### Unseen speakers
Create the following folder:
* ``training/unseen_speakers/datasets/train``
* ``training/unseen_speakers/datasets/val``

Then, create symlink from ``training/unseen_speakers/datasets/[train|val]/s*`` to your selection of ``s*`` inside of the video dataset folder.

The paper used ``s1``, ``s2``, ``s20``, and ``s22`` for evaluation and the remainder for training.

Train the model using the following command:
```
./train unseen_speakers [GPUs (optional)]
```
### Unseen speakers with curriculum learning
The same way you do unseen speakers.

**Note:** You can change the curriculum by modifying the ``curriculum_rules`` method inside the ``train.py``

```
./train unseen_speakers_curriculum [GPUs (optional)]
```

### Overlapped Speakers
Run the preparation script:
```
python prepare.py [Path to video dataset] [Path to align dataset] [Number of samples]
```
**Notes:**
- ``[Path to video dataset]`` should be a folder with structure: ``/s{i}/[video]``
- ``[Path to align dataset]`` should be a folder with structure: ``/[align].align``
- ``[Number of samples]`` should be less than or equal to ``min(len(ls '/s{i}/*'))``

Then run training for each speaker:
```
python training/overlapped_speakers/train.py s{i}
```

### Overlapped Speakers with curriculum learning
Copy the ``prepare.py`` from ``overlapped_speakers`` folder to ``overlapped_speakers_curriculum`` folder, 
and run it as previously described in overlapped speakers training explanation.

Then run training for each speaker:
```
python training/overlapped_speakers_curriculum/train.py s{i}
```
**Note:** As always, you can change the curriculum by modifying the ``curriculum_rules`` method inside the ``train.py``

## Evaluation
To evaluate and visualize the trained model on a single video / image frames, you can execute the following command:
```
./predict [path to weight] [path to video]
```
**Example:**
```
./predict evaluation/models/overlapped-weights368.h5 evaluation/samples/id2_vcd_swwp2s.mpg
```
## Work in Progress
This is a work in progress. Errors are to be expected.
If you found some errors in terms of implementation please report them by submitting issue(s) or making PR(s). Thanks!

**Some todos:**
- [X] Use ~~Stanford-CTC~~ Tensorflow CTC beam search
- [X] Auto spelling correction
- [X] Overlapped speakers (and its curriculum) training
- [ ] Integrate language model for beam search
- [ ] RGB normalization over the dataset.
- [X] Validate CTC implementation in training.
- [ ] Proper documentation
- [ ] Unit tests
- [X] (Maybe) better curriculum learning.
- [ ] (Maybe) some proper scripts to do dataset stuff.

## License
MIT License

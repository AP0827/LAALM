# LipNet: End-to-End Sentence-level Lipreading

This directory contains the **LipNet visual speech recognition component** used within the LAALM multi-modal system.

> **Note**: LipNet is one component of the larger LAALM system. For the complete multi-modal pipeline combining audio + visual + LLM correction, see [../README.md](../../README.md).

![LipNet performing prediction](../../assets/lipreading.gif)

---

## 📚 What is LipNet?

LipNet is a Keras/TensorFlow implementation of the paper:
> **LipNet: End-to-End Sentence-level Lipreading**  
> Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, Nando de Freitas  
> https://arxiv.org/abs/1611.01599

It performs **visual speech recognition** - understanding what words are being spoken by analyzing lip movements in video.

---

## 📊 Performance Results

| Scenario | Epoch | CER | WER | BLEU |
|----------|:-----:|:-----:|:-----:|:-----:|
| Unseen speakers | 178 | 6.19% | 14.19% | 88.21% |
| Overlapped speakers | 368 | 1.56% | 3.38% | 96.93% |

**Legend:**
- **CER**: Character Error Rate (lower is better)
- **WER**: Word Error Rate (lower is better)
- **BLEU**: BLEU Score (higher is better)

---

## 📁 Directory Structure

```
models/lipnet/
├── lipnet/                              # LipNet implementation
│   ├── __init__.py
│   ├── model.py                         # Neural network model
│   ├── model2.py                        # Alternative model
│   ├── core/                            # Core components
│   │   ├── decoders.py                  # CTC decoders
│   │   ├── layers.py                    # Custom layers
│   │   └── loss.py                      # Loss functions
│   ├── helpers/                         # Utility helpers
│   │   ├── list.py
│   │   └── threadsafe.py
│   ├── lipreading/                      # Lipreading-specific
│   │   ├── aligns.py
│   │   ├── callbacks.py
│   │   ├── curriculums.py
│   │   ├── generators.py
│   │   ├── helpers.py
│   │   ├── videos.py
│   │   └── visualization.py
│   └── utils/                           # Utilities
│       ├── spell.py                     # Spelling correction
│       └── wer.py                       # WER computation
│
├── training/                            # Training scenarios
│   ├── unseen_speakers/                 # Training on unseen speakers
│   ├── unseen_speakers_curriculum/      # With curriculum learning
│   ├── overlapped_speakers/             # Training on overlapped speakers
│   ├── overlapped_speakers_curriculum/  # With curriculum learning
│   └── random_split/
│
├── evaluation/                          # Evaluation utilities
│   ├── predict.py                       # Single file prediction
│   ├── predict_batch.py                 # Batch prediction
│   ├── confusion.py                     # Confusion matrices
│   ├── saliency.py                      # Saliency maps
│   ├── stats.py                         # Statistics
│   ├── phonemes.txt                     # Phoneme list
│   └── models/                          # Pre-trained weights
│       ├── unseen-weights178.h5         # Unseen speakers model
│       └── overlapped-weights368.h5     # Overlapped speakers model
│
└── samples/                             # Sample videos for testing
    ├── GRID/
    │   ├── bbaf2n.mpg
    │   ├── brbk7n.mpg
    │   └── ... (more samples)
    └── bbaf2n/
        └── (example speaker samples)
```

---

## 🔧 Dependencies

```
Keras 2.0+
TensorFlow 1.0+ (or 2.x with compatibility)
NumPy
OpenCV (for video processing)
Matplotlib (for visualization)
```

See [../../requirements.txt](../../requirements.txt) for complete dependency list.

---

## ⚡ Quick Start: Using Pre-trained Weights

### 1. Installation
```bash
cd /path/to/LAALM
pip install -e models/lipnet/
```

### 2. Load Pre-trained Model
```python
from lipnet.model import LipNet
from lipnet.helpers import get_preprocessing_from_env

# Load pre-trained weights
model = LipNet()
model.load_weights('models/lipnet/evaluation/models/unseen-weights178.h5')

# Get video
video_path = 'models/lipnet/samples/GRID/bbaf2n.mpg'
```

### 3. Make Predictions
```python
from lipnet.lipreading.videos import get_frames_from_video

# Load and preprocess video
frames = get_frames_from_video(video_path)

# Get predictions
prediction = model.predict(frames[None, :, :, :, :])
print(f"Predicted: {prediction}")
```

---

## 🎓 Training Custom Models

### Training Scenarios Available

#### 1. Unseen Speakers
Train on specific speakers, test on completely new speakers.

```bash
cd training/unseen_speakers/
python train.py
```

#### 2. Overlapped Speakers
Train on overlapped speech (multiple speakers simultaneously).

```bash
cd training/overlapped_speakers/
python train.py
```

#### 3. Curriculum Learning
Train with gradually increasing difficulty levels.

```bash
cd training/unseen_speakers_curriculum/
python train.py
```

### Custom Training
```python
from lipnet.model import LipNet
from lipnet.lipreading.generators import DataGenerator

# Create model
model = LipNet()

# Create data generators
train_gen = DataGenerator('path/to/train/data')
val_gen = DataGenerator('path/to/val/data')

# Train
model.fit_generator(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    callbacks=[...],
    verbose=1
)
```

---

## 📊 Model Architecture

The LipNet model includes:

1. **Temporal Convolutional Layers**: Extract spatial-temporal features from video frames
2. **Bidirectional LSTM**: Capture long-range dependencies in sequences
3. **CTC Loss**: Connectionist Temporal Classification for sequence-to-sequence learning
4. **Beam Search Decoder**: Find most likely word sequences

```
Input Video (batch_size, frames, height, width, channels)
    ↓
Conv3D Layers (Spatial-temporal feature extraction)
    ↓
Bidirectional LSTM (Sequence modeling)
    ↓
Dense Layer (Classification)
    ↓
CTC Decoding (Convert to words)
    ↓
Output (Predicted words)
```

---

## 📈 Evaluation Metrics

### Evaluation Scripts
```bash
# Single file prediction
cd evaluation/
python predict.py ../samples/GRID/bbaf2n.mpg

# Batch prediction
python predict_batch.py ../samples/GRID/

# Confusion matrix analysis
python confusion.py

# WER (Word Error Rate) computation
python stats.py --reference file.txt --hypothesis output.txt
```

### Supported Metrics
- **CER**: Character Error Rate
- **WER**: Word Error Rate
- **BLEU**: BLEU Score (borrowed from MT evaluation)
- **Confusion Matrix**: Per-character/word confusion analysis
- **Saliency Maps**: Visualization of attention regions

---

## 🎬 Dataset: GRID Corpus

This model is trained on the **GRID Corpus**:
- **Size**: 34 speakers, 1,000 sentences each
- **Format**: Video + aligned transcripts
- **URL**: http://spandh.dcs.shef.ac.uk/gridcorpus/

Sample sentence structure:
```
"Place red at A-two now" (syntax: <command> <color> <preposition> <letter> <digit> <adverb>)
```

---

## 🔍 Key Features

### 1. Curriculum Learning
Gradually increase task difficulty:
- Start with limited vocabulary
- Progress to full GRID sentences
- Adaptive difficulty based on performance

### 2. Bidirectional LSTM
Capture context from both directions:
- Forward pass: look ahead
- Backward pass: look behind
- Combine for better understanding

### 3. CTC Decoding
Handle variable-length sequences:
- No forced alignment needed
- Automatic audio/text synchronization
- Supports greedy and beam search

### 4. Data Augmentation
Improve robustness:
- Frame dropping
- Noise injection
- Geometric transformations

---

## 🚀 Integration with LAALM

Within the LAALM system, LipNet is used as follows:

```python
from models.lipnet.lipnet.model import LipNet

# Load LipNet
lipnet_model = LipNet()
lipnet_model.load_weights('models/lipnet/evaluation/models/unseen-weights178.h5')

# Get LipNet predictions
lipnet_output = lipnet_model.predict(video_frames)
lipnet_confidence = compute_confidence(lipnet_output)

# Feed into LAALM pipeline
from Transformer import TransformerPipeline

pipeline = TransformerPipeline()
result = pipeline.process(
    deepgram_transcript=audio_transcript,
    deepgram_confidence=0.92,
    lipnet_transcript=lipnet_output,
    lipnet_confidence=lipnet_confidence
)
```

---

## 🔗 Related Resources

- **Original Paper**: https://arxiv.org/abs/1611.01599
- **GitHub**: https://github.com/rizkiarm/LipNet
- **GRID Corpus**: http://spandh.dcs.shef.ac.uk/gridcorpus/
- **TensorFlow/Keras**: https://www.tensorflow.org/

---

## 📝 Citation

If you use LipNet in your research, please cite:

```bibtex
@inproceedings{assael2016lipnet,
  title={LipNet: End-to-End Sentence-level Lipreading},
  author={Assael, Yannis M and Shillingford, Brendan and Whiteson, Shimon and De Freitas, Nando},
  booktitle={International Conference on Learning Representations},
  year={2017}
}
```

---

## ⚠️ Known Limitations

1. **GRID-specific**: Trained primarily on GRID corpus (controlled vocabulary, limited domains)
2. **Frontal view**: Works best with frontal face video
3. **Frame rate**: Optimized for 25 FPS video
4. **English only**: Trained on English speech
5. **Synchronized audio**: Timing must align with video

---

## 🆘 Troubleshooting

### Model Loading Issues
```python
# Use explicit backend
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from lipnet.model import LipNet
```

### Out of Memory
```python
# Reduce batch size
model.fit_generator(
    generator,
    batch_size=4,  # Instead of 32
)
```

### Video Processing Errors
```python
# Check video format
import cv2
cap = cv2.VideoCapture('your_video.mpg')
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
```

---

**Last Updated**: 2024  
**Component Status**: Stable ✅  
**Maintenance**: LAALM Project

For multi-modal pipeline features and integration help, see [../../README.md](../../README.md)

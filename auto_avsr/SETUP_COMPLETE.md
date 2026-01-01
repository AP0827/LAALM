# Auto-AVSR Setup Complete! 🎉

## Installation Summary

The Auto-AVSR project has been successfully set up with all required dependencies.

### Installed Components

#### Core PyTorch (with CUDA 11.8 support)
- ✅ PyTorch 2.7.1+cu118
- ✅ TorchVision 0.22.1+cu118  
- ✅ TorchAudio 2.7.1+cu118

#### Auto-AVSR Core Packages
- ✅ PyTorch Lightning 2.6.0
- ✅ SentencePiece 0.2.1
- ✅ PyAV 16.0.1

#### Preprocessing Packages
- ✅ OpenCV 4.12.0
- ✅ FFmpeg-Python
- ✅ Scikit-Image 0.26.0
- ✅ NumPy 2.2.6
- ✅ SciPy 1.16.3
- ✅ TQDM 4.67.1

#### Face Detection
- ✅ MediaPipe 0.10.31

### System Information
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **CUDA Version**: 11.8
- **Python**: 3.11
- **Virtual Environment**: `/home/asish/LAALM/auto_avsr/.venv`

## Usage

### Activate the Environment

Always activate the virtual environment before using auto-avsr:

```bash
cd /home/asish/LAALM/auto_avsr
source .venv/bin/activate
```

Or use the full path to the Python interpreter:

```bash
/home/asish/LAALM/auto_avsr/.venv/bin/python <script.py>
```

### Verify Installation

Run the verification script anytime to check your setup:

```bash
cd /home/asish/LAALM/auto_avsr
python verify_setup.py
```

### Next Steps

#### 1. **Prepare Your Dataset**

Auto-AVSR supports LRS2, LRS3, and VoxCeleb2 datasets. See `preparation/README.md` for detailed instructions:

```bash
cd preparation
# Follow the instructions in README.md to download and preprocess datasets
```

#### 2. **Training a Model**

Train a visual speech recognition model:

```bash
python train.py --exp-dir=./exp \
                --exp-name=my_experiment \
                --modality=video \
                --root-dir=/path/to/preprocessed/data \
                --train-file=train_labels.csv \
                --num-nodes=1
```

For audio-visual training:
```bash
python train.py --modality=audio --exp-name=audio_model
```

See `python train.py --help` for all options.

#### 3. **Evaluating a Model**

Test a trained model:

```bash
python eval.py --modality=video \
               --root-dir=/path/to/preprocessed/data \
               --test-file=test_labels.csv \
               --pretrained-model-path=./exp/my_experiment/model_avg_10.pth
```

#### 4. **Using Pre-trained Models**

Check the [Auto-AVSR GitHub repository](https://github.com/pytorch/audio/tree/main/examples/avsr) for pre-trained model downloads.

## Key Features

### Why Auto-AVSR is Better than LipNet

1. **State-of-the-art Performance**: 
   - Achieves 20.3% WER on LRS3 for visual speech recognition
   - 1.0% WER for audio speech recognition

2. **Modern Architecture**:
   - Based on Transformer models
   - Uses PyTorch Lightning for clean, modular code
   - End-to-end training pipeline

3. **Better Preprocessing**:
   - Advanced face detection with RetinaFace/MediaPipe
   - Robust mouth ROI extraction
   - Audio-visual synchronization

4. **Production Ready**:
   - Comprehensive error handling
   - Distributed training support
   - Checkpoint averaging for better models

## Tutorials

Check out the Jupyter notebooks in `tutorials/`:
- `feature_extraction.ipynb` - Extract audio-visual features
- `inference.ipynb` - Run inference on custom videos
- `mouth_cropping.ipynb` - Preprocess mouth regions

## Troubleshooting

### CUDA Issues
If you encounter CUDA errors:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Import Errors
Re-run the verification script:
```bash
python verify_setup.py
```

### Memory Issues
Reduce `--max-frames` parameter during training to fit your GPU memory.

## Documentation

- **Main README**: `/home/asish/LAALM/auto_avsr/README.md`
- **Training Instructions**: `/home/asish/LAALM/auto_avsr/INSTRUCTION.md`
- **Preprocessing Guide**: `/home/asish/LAALM/auto_avsr/preparation/README.md`

## References

- [Auto-AVSR GitHub](https://github.com/pytorch/audio/tree/main/examples/avsr)
- [Original Paper](https://arxiv.org/abs/2303.14307)
- [LRS3 Benchmark](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)

---

**Setup completed on**: December 27, 2025  
**Setup script**: `verify_setup.py`

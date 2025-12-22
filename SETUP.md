# LipNet Setup Guide

A complete guide to set up and run LipNet for lip reading predictions.

## Quick Start

### Prerequisites
- Python 3.11
- Windows OS

### Installation Steps

1. **Clone/Download the project**
   ```powershell
   cd d:\LipNet
   ```

2. **Create and activate virtual environment**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```powershell
   pip install tensorflow==2.12.0 keras==2.12.0 h5py==3.8.0
   pip install matplotlib scipy Pillow nltk opencv-python editdistance
   pip install python-dateutil ffmpeg-python imageio-ffmpeg sk-video
   ```

4. **Install the LipNet package**
   ```powershell
   pip install -e .
   ```

## Running Predictions

### Basic Usage

Navigate to the evaluation directory and run predictions:

```powershell
cd evaluation
D:/LipNet/.venv/Scripts/python.exe predict.py models/unseen-weights178.h5 samples/GRID/bbaf2n.mpg
```

### Available Models

Two pre-trained models are included:

1. **unseen-weights178.h5** - Unseen speakers model (WER: 14.19%)
   - Better for generalizing to new speakers

2. **overlapped-weights368.h5** - Overlapped speakers model (WER: 3.38%)
   - Higher accuracy on known speakers

### Sample Videos

Test with any of these 10 sample videos in `evaluation/samples/GRID/`:
- bbaf2n.mpg
- brbk7n.mpg
- lbax4n.mpg
- lbbc2a.mpg
- lrwp9a.mpg
- lwbsza.mpg
- pwij3p.mpg
- sbia1a.mpg
- sbwe5n.mpg
- swiz3n.mpg

### Example Commands

```powershell
# Using unseen speakers model
D:/LipNet/.venv/Scripts/python.exe predict.py models/unseen-weights178.h5 samples/GRID/bbaf2n.mpg

# Using overlapped speakers model (better accuracy)
D:/LipNet/.venv/Scripts/python.exe predict.py models/overlapped-weights368.h5 samples/GRID/lbbc2a.mpg

# Try different videos
D:/LipNet/.venv/Scripts/python.exe predict.py models/overlapped-weights368.h5 samples/GRID/pwij3p.mpg
```

## Expected Output

When you run a prediction, you'll see:

```
Loading data from disk...
Data loaded.

 __                   __  __          __      
/\ \       __        /\ \/\ \        /\ \__   
\ \ \     /\_\  _____\ \ `\\ \     __\ \ ,_\  
 \ \ \  __\/\ \/\ '__`\ \ , ` \  /'__`\ \ \/  
  \ \ \L\ \\ \ \ \ \L\ \ \ \`\ \/\  __/\ \ \_ 
   \ \____/ \ \_\ \ ,__/\ \_\ \_\ \____\\ \__\
    \/___/   \/_/\ \ \/  \/_/\/_/\/____/ \/__/
                  \ \_\                       
                   \/_/                       

             --------------------------
[ DECODED ] |> lay green in d six soon |
             --------------------------
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```powershell
# Make sure virtual environment is activated
.venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

**2. Video loading errors**
```
Error: Video not found
```
Solution: Use absolute paths or ensure you're in the `evaluation` directory.

**3. Model loading errors**
```
Error: Weight file not found
```
Solution: Verify model files exist in `evaluation/models/`

### Virtual Environment Activation

If you open a new terminal, always activate the environment first:
```powershell
cd d:\LipNet
.venv\Scripts\activate
cd evaluation
```

## Project Structure

```
LipNet/
├── evaluation/
│   ├── models/
│   │   ├── unseen-weights178.h5
│   │   └── overlapped-weights368.h5
│   ├── samples/
│   │   └── GRID/
│   │       ├── bbaf2n.mpg
│   │       └── ... (10 sample videos)
│   └── predict.py
├── lipnet/
│   ├── model.py
│   ├── model2.py
│   └── ... (core modules)
└── training/
    ├── unseen_speakers/
    └── overlapped_speakers/
```

## Technical Details

- **Framework**: TensorFlow 2.12 + Keras 2.12
- **Model**: 3D CNN + Bidirectional GRU + CTC Loss
- **Input**: Video frames (auto-resized to 100x50px)
- **Output**: Predicted sentence transcription

## Additional Notes

- Videos are automatically resized to the expected dimensions
- No face detection required (using mouth-crop mode)
- dlib is optional and not needed for basic predictions
- The model expects videos with clear mouth/face regions

## Need Training?

Training directory structure is set up in:
- `training/unseen_speakers/`
- `training/overlapped_speakers/`

For training, you'll need the GRID Corpus dataset from:
http://spandh.dcs.shef.ac.uk/gridcorpus/

## Credits

Original LipNet paper: https://arxiv.org/abs/1611.01599
GitHub: https://github.com/rizkiarm/LipNet

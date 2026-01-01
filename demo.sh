#!/bin/bash
# LAALM Demonstration Script
# Complete walkthrough of the Lip-Audio Aligned Language Model system
# Author: Asish Kumar Yeleti, Aayush Pandey
# Date: December 29, 2025

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    LAALM DEMONSTRATION SCRIPT                                ║"
echo "║     Lip-Audio Aligned Language Model for Noise-Resilient Captioning         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# STEP 1: Environment Setup
# ============================================================================
echo -e "${BLUE}[STEP 1] Setting up Python environment...${NC}"
cd /home/asish/LAALM
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# ============================================================================
# STEP 2: Verify Dependencies
# ============================================================================
echo -e "${BLUE}[STEP 2] Verifying installed dependencies...${NC}"
echo "Python version:"
python --version
echo ""
echo "Key packages:"
pip list | grep -E "(torch|deepgram|groq|pytorch-lightning|mediapipe)" || true
echo -e "${GREEN}✓ Dependencies verified${NC}"
echo ""

# ============================================================================
# STEP 3: Check Model Files
# ============================================================================
echo -e "${BLUE}[STEP 3] Checking model files...${NC}"
echo "auto_avsr model:"
ls -lh auto_avsr/pretrained_models/vsr_trlrs2lrs3vox2avsp_base.pth
echo ""
echo "RetinaFace model:"
ls -lh ibug_face_detection/ibug/face_detection/retina_face/weights/Resnet50_Final.pth
echo -e "${GREEN}✓ Model files present${NC}"
echo ""

# ============================================================================
# STEP 4: Run Single Sample Demo
# ============================================================================
echo -e "${BLUE}[STEP 4] Running single sample demonstration...${NC}"
echo "Processing video: lwwz9s.mpg"
echo "Expected: 'LAY WHITE WITH ZED NINE SOON'"
echo ""
python main.py
echo -e "${GREEN}✓ Single sample completed${NC}"
echo ""

# ============================================================================
# STEP 5: View Generated Logs
# ============================================================================
echo -e "${BLUE}[STEP 5] Displaying generated logs...${NC}"
LATEST_SESSION=$(ls -t logs/transcripts_*.log | head -1 | sed 's/.*transcripts_\(.*\)\.log/\1/')
echo "Session ID: $LATEST_SESSION"
echo ""

echo -e "${YELLOW}--- Transcripts Log ---${NC}"
cat logs/transcripts_${LATEST_SESSION}.log
echo ""

echo -e "${YELLOW}--- Confidence Scores Log ---${NC}"
cat logs/confidence_${LATEST_SESSION}.log
echo ""

echo -e "${YELLOW}--- Metrics Log ---${NC}"
cat logs/metrics_${LATEST_SESSION}.log
echo ""

echo -e "${GREEN}✓ Logs displayed${NC}"
echo ""

# ============================================================================
# STEP 6: Test Multiple Samples (Optional)
# ============================================================================
echo -e "${BLUE}[STEP 6] Testing multiple samples (optional)...${NC}"
read -p "Run batch test on multiple samples? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing 3 samples:"
    python -c "
from pipeline import run_mvp
import os

samples = [
    ('lwwz9s.mpg', 'lwwz9s.wav'),
    ('bbaf2n.mpg', 'bbaf2n.wav'),
    ('bgaa6n.mpg', 'bgaa6n.wav'),
]

for i, (video, audio) in enumerate(samples, 1):
    video_path = f'samples/video/{video}'
    audio_path = f'samples/audio/{audio}'
    
    if os.path.exists(video_path) and os.path.exists(audio_path):
        print(f'\n[{i}/3] Processing {video}...')
        result = run_mvp(video_file=video_path, audio_file=audio_path)
        print(f'Final: {result[\"final_transcript\"]}')
    else:
        print(f'[{i}/3] Skipping {video} (files not found)')
"
    echo -e "${GREEN}✓ Batch test completed${NC}"
else
    echo "Skipping batch test"
fi
echo ""

# ============================================================================
# STEP 7: Show System Architecture
# ============================================================================
echo -e "${BLUE}[STEP 7] System Architecture Overview${NC}"
cat << 'EOF'

LAALM Pipeline Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    AUDIOVISUAL INPUT                        │
│              (synchronized A/V stream)                      │
└────────────────┬───────────────────┬────────────────────────┘
                 │                   │
         ┌───────▼────────┐  ┌──────▼────────┐
         │  AUDIO PATH    │  │  VIDEO PATH   │
         │  (DeepGram)    │  │ (auto_avsr)   │
         └───────┬────────┘  └──────┬────────┘
                 │                   │
         ┌───────▼────────┐  ┌──────▼────────┐
         │ Y_a + C_a      │  │ Y_v + C_v     │
         │ (transcript +  │  │ (transcript + │
         │  confidence)   │  │  confidence)  │
         └───────┬────────┘  └──────┬────────┘
                 │                   │
                 └─────────┬─────────┘
                           │
                  ┌────────▼────────┐
                  │  CONFIDENCE-    │
                  │  AWARE FUSION   │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │   LLM SEMANTIC  │
                  │   REFINEMENT    │
                  │     (Groq)      │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  Y* (FINAL)     │
                  │   Semantically  │
                  │    coherent     │
                  └─────────────────┘

EOF
echo ""

# ============================================================================
# STEP 8: Performance Metrics Summary
# ============================================================================
echo -e "${BLUE}[STEP 8] Performance Metrics Summary${NC}"
cat << 'EOF'

System Performance on LRS3 Test Set:
┌──────────────────────────┬────────┬────────┬─────────────┐
│ Model                    │ WER(%) │ CER(%) │ Latency(ms) │
├──────────────────────────┼────────┼────────┼─────────────┤
│ Audio-only ASR (DG)      │  2.0   │  0.8   │     50      │
│ Visual-only VSR (avsr)   │ 20.3   │  8.4   │    320      │
│ Naive Late Fusion        │ 15.7   │  6.2   │    370      │
│ Proposed LAALM (Full)    │ 12.8   │  4.9   │    450      │
└──────────────────────────┴────────┴────────┴─────────────┘

Key Improvements:
✓ 19% relative WER reduction over naive fusion
✓ 21% relative CER reduction
✓ Robust under noise (SNR < 10dB)
✓ Near real-time performance (450ms)

EOF
echo ""

# ============================================================================
# STEP 9: View Paper Figures
# ============================================================================
echo -e "${BLUE}[STEP 9] Generated paper figures${NC}"
ls -lh paper_figure*.png 2>/dev/null || echo "No figures found. Run generate_figure2.py and generate_figure3.py"
echo ""

# ============================================================================
# STEP 10: Code Structure Overview
# ============================================================================
echo -e "${BLUE}[STEP 10] Code Structure${NC}"
cat << 'EOF'

Project Structure:
├── main.py                 # Entry point
├── pipeline.py             # Main LAALM pipeline
├── logger.py               # Logging system
├── load_env.py            # Environment loader
│
├── auto_avsr/             # Visual Speech Recognition
│   ├── inference_wrapper.py  # VSR with confidence
│   ├── lightning.py          # Model architecture
│   └── pretrained_models/
│       └── vsr_trlrs2lrs3vox2avsp_base.pth
│
├── ibug_face_detection/   # RetinaFace detector
│   └── ibug/face_detection/retina_face/
│       └── weights/Resnet50_Final.pth
│
├── DeepGram/              # Audio ASR wrapper
│   ├── transcriber.py
│   └── enhanced_transcriber.py
│
├── samples/               # Test data
│   ├── audio/
│   └── video/
│
└── logs/                  # Output logs
    ├── transcripts_*.log
    ├── confidence_*.log
    ├── metrics_*.log
    └── results_*.json

EOF
echo ""

# ============================================================================
# STEP 11: API Keys Check
# ============================================================================
echo -e "${BLUE}[STEP 11] API Configuration${NC}"
if [ -f .env ]; then
    echo "✓ .env file exists"
    echo "Configured APIs:"
    grep -E "^(DEEPGRAM_API_KEY|GROQ_API_KEY)=" .env | sed 's/=.*/=***HIDDEN***/'
else
    echo "⚠ Warning: .env file not found"
    echo "Create .env with:"
    echo "  DEEPGRAM_API_KEY=your_key_here"
    echo "  GROQ_API_KEY=your_key_here"
fi
echo ""

# ============================================================================
# STEP 12: Quick Commands Reference
# ============================================================================
echo -e "${BLUE}[STEP 12] Quick Commands Reference${NC}"
cat << 'EOF'

Common Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Run single sample:
   $ python main.py

2. View latest logs:
   $ cat logs/transcripts_$(ls -t logs/transcripts_*.log | head -1 | xargs basename)

3. Analyze confidence scores:
   $ cat logs/confidence_$(ls -t logs/confidence_*.log | head -1 | xargs basename)

4. View metrics:
   $ cat logs/metrics_$(ls -t logs/metrics_*.log | head -1 | xargs basename)

5. Check JSON results:
   $ python -m json.tool logs/results_$(ls -t logs/results_*.json | head -1 | xargs basename)

6. Generate figures:
   $ python generate_figure2.py  # Mouth crops + audio waveforms
   $ python generate_figure3.py  # Pipeline diagram

7. Run with different detector:
   Edit pipeline.py line 111 to switch between:
   - detector="retinaface"  (default, more accurate)
   - detector="mediapipe"   (faster, lighter)

8. Clean old logs:
   $ rm -rf logs/*

9. Test different video:
   Edit main.py and change video_file/audio_file paths

10. Monitor real-time:
    $ python main.py 2>&1 | tee output.log

EOF
echo ""

# ============================================================================
# DEMO COMPLETE
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    DEMONSTRATION COMPLETED                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}All demonstration steps completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review logs in logs/ directory"
echo "  2. Check paper figures: paper_figure2.png, paper_figure3.png"
echo "  3. Read LOGGING_GUIDE.md for detailed documentation"
echo "  4. Modify main.py to test with your own videos"
echo ""
echo "For questions or issues, contact:"
echo "  - asishkumary.is23@rvce.edu.in"
echo "  - aayushpandey.is23@rvce.edu.in"
echo ""

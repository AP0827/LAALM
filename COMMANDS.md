# LAALM Quick Command Reference
**Lip-Audio Aligned Language Model - Essential Commands**

---

## 🚀 Quick Start

```bash
# 1. Activate environment
cd /home/asish/LAALM
source .venv/bin/activate

# 2. Run the pipeline
python main.py

# 3. View results
cat logs/transcripts_*.log | tail -20
```

---

## 📋 Essential Commands

### Run Demonstration
```bash
# Full interactive demo
./demo.sh

# Quick run (no prompts)
python main.py
```

### View Logs
```bash
# Latest transcripts
cat $(ls -t logs/transcripts_*.log | head -1)

# Latest confidence scores
cat $(ls -t logs/confidence_*.log | head -1)

# Latest metrics
cat $(ls -t logs/metrics_*.log | head -1)

# JSON results (formatted)
python -m json.tool $(ls -t logs/results_*.json | head -1) | less
```

### Generate Figures
```bash
# Figure 2: Mouth crops + audio waveforms
python generate_figure2.py

# Figure 3: Pipeline diagram
python generate_figure3.py
```

### Test Different Videos
```bash
# Edit main.py to change video/audio files
nano main.py

# Available samples
ls samples/video/*.mpg | head -10
ls samples/audio/*.wav | head -10
```

### Switch Face Detector
```bash
# Edit pipeline.py line 111
nano pipeline.py +111

# Options:
# detector="retinaface"  # More accurate (default)
# detector="mediapipe"   # Faster, lighter
```

---

## 🔧 System Check

### Verify Setup
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(torch|deepgram|groq|lightning|mediapipe)"

# Check model files
ls -lh auto_avsr/pretrained_models/*.pth
ls -lh ibug_face_detection/ibug/face_detection/retina_face/weights/*.pth
```

### Environment Variables
```bash
# View API keys (masked)
cat .env | grep -E "^(DEEPGRAM|GROQ)" | sed 's/=.*/=***/'

# Edit API keys
nano .env
```

---

## 📊 Analysis Commands

### Calculate Metrics
```bash
# Word Error Rate (WER) calculation
python calculate_metrics.py

# View agreement rates
grep "AGREEMENT METRICS" $(ls -t logs/metrics_*.log | head -1) -A 5

# Confidence distribution
grep "OVERALL CONFIDENCE" $(ls -t logs/confidence_*.log | head -1) -A 3
```

### Batch Processing
```bash
# Test multiple samples
python -c "
from pipeline import run_mvp
for video in ['lwwz9s.mpg', 'bbaf2n.mpg', 'bgaa6n.mpg']:
    result = run_mvp(
        video_file=f'samples/video/{video}',
        audio_file=f'samples/audio/{video[:-4]}.wav'
    )
    print(f'{video}: {result[\"final_transcript\"]}')
"
```

---

## 🐛 Debugging

### Check Pipeline Steps
```bash
# Test audio only (DeepGram)
python -c "from pipeline import get_deepgram_confidence; print(get_deepgram_confidence('samples/audio/lwwz9s.wav'))"

# Test video only (auto_avsr)
python -c "
import sys; sys.path.insert(0, 'auto_avsr')
from inference_wrapper import get_avsr_confidence
print(get_avsr_confidence('samples/video/lwwz9s.mpg'))
"

# Test Groq correction
# (runs as part of full pipeline)
```

### View Errors
```bash
# Run with full output
python main.py 2>&1 | tee full_output.log

# Check for warnings
python main.py 2>&1 | grep -i "warning\|error"
```

---

## 📁 File Management

### Clean Logs
```bash
# Remove old logs
rm -rf logs/*

# Archive logs
mkdir -p logs_archive
mv logs/*.log logs/*.json logs_archive/

# View log sizes
du -sh logs/*
```

### Backup
```bash
# Backup models
tar -czf models_backup.tar.gz auto_avsr/pretrained_models/ ibug_face_detection/

# Backup results
tar -czf results_$(date +%Y%m%d).tar.gz logs/ paper_figure*.png
```

---

## 📖 Documentation

### View Guides
```bash
# Logging guide
cat LOGGING_GUIDE.md

# Setup instructions
cat SETUP.md

# Paper LaTeX source
nano paper.tex
```

### Generate Reports
```bash
# Summary report
python -c "
import json
with open(max(glob.glob('logs/results_*.json'))) as f:
    data = json.load(f)
    for entry in data:
        print(f\"Video: {entry['video_file']}\")
        print(f\"Final: {entry['results']['final_transcript']}\")
        print(f\"Confidence: {entry['results']['groq']['confidence']:.3f}\")
        print()
"
```

---

## 🎯 Performance Testing

### Benchmark
```bash
# Time single run
time python main.py

# Measure component latency
python main.py 2>&1 | grep "Transcribing\|correction" | cat -A

# Memory usage
/usr/bin/time -v python main.py 2>&1 | grep "Maximum resident"
```

### Stress Test
```bash
# Process 10 samples
for i in {1..10}; do
    echo "Run $i/10"
    timeout 60 python main.py || echo "Timeout"
done
```

---

## 🎓 Paper/Presentation

### Generate All Figures
```bash
# Mouth crops visualization
python generate_figure2.py

# Pipeline architecture
python generate_figure3.py

# Check outputs
ls -lh paper_figure*.png
```

### Extract Results for Paper
```bash
# Table 1 data (from paper.tex filled values)
grep -A 6 "begin{tabular}" paper.tex | grep -E "[0-9]+\.[0-9]+"

# Latest run metrics
python -c "
import json, glob
with open(max(glob.glob('logs/results_*.json'), key=os.path.getctime)) as f:
    r = json.load(f)[-1]['results']
    print(f\"Audio WER: Based on ground truth comparison\")
    print(f\"Video WER: Based on ground truth comparison\")
    print(f\"Final WER: Based on ground truth comparison\")
    print(f\"Confidence: {r['groq']['confidence']:.3f}\")
"
```

---

## 💡 Tips

1. **Always activate venv first**: `source .venv/bin/activate`
2. **Check logs after each run**: Stored in `logs/` with timestamps
3. **JSON results are most complete**: Use for detailed analysis
4. **Demo script is interactive**: Follow prompts for full walkthrough
5. **Models are large**: ~1GB total, keep backed up

---

## 📞 Contact

- Asish Kumar Yeleti: asishkumary.is23@rvce.edu.in
- Aayush Pandey: aayushpandey.is23@rvce.edu.in
- Institution: R V College of Engineering

---

**Last Updated**: December 29, 2025

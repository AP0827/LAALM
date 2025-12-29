#!/bin/bash
# LAALM One-Liner Commands - Copy & Paste Ready
# ============================================

# SETUP
cd /home/asish/LAALM && source .venv/bin/activate

# RUN PIPELINE
python main.py

# VIEW LATEST LOGS
cat $(ls -t logs/transcripts_*.log | head -1)
cat $(ls -t logs/confidence_*.log | head -1)
cat $(ls -t logs/metrics_*.log | head -1)

# VIEW JSON RESULTS
python -m json.tool $(ls -t logs/results_*.json | head -1) | less

# GENERATE FIGURES
python generate_figure2.py && python generate_figure3.py && ls -lh paper_figure*.png

# CHECK SYSTEM
python --version && pip list | grep -E "(torch|deepgram|groq)" && ls -lh auto_avsr/pretrained_models/*.pth

# RUN BATCH TEST (3 samples)
python -c "from pipeline import run_mvp; [print(f'{v}: {run_mvp(video_file=f\"samples/video/{v}\", audio_file=f\"samples/audio/{v[:-4]}.wav\")[\"final_transcript\"]}') for v in ['lwwz9s.mpg', 'bbaf2n.mpg', 'bgaa6n.mpg']]"

# CALCULATE WER
python calculate_metrics.py

# FULL DEMO
./demo.sh

# CLEAN LOGS
rm -rf logs/*

# BACKUP RESULTS
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz logs/ paper_figure*.png

# CHECK API KEYS
cat .env | grep -E "^(DEEPGRAM|GROQ)" | sed 's/=.*/=***/'

# MONITOR REAL-TIME
python main.py 2>&1 | tee output.log

# BENCHMARK PERFORMANCE
time python main.py

# EXTRACT FINAL TRANSCRIPT ONLY
python main.py 2>&1 | grep "^📄 Final Transcript:"

# VIEW AGREEMENT METRICS
grep -A 4 "AGREEMENT METRICS" $(ls -t logs/metrics_*.log | head -1)

# VIEW CORRECTIONS
grep -A 20 "CORRECTIONS APPLIED" $(ls -t logs/metrics_*.log | head -1)

# COUNT TOTAL RUNS
ls logs/results_*.json | wc -l

# SHOW CONFIDENCE TRENDS
for f in logs/confidence_*.log; do echo "$f:"; grep "Final:" $f; done

# LIST AVAILABLE SAMPLES
ls samples/video/*.mpg | head -20 | xargs -n 1 basename

# TEST SINGLE MODALITY
python -c "from pipeline import get_deepgram_confidence; print(get_deepgram_confidence('samples/audio/lwwz9s.wav')['transcript'])"

# QUICK PERFORMANCE SUMMARY
python -c "import json; r=json.load(open(max(__import__('glob').glob('logs/results_*.json'), key=__import__('os').path.getctime)))[-1]['results']; print(f\"Audio: {r['deepgram']['overall_confidence']:.3f} | Video: {r['avsr']['overall_confidence']:.3f} | Final: {r['groq']['confidence']:.3f}\")"

# DOCUMENTATION
cat COMMANDS.md | less
cat LOGGING_GUIDE.md | less

# HELP
./demo.sh --help 2>/dev/null || echo "Run: ./demo.sh for full demonstration"

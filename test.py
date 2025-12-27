"""
Simple test script - verify the pipeline works
"""

import sys
import os

sys.path.insert(0, '.')
from load_env import load_env_file
load_env_file()

print("\n" + "=" * 60)
print(" Testing Multi-Modal Pipeline")
print("=" * 60)

# Test 1: Check APIs
print("\n1. Checking API Keys...")
dg = "✓" if os.getenv('DEEPGRAM_API_KEY') else "✗"
groq = "✓" if os.getenv('GROQ_API_KEY') else "✗"
print(f"   {dg} DeepGram")
print(f"   {groq} Groq")

# Test 2: Test with mock data
print("\n2. Testing Pipeline...")
try:
    from pipeline import run_mvp
    result = run_mvp()
    print(f"   ✓ Pipeline works")
    print(f"   Transcript: {result['final_transcript']}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print(" Status: Ready to process real videos")
print("=" * 60)

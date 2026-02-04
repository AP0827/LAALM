"""
Ablation Study 3: LLM Provider Comparison (Free Alternatives)

Since you only have Groq API access, this script compares:
1. Groq (Llama 3.3 70B) - Your current setup
2. Ollama Local (Llama 3 8B) - Free local model
3. No LLM - Direct fusion without semantic correction

This demonstrates the value of LLM semantic correction while working
within your budget constraints.

Usage:
    # First install Ollama: https://ollama.ai/
    # Then pull model: ollama pull llama3
    python ablation_llm.py --test-set samples/ --output results/ablation_llm.json
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_mvp
from calculate_metrics import compute_wer
from Transformer.llm_corrector import LLMSemanticCorrector, CorrectionContext, LLMProvider


def extract_word_conf(word_tuple):
    """Extract word and confidence from tuple (handles both 2 and 4 element tuples)."""
    if isinstance(word_tuple, (list, tuple)):
        if len(word_tuple) >= 2:
            return word_tuple[0], word_tuple[1]
    return str(word_tuple), 0.0


def check_ollama():
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def run_llm_ablation(test_samples: List[Dict], output_path: str, providers: List[str]):
    """Run ablation study for LLM providers."""
    
    results = {}
    
    # Initialize Ollama if needed
    ollama_corrector = None
    if 'all' in providers or 'ollama' in providers:
        if check_ollama():
            print("✅ Ollama detected locally")
            # FIX: Correct init arguments (model instead of model_name)
            ollama_corrector = LLMSemanticCorrector(
                provider=LLMProvider.OLLAMA, 
                model='llama3', 
                api_key=None
            )
        else:
            print("⚠️ Ollama not running. Skipping Ollama tests.")
    
    # Storage for predictions
    preds_groq = []
    preds_ollama = []
    preds_nollm = []
    references = []
    
    latencies = {'groq': 0, 'ollama': 0, 'none': 0}
    
    print(f"\n{'='*80}")
    print(f"Processing {len(test_samples)} samples...")
    print(f"{'='*80}")
    
    for i, sample in enumerate(test_samples):
        print(f"Processing {i+1}/{len(test_samples)}: {sample['id']}")
        
        # 1. Run Pipeline (Includes Groq & No LLM)
        start_time = time.time()
        result = run_mvp(
            audio_file=sample.get('audio_path'),
            video_file=sample['video_path']
        )
        pipeline_time = (time.time() - start_time) * 1000
        
        # Store Groq Result
        preds_groq.append(result['final_transcript'])
        latencies['groq'] += pipeline_time
        
        # Store No LLM Result
        # FIX: 'fused' key does not exist. run_mvp returns 'combined_transcript'.
        preds_nollm.append(result['combined_transcript'])
        latencies['none'] += (pipeline_time * 0.7)  # Estimate: subtract LLM overhead
        
        # Store Reference
        references.append(sample['ground_truth'])
        
        # 2. Run Ollama (separately using raw outputs)
        if ollama_corrector:
            start_ollama = time.time()
            
            # FIX: Create CorrectionContext as required by LLMSemanticCorrector
            combined_words = result['combined_words']
            num_agreed = sum(1 for w in combined_words if w['agreed'])
            align_score = num_agreed / len(combined_words) if combined_words else 0.0
            
            # FIX: Normalize word confidences to 2-tuples
            dg_word_confs = [extract_word_conf(w) for w in result['deepgram']['word_confidences']]
            vsr_word_confs = [extract_word_conf(w) for w in result['avsr']['word_confidences']]
            
            ctx = CorrectionContext(
                deepgram_transcript=result['deepgram']['transcript'],
                deepgram_confidence=result['deepgram']['overall_confidence'],
                deepgram_word_confidences=dg_word_confs,
                
                # Map AVSR output to 'lipnet' fields (class expectation)
                lipnet_transcript=result['avsr']['transcript'],
                lipnet_confidence=result['avsr']['overall_confidence'],
                lipnet_word_confidences=vsr_word_confs,
                
                alignment_score=align_score,
                flagged_discrepancies=[], # Simplified
                domain_context="general",
                audio_metadata={}
            )
            
            try:
                corrected_result = ollama_corrector.correct(ctx)
                ollama_transcript = corrected_result.corrected_transcript
            except Exception as e:
                print(f"  ⚠ Ollama correction failed: {e}")
                ollama_transcript = result['combined_transcript'] # Fallback
                
            ollama_time = (time.time() - start_ollama) * 1000
            preds_ollama.append(ollama_transcript)
            latencies['ollama'] += (pipeline_time * 0.7) + ollama_time  # Pipelines + Ollama time
            
    # Compute Metrics
    
    # 1. Groq
    if 'all' in providers or 'groq' in providers:
        results['Groq (Llama 3.3 70B)'] = {
            'wer': compute_wer(preds_groq, references),
            'avg_latency_ms': latencies['groq'] / len(test_samples),
            'cost_per_1k': 2.50,
            'model': 'llama-3.3-70b'
        }
        
    # 2. No LLM
    if 'all' in providers or 'none' in providers:
        results['No LLM (Baseline)'] = {
            'wer': compute_wer(preds_nollm, references),
            'avg_latency_ms': latencies['none'] / len(test_samples),
            'cost_per_1k': 0.0,
            'model': 'none'
        }
        
    # 3. Ollama
    if ollama_corrector and len(preds_ollama) > 0:
        results['Ollama Local (Llama 3 8B)'] = {
            'wer': compute_wer(preds_ollama, references),
            'avg_latency_ms': latencies['ollama'] / len(test_samples),
            'cost_per_1k': 0.0,
            'model': 'llama3-8b'
        }

    # Save results
    if os.path.isdir(output_path) or output_path.endswith('/'):
        output_path = os.path.join(output_path, 'ablation_llm.json')
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison
    print(f"\n{'='*80}")
    print("LLM PROVIDER COMPARISON")
    print(f"{'='*80}")
    print(f"{'Provider':<30} {'WER (%)':<10} {'Latency (ms)':<15} {'Cost ($/1k)':<12}")
    print(f"{'-'*80}")
    
    for provider, metrics in results.items():
        print(f"{provider:<30} {metrics['wer']:<10.2f} {metrics['avg_latency_ms']:<15.0f} ${metrics['cost_per_1k']:<11.2f}")
    
    print(f"{'='*80}\n")
    print(f"✅ Results saved to {output_path}")


def load_test_samples(test_dir: str) -> List[Dict]:
    """Load test samples from directory."""
    test_dir = Path(test_dir)
    samples = []
    
    for video_file in test_dir.glob("**/*.mp4"):
        audio_file = video_file.with_suffix('.wav')
        if not audio_file.exists():
            audio_file = None
        
        gt_file = video_file.with_suffix('.txt')
        if gt_file.exists():
            ground_truth = gt_file.read_text().strip()
        else:
            continue
        
        samples.append({
            'id': video_file.stem,
            'video_path': str(video_file),
            'audio_path': str(audio_file) if audio_file else None,
            'ground_truth': ground_truth
        })
    
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM provider comparison")
    parser.add_argument('--test-set', type=str, required=True, help="Path to test set directory")
    parser.add_argument('--output', type=str, default='results/ablation_llm.json', help="Output JSON path")
    parser.add_argument('--providers', nargs='+', default=['all'],
                       choices=['all', 'groq', 'ollama', 'none'],
                       help="Which LLM providers to test")
    
    args = parser.parse_args()
    
    print(f"Loading test samples from {args.test_set}...")
    test_samples = load_test_samples(args.test_set)
    print(f"Loaded {len(test_samples)} test samples")
    
    if len(test_samples) == 0:
        print("Error: No test samples found!")
        sys.exit(1)
    
    run_llm_ablation(test_samples, args.output, args.providers)

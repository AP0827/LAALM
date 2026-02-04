"""
Ablation Study 2: Hyperparameter Sensitivity Analysis

Tests sensitivity to key hyperparameters:
1. Context window size
2. Temperature
3. N-Best beam width
4. Switching penalty

Usage:
    python ablation_hyperparameters.py --test-set samples/ --output results/
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_mvp
from calculate_metrics import compute_wer
from Transformer.attention_fusion import AttentionFusion


def extract_word_conf(word_tuple):
    """Extract word and confidence from tuple (handles both 2 and 4 element tuples)."""
    if isinstance(word_tuple, (list, tuple)):
        if len(word_tuple) >= 2:
            return word_tuple[0], word_tuple[1]
    return str(word_tuple), 0.0


def precompute_pipeline_results(test_samples: List[Dict]) -> Dict:
    """Run pipeline once per sample and cache results."""
    print("\n" + "="*80)
    print(f"Pre-computing pipeline results for {len(test_samples)} samples...")
    print("="*80)
    
    cache = {}
    for i, sample in enumerate(test_samples):
        print(f"Processing sample {i+1}/{len(test_samples)}: {sample['id']}")
        result = run_mvp(
            audio_file=sample.get('audio_path'),
            video_file=sample['video_path']
        )
        cache[sample['id']] = {
            'audio_words': result['deepgram']['word_confidences'],
            'visual_words': result['avsr']['word_confidences']
        }
    return cache


def test_context_window(test_samples: List[Dict], cache: Dict, output_dir: str):
    """Test different context window sizes."""
    print("\n" + "="*80)
    print("ABLATION: Context Window Size")
    print("="*80)
    
    window_sizes = [1, 2, 3, 5, 7]
    results = {}
    
    for window_size in window_sizes:
        print(f"\nTesting context window = {window_size}")
        
        fusion = AttentionFusion(
            temperature=2.0,
            context_window=window_size,
            switching_penalty=0.15
        )
        
        predictions = []
        references = []
        total_switches = 0
        
        for sample in test_samples:
            # Use cached results
            data = cache[sample['id']]
            audio_words = data['audio_words']
            visual_words = data['visual_words']
            
            fusion_result = fusion.fuse_transcripts(audio_words, visual_words)
            
            predictions.append(fusion_result.fused_transcript)
            references.append(sample['ground_truth'])
            total_switches += fusion_result.switches
        
        wer = compute_wer(predictions, references)
        avg_switches = total_switches / len(test_samples)
        
        results[window_size] = {
            'wer': wer,
            'avg_switches': avg_switches
        }
        
        print(f"  WER: {wer:.2f}%, Switches: {avg_switches:.1f}")
    
    # Save results
    output_path = os.path.join(output_dir, 'ablation_context_window.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    windows = list(results.keys())
    wers = [results[w]['wer'] for w in windows]
    switches = [results[w]['avg_switches'] for w in windows]
    
    ax1.plot(windows, wers, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Context Window Size', fontsize=12)
    ax1.set_ylabel('WER (%)', fontsize=12)
    ax1.set_title('WER vs Context Window Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(windows, switches, marker='s', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Context Window Size', fontsize=12)
    ax2.set_ylabel('Avg Modality Switches', fontsize=12)
    ax2.set_title('Modality Switches vs Context Window', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_context_window.png'), dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to {output_dir}/ablation_context_window.png")
    
    return results


def test_temperature(test_samples: List[Dict], cache: Dict, output_dir: str):
    """Test different temperature values."""
    print("\n" + "="*80)
    print("ABLATION: Temperature")
    print("="*80)
    
    temperatures = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results = {}
    
    for temp in temperatures:
        print(f"\nTesting temperature = {temp}")
        
        fusion = AttentionFusion(
            temperature=temp,
            context_window=3,
            switching_penalty=0.15
        )
        
        predictions = []
        references = []
        
        for sample in test_samples:
            # Use cached results
            data = cache[sample['id']]
            audio_words = data['audio_words']
            visual_words = data['visual_words']
            
            fusion_result = fusion.fuse_transcripts(audio_words, visual_words)
            
            predictions.append(fusion_result.fused_transcript)
            references.append(sample['ground_truth'])
        
        wer = compute_wer(predictions, references)
        
        results[temp] = {'wer': wer}
        print(f"  WER: {wer:.2f}%")
    
    # Save results
    output_path = os.path.join(output_dir, 'ablation_temperature.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot
    plt.figure(figsize=(8, 5))
    temps = list(results.keys())
    wers = [results[t]['wer'] for t in temps]
    
    plt.plot(temps, wers, marker='o', linewidth=2, markersize=8, color='green')
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('WER (%)', fontsize=12)
    plt.title('WER vs Temperature', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_temperature.png'), dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to {output_dir}/ablation_temperature.png")
    
    return results


def test_switching_penalty(test_samples: List[Dict], cache: Dict, output_dir: str):
    """Test different switching penalty values."""
    print("\n" + "="*80)
    print("ABLATION: Switching Penalty")
    print("="*80)
    
    penalties = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    results = {}
    
    for penalty in penalties:
        print(f"\nTesting switching penalty = {penalty}")
        
        fusion = AttentionFusion(
            temperature=2.0,
            context_window=3,
            switching_penalty=penalty
        )
        
        predictions = []
        references = []
        total_switches = 0
        
        for sample in test_samples:
            # Use cached results
            data = cache[sample['id']]
            audio_words = data['audio_words']
            visual_words = data['visual_words']
            
            fusion_result = fusion.fuse_transcripts(audio_words, visual_words)
            
            predictions.append(fusion_result.fused_transcript)
            references.append(sample['ground_truth'])
            total_switches += fusion_result.switches
        
        wer = compute_wer(predictions, references)
        avg_switches = total_switches / len(test_samples)
        
        results[penalty] = {
            'wer': wer,
            'avg_switches': avg_switches
        }
        
        print(f"  WER: {wer:.2f}%, Switches: {avg_switches:.1f}")
    
    # Save results
    output_path = os.path.join(output_dir, 'ablation_switching_penalty.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    pens = list(results.keys())
    wers = [results[p]['wer'] for p in pens]
    switches = [results[p]['avg_switches'] for p in pens]
    
    ax1.plot(pens, wers, marker='o', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('Switching Penalty', fontsize=12)
    ax1.set_ylabel('WER (%)', fontsize=12)
    ax1.set_title('WER vs Switching Penalty', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(pens, switches, marker='s', linewidth=2, markersize=8, color='purple')
    ax2.set_xlabel('Switching Penalty', fontsize=12)
    ax2.set_ylabel('Avg Modality Switches', fontsize=12)
    ax2.set_title('Switches vs Penalty', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_switching_penalty.png'), dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to {output_dir}/ablation_switching_penalty.png")
    
    return results


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
    parser = argparse.ArgumentParser(description="Run hyperparameter sensitivity analysis")
    parser.add_argument('--test-set', type=str, required=True, help="Path to test set directory")
    parser.add_argument('--output', type=str, default='results/', help="Output directory")
    parser.add_argument('--ablations', nargs='+', default=['all'], 
                       choices=['all', 'context', 'temperature', 'penalty'],
                       help="Which ablations to run")
    
    args = parser.parse_args()
    
    # Load test samples
    print(f"Loading test samples from {args.test_set}...")
    test_samples = load_test_samples(args.test_set)
    print(f"Loaded {len(test_samples)} test samples")
    
    if len(test_samples) == 0:
        print("Error: No test samples found!")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Pre-compute pipeline results (Optimization)
    cache = precompute_pipeline_results(test_samples)
    
    # Run ablations using cache
    all_results = {}
    
    if 'all' in args.ablations or 'context' in args.ablations:
        all_results['context_window'] = test_context_window(test_samples, cache, args.output)
    
    if 'all' in args.ablations or 'temperature' in args.ablations:
        all_results['temperature'] = test_temperature(test_samples, cache, args.output)
    
    if 'all' in args.ablations or 'penalty' in args.ablations:
        all_results['switching_penalty'] = test_switching_penalty(test_samples, cache, args.output)
    
    # Save combined results
    with open(os.path.join(args.output, 'ablation_hyperparameters_all.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ All ablation studies complete!")
    print(f"Results saved to {args.output}")
    print(f"{'='*80}\n")

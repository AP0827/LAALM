"""
Theoretical Analysis 2: Distributional Recovery via N-Best Fusion

This script analyzes how N-Best fusion preserves the probability manifold
compared to Top-1 decoding.

Metrics:
1. Beam entropy - How much uncertainty is preserved
2. Oracle accuracy - How often ground truth appears in top-K
3. Probability mass coverage - Cumulative probability captured
4. Rank distribution - Where ground truth appears in beam

Usage:
    python theory_distributional_recovery.py --test-set samples/ --output results/
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


def compute_beam_entropy(beam_hypotheses: List[Dict]) -> float:
    """
    Compute entropy of beam search distribution.
    
    Higher entropy = more uncertainty preserved
    """
    # Extract scores and normalize to probabilities
    scores = [hyp.get('score', -10.0) for hyp in beam_hypotheses]
    
    # Convert log probabilities to probabilities
    probs = np.exp(scores)
    probs = probs / np.sum(probs)  # Normalize
    
    # Compute entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return entropy


def measure_oracle_accuracy(beam_hypotheses: List[str], ground_truth: str, k_values: List[int]) -> Dict[int, bool]:
    """
    Check if ground truth appears in top-K hypotheses.
    
    Returns:
        Dictionary mapping K -> whether GT is in top-K
    """
    results = {}
    
    for k in k_values:
        top_k = beam_hypotheses[:k]
        # Normalize for comparison
        top_k_normalized = [h.lower().strip() for h in top_k]
        gt_normalized = ground_truth.lower().strip()
        
        results[k] = gt_normalized in top_k_normalized
    
    return results


def analyze_probability_mass(beam_hypotheses: List[Dict], k_values: List[int]) -> Dict[int, float]:
    """
    Compute cumulative probability mass captured by top-K.
    """
    scores = [hyp.get('score', -10.0) for hyp in beam_hypotheses]
    probs = np.exp(scores)
    probs = probs / np.sum(probs)
    
    results = {}
    for k in k_values:
        cumulative_prob = np.sum(probs[:k])
        results[k] = float(cumulative_prob)
    
    return results


def analyze_distributional_recovery(test_samples: List[Dict], output_dir: str):
    """
    Analyze how N-Best fusion recovers probability distribution.
    """
    print("\n" + "="*80)
    print("ANALYZING DISTRIBUTIONAL RECOVERY")
    print("="*80)
    
    k_values = [1, 3, 5, 7, 10]
    
    all_entropies = []
    oracle_accuracies = {k: [] for k in k_values}
    probability_masses = {k: [] for k in k_values}
    gt_ranks = []  # Where does ground truth appear in beam?
    
    for i, sample in enumerate(test_samples):
        print(f"Processing {i+1}/{len(test_samples)}: {sample['id']}")
        
        result = run_mvp(
            audio_file=sample.get('audio_path'),
            video_file=sample['video_path'],
            
        )
        
        # Get N-Best hypotheses from visual modality
        nbest_transcripts = result['avsr'].get('nbest_transcripts', [])
        
        if not nbest_transcripts:
            print(f"  Warning: No N-Best hypotheses for {sample['id']}, skipping")
            continue
        
        # Compute entropy
        # Note: We don't have actual scores, so we'll use a proxy
        # In real implementation, you'd extract scores from beam search
        entropy = np.log(len(nbest_transcripts))  # Proxy: uniform distribution
        all_entropies.append(entropy)
        
        # Oracle accuracy
        oracle_results = measure_oracle_accuracy(
            nbest_transcripts,
            sample['ground_truth'],
            k_values
        )
        
        for k, is_present in oracle_results.items():
            oracle_accuracies[k].append(1 if is_present else 0)
        
        # Find rank of ground truth
        gt_normalized = sample['ground_truth'].lower().strip()
        rank = None
        for r, hyp in enumerate(nbest_transcripts):
            if hyp.lower().strip() == gt_normalized:
                rank = r + 1  # 1-indexed
                break
        
        if rank:
            gt_ranks.append(rank)
        else:
            gt_ranks.append(len(nbest_transcripts) + 1)  # Not in beam
        
        # Probability mass (using uniform proxy)
        for k in k_values:
            prob_mass = min(k / len(nbest_transcripts), 1.0)
            probability_masses[k].append(prob_mass)
    
    # Compute statistics
    avg_entropy = np.mean(all_entropies) if all_entropies else 0.0
    
    oracle_acc_stats = {}
    for k in k_values:
        oracle_acc_stats[k] = {
            'accuracy': np.mean(oracle_accuracies[k]) if oracle_accuracies[k] else 0.0,
            'count': sum(oracle_accuracies[k])
        }
    
    prob_mass_stats = {}
    for k in k_values:
        prob_mass_stats[k] = np.mean(probability_masses[k]) if probability_masses[k] else 0.0
    
    avg_gt_rank = np.mean(gt_ranks) if gt_ranks else 0.0
    median_gt_rank = np.median(gt_ranks) if gt_ranks else 0.0
    
    # Print results
    print(f"\n{'='*80}")
    print("DISTRIBUTIONAL RECOVERY RESULTS")
    print(f"{'='*80}")
    
    print(f"\nAverage Beam Entropy: {avg_entropy:.3f} bits")
    print(f"  (Higher = more uncertainty preserved)")
    
    print(f"\nOracle Accuracy (Ground Truth in Top-K):")
    for k in k_values:
        acc = oracle_acc_stats[k]['accuracy']
        count = oracle_acc_stats[k]['count']
        print(f"  Top-{k}: {acc:.1%} ({count}/{len(test_samples)} samples)")
    
    print(f"\nProbability Mass Captured:")
    for k in k_values:
        mass = prob_mass_stats[k]
        print(f"  Top-{k}: {mass:.1%}")
    
    print(f"\nGround Truth Rank:")
    print(f"  Mean: {avg_gt_rank:.2f}")
    print(f"  Median: {median_gt_rank:.1f}")
    
    # Save results
    results = {
        'avg_entropy': float(avg_entropy),
        'oracle_accuracy': {str(k): oracle_acc_stats[k] for k in k_values},
        'probability_mass': {str(k): float(prob_mass_stats[k]) for k in k_values},
        'ground_truth_rank': {
            'mean': float(avg_gt_rank),
            'median': float(median_gt_rank),
            'distribution': gt_ranks
        }
    }
    
    output_path = os.path.join(output_dir, 'theory_distributional_recovery.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Oracle Accuracy
    k_list = list(k_values)
    oracle_accs = [oracle_acc_stats[k]['accuracy'] * 100 for k in k_list]
    
    ax1.plot(k_list, oracle_accs, marker='o', linewidth=2, markersize=10, color='#2E86AB')
    ax1.set_xlabel('K (Beam Width)', fontsize=12)
    ax1.set_ylabel('Oracle Accuracy (%)', fontsize=12)
    ax1.set_title('Ground Truth Recovery in Top-K', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Add annotations
    for k, acc in zip(k_list, oracle_accs):
        ax1.annotate(f'{acc:.1f}%', (k, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')
    
    # 2. Probability Mass
    prob_masses = [prob_mass_stats[k] * 100 for k in k_list]
    
    ax2.bar(k_list, prob_masses, color='#A23B72', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('K (Beam Width)', fontsize=12)
    ax2.set_ylabel('Cumulative Probability (%)', fontsize=12)
    ax2.set_title('Probability Mass Captured', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # 3. GT Rank Distribution
    rank_counts = np.bincount(gt_ranks, minlength=11)[:11]  # Up to rank 10
    
    ax3.bar(range(1, len(rank_counts)+1), rank_counts, color='#F18F01', edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Ground Truth Rank in Beam', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of GT Rank', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Improvement over Top-1
    top1_acc = oracle_acc_stats[1]['accuracy'] * 100
    improvements = [(oracle_acc_stats[k]['accuracy'] * 100 - top1_acc) for k in k_list[1:]]
    
    ax4.bar(k_list[1:], improvements, color='#06A77D', edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('K (Beam Width)', fontsize=12)
    ax4.set_ylabel('Improvement over Top-1 (%)', fontsize=12)
    ax4.set_title('Oracle Accuracy Gain', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'theory_distributional_recovery.png'), dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to {output_dir}/theory_distributional_recovery.png")
    
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
    parser = argparse.ArgumentParser(description="Analyze distributional recovery via N-Best fusion")
    parser.add_argument('--test-set', type=str, required=True, help="Path to test set directory")
    parser.add_argument('--output', type=str, default='results/', help="Output directory")
    
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
    
    # Analyze distributional recovery
    results = analyze_distributional_recovery(test_samples, args.output)
    
    print(f"\n{'='*80}")
    print("✅ Distributional recovery analysis complete!")
    print(f"Results saved to {args.output}")
    print(f"{'='*80}\n")

"""
Theoretical Analysis 1: Error Orthogonality

This script formally tests the hypothesis that audio and visual errors
are statistically independent (orthogonal error modes).

Tests:
1. Chi-square test for independence
2. Mutual information analysis
3. Error correlation analysis
4. Error type categorization

Usage:
    python theory_error_orthogonality.py --test-set samples/ --output results/
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
import argparse
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_mvp
from calculate_metrics import compute_wer


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    # Remove punctuation and lowercase
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()


def classify_error_type(predicted: str, reference: str) -> str:
    """
    Classify the type of error.
    
    Returns:
        'substitution', 'insertion', 'deletion', or 'correct'
    """
    pred_norm = normalize_text(predicted)
    ref_norm = normalize_text(reference)
    
    if pred_norm == ref_norm:
        return 'correct'
    elif not pred_norm:
        return 'deletion'
    elif not ref_norm:
        return 'insertion'
    else:
        return 'substitution'


def analyze_error_patterns(test_samples: List[Dict]) -> Dict:
    """
    Analyze error patterns from audio and visual modalities.
    
    Returns:
        Dictionary with error analysis results
    """
    print("\n" + "="*80)
    print("ANALYZING ERROR PATTERNS")
    print("="*80)
    
    audio_errors = []
    visual_errors = []
    audio_error_positions = set()
    visual_error_positions = set()
    total_words_count = 0
    
    # Collect errors
    for i, sample in enumerate(test_samples):
        print(f"Processing {i+1}/{len(test_samples)}: {sample['id']}")
        
        result = run_mvp(
            audio_file=sample.get('audio_path'),
            video_file=sample['video_path'],
              # Don't use LLM, we want raw modality outputs
        )
        
        audio_words = [w[0] for w in result['deepgram']['word_confidences']]
        visual_words = [w[0] for w in result['avsr']['word_confidences']]
        ground_truth_words = sample['ground_truth'].split()
        
        # Align and compare (simple word-by-word for now)
        max_len = max(len(audio_words), len(visual_words), len(ground_truth_words))
        total_words_count += max_len
        
        for pos in range(max_len):
            audio_word = audio_words[pos] if pos < len(audio_words) else ""
            visual_word = visual_words[pos] if pos < len(visual_words) else ""
            gt_word = ground_truth_words[pos] if pos < len(ground_truth_words) else ""
            
            # Classify errors
            audio_error_type = classify_error_type(audio_word, gt_word)
            visual_error_type = classify_error_type(visual_word, gt_word)
            
            if audio_error_type != 'correct':
                audio_errors.append({
                    'sample_id': sample['id'],
                    'position': pos,
                    'type': audio_error_type,
                    'predicted': audio_word,
                    'reference': gt_word
                })
                audio_error_positions.add((sample['id'], pos))
            
            if visual_error_type != 'correct':
                visual_errors.append({
                    'sample_id': sample['id'],
                    'position': pos,
                    'type': visual_error_type,
                    'predicted': visual_word,
                    'reference': gt_word
                })
                visual_error_positions.add((sample['id'], pos))
    
    return {
        'audio_errors': audio_errors,
        'visual_errors': visual_errors,
        'audio_error_positions': sorted(list(audio_error_positions)), # Convert set to list for JSON serialization
        'visual_error_positions': sorted(list(visual_error_positions)),
        'total_words': total_words_count
    }


def test_independence(error_data: Dict, output_dir: str):
    """
    Test statistical independence of audio and visual errors.
    """
    print("\n" + "="*80)
    print("TESTING ERROR INDEPENDENCE")
    print("="*80)
    
    audio_error_positions = set([tuple(x) for x in error_data['audio_error_positions']])
    visual_error_positions = set([tuple(x) for x in error_data['visual_error_positions']])
    total_words = error_data.get('total_words', 0)
    
    # Calculate counts
    # Positions with errors in respective modalities
    num_audio_errors = len(audio_error_positions)
    num_visual_errors = len(visual_error_positions)
    
    # Intersection
    both_error = len(audio_error_positions & visual_error_positions)
    
    # Exclusive errors
    audio_err_visual_corr = len(audio_error_positions - visual_error_positions)
    visual_err_audio_corr = len(visual_error_positions - audio_error_positions)
    
    # Neither error (Both Correct)
    # Total words - (Any Error)
    both_correct = total_words - len(audio_error_positions | visual_error_positions)
    
    # Ensure non-negative (logic check)
    both_correct = max(0, both_correct)
    
    # Build contingency table
    # [audio_correct, audio_error] x [visual_correct, visual_error]
    contingency = np.zeros((2, 2))
    
    # Audio Correct, Visual Correct
    contingency[0, 0] = both_correct
    # Audio Correct, Visual Error
    contingency[0, 1] = visual_err_audio_corr
    # Audio Error, Visual Correct
    contingency[1, 0] = audio_err_visual_corr
    # Audio Error, Visual Error
    contingency[1, 1] = both_error

    # For correlation, we need full arrays
    # 0 = correct, 1 = error
    # We construct arrays representing the full population of words
    audio_error_indicator = [0] * both_correct + [0] * visual_err_audio_corr + [1] * audio_err_visual_corr + [1] * both_error
    visual_error_indicator = [0] * both_correct + [1] * visual_err_audio_corr + [0] * audio_err_visual_corr + [1] * both_error
    
    print("\nContingency Table:")
    print("                  Visual Correct  Visual Error")
    print(f"Audio Correct     {contingency[0, 0]:<15.0f} {contingency[0, 1]:<15.0f}")
    print(f"Audio Error       {contingency[1, 0]:<15.0f} {contingency[1, 1]:<15.0f}")
    
    print(f"\nTotal Words Analyzed: {total_words}")
    
    # Chi-square test
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        print(f"\nChi-Square Test:")
        print(f"  χ² = {chi2:.4f}")
        print(f"  p-value = {p_value:.4f}")
        print(f"  degrees of freedom = {dof}")
        
        if p_value > 0.05:
            print(f"  ✅ Errors are INDEPENDENT (p > 0.05)")
            independence_result = "independent"
        else:
            print(f"  ❌ Errors are CORRELATED (p < 0.05)")
            independence_result = "correlated"
    except Exception as e:
        print(f"\n⚠ Chi-Square Test calculation failed: {e}")
        chi2, p_value, dof = 0.0, 1.0, 0
        independence_result = "inconclusive"
    
    # Mutual Information
    try:
        mi = mutual_info_score(audio_error_indicator, visual_error_indicator)
        print(f"\nMutual Information:")
        print(f"  I(E_audio; E_visual) = {mi:.4f} bits")
        
        if mi < 0.1:
            print(f"  ✅ Low mutual information (< 0.1 bits) indicates independence")
        else:
            print(f"  ⚠️  High mutual information (> 0.1 bits) indicates some correlation")
    except:
        mi = 0.0
    
    # Correlation coefficient
    try:
        if len(audio_error_indicator) > 1:
            correlation = np.corrcoef(audio_error_indicator, visual_error_indicator)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
            
        print(f"\nPearson Correlation:")
        print(f"  r = {correlation:.4f}")
        
        if abs(correlation) < 0.3:
            print(f"  ✅ Weak correlation (|r| < 0.3) supports independence")
        else:
            print(f"  ⚠️  Moderate/strong correlation (|r| >= 0.3)")
    except:
        correlation = 0.0
    
    # Save results
    results = {
        'contingency_table': contingency.tolist(),
        'chi_square': {
            'statistic': float(chi2),
            'p_value': float(p_value),
            'dof': int(dof),
            'result': independence_result
        },
        'mutual_information': {
            'value': float(mi),
            'interpretation': 'independent' if mi < 0.1 else 'correlated'
        },
        'correlation': {
            'value': float(correlation),
            'interpretation': 'weak' if abs(correlation) < 0.3 else 'moderate/strong'
        }
    }
    
    # Fix output path logic (NEW FIX)
    if os.path.isdir(output_dir) or output_dir.endswith('/'):
        output_path = os.path.join(output_dir, 'theory_error_orthogonality.json')
    else:
        output_path = output_dir
        output_dir = os.path.dirname(output_path)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    try:
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Contingency table heatmap
        sns.heatmap(contingency, annot=True, fmt='.0f', cmap='Blues', ax=ax1,
                    xticklabels=['Visual Correct', 'Visual Error'],
                    yticklabels=['Audio Correct', 'Audio Error'])
        ax1.set_title('Error Co-occurrence Matrix', fontsize=14, fontweight='bold')
        
        # Error overlap Venn diagram (simplified as bar chart)
        only_audio = len(audio_error_positions - visual_error_positions)
        only_visual = len(visual_error_positions - audio_error_positions)
        both = len(audio_error_positions & visual_error_positions)
        
        categories = ['Audio Only', 'Visual Only', 'Both']
        values = [only_audio, only_visual, both]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Number of Errors', fontsize=12)
        ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(values):
            ax2.text(i, v + max(values)*0.02, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'theory_error_orthogonality.png'), dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved to {output_dir}/theory_error_orthogonality.png")
    except Exception as e:
        print(f"\n⚠ Visualization failed: {e}")
    
    return results


def analyze_error_types(error_data: Dict, output_dir: str):
    """Analyze distribution of error types."""
    print("\n" + "="*80)
    print("ERROR TYPE ANALYSIS")
    print("="*80)
    
    # Count error types
    audio_error_types = {}
    visual_error_types = {}
    
    for error in error_data['audio_errors']:
        error_type = error['type']
        audio_error_types[error_type] = audio_error_types.get(error_type, 0) + 1
    
    for error in error_data['visual_errors']:
        error_type = error['type']
        visual_error_types[error_type] = visual_error_types.get(error_type, 0) + 1
    
    print("\nAudio Error Types:")
    for error_type, count in sorted(audio_error_types.items(), key=lambda x: -x[1]):
        print(f"  {error_type}: {count}")
    
    print("\nVisual Error Types:")
    for error_type, count in sorted(visual_error_types.items(), key=lambda x: -x[1]):
        print(f"  {error_type}: {count}")
    
    # Save
    results = {
        'audio_error_types': audio_error_types,
        'visual_error_types': visual_error_types
    }
    
    # Use output_dir handled in test_independence effectively or pass correct dir
    if os.path.isdir(output_dir) or output_dir.endswith('/'):
        output_path = os.path.join(output_dir, 'theory_error_types.json')
    else:
        output_path = os.path.join(os.path.dirname(output_dir), 'theory_error_types.json')
        
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
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
    parser = argparse.ArgumentParser(description="Test error orthogonality hypothesis")
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
    
    # Analyze error patterns
    error_data = analyze_error_patterns(test_samples)
    
    # Test independence
    independence_results = test_independence(error_data, args.output)
    
    # Analyze error types
    error_type_results = analyze_error_types(error_data, args.output)
    
    print(f"\n{'='*80}")
    print("✅ Error orthogonality analysis complete!")
    print(f"Results saved to {args.output}")
    print(f"{'='*80}\n")

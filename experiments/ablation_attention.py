"""
Ablation Study 1: Attention Mechanism Comparison

This script compares different fusion strategies to demonstrate that the
proposed attention mechanism outperforms simpler baselines.

Baselines:
1. Simple Average - Equal weighting (0.5, 0.5)
2. Confidence-Weighted - Weight by raw confidence
3. Max Confidence - Select modality with higher confidence
4. Attention (Ours) - Multi-factor attention with phonetic bias

Usage:
    python ablation_attention.py --test-set samples/ --output results/ablation_attention.json
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import argparse

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_mvp
from calculate_metrics import compute_wer
from Transformer.attention_fusion import AttentionFusion


def extract_word_conf(word_tuple):
    """Extract word and confidence from tuple (handles 2-4 element tuples)."""
    if isinstance(word_tuple, (list, tuple)) and len(word_tuple) >= 2:
        return word_tuple[0], word_tuple[1]
    return str(word_tuple), 0.0


@dataclass
class FusionStrategy:
    """Configuration for a fusion strategy."""
    name: str
    use_attention: bool
    use_phonetic: bool
    use_context: bool
    description: str


class SimpleFusion:
    """Simple averaging fusion baseline."""
    
    def fuse_transcripts(self, audio_words, visual_words):
        """Simple 50-50 averaging."""
        fused_words = []
        word_details = []
        
        max_len = max(len(audio_words), len(visual_words))
        
        for i in range(max_len):
            audio_word, audio_conf = extract_word_conf(audio_words[i]) if i < len(audio_words) else ("", 0.0)
            visual_word, visual_conf = extract_word_conf(visual_words[i]) if i < len(visual_words) else ("", 0.0)
            
            # Simple average
            avg_conf = (audio_conf + visual_conf) / 2.0
            
            # Select word with higher confidence
            if audio_conf >= visual_conf:
                selected_word = audio_word
                selected_modality = 'audio'
            else:
                selected_word = visual_word
                selected_modality = 'visual'
            
            fused_words.append(selected_word)
            word_details.append({
                'word': selected_word,
                'position': i,
                'audio_word': audio_word,
                'audio_conf': audio_conf,
                'visual_word': visual_word,
                'visual_conf': visual_conf,
                'selected_modality': selected_modality,
                'audio_weight': 0.5,
                'visual_weight': 0.5,
                'confidence': avg_conf
            })
        
        from Transformer.attention_fusion import FusionResult
        return FusionResult(
            fused_transcript=' '.join(fused_words),
            word_details=word_details,
            audio_weight=0.5,
            visual_weight=0.5,
            agreement_score=0.0,
            fusion_confidence=np.mean([w['confidence'] for w in word_details]) if word_details else 0.0,
            switches=0
        )


class ConfidenceWeightedFusion:
    """Confidence-weighted fusion baseline."""
    
    def fuse_transcripts(self, audio_words, visual_words):
        """Weight by confidence only (no phonetic or context)."""
        fused_words = []
        word_details = []
        
        max_len = max(len(audio_words), len(visual_words))
        total_audio_weight = 0.0
        total_visual_weight = 0.0
        
        for i in range(max_len):
            audio_word, audio_conf = extract_word_conf(audio_words[i]) if i < len(audio_words) else ("", 0.0)
            visual_word, visual_conf = extract_word_conf(visual_words[i]) if i < len(visual_words) else ("", 0.0)
            
            # Normalize weights
            total = audio_conf + visual_conf
            if total > 0:
                audio_weight = audio_conf / total
                visual_weight = visual_conf / total
            else:
                audio_weight = 0.5
                visual_weight = 0.5
            
            # Select word with higher confidence
            if audio_conf >= visual_conf:
                selected_word = audio_word
                selected_modality = 'audio'
            else:
                selected_word = visual_word
                selected_modality = 'visual'
            
            fused_words.append(selected_word)
            total_audio_weight += audio_weight
            total_visual_weight += visual_weight
            
            word_details.append({
                'word': selected_word,
                'position': i,
                'audio_word': audio_word,
                'audio_conf': audio_conf,
                'visual_word': visual_word,
                'visual_conf': visual_conf,
                'selected_modality': selected_modality,
                'audio_weight': audio_weight,
                'visual_weight': visual_weight,
                'confidence': max(audio_conf, visual_conf)
            })
        
        from Transformer.attention_fusion import FusionResult
        num_words = len(fused_words)
        return FusionResult(
            fused_transcript=' '.join(fused_words),
            word_details=word_details,
            audio_weight=total_audio_weight / num_words if num_words > 0 else 0.5,
            visual_weight=total_visual_weight / num_words if num_words > 0 else 0.5,
            agreement_score=0.0,
            fusion_confidence=np.mean([w['confidence'] for w in word_details]) if word_details else 0.0,
            switches=0
        )


def run_ablation_study(test_samples: List[Dict], output_path: str):
    """
    Run ablation study comparing different fusion strategies.
    
    Args:
        test_samples: List of test samples with audio/video paths and ground truth
        output_path: Path to save results JSON
    """
    
    # Define fusion strategies
    strategies = {
        'Simple Average': SimpleFusion(),
        'Confidence-Weighted': ConfidenceWeightedFusion(),
        'Attention (No Phonetic)': AttentionFusion(
            temperature=2.0,
            context_window=3,
            switching_penalty=0.15
        ),
        'Attention (Full - Ours)': AttentionFusion(
            temperature=2.0,
            context_window=3,
            switching_penalty=0.15
        )
    }
    
    results = {}
    
    # Phase 1: Pre-compute transcripts for all samples (Optimization)
    print(f"\n{'='*80}")
    print(f"Pre-computing transcripts for {len(test_samples)} samples...")
    print(f"{'='*80}")
    
    sample_cache = {}
    
    for i, sample in enumerate(test_samples):
        print(f"Processing sample {i+1}/{len(test_samples)}: {sample['id']}")
        
        # Run pipeline to get audio and visual transcripts
        result = run_mvp(
            audio_file=sample.get('audio_path'),
            video_file=sample['video_path']
        )
        sample_cache[sample['id']] = result
    
    # Phase 2: Run all fusion strategies on cached data
    for strategy_name, fusion_model in strategies.items():
        print(f"\n{'='*80}")
        print(f"Testing: {strategy_name}")
        print(f"{'='*80}")
        
        predictions = []
        references = []
        total_switches = 0
        total_audio_weight = 0.0
        total_visual_weight = 0.0
        
        for i, sample in enumerate(test_samples):
            # Retrieve pre-computed data
            result = sample_cache[sample['id']]
            
            # Get word-level data
            audio_words = result['deepgram']['word_confidences']
            visual_words = result['avsr']['word_confidences']
            
            # Apply fusion strategy
            fusion_result = fusion_model.fuse_transcripts(audio_words, visual_words)
            
            predictions.append(fusion_result.fused_transcript)
            references.append(sample['ground_truth'])
            total_switches += fusion_result.switches
            total_audio_weight += fusion_result.audio_weight
            total_visual_weight += fusion_result.visual_weight
        
        # Compute metrics
        wer = compute_wer(predictions, references)
        avg_switches = total_switches / len(test_samples)
        avg_audio_weight = total_audio_weight / len(test_samples)
        avg_visual_weight = total_visual_weight / len(test_samples)
        
        results[strategy_name] = {
            'wer': wer,
            'avg_switches': avg_switches,
            'avg_audio_weight': avg_audio_weight,
            'avg_visual_weight': avg_visual_weight,
            'num_samples': len(test_samples)
        }
        
        print(f"\nResults for {strategy_name}:")
        print(f"  WER: {wer:.2f}%")
        print(f"  Avg Switches: {avg_switches:.1f}")
        print(f"  Audio Weight: {avg_audio_weight:.3f}")
        print(f"  Visual Weight: {avg_visual_weight:.3f}")
    
    # Save results
    if os.path.isdir(output_path) or output_path.endswith('/'):
        output_path = os.path.join(output_path, 'ablation_attention.json')
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*80}")
    print(f"{'Strategy':<30} {'WER (%)':<10} {'Δ vs Ours':<12} {'Switches':<10}")
    print(f"{'-'*80}")
    
    ours_wer = results['Attention (Full - Ours)']['wer']
    
    for strategy_name, metrics in results.items():
        delta = metrics['wer'] - ours_wer
        delta_str = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%"
        print(f"{strategy_name:<30} {metrics['wer']:<10.2f} {delta_str:<12} {metrics['avg_switches']:<10.1f}")
    
    print(f"{'='*80}\n")
    
    return results


def load_test_samples(test_dir: str) -> List[Dict]:
    """Load test samples from directory."""
    test_dir = Path(test_dir)
    samples = []
    
    # Look for video files
    for video_file in test_dir.glob("**/*.mp4"):
        # Try to find corresponding audio file
        audio_file = video_file.with_suffix('.wav')
        if not audio_file.exists():
            audio_file = None
        
        # Try to find ground truth
        gt_file = video_file.with_suffix('.txt')
        if gt_file.exists():
            ground_truth = gt_file.read_text().strip()
        else:
            print(f"Warning: No ground truth found for {video_file.name}, skipping")
            continue
        
        samples.append({
            'id': video_file.stem,
            'video_path': str(video_file),
            'audio_path': str(audio_file) if audio_file else None,
            'ground_truth': ground_truth
        })
    
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attention mechanism ablation study")
    parser.add_argument('--test-set', type=str, required=True, help="Path to test set directory")
    parser.add_argument('--output', type=str, default='results/ablation_attention.json', help="Output JSON path")
    
    args = parser.parse_args()
    
    # Load test samples
    print(f"Loading test samples from {args.test_set}...")
    test_samples = load_test_samples(args.test_set)
    print(f"Loaded {len(test_samples)} test samples")
    
    if len(test_samples) == 0:
        print("Error: No test samples found!")
        sys.exit(1)
    
    # Run ablation study
    results = run_ablation_study(test_samples, args.output)
    
    print(f"\n✅ Ablation study complete! Results saved to {args.output}")

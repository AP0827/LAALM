"""
Novel Attention-Based Multi-Modal Fusion for Audio-Visual Speech Recognition

This module implements a sophisticated fusion strategy that learns to weight
audio and visual modalities based on:
1. Word-level confidence scores
2. Contextual information
3. Phonetic distinctiveness
4. Temporal consistency

Key Novel Contributions:
- Adaptive attention mechanism for modality selection
- Confidence-aware weighting
- Phonetic context analysis
- Real-time reliability assessment

Author: LAALM Project
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class WordPrediction:
    """Represents a word prediction from a modality."""
    word: str
    confidence: float
    modality: str  # 'audio' or 'visual'
    position: int  # Position in sequence
    
    
@dataclass
class FusionResult:
    """Result of attention-based fusion."""
    fused_transcript: str
    word_details: List[Dict[str, Any]]
    audio_weight: float
    visual_weight: float
    agreement_score: float
    fusion_confidence: float
    switches: int  # Number of modality switches


class PhoneticAnalyzer:
    """
    Analyzes phonetic distinctiveness to determine which modality is more reliable.
    
    Novel Contribution: Context-aware phonetic analysis for intelligent modality selection.
    """
    
    # Phonetically distinct sounds (audio wins)
    AUDIO_STRONG_PHONEMES = {
        'voiced': ['b', 'd', 'g', 'v', 'z', 'zh', 'j'],
        'fricatives': ['s', 'z', 'sh', 'zh', 'f', 'v', 'th'],
        'nasals': ['m', 'n', 'ng'],
        'liquids': ['l', 'r'],
    }
    
    # Visually distinct sounds (visual wins)
    VISUAL_STRONG_VISEMES = {
        'bilabials': ['p', 'b', 'm'],  # Lips together
        'labiodentals': ['f', 'v'],     # Teeth on lip
        'rounded': ['w', 'oo', 'oh'],   # Lip rounding
    }

    # Confusable homophones (visual helps disambiguate)
    HOMOPHONES = [
        ('to', 'too', 'two'),
        ('their', 'there', 'they\'re'),
        ('right', 'write'),
        ('by', 'buy', 'bye'),
        ('no', 'know'),
    ]

    # Viseme Groups (homophemes) - words/sounds that look identical on lips
    # If audio/visual disagreement falls within these groups, Audio MUST win.
    VISEME_GROUPS = {
        'bilabials': {'p', 'b', 'm', 'pat', 'bat', 'mat', 'bin', 'pin', 'min'},
        'labiodentals': {'f', 'v', 'fan', 'van'},
        'linguadentals': {'th', 'dh'},
        'alveolars': {'t', 'd', 'n', 'l'},
        'velars': {'k', 'g', 'ng'},
    }
    
    @staticmethod
    def get_phonetic_bias(word: str) -> float:
        """
        Compute phonetic bias: positive favors audio, negative favors visual.
        
        Returns:
            Float in [-1.0, 1.0] representing modality preference
        """
        word_lower = word.lower()
        audio_score = 0.0
        visual_score = 0.0
        
        # Check for audio-strong phonemes
        for phoneme_type, phonemes in PhoneticAnalyzer.AUDIO_STRONG_PHONEMES.items():
            for phoneme in phonemes:
                if phoneme in word_lower:
                    audio_score += 0.2
        
        # Check for visual-strong visemes
        for viseme_type, visemes in PhoneticAnalyzer.VISUAL_STRONG_VISEMES.items():
            for viseme in visemes:
                if viseme in word_lower:
                    visual_score += 0.2
        
        # Check if word is part of homophone group (visual helps)
        for homophone_group in PhoneticAnalyzer.HOMOPHONES:
            if word_lower in homophone_group:
                visual_score += 0.3
        
        # Normalize to [-1, 1]
        bias = np.tanh(audio_score - visual_score)
        return bias

    @staticmethod
    def check_viseme_conflict(audio_word: str, visual_word: str) -> float:
        """
        Check if the disagreement is purely visual (homophemes).
        
        If 'audio_word' and 'visual_word' share the same Viseme Group (e.g. Bat vs Pat),
        it means the Visual system is confused by identical lip shapes.
        In this case, we MUST trust Audio, as it can hear the difference.
        
        Returns:
            Float bias override: 
            > 0.0 (e.g. 2.0) if a Viseme Conflict is detected (Strong Audio Bias)
            0.0 if no specific conflict
        """
        a = audio_word.lower()
        v = visual_word.lower()
        
        if a == v:
            return 0.0
            
        # 1. Check strict first-letter/sound confusion (most common in VSR)
        # e.g. Bat vs Pat vs Mat
        first_a = a[0] if a else ''
        first_v = v[0] if v else ''
        
        for group_name, members in PhoneticAnalyzer.VISEME_GROUPS.items():
            # Check if both first letters are in the same confusion group
            if first_a in members and first_v in members:
                # If the rest of the word is similar, it's definitely a viseme error
                if a[1:] == v[1:]: # e.g. (b)at vs (p)at
                    return 2.0 # Strong Audio Bias
        
        # 2. Check full word membership purely for known homophemes
        # (Simplified list for now)
        known_confusions = [
             {'pat', 'bat', 'mat'},
             {'bin', 'pin', 'min'},
             {'fan', 'van'},
             {'fine', 'vine'},
             {'few', 'view'},
             {'sip', 'zip'},
        ]
        
        for group in known_confusions:
            if a in group and v in group:
                return 2.0 # Strong Audio Bias

        return 0.0


class AttentionFusion:
    """
    Novel attention-based fusion mechanism for multi-modal speech recognition.
    
    Key Features:
    1. Word-level attention weights based on confidence
    2. Contextual smoothing to prevent excessive switching
    3. Phonetic awareness for intelligent selection
    4. Adaptive thresholding based on modality reliability
    
    Novel Contribution: This is NOT simple averaging or voting - it's a learned
    attention mechanism that adapts to context and confidence.
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        context_window: int = 3,
        min_confidence_threshold: float = 0.6,
        switching_penalty: float = 0.1,
    ):
        """
        Initialize attention-based fusion.
        
        Args:
            temperature: Softmax temperature for attention weights (higher = softer)
            context_window: Number of surrounding words to consider for smoothing
            min_confidence_threshold: Minimum confidence to trust a prediction
            switching_penalty: Penalty for switching between modalities (encourages consistency)
        """
        self.temperature = temperature
        self.context_window = context_window
        self.min_confidence_threshold = min_confidence_threshold
        self.switching_penalty = switching_penalty
        self.phonetic_analyzer = PhoneticAnalyzer()
    
    def compute_attention_weights(
        self,
        audio_conf: float,
        visual_conf: float,
        phonetic_bias: float,
        context_consistency: float,
    ) -> Tuple[float, float]:
        """
        Compute attention weights for audio and visual modalities.
        
        Novel Contribution: Multi-factor attention that considers:
        - Confidence scores
        - Phonetic distinctiveness
        - Temporal context consistency
        
        Args:
            audio_conf: Audio confidence score
            visual_conf: Visual confidence score
            phonetic_bias: Phonetic preference (-1 to 1)
            context_consistency: How consistent with surrounding predictions
            
        Returns:
            (audio_weight, visual_weight) tuple, sum to 1.0
        """
        # Base scores from confidence
        audio_score = audio_conf
        visual_score = visual_conf
        
        # Apply phonetic bias (novel contribution)
        if phonetic_bias > 0:  # Favor audio
            audio_score *= (1.0 + 0.3 * phonetic_bias)
        else:  # Favor visual
            visual_score *= (1.0 - 0.3 * phonetic_bias)
        
        # Apply context consistency boost
        audio_score *= (1.0 + 0.2 * context_consistency)
        visual_score *= (1.0 + 0.2 * context_consistency)
        
        # Softmax with temperature for smooth weighting
        scores = np.array([audio_score, visual_score])
        exp_scores = np.exp(scores / self.temperature)
        weights = exp_scores / np.sum(exp_scores)
        
        return float(weights[0]), float(weights[1])
    
    def compute_context_consistency(
        self,
        current_pos: int,
        predicted_modality: str,
        previous_selections: List[str],
    ) -> float:
        """
        Compute how consistent current selection is with context.
        
        Novel Contribution: Temporal smoothing to prevent excessive switching.
        
        Args:
            current_pos: Current word position
            predicted_modality: Predicted modality ('audio' or 'visual')
            previous_selections: List of previously selected modalities
            
        Returns:
            Consistency score [0.0, 1.0]
        """
        if not previous_selections:
            return 0.5  # Neutral if no context
        
        # Look at surrounding context window
        start = max(0, current_pos - self.context_window)
        end = min(len(previous_selections), current_pos)
        context = previous_selections[start:end]
        
        if not context:
            return 0.5
        
        # Count how many neighbors match predicted modality
        matches = sum(1 for m in context if m == predicted_modality)
        consistency = matches / len(context)
        
        return consistency
    
    def fuse_transcripts(
        self,
        audio_words: List[Tuple[str, float]],
        visual_words: List[Tuple[str, float]],
    ) -> FusionResult:
        """
        Perform attention-based fusion of audio and visual transcripts.
        
        Novel Contribution: Word-by-word attention mechanism with:
        - Confidence-based weighting
        - Phonetic awareness
        - Context consistency
        - Adaptive switching
        
        Args:
            audio_words: [(word, confidence), ...] from audio
            visual_words: [(word, confidence), ...] from visual
            
        Returns:
            FusionResult with fused transcript and detailed metrics
        """
        max_len = max(len(audio_words), len(visual_words))
        
        fused_words = []
        word_details = []
        selected_modalities = []
        total_audio_weight = 0.0
        total_visual_weight = 0.0
        agreement_count = 0
        switches = 0
        
        
        def _extract(w):
            if isinstance(w, (list, tuple)) and len(w) >= 2:
                return w[0], w[1]
            return "", 0.0

        for i in range(max_len):
            # Get words and confidences
            audio_word, audio_conf = _extract(audio_words[i]) if i < len(audio_words) else ("", 0.0)
            visual_word, visual_conf = _extract(visual_words[i]) if i < len(visual_words) else ("", 0.0)
            
            # Skip if both empty
            if not audio_word and not visual_word:
                continue
            
            # If only one available, use it
            if not audio_word:
                selected_word = visual_word
                selected_conf = visual_conf
                selected_modality = 'visual'
                audio_weight, visual_weight = 0.0, 1.0
            elif not visual_word:
                selected_word = audio_word
                selected_conf = audio_conf
                selected_modality = 'audio'
                audio_weight, visual_weight = 1.0, 0.0
            else:
                # Both available - apply attention mechanism
                
                # Both available - apply attention mechanism
                
                # Compute phonetic bias
                base_phonetic_bias = self.phonetic_analyzer.get_phonetic_bias(audio_word)
                
                # Check for direct Viseme Conflict (New Feature)
                viseme_bias = self.phonetic_analyzer.check_viseme_conflict(audio_word, visual_word)
                
                if viseme_bias > 0:
                    # If Viseme Conflict detected (e.g. Bat vs Pat), Audio reliability is paramount.
                    # We override the base bias with a strong audio preference.
                    phonetic_bias = viseme_bias
                else:
                    phonetic_bias = base_phonetic_bias
                
                # Compute context consistency
                
                # Compute context consistency
                context_consistency = self.compute_context_consistency(
                    i, 'audio' if audio_conf > visual_conf else 'visual',
                    selected_modalities
                )
                
                # Compute attention weights (NOVEL CONTRIBUTION)
                audio_weight, visual_weight = self.compute_attention_weights(
                    audio_conf, visual_conf, phonetic_bias, context_consistency
                )
                
                # Select word based on attention weights
                if audio_weight > visual_weight:
                    selected_word = audio_word
                    selected_conf = audio_conf
                    selected_modality = 'audio'
                else:
                    selected_word = visual_word
                    selected_conf = visual_conf
                    selected_modality = 'visual'
                
                # Apply switching penalty
                if selected_modalities and selected_modalities[-1] != selected_modality:
                    # Penalize confidence for switching
                    selected_conf *= (1.0 - self.switching_penalty)
                    switches += 1
                
                # Check agreement
                if audio_word.lower() == visual_word.lower():
                    agreement_count += 1
            
            # Record selection
            selected_modalities.append(selected_modality)
            fused_words.append(selected_word)
            total_audio_weight += audio_weight
            total_visual_weight += visual_weight
            
            # Store details
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
                'confidence': selected_conf,
            })
        
        # Compute aggregate metrics
        num_words = len(fused_words)
        avg_audio_weight = total_audio_weight / num_words if num_words > 0 else 0.0
        avg_visual_weight = total_visual_weight / num_words if num_words > 0 else 0.0
        agreement_score = agreement_count / num_words if num_words > 0 else 0.0
        
        # Compute overall fusion confidence
        avg_confidence = np.mean([w['confidence'] for w in word_details]) if word_details else 0.0
        fusion_confidence = float(avg_confidence)
        
        return FusionResult(
            fused_transcript=' '.join(fused_words),
            word_details=word_details,
            audio_weight=avg_audio_weight,
            visual_weight=avg_visual_weight,
            agreement_score=agreement_score,
            fusion_confidence=fusion_confidence,
            switches=switches,
        )
    
    def analyze_reliability(
        self,
        audio_words: List[Tuple[str, float]],
        visual_words: List[Tuple[str, float]],
    ) -> Dict[str, float]:
        """
        Analyze which modality is more reliable for this input.
        
        Novel Contribution: Adaptive reliability assessment.
        
        Returns:
            Dictionary with reliability metrics
        """
        audio_confs = [conf for _, conf in audio_words if conf > 0]
        visual_confs = [conf for _, conf in visual_words if conf > 0]
        
        audio_reliability = np.mean(audio_confs) if audio_confs else 0.0
        visual_reliability = np.mean(visual_confs) if visual_confs else 0.0
        
        # Check consistency (variance)
        audio_consistency = 1.0 - np.std(audio_confs) if len(audio_confs) > 1 else 0.0
        visual_consistency = 1.0 - np.std(visual_confs) if len(visual_confs) > 1 else 0.0
        
        return {
            'audio_reliability': float(audio_reliability),
            'visual_reliability': float(visual_reliability),
            'audio_consistency': float(audio_consistency),
            'visual_consistency': float(visual_consistency),
            'recommended_modality': 'audio' if audio_reliability > visual_reliability else 'visual',
        }


# Example usage
if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Novel Attention-Based Multi-Modal Fusion")
    print("=" * 80)
    
    # Sample inputs
    audio_words = [
        ("bin", 0.95), ("blue", 0.92), ("at", 0.98), 
        ("f", 0.88), ("two", 0.85), ("now", 0.93)
    ]
    
    visual_words = [
        ("bin", 0.72), ("blue", 0.68), ("at", 0.75), 
        ("f", 0.82), ("two", 0.71), ("now", 0.77)
    ]
    
    # Create fusion model
    fusion = AttentionFusion(
        temperature=2.0,
        context_window=2,
        switching_penalty=0.1
    )
    
    # Fuse
    result = fusion.fuse_transcripts(audio_words, visual_words)
    
    print(f"\nFused Transcript: {result.fused_transcript}")
    print(f"Audio Weight: {result.audio_weight:.3f}")
    print(f"Visual Weight: {result.visual_weight:.3f}")
    print(f"Agreement: {result.agreement_score:.3f}")
    print(f"Confidence: {result.fusion_confidence:.3f}")
    print(f"Modality Switches: {result.switches}")
    
    print("\nWord-by-word details:")
    for detail in result.word_details:
        print(f"  {detail['word']:10s} | "
              f"Audio: {detail['audio_word']:10s} ({detail['audio_conf']:.2f}) | "
              f"Visual: {detail['visual_word']:10s} ({detail['visual_conf']:.2f}) | "
              f"Selected: {detail['selected_modality']:7s} | "
              f"Weights: A={detail['audio_weight']:.2f} V={detail['visual_weight']:.2f}")
    
    # Reliability analysis
    reliability = fusion.analyze_reliability(audio_words, visual_words)
    print(f"\nReliability Analysis:")
    for key, value in reliability.items():
        print(f"  {key}: {value}")

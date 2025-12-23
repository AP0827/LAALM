"""Multi-modal fusion of DeepGram and LipNet outputs with confidence weighting."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModalityOutput:
    """Represents output from a single modality (DeepGram or LipNet)."""
    modality: str  # "deepgram" or "lipnet"
    transcript: str
    word_confidences: Optional[List[Tuple[str, float]]] = None  # [(word, confidence), ...]
    character_confidences: Optional[List[float]] = None  # Per-timestep probs
    overall_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Result of multi-modal fusion."""
    deepgram_output: ModalityOutput
    lipnet_output: ModalityOutput
    fused_transcript: str
    fused_word_confidences: List[Tuple[str, float]]  # Weighted confidences
    fusion_weights: Dict[str, float]  # {deepgram_weight, lipnet_weight}
    alignment_score: float  # How well outputs align (0-1)
    flagged_discrepancies: List[Dict[str, Any]]  # Words/phrases that differ
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModalityFuser:
    """Fuses outputs from multiple modalities (DeepGram audio + LipNet visual)."""
    
    def __init__(
        self,
        deepgram_weight: Optional[float] = None,
        lipnet_weight: Optional[float] = None,
        confidence_weighted: bool = True,
    ):
        """
        Initialize the modality fuser.
        
        Args:
            deepgram_weight: Weight for DeepGram modality (default: auto-computed).
            lipnet_weight: Weight for LipNet modality (default: auto-computed).
            confidence_weighted: If True, weights are adjusted by confidence scores.
        """
        self.deepgram_weight = deepgram_weight
        self.lipnet_weight = lipnet_weight
        self.confidence_weighted = confidence_weighted
        
        # Normalize weights if both provided
        if deepgram_weight is not None and lipnet_weight is not None:
            total = deepgram_weight + lipnet_weight
            self.deepgram_weight = deepgram_weight / total
            self.lipnet_weight = lipnet_weight / total
    
    def compute_dynamic_weights(
        self,
        deepgram_output: ModalityOutput,
        lipnet_output: ModalityOutput,
    ) -> Tuple[float, float]:
        """
        Compute weights based on confidence scores.
        
        Args:
            deepgram_output: Output from DeepGram.
            lipnet_output: Output from LipNet.
            
        Returns:
            Tuple of (deepgram_weight, lipnet_weight) that sum to 1.0
        """
        dg_conf = deepgram_output.overall_confidence
        lipnet_conf = lipnet_output.overall_confidence
        
        total_conf = dg_conf + lipnet_conf
        if total_conf == 0:
            return 0.5, 0.5
        
        return dg_conf / total_conf, lipnet_conf / total_conf
    
    def tokenize_transcript(self, transcript: str) -> List[str]:
        """
        Tokenize transcript into words.
        
        Args:
            transcript: Transcript text.
            
        Returns:
            List of words.
        """
        return transcript.lower().split()
    
    def compute_alignment_score(
        self,
        transcript1: str,
        transcript2: str,
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Compute alignment score between two transcripts using simple token matching.
        
        Args:
            transcript1: First transcript.
            transcript2: Second transcript.
            
        Returns:
            Tuple of (alignment_score [0-1], mismatches [(idx1, idx2), ...])
        """
        words1 = self.tokenize_transcript(transcript1)
        words2 = self.tokenize_transcript(transcript2)
        
        # Simple word-level matching
        matches = sum(1 for w1, w2 in zip(words1, words2) if w1 == w2)
        total = max(len(words1), len(words2))
        
        alignment_score = matches / total if total > 0 else 0.0
        
        # Find mismatches
        mismatches = []
        for i, (w1, w2) in enumerate(zip(words1, words2)):
            if w1 != w2:
                mismatches.append((i, i))
        
        return alignment_score, mismatches
    
    def fuse_word_confidences(
        self,
        dg_words: List[Tuple[str, float]],
        lipnet_words: List[Tuple[str, float]],
        dg_weight: float,
        lipnet_weight: float,
    ) -> List[Tuple[str, float]]:
        """
        Fuse word-level confidences from both modalities.
        
        Args:
            dg_words: [(word, confidence), ...] from DeepGram.
            lipnet_words: [(word, confidence), ...] from LipNet.
            dg_weight: Weight for DeepGram.
            lipnet_weight: Weight for LipNet.
            
        Returns:
            List of (word, fused_confidence) tuples.
        """
        # Pad shorter list
        max_len = max(len(dg_words), len(lipnet_words))
        
        fused = []
        for i in range(max_len):
            # Get word and confidence from each modality (or use defaults)
            if i < len(dg_words):
                dg_word, dg_conf = dg_words[i]
            else:
                dg_word, dg_conf = "", 0.0
            
            if i < len(lipnet_words):
                lipnet_word, lipnet_conf = lipnet_words[i]
            else:
                lipnet_word, lipnet_conf = "", 0.0
            
            # Use word from whichever has higher confidence
            word = dg_word if dg_conf >= lipnet_conf else lipnet_word
            word = word or dg_word or lipnet_word  # Fallback
            
            # Fuse confidences
            fused_conf = (dg_conf * dg_weight) + (lipnet_conf * lipnet_weight)
            fused.append((word, fused_conf))
        
        return fused
    
    def flag_discrepancies(
        self,
        dg_words: List[Tuple[str, float]],
        lipnet_words: List[Tuple[str, float]],
        confidence_diff_threshold: float = 0.3,
        word_diff_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Flag significant discrepancies between modalities.
        
        Args:
            dg_words: Words from DeepGram.
            lipnet_words: Words from LipNet.
            confidence_diff_threshold: Flag if confidence differs by this amount.
            word_diff_threshold: Alignment score below which to flag.
            
        Returns:
            List of discrepancy dictionaries with details.
        """
        discrepancies = []
        
        for i in range(min(len(dg_words), len(lipnet_words))):
            dg_word, dg_conf = dg_words[i]
            lipnet_word, lipnet_conf = lipnet_words[i]
            
            conf_diff = abs(dg_conf - lipnet_conf)
            words_differ = dg_word.lower() != lipnet_word.lower()
            
            if conf_diff > confidence_diff_threshold or words_differ:
                discrepancies.append({
                    "position": i,
                    "deepgram_word": dg_word,
                    "deepgram_confidence": dg_conf,
                    "lipnet_word": lipnet_word,
                    "lipnet_confidence": lipnet_conf,
                    "confidence_diff": conf_diff,
                    "words_differ": words_differ,
                    "severity": "high" if words_differ else "medium",
                })
        
        # Handle length mismatches
        if len(dg_words) != len(lipnet_words):
            discrepancies.append({
                "position": "length_mismatch",
                "deepgram_length": len(dg_words),
                "lipnet_length": len(lipnet_words),
                "severity": "low" if abs(len(dg_words) - len(lipnet_words)) <= 1 else "medium",
            })
        
        return discrepancies
    
    def fuse(
        self,
        deepgram_output: ModalityOutput,
        lipnet_output: ModalityOutput,
    ) -> FusionResult:
        """
        Fuse outputs from both modalities.
        
        Args:
            deepgram_output: Output from DeepGram module.
            lipnet_output: Output from LipNet module.
            
        Returns:
            FusionResult with fused transcript and metrics.
        """
        # Compute or use provided weights
        if self.confidence_weighted:
            dg_weight, lipnet_weight = self.compute_dynamic_weights(
                deepgram_output, lipnet_output
            )
        else:
            dg_weight = self.deepgram_weight or 0.5
            lipnet_weight = self.lipnet_weight or 0.5
        
        # Compute alignment
        alignment_score, mismatches = self.compute_alignment_score(
            deepgram_output.transcript,
            lipnet_output.transcript,
        )
        
        # Fuse word-level confidences
        dg_words = deepgram_output.word_confidences or []
        lipnet_words = lipnet_output.word_confidences or []
        
        fused_words = self.fuse_word_confidences(
            dg_words, lipnet_words, dg_weight, lipnet_weight
        )
        
        # Create fused transcript (prefer higher-confidence version)
        if alignment_score > 0.8:  # High agreement
            fused_transcript = deepgram_output.transcript
        elif alignment_score > 0.5:  # Moderate agreement, use higher confidence
            fused_transcript = (
                deepgram_output.transcript 
                if deepgram_output.overall_confidence >= lipnet_output.overall_confidence
                else lipnet_output.transcript
            )
        else:  # Low agreement, concatenate with confidence markers
            fused_transcript = (
                f"[DG: {deepgram_output.transcript} (conf: {deepgram_output.overall_confidence:.2f})] "
                f"[LN: {lipnet_output.transcript} (conf: {lipnet_output.overall_confidence:.2f})]"
            )
        
        # Flag discrepancies
        flagged = self.flag_discrepancies(dg_words, lipnet_words)
        
        return FusionResult(
            deepgram_output=deepgram_output,
            lipnet_output=lipnet_output,
            fused_transcript=fused_transcript,
            fused_word_confidences=fused_words,
            fusion_weights={
                "deepgram": float(dg_weight),
                "lipnet": float(lipnet_weight),
            },
            alignment_score=float(alignment_score),
            flagged_discrepancies=flagged,
            metadata={
                "fusion_method": "confidence_weighted" if self.confidence_weighted else "equal",
                "alignment_mismatches": len(mismatches),
            },
        )
    
    def get_fusion_report(self, fusion_result: FusionResult) -> str:
        """
        Generate a human-readable report of the fusion result.
        
        Args:
            fusion_result: Result from fuse().
            
        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 80,
            "MULTI-MODAL FUSION REPORT",
            "=" * 80,
            "",
            f"DeepGram Transcript:  {fusion_result.deepgram_output.transcript}",
            f"  Overall Confidence:  {fusion_result.deepgram_output.overall_confidence:.3f}",
            f"  Weight in Fusion:    {fusion_result.fusion_weights['deepgram']:.3f}",
            "",
            f"LipNet Transcript:    {fusion_result.lipnet_output.transcript}",
            f"  Overall Confidence:  {fusion_result.lipnet_output.overall_confidence:.3f}",
            f"  Weight in Fusion:    {fusion_result.fusion_weights['lipnet']:.3f}",
            "",
            f"Alignment Score:      {fusion_result.alignment_score:.3f}",
            f"Fused Transcript:     {fusion_result.fused_transcript}",
            "",
        ]
        
        if fusion_result.flagged_discrepancies:
            lines.append("FLAGGED DISCREPANCIES:")
            for disc in fusion_result.flagged_discrepancies:
                if "position" in disc and isinstance(disc["position"], int):
                    lines.append(
                        f"  Position {disc['position']} ({disc['severity'].upper()}):"
                    )
                    if disc.get("words_differ"):
                        lines.append(
                            f"    DeepGram: '{disc['deepgram_word']}' (conf: {disc['deepgram_confidence']:.3f})"
                        )
                        lines.append(
                            f"    LipNet:   '{disc['lipnet_word']}' (conf: {disc['lipnet_confidence']:.3f})"
                        )
                    else:
                        lines.append(
                            f"    Confidence Diff: {disc['confidence_diff']:.3f}"
                        )
                elif "length_mismatch" in disc:
                    lines.append(
                        f"  Length Mismatch: DeepGram={disc['deepgram_length']}, "
                        f"LipNet={disc['lipnet_length']}"
                    )
            lines.append("")
        
        lines.extend([
            "=" * 80,
            "",
        ])
        
        return "\n".join(lines)

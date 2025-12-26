"""Transformer module initialization and main pipeline integration."""

from typing import Dict, Any, Optional
from .fusion import ModalityOutput, FusionResult, ModalityFuser
from .llm_corrector import CorrectionContext, CorrectionResult, LLMSemanticCorrector, LLMProvider


__all__ = [
    "ModalityOutput",
    "FusionResult",
    "ModalityFuser",
    "CorrectionContext",
    "CorrectionResult",
    "LLMSemanticCorrector",
    "LLMProvider",
    "TransformerPipeline",
]


class TransformerPipeline:
    """
    End-to-end pipeline combining DeepGram + LipNet with LLM refinement.
    
    Workflow:
    1. Get outputs from DeepGram (audio transcription + word-level confidence)
    2. Get outputs from LipNet (visual transcription + character-level confidence)
    3. Fuse both using confidence-weighted combination
    4. Apply LLM-based semantic correction considering probabilities
    5. Return refined transcript with confidence metrics
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider = LLMProvider.GROQ,
        llm_model: str = "mixtral-8x7b-32768",
        llm_api_key: Optional[str] = None,
        use_confidence_weighting: bool = True,
        llm_enabled: bool = True,
    ):
        """
        Initialize the transformer pipeline.
        
        Args:
            llm_provider: LLM provider to use for semantic correction (default: Groq).
            llm_model: LLM model name (default: mixtral-8x7b-32768 for Groq).
            llm_api_key: API key for LLM provider (uses GROQ_API_KEY env var if not provided).
            use_confidence_weighting: Whether to weight fusion by confidence.
            llm_enabled: Whether to apply LLM correction.
        """
        self.fuser = ModalityFuser(confidence_weighted=use_confidence_weighting)
        self.llm_enabled = llm_enabled
        
        if llm_enabled:
            self.corrector = LLMSemanticCorrector(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
            )
        else:
            self.corrector = None
    
    def process(
        self,
        deepgram_transcript: str,
        deepgram_confidence: float,
        deepgram_word_confidences: list,
        lipnet_transcript: str,
        lipnet_confidence: float,
        lipnet_word_confidences: list,
        domain_context: Optional[str] = None,
        audio_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process outputs from both modalities through the full pipeline.
        
        Args:
            deepgram_transcript: Transcript from DeepGram.
            deepgram_confidence: Overall confidence from DeepGram.
            deepgram_word_confidences: [(word, confidence), ...] from DeepGram.
            lipnet_transcript: Transcript from LipNet.
            lipnet_confidence: Overall confidence from LipNet.
            lipnet_word_confidences: [(word, confidence), ...] from LipNet.
            domain_context: Optional domain context (e.g., "medical", "legal").
            audio_metadata: Optional metadata about the audio.
            
        Returns:
            Dictionary with full pipeline results.
        """
        # Create modality outputs
        dg_output = ModalityOutput(
            modality="deepgram",
            transcript=deepgram_transcript,
            word_confidences=deepgram_word_confidences,
            overall_confidence=deepgram_confidence,
            metadata={"source": "audio"},
        )
        
        lipnet_output = ModalityOutput(
            modality="lipnet",
            transcript=lipnet_transcript,
            word_confidences=lipnet_word_confidences,
            overall_confidence=lipnet_confidence,
            metadata={"source": "visual"},
        )
        
        # Step 1: Fuse modalities
        fusion_result = self.fuser.fuse(dg_output, lipnet_output)
        
        # Step 2: Apply LLM correction if enabled
        correction_result = None
        if self.llm_enabled:
            context = CorrectionContext(
                deepgram_transcript=deepgram_transcript,
                deepgram_confidence=deepgram_confidence,
                deepgram_word_confidences=deepgram_word_confidences,
                lipnet_transcript=lipnet_transcript,
                lipnet_confidence=lipnet_confidence,
                lipnet_word_confidences=lipnet_word_confidences,
                alignment_score=fusion_result.alignment_score,
                flagged_discrepancies=fusion_result.flagged_discrepancies,
                domain_context=domain_context,
                audio_metadata=audio_metadata,
            )
            correction_result = self.corrector.correct(context)
        
        return {
            "deepgram_output": {
                "transcript": deepgram_transcript,
                "confidence": deepgram_confidence,
                "word_confidences": deepgram_word_confidences,
            },
            "lipnet_output": {
                "transcript": lipnet_transcript,
                "confidence": lipnet_confidence,
                "word_confidences": lipnet_word_confidences,
            },
            "fusion_result": {
                "fused_transcript": fusion_result.fused_transcript,
                "alignment_score": fusion_result.alignment_score,
                "fusion_weights": fusion_result.fusion_weights,
                "fused_word_confidences": fusion_result.fused_word_confidences,
                "flagged_discrepancies": fusion_result.flagged_discrepancies,
            },
            "correction_result": {
                "corrected_transcript": correction_result.corrected_transcript if correction_result else None,
                "corrections_made": correction_result.corrections_made if correction_result else [],
                "confidence_in_corrections": correction_result.confidence_in_corrections if correction_result else None,
                "explanation": correction_result.explanation if correction_result else None,
            } if self.llm_enabled else None,
            "final_transcript": (
                correction_result.corrected_transcript
                if self.llm_enabled and correction_result
                else fusion_result.fused_transcript
            ),
            "metadata": {
                "domain": domain_context,
                "llm_enabled": self.llm_enabled,
                "fusion_method": "confidence_weighted",
            },
        }
    
    def get_full_report(self, result: Dict[str, Any]) -> str:
        """
        Generate a comprehensive report of the full pipeline.
        
        Args:
            result: Result from process().
            
        Returns:
            Formatted multi-section report.
        """
        lines = [
            "\n",
            "╔" + "=" * 78 + "╗",
            "║" + " " * 20 + "TRANSFORMER PIPELINE FULL REPORT" + " " * 26 + "║",
            "╚" + "=" * 78 + "╝",
            "",
        ]
        
        # DeepGram section
        lines.extend([
            "AUDIO MODALITY (DeepGram)",
            "-" * 80,
            f"Transcript:  {result['deepgram_output']['transcript']}",
            f"Confidence:  {result['deepgram_output']['confidence']:.3f}",
            "",
        ])
        
        # LipNet section
        lines.extend([
            "VISUAL MODALITY (LipNet)",
            "-" * 80,
            f"Transcript:  {result['lipnet_output']['transcript']}",
            f"Confidence:  {result['lipnet_output']['confidence']:.3f}",
            "",
        ])
        
        # Fusion section
        fusion = result["fusion_result"]
        lines.extend([
            "MULTI-MODAL FUSION",
            "-" * 80,
            f"Alignment Score:    {fusion['alignment_score']:.3f}",
            f"Fusion Weights:     DeepGram={fusion['fusion_weights']['deepgram']:.3f}, "
            f"LipNet={fusion['fusion_weights']['lipnet']:.3f}",
            f"Fused Transcript:   {fusion['fused_transcript']}",
            "",
        ])
        
        if fusion["flagged_discrepancies"]:
            lines.extend([
                "Flagged Discrepancies:",
            ])
            for disc in fusion["flagged_discrepancies"]:
                if isinstance(disc.get("position"), int):
                    lines.append(
                        f"  Position {disc['position']}: "
                        f"'{disc.get('deepgram_word', '?')}' vs '{disc.get('lipnet_word', '?')}' "
                        f"({disc['severity']})"
                    )
            lines.append("")
        
        # Correction section
        if result["correction_result"]:
            correction = result["correction_result"]
            lines.extend([
                "LLM SEMANTIC CORRECTION",
                "-" * 80,
                f"Corrected Transcript: {correction['corrected_transcript']}",
                f"Confidence:           {correction['confidence_in_corrections']:.3f}",
                "",
            ])
            
            if correction["corrections_made"]:
                lines.append("Corrections Applied:")
                for corr in correction["corrections_made"]:
                    lines.append(
                        f"  • '{corr.get('original_phrase', '')}' → "
                        f"'{corr.get('corrected_phrase', '')}'"
                    )
                lines.append("")
            
            if correction["explanation"]:
                lines.extend([
                    "Explanation:",
                    correction["explanation"],
                    "",
                ])
        
        # Final result
        lines.extend([
            "FINAL OUTPUT",
            "=" * 80,
            f"Final Transcript: {result['final_transcript']}",
            "",
        ])
        
        return "\n".join(lines)

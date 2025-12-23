"""LLM-based semantic correction using both modality outputs and confidence scores."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    LOCAL = "local"


@dataclass
class CorrectionContext:
    """Context for LLM correction."""
    deepgram_transcript: str
    deepgram_confidence: float
    deepgram_word_confidences: List[Tuple[str, float]]
    
    lipnet_transcript: str
    lipnet_confidence: float
    lipnet_word_confidences: List[Tuple[str, float]]
    
    alignment_score: float
    flagged_discrepancies: List[Dict[str, Any]]
    domain_context: Optional[str] = None  # e.g., "medical", "legal", "casual"
    audio_metadata: Optional[Dict[str, Any]] = None


@dataclass
class CorrectionResult:
    """Result of LLM-based correction."""
    corrected_transcript: str
    original_fused_transcript: str
    corrections_made: List[Dict[str, Any]]
    confidence_in_corrections: float
    explanation: str
    metadata: Dict[str, Any]


class LLMSemanticCorrector:
    """Uses LLM to semantically correct and refine multi-modal outputs."""
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ):
        """
        Initialize the LLM-based semantic corrector.
        
        Args:
            provider: LLM provider to use.
            model: Model name/ID.
            api_key: API key for the provider.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens in response.
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._init_client()
    
    def _init_client(self):
        """Initialize the LLM client based on provider."""
        if self.provider == LLMProvider.OPENAI:
            try:
                import openai
                openai.api_key = self.api_key
                self.client = openai.ChatCompletion
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        
        elif self.provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        
        elif self.provider == LLMProvider.GOOGLE:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "google-generativeai package required: pip install google-generativeai"
                )
        
        elif self.provider == LLMProvider.OLLAMA:
            # For local Ollama, we'll use requests
            try:
                import requests
                self.client = requests
                self.ollama_base_url = "http://localhost:11434"
            except ImportError:
                raise ImportError("requests package required: pip install requests")
        
        elif self.provider == LLMProvider.LOCAL:
            # For completely local inference (e.g., llama.cpp, transformers)
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model)
                self.model_obj = AutoModelForCausalLM.from_pretrained(self.model)
            except ImportError:
                raise ImportError(
                    "transformers package required: pip install transformers torch"
                )
    
    def build_correction_prompt(
        self,
        context: CorrectionContext,
        include_confidence_analysis: bool = True,
        include_discrepancy_analysis: bool = True,
    ) -> str:
        """
        Build a detailed prompt for the LLM with context.
        
        Args:
            context: CorrectionContext with multi-modal data.
            include_confidence_analysis: Whether to include confidence-based guidance.
            include_discrepancy_analysis: Whether to analyze discrepancies.
            
        Returns:
            Formatted prompt string.
        """
        prompt_parts = [
            "You are an expert speech transcription refinement system.",
            "Your task is to semantically correct and improve transcripts from multiple modalities.",
            "",
            "CONTEXT:",
            f"- Domain: {context.domain_context or 'general'}",
            f"- Audio Duration: {context.audio_metadata.get('duration', 'unknown') if context.audio_metadata else 'unknown'}s",
            "",
            "AUDIO TRANSCRIPTION (DeepGram - acoustic model):",
            f"  Transcript: {context.deepgram_transcript}",
            f"  Overall Confidence: {context.deepgram_confidence:.3f}",
        ]
        
        if include_confidence_analysis and context.deepgram_word_confidences:
            low_conf_words = [
                (w, c) for w, c in context.deepgram_word_confidences if c < 0.7
            ]
            if low_conf_words:
                prompt_parts.append(f"  Low-Confidence Words (< 0.7): {low_conf_words}")
        
        prompt_parts.extend([
            "",
            "VISUAL TRANSCRIPTION (LipNet - visual model):",
            f"  Transcript: {context.lipnet_transcript}",
            f"  Overall Confidence: {context.lipnet_confidence:.3f}",
        ])
        
        if include_confidence_analysis and context.lipnet_word_confidences:
            low_conf_words = [
                (w, c) for w, c in context.lipnet_word_confidences if c < 0.7
            ]
            if low_conf_words:
                prompt_parts.append(f"  Low-Confidence Words (< 0.7): {low_conf_words}")
        
        prompt_parts.extend([
            "",
            f"ALIGNMENT SCORE: {context.alignment_score:.3f} (0=no agreement, 1=perfect agreement)",
        ])
        
        if include_discrepancy_analysis and context.flagged_discrepancies:
            high_severity = [d for d in context.flagged_discrepancies if d.get("severity") == "high"]
            if high_severity:
                prompt_parts.append("")
                prompt_parts.append("FLAGGED HIGH-SEVERITY DISCREPANCIES:")
                for disc in high_severity:
                    if isinstance(disc.get("position"), int):
                        prompt_parts.append(
                            f"  Position {disc['position']}: "
                            f"'{disc['deepgram_word']}' (audio, conf={disc['deepgram_confidence']:.2f}) vs "
                            f"'{disc['lipnet_word']}' (visual, conf={disc['lipnet_confidence']:.2f})"
                        )
        
        prompt_parts.extend([
            "",
            "TASK:",
            "1. Analyze the semantic meaning across both transcripts.",
            "2. Identify and correct likely errors based on confidence scores.",
            "3. For low-confidence words, suggest corrections if possible.",
            "4. When modalities disagree, determine which is more likely correct.",
            "5. Preserve technical terms, proper nouns, and domain-specific language.",
            "",
            "CONSTRAINTS:",
            "- Prefer high-confidence words from either modality.",
            "- If alignment is low (< 0.5), be more conservative with corrections.",
            "- Maintain semantic coherence - ensure corrections make sense in context.",
            "- Don't over-correct minor discrepancies.",
            "",
            "RESPONSE FORMAT:",
            "Provide a JSON response with this structure:",
            "{",
            '  "corrected_transcript": "the refined transcript",',
            '  "corrections": [',
            '    {"original_phrase": "...", "corrected_phrase": "...", "reason": "...", "confidence": 0.95},',
            "  ],",
            '  "explanation": "summary of changes made",',
            '  "confidence_in_corrections": 0.85,',
            '  "notes": "any additional notes"',
            "}",
        ])
        
        return "\n".join(prompt_parts)
    
    def correct_with_openai(self, prompt: str) -> CorrectionResult:
        """Use OpenAI API for correction."""
        try:
            import openai
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert transcription refinement system that outputs valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)
            
            return CorrectionResult(
                corrected_transcript=result_json.get("corrected_transcript", ""),
                original_fused_transcript="",
                corrections_made=result_json.get("corrections", []),
                confidence_in_corrections=float(result_json.get("confidence_in_corrections", 0.5)),
                explanation=result_json.get("explanation", ""),
                metadata={
                    "provider": "openai",
                    "model": self.model,
                    "notes": result_json.get("notes", ""),
                },
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response was not valid JSON: {e}")
    
    def correct_with_anthropic(self, prompt: str) -> CorrectionResult:
        """Use Anthropic Claude API for correction."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            
            result_text = response.content[0].text
            result_json = json.loads(result_text)
            
            return CorrectionResult(
                corrected_transcript=result_json.get("corrected_transcript", ""),
                original_fused_transcript="",
                corrections_made=result_json.get("corrections", []),
                confidence_in_corrections=float(result_json.get("confidence_in_corrections", 0.5)),
                explanation=result_json.get("explanation", ""),
                metadata={
                    "provider": "anthropic",
                    "model": self.model,
                    "notes": result_json.get("notes", ""),
                },
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response was not valid JSON: {e}")
    
    def correct_with_ollama(self, prompt: str) -> CorrectionResult:
        """Use local Ollama instance for correction."""
        try:
            response = self.client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False,
                },
            )
            
            response.raise_for_status()
            result_text = response.json()["response"]
            result_json = json.loads(result_text)
            
            return CorrectionResult(
                corrected_transcript=result_json.get("corrected_transcript", ""),
                original_fused_transcript="",
                corrections_made=result_json.get("corrections", []),
                confidence_in_corrections=float(result_json.get("confidence_in_corrections", 0.5)),
                explanation=result_json.get("explanation", ""),
                metadata={
                    "provider": "ollama",
                    "model": self.model,
                    "notes": result_json.get("notes", ""),
                },
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response was not valid JSON: {e}")
    
    def correct(
        self,
        context: CorrectionContext,
    ) -> CorrectionResult:
        """
        Perform LLM-based semantic correction.
        
        Args:
            context: CorrectionContext with multi-modal data.
            
        Returns:
            CorrectionResult with corrected transcript and metadata.
        """
        prompt = self.build_correction_prompt(context)
        
        try:
            if self.provider == LLMProvider.OPENAI:
                result = self.correct_with_openai(prompt)
            elif self.provider == LLMProvider.ANTHROPIC:
                result = self.correct_with_anthropic(prompt)
            elif self.provider == LLMProvider.OLLAMA:
                result = self.correct_with_ollama(prompt)
            else:
                raise NotImplementedError(f"Provider {self.provider} not yet implemented")
            
            # Store original for comparison
            result.original_fused_transcript = (
                context.deepgram_transcript
                if context.deepgram_confidence >= context.lipnet_confidence
                else context.lipnet_transcript
            )
            
            return result
        
        except Exception as e:
            # Fallback: return the higher-confidence transcript unchanged
            return CorrectionResult(
                corrected_transcript=(
                    context.deepgram_transcript
                    if context.deepgram_confidence >= context.lipnet_confidence
                    else context.lipnet_transcript
                ),
                original_fused_transcript="",
                corrections_made=[],
                confidence_in_corrections=max(context.deepgram_confidence, context.lipnet_confidence),
                explanation=f"LLM correction failed: {str(e)}. Returning higher-confidence modality.",
                metadata={"error": str(e), "fallback": True},
            )
    
    def get_correction_report(self, result: CorrectionResult) -> str:
        """
        Generate a human-readable report of the correction.
        
        Args:
            result: CorrectionResult from correct().
            
        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 80,
            "LLM SEMANTIC CORRECTION REPORT",
            "=" * 80,
            "",
            f"Original Transcript:  {result.original_fused_transcript}",
            f"Corrected Transcript: {result.corrected_transcript}",
            f"Confidence:           {result.confidence_in_corrections:.3f}",
            "",
        ]
        
        if result.corrections_made:
            lines.append("CORRECTIONS APPLIED:")
            for i, corr in enumerate(result.corrections_made, 1):
                lines.append(f"  {i}. '{corr.get('original_phrase', '')}' → '{corr.get('corrected_phrase', '')}'")
                if corr.get("reason"):
                    lines.append(f"     Reason: {corr['reason']}")
                if corr.get("confidence"):
                    lines.append(f"     Confidence: {corr['confidence']:.3f}")
            lines.append("")
        
        if result.explanation:
            lines.append("EXPLANATION:")
            lines.append(result.explanation)
            lines.append("")
        
        if result.metadata.get("error"):
            lines.append("⚠ NOTE: This is a fallback result due to LLM error")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)

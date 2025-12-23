"""Word-level confidence extraction from DeepGram responses."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class WordConfidence:
    """Represents word-level confidence data from DeepGram."""
    word: str
    start: float
    end: float
    confidence: float
    duration: float
    
    def __post_init__(self):
        """Calculate duration if not provided."""
        if self.duration == 0:
            self.duration = self.end - self.start


@dataclass
class TranscriptMetrics:
    """Aggregated metrics from transcript-level and word-level data."""
    transcript: str
    overall_confidence: float
    mean_word_confidence: float
    median_word_confidence: float
    std_word_confidence: float
    min_word_confidence: float
    max_word_confidence: float
    low_confidence_ratio: float  # Fraction of words with confidence < threshold
    low_confidence_threshold: float
    total_words: int
    transcript_duration: float
    transcript_completeness: float
    words: List[WordConfidence]


class WordConfidenceExtractor:
    """Extracts and analyzes word-level confidence from DeepGram responses."""
    
    def __init__(self, low_confidence_threshold: float = 0.7):
        """
        Initialize the word confidence extractor.
        
        Args:
            low_confidence_threshold: Confidence threshold below which words are
                                     considered "low confidence". Default 0.7 (70%).
        """
        self.low_confidence_threshold = low_confidence_threshold
    
    def extract_word_confidences(
        self,
        response: Dict[str, Any],
        alternative_index: int = 0,
    ) -> List[WordConfidence]:
        """
        Extract word-level confidence scores from DeepGram response.
        
        Args:
            response: DeepGram API response (dict or ListenV1Response object).
            alternative_index: Which alternative to extract (default: best alternative).
            
        Returns:
            List of WordConfidence objects with timing and confidence data.
            
        Raises:
            ValueError: If response format is invalid or words not found.
        """
        # Convert response object to dict if needed
        if hasattr(response, 'model_dump'):
            response_dict = response.model_dump()
        elif hasattr(response, 'dict'):
            response_dict = response.dict()
        else:
            response_dict = response
        
        try:
            results = response_dict.get("results", {})
            channels = results.get("channels", [])
            
            if not channels:
                raise ValueError("No channels found in response")
            
            alternatives = channels[0].get("alternatives", [])
            if not alternatives or len(alternatives) <= alternative_index:
                raise ValueError(
                    f"Alternative {alternative_index} not found. "
                    f"Only {len(alternatives)} alternative(s) available."
                )
            
            alternative = alternatives[alternative_index]
            words_data = alternative.get("words", [])
            
            if not words_data:
                raise ValueError("No words found in alternative")
            
            word_confidences = []
            for word_data in words_data:
                wc = WordConfidence(
                    word=word_data.get("word", ""),
                    start=float(word_data.get("start", 0)),
                    end=float(word_data.get("end", 0)),
                    confidence=float(word_data.get("confidence", 0)),
                    duration=0,  # Will be calculated in __post_init__
                )
                word_confidences.append(wc)
            
            return word_confidences
        
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise ValueError(f"Failed to extract word confidences from response: {e}")
    
    def compute_metrics(
        self,
        response: Dict[str, Any],
        alternative_index: int = 0,
    ) -> TranscriptMetrics:
        """
        Compute comprehensive word-level confidence metrics from response.
        
        Args:
            response: DeepGram API response.
            alternative_index: Which alternative to use.
            
        Returns:
            TranscriptMetrics object with all computed metrics.
        """
        # Extract word-level data
        words = self.extract_word_confidences(response, alternative_index)
        
        if not words:
            raise ValueError("No words extracted from response")
        
        # Get overall transcript confidence
        response_dict = (
            response.model_dump() if hasattr(response, 'model_dump') else
            response.dict() if hasattr(response, 'dict') else
            response
        )
        
        results = response_dict.get("results", {})
        channels = results.get("channels", [])
        alternatives = channels[0].get("alternatives", [])
        alternative = alternatives[alternative_index]
        
        overall_confidence = float(alternative.get("confidence", 0))
        transcript = alternative.get("transcript", "")
        
        # Compute confidence-based metrics
        confidences = [w.confidence for w in words]
        mean_confidence = float(np.mean(confidences))
        median_confidence = float(np.median(confidences))
        std_confidence = float(np.std(confidences))
        min_confidence = float(np.min(confidences))
        max_confidence = float(np.max(confidences))
        
        low_conf_words = sum(1 for c in confidences if c < self.low_confidence_threshold)
        low_conf_ratio = low_conf_words / len(confidences) if confidences else 0.0
        
        # Compute timing-based metrics
        if words:
            start_time = min(w.start for w in words)
            end_time = max(w.end for w in words)
            transcript_duration = end_time - start_time
        else:
            transcript_duration = 0.0
        
        # Get audio metadata duration if available
        metadata = response_dict.get("metadata", {})
        audio_duration = float(metadata.get("duration", transcript_duration))
        
        # Transcript completeness: what fraction of audio was transcribed
        transcript_completeness = (
            transcript_duration / audio_duration 
            if audio_duration > 0 else 0.0
        )
        
        return TranscriptMetrics(
            transcript=transcript,
            overall_confidence=overall_confidence,
            mean_word_confidence=mean_confidence,
            median_word_confidence=median_confidence,
            std_word_confidence=std_confidence,
            min_word_confidence=min_confidence,
            max_word_confidence=max_confidence,
            low_confidence_ratio=low_conf_ratio,
            low_confidence_threshold=self.low_confidence_threshold,
            total_words=len(words),
            transcript_duration=transcript_duration,
            transcript_completeness=transcript_completeness,
            words=words,
        )
    
    def get_low_confidence_words(
        self,
        words: List[WordConfidence],
        threshold: Optional[float] = None,
    ) -> List[WordConfidence]:
        """
        Filter words below confidence threshold.
        
        Args:
            words: List of WordConfidence objects.
            threshold: Confidence threshold. Uses instance default if None.
            
        Returns:
            List of words with confidence below threshold.
        """
        threshold = threshold or self.low_confidence_threshold
        return [w for w in words if w.confidence < threshold]
    
    def get_word_probability_range(
        self,
        words: List[WordConfidence],
    ) -> Dict[str, float]:
        """
        Get probability statistics for display/analysis.
        
        Args:
            words: List of WordConfidence objects.
            
        Returns:
            Dictionary with min, max, mean, median probabilities.
        """
        if not words:
            return {"min": 0, "max": 0, "mean": 0, "median": 0}
        
        confidences = [w.confidence for w in words]
        return {
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "mean": float(np.mean(confidences)),
            "median": float(np.median(confidences)),
        }
    
    def format_words_with_confidence(
        self,
        words: List[WordConfidence],
        include_timing: bool = True,
    ) -> str:
        """
        Format words with confidence scores for display.
        
        Args:
            words: List of WordConfidence objects.
            include_timing: Whether to include start/end timing.
            
        Returns:
            Formatted string representation.
        """
        lines = []
        for w in words:
            if include_timing:
                lines.append(
                    f"{w.word:<15} | confidence: {w.confidence:.3f} | "
                    f"[{w.start:.2f}s - {w.end:.2f}s]"
                )
            else:
                lines.append(f"{w.word:<15} | confidence: {w.confidence:.3f}")
        
        return "\n".join(lines)

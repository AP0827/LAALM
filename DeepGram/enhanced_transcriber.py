"""Integration of word-level confidence extraction into DeepGram pipeline."""

from .word_confidence import WordConfidenceExtractor, TranscriptMetrics
from .transcriber import AudioTranscriber
from .config import DeepGramConfig


class DeepGramWithConfidence:
    """
    Enhanced DeepGram transcriber that automatically extracts 
    word-level confidence metrics alongside transcription.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize enhanced transcriber.
        
        Args:
            api_key: DeepGram API key.
        """
        config = DeepGramConfig(api_key=api_key)
        self.transcriber = AudioTranscriber(config=config)
        self.confidence_extractor = WordConfidenceExtractor(low_confidence_threshold=0.75)
    
    def transcribe_file_with_confidence(
        self,
        file_path: str,
        include_utterances: bool = True,
    ) -> dict:
        """
        Transcribe file and extract word-level confidence metrics.
        
        Args:
            file_path: Path to audio file.
            include_utterances: Whether to include utterances.
            
        Returns:
            Dictionary with transcript, confidence metrics, and raw response.
        """
        # Get transcription
        response = self.transcriber.transcribe_file(file_path, include_utterances)
        
        # Extract word confidences
        metrics = self.confidence_extractor.compute_metrics(response)
        
        # Extract words as list of tuples for downstream processing
        # Updated to include timing for alignment: (word, confidence, start, end)
        word_confidences = [(w.word, w.confidence, w.start, w.end) for w in metrics.words]
        
        return {
            "raw_response": response,
            "transcript": metrics.transcript,
            "overall_confidence": metrics.overall_confidence,
            "word_confidences": word_confidences,
            "metrics": {
                "mean_confidence": metrics.mean_word_confidence,
                "median_confidence": metrics.median_word_confidence,
                "std_confidence": metrics.std_word_confidence,
                "min_confidence": metrics.min_word_confidence,
                "max_confidence": metrics.max_word_confidence,
                "low_confidence_ratio": metrics.low_confidence_ratio,
                "low_confidence_threshold": metrics.low_confidence_threshold,
                "total_words": metrics.total_words,
                "transcript_duration": metrics.transcript_duration,
                "transcript_completeness": metrics.transcript_completeness,
            },
            "low_confidence_words": [
                (w.word, w.confidence) 
                for w in metrics.words 
                if w.confidence < metrics.low_confidence_threshold
            ],
        }
    
    def transcribe_url_with_confidence(
        self,
        url: str,
        include_utterances: bool = True,
    ) -> dict:
        """
        Transcribe from URL and extract word-level confidence metrics.
        
        Args:
            url: URL of audio file.
            include_utterances: Whether to include utterances.
            
        Returns:
            Dictionary with transcript, confidence metrics, and raw response.
        """
        # Get transcription
        response = self.transcriber.transcribe_url(url, include_utterances)
        
        # Extract word confidences
        metrics = self.confidence_extractor.compute_metrics(response)
        
        # Extract words as list of tuples
        # Updated to include timing for alignment: (word, confidence, start, end)
        word_confidences = [(w.word, w.confidence, w.start, w.end) for w in metrics.words]
        
        return {
            "raw_response": response,
            "transcript": metrics.transcript,
            "overall_confidence": metrics.overall_confidence,
            "word_confidences": word_confidences,
            "metrics": {
                "mean_confidence": metrics.mean_word_confidence,
                "median_confidence": metrics.median_word_confidence,
                "std_confidence": metrics.std_word_confidence,
                "min_confidence": metrics.min_word_confidence,
                "max_confidence": metrics.max_word_confidence,
                "low_confidence_ratio": metrics.low_confidence_ratio,
                "low_confidence_threshold": metrics.low_confidence_threshold,
                "total_words": metrics.total_words,
                "transcript_duration": metrics.transcript_duration,
                "transcript_completeness": metrics.transcript_completeness,
            },
            "low_confidence_words": [
                (w.word, w.confidence) 
                for w in metrics.words 
                if w.confidence < metrics.low_confidence_threshold
            ],
        }
    
    def get_confidence_report(self, result: dict) -> str:
        """
        Generate a human-readable confidence report.
        
        Args:
            result: Result from transcribe_file_with_confidence or transcribe_url_with_confidence.
            
        Returns:
            Formatted report string.
        """
        metrics = result["metrics"]
        words = result["word_confidences"]
        
        lines = [
            "=" * 80,
            "DEEPGRAM TRANSCRIPTION WITH CONFIDENCE METRICS",
            "=" * 80,
            "",
            f"Transcript: {result['transcript']}",
            "",
            "CONFIDENCE METRICS:",
            f"  Overall Confidence:       {result['overall_confidence']:.3f}",
            f"  Mean Word Confidence:     {metrics['mean_confidence']:.3f}",
            f"  Median Word Confidence:   {metrics['median_confidence']:.3f}",
            f"  Std Dev:                  {metrics['std_confidence']:.3f}",
            f"  Min/Max:                  {metrics['min_confidence']:.3f} / {metrics['max_confidence']:.3f}",
            f"  Low-Confidence Words:     {len(result['low_confidence_words'])} ({metrics['low_confidence_ratio']:.1%})",
            f"  Total Words:              {metrics['total_words']}",
            "",
            f"TIMING:",
            f"  Transcript Duration:      {metrics['transcript_duration']:.2f}s",
            f"  Transcript Completeness:  {metrics['transcript_completeness']:.1%}",
            "",
        ]
        
        if result["low_confidence_words"]:
            lines.append("LOW-CONFIDENCE WORDS (< 0.75):")
            for word, conf in result["low_confidence_words"]:
                lines.append(f"  • '{word}': {conf:.3f}")
            lines.append("")
        
        lines.append("WORD-LEVEL BREAKDOWN:")
        lines.append(self.confidence_extractor.format_words_with_confidence(
            [
                type('WordConfidence', (), {
                    'word': w[0], 
                    'confidence': w[1], 
                    'start': 0, 
                    'end': 0
                })()
                for w in words
            ],
            include_timing=False
        ))
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize with enhanced confidence extraction
    dg = DeepGramWithConfidence(api_key=os.getenv("DEEPGRAM_API_KEY"))
    
    # Transcribe with confidence metrics
    result = dg.transcribe_file_with_confidence("audio.wav")
    
    # Print report
    print(dg.get_confidence_report(result))
    
    # Access programmatically
    print(f"\nTranscript: {result['transcript']}")
    print(f"Mean Confidence: {result['metrics']['mean_confidence']:.3f}")
    print(f"Low-Confidence Ratio: {result['metrics']['low_confidence_ratio']:.1%}")

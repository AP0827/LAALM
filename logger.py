"""
Logging utilities for LAALM pipeline.
Provides structured logging for transcripts, metrics, and confidence scores.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any


class LAALMLogger:
    """Comprehensive logger for LAALM pipeline outputs and metrics."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize logger with separate log files for different data types."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup different loggers
        self.transcript_logger = self._setup_logger(
            "transcripts", 
            os.path.join(log_dir, f"transcripts_{self.session_id}.log")
        )
        
        self.metrics_logger = self._setup_logger(
            "metrics",
            os.path.join(log_dir, f"metrics_{self.session_id}.log")
        )
        
        self.confidence_logger = self._setup_logger(
            "confidence",
            os.path.join(log_dir, f"confidence_{self.session_id}.log")
        )
        
        # JSON output file for structured data
        self.json_output = os.path.join(log_dir, f"results_{self.session_id}.json")
        self.results_history = []
        
    def _setup_logger(self, name: str, log_file: str) -> logging.Logger:
        """Setup individual logger with file handler."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers = []
        
        # File handler
        handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def log_transcripts(self, result: Dict[str, Any], video_file: str = ""):
        """Log all transcription outputs."""
        separator = "=" * 80
        
        self.transcript_logger.info(separator)
        self.transcript_logger.info(f"VIDEO: {video_file}")
        self.transcript_logger.info(separator)
        self.transcript_logger.info(f"Audio (DeepGram):  {result['deepgram']['transcript']}")
        self.transcript_logger.info(f"Video (auto_avsr): {result['avsr']['transcript']}")
        self.transcript_logger.info(f"Combined:          {result['combined_transcript']}")
        self.transcript_logger.info(f"Final (Groq):      {result['final_transcript']}")
        self.transcript_logger.info(separator + "\n")
    
    def log_confidence_scores(self, result: Dict[str, Any], video_file: str = ""):
        """Log detailed confidence scores."""
        separator = "=" * 80
        
        self.confidence_logger.info(separator)
        self.confidence_logger.info(f"VIDEO: {video_file}")
        self.confidence_logger.info(separator)
        
        # Overall confidences
        self.confidence_logger.info("OVERALL CONFIDENCE SCORES:")
        self.confidence_logger.info(f"  Audio:  {result['deepgram']['overall_confidence']:.3f}")
        self.confidence_logger.info(f"  Video:  {result['avsr']['overall_confidence']:.3f}")
        self.confidence_logger.info(f"  Final:  {result['groq']['confidence']:.3f}")
        
        # Word-level confidences
        self.confidence_logger.info("\nWORD-LEVEL CONFIDENCE SCORES:")
        self.confidence_logger.info(f"  Audio words: {result['deepgram']['word_confidences']}")
        self.confidence_logger.info(f"  Video words: {result['avsr']['word_confidences']}")
        
        # Combined analysis
        if 'combined_words' in result:
            self.confidence_logger.info("\nCOMBINED WORD ANALYSIS:")
            for word_data in result['combined_words']:
                word = word_data['word']
                dg_conf = word_data['deepgram']['confidence']
                av_conf = word_data['avsr']['confidence']
                avg_conf = word_data['average_confidence']
                agreement = "✓" if word_data['agreement'] else "✗"
                low_conf = "⚠" if word_data['low_confidence'] else "✓"
                
                self.confidence_logger.info(
                    f"  {word:15s} | DG: {dg_conf:.2f} | VSR: {av_conf:.2f} | "
                    f"Avg: {avg_conf:.2f} | Agree: {agreement} | Conf: {low_conf}"
                )
        
        self.confidence_logger.info(separator + "\n")
    
    def log_metrics(self, result: Dict[str, Any], video_file: str = "", 
                   ground_truth: str = None):
        """Log performance metrics."""
        separator = "=" * 80
        
        self.metrics_logger.info(separator)
        self.metrics_logger.info(f"VIDEO: {video_file}")
        self.metrics_logger.info(separator)
        
        # Timestamp and session
        self.metrics_logger.info(f"Session ID: {self.session_id}")
        self.metrics_logger.info(f"Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Transcription lengths
        audio_words = len(result['deepgram']['transcript'].split())
        video_words = len(result['avsr']['transcript'].split())
        final_words = len(result['final_transcript'].split())
        
        self.metrics_logger.info("\nTRANSCRIPT STATISTICS:")
        self.metrics_logger.info(f"  Audio word count:  {audio_words}")
        self.metrics_logger.info(f"  Video word count:  {video_words}")
        self.metrics_logger.info(f"  Final word count:  {final_words}")
        
        # Agreement metrics
        if 'combined_words' in result:
            total_words = len(result['combined_words'])
            agreements = sum(1 for w in result['combined_words'] if w['agreement'])
            disagreements = total_words - agreements
            low_conf = sum(1 for w in result['combined_words'] if w['low_confidence'])
            
            self.metrics_logger.info("\nAGREEMENT METRICS:")
            self.metrics_logger.info(f"  Total words:       {total_words}")
            self.metrics_logger.info(f"  Agreements:        {agreements} ({agreements/total_words*100:.1f}%)")
            self.metrics_logger.info(f"  Disagreements:     {disagreements} ({disagreements/total_words*100:.1f}%)")
            self.metrics_logger.info(f"  Low confidence:    {low_conf} ({low_conf/total_words*100:.1f}%)")
        
        # Corrections applied
        if 'groq' in result and 'corrections' in result['groq']:
            num_corrections = len(result['groq']['corrections'])
            self.metrics_logger.info(f"\nCORRECTIONS APPLIED: {num_corrections}")
            for i, corr in enumerate(result['groq']['corrections'], 1):
                self.metrics_logger.info(f"  {i}. '{corr.get('original_phrase', 'N/A')}' → "
                                        f"'{corr.get('corrected_phrase', 'N/A')}'")
                self.metrics_logger.info(f"     Reason: {corr.get('reason', 'N/A')}")
        
        # Ground truth comparison (if provided)
        if ground_truth:
            self.metrics_logger.info(f"\nGROUND TRUTH: {ground_truth}")
            # Could add WER/CER calculation here
        
        self.metrics_logger.info(separator + "\n")
    
    def save_json_result(self, result: Dict[str, Any], video_file: str = ""):
        """Save structured result to JSON file."""
        # Create a JSON-serializable copy
        json_safe_result = self._make_json_safe(result)
        
        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "video_file": video_file,
            "results": json_safe_result
        }
        
        self.results_history.append(result_entry)
        
        # Write to JSON file
        with open(self.json_output, 'w', encoding='utf-8') as f:
            json.dump(self.results_history, f, indent=2, ensure_ascii=False)
    
    def _make_json_safe(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert other objects to string representation
            return str(obj)
    
    def log_all(self, result: Dict[str, Any], video_file: str = "", 
                ground_truth: str = None):
        """Convenience method to log everything at once."""
        self.log_transcripts(result, video_file)
        self.log_confidence_scores(result, video_file)
        self.log_metrics(result, video_file, ground_truth)
        self.save_json_result(result, video_file)
        
        print(f"\n📝 Logs saved to {self.log_dir}/")
        print(f"   - Transcripts: transcripts_{self.session_id}.log")
        print(f"   - Confidence:  confidence_{self.session_id}.log")
        print(f"   - Metrics:     metrics_{self.session_id}.log")
        print(f"   - JSON data:   results_{self.session_id}.json")


# Singleton instance
_logger_instance = None

def get_logger(log_dir: str = "logs") -> LAALMLogger:
    """Get or create the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = LAALMLogger(log_dir)
    return _logger_instance

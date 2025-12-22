"""Caption formatting module for WebVTT and SRT formats."""

from typing import Dict, Any, List, Optional
from pathlib import Path
from deepgram_captions import DeepgramConverter, webvtt, srt


class CaptionFormatter:
    """Formats transcription responses into WebVTT or SRT captions."""
    
    @staticmethod
    def _to_dict(response):
        """Convert response object to dict if needed."""
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        elif hasattr(response, 'dict'):
            return response.dict()
        else:
            return response
    
    @staticmethod
    def format_webvtt(response: Dict[str, Any]) -> str:
        """
        Convert DeepGram response to WebVTT format.
        
        Args:
            response: DeepGram API response with utterances.
            
        Returns:
            WebVTT formatted string.
        """
        response_dict = CaptionFormatter._to_dict(response)
        transcription = DeepgramConverter(response_dict)
        captions = webvtt(transcription)
        return captions
    
    @staticmethod
    def format_srt(response: Dict[str, Any]) -> str:
        """
        Convert DeepGram response to SRT format.
        
        Args:
            response: DeepGram API response with utterances.
            
        Returns:
            SRT formatted string.
        """
        response_dict = CaptionFormatter._to_dict(response)
        transcription = DeepgramConverter(response_dict)
        captions = srt(transcription)
        return captions
    
    @staticmethod
    def save_captions(
        caption_content: str,
        output_path: str,
        format_type: str = "vtt",
    ) -> Path:
        """
        Save captions to a file.
        
        Args:
            caption_content: Formatted caption string.
            output_path: Path where to save the caption file.
            format_type: Caption format type ('vtt' or 'srt').
            
        Returns:
            Path object pointing to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(caption_content)
        
        return output_path
    
    @staticmethod
    def response_to_captions(
        response: Dict[str, Any],
        output_path: str,
        format_type: str = "vtt",
    ) -> Path:
        """
        Convert DeepGram response directly to caption file.
        
        Args:
            response: DeepGram API response with utterances.
            output_path: Path where to save the caption file.
            format_type: Caption format type ('vtt' or 'srt'). Defaults to 'vtt'.
            
        Returns:
            Path object pointing to the saved file.
            
        Raises:
            ValueError: If format_type is not 'vtt' or 'srt'.
        """
        if format_type.lower() == "vtt":
            captions = CaptionFormatter.format_webvtt(response)
        elif format_type.lower() == "srt":
            captions = CaptionFormatter.format_srt(response)
        else:
            raise ValueError(f"Unsupported format type: {format_type}. Use 'vtt' or 'srt'.")
        
        return CaptionFormatter.save_captions(captions, output_path, format_type)

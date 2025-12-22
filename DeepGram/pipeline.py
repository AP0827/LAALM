"""Main pipeline module for audio transcription and captioning."""

from typing import Optional, Tuple
from pathlib import Path

from .transcriber import AudioTranscriber
from .caption_formatter import CaptionFormatter
from .config import DeepGramConfig


class TranscriptionPipeline:
    """End-to-end pipeline for audio transcription and caption generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the transcription pipeline.
        
        Args:
            api_key: DeepGram API key. If None, uses DEEPGRAM_API_KEY env var.
        """
        config = DeepGramConfig(api_key=api_key)
        self.transcriber = AudioTranscriber(config=config)
        self.formatter = CaptionFormatter()
    
    def transcribe_and_caption_file(
        self,
        audio_file_path: str,
        output_dir: str = "./output",
        caption_format: str = "vtt",
        save_transcript: bool = True,
    ) -> dict:
        """
        Transcribe a local audio file and generate captions.
        
        Args:
            audio_file_path: Path to the audio file to transcribe.
            output_dir: Directory to save output files.
            caption_format: Caption format ('vtt' or 'srt'). Defaults to 'vtt'.
            save_transcript: Whether to save plain text transcript.
            
        Returns:
            Dictionary containing paths to generated files and transcript text.
            
        Example:
            >>> pipeline = TranscriptionPipeline()
            >>> result = pipeline.transcribe_and_caption_file("audio.mp3")
            >>> print(result['caption_file'])
            >>> print(result['transcript'])
        """
        audio_path = Path(audio_file_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Transcribe audio
        print(f"Transcribing audio file: {audio_file_path}")
        response = self.transcriber.transcribe_file(
            audio_file_path,
            include_utterances=True,
        )
        
        # Get transcript text
        transcript = self.transcriber.get_transcript_text(response)
        
        # Generate captions
        print(f"Generating {caption_format.upper()} captions...")
        caption_filename = f"{audio_path.stem}.{caption_format}"
        caption_path = output_path / caption_filename
        self.formatter.response_to_captions(response, str(caption_path), caption_format)
        
        result = {
            "caption_file": str(caption_path),
            "transcript": transcript,
        }
        
        # Save transcript if requested
        if save_transcript:
            transcript_path = output_path / f"{audio_path.stem}_transcript.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            result["transcript_file"] = str(transcript_path)
        
        print(f"Transcription complete!")
        print(f"  Caption file: {caption_path}")
        if save_transcript:
            print(f"  Transcript file: {result['transcript_file']}")
        
        return result
    
    def transcribe_and_caption_url(
        self,
        audio_url: str,
        output_filename: str,
        output_dir: str = "./output",
        caption_format: str = "vtt",
        save_transcript: bool = True,
    ) -> dict:
        """
        Transcribe audio from a URL and generate captions.
        
        Args:
            audio_url: URL of the audio file to transcribe.
            output_filename: Base filename for output files (without extension).
            output_dir: Directory to save output files.
            caption_format: Caption format ('vtt' or 'srt'). Defaults to 'vtt'.
            save_transcript: Whether to save plain text transcript.
            
        Returns:
            Dictionary containing paths to generated files and transcript text.
            
        Example:
            >>> pipeline = TranscriptionPipeline()
            >>> result = pipeline.transcribe_and_caption_url(
            ...     "https://example.com/audio.wav",
            ...     "my_audio"
            ... )
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Transcribe audio from URL
        print(f"Transcribing audio from URL: {audio_url}")
        response = self.transcriber.transcribe_url(
            audio_url,
            include_utterances=True,
        )
        
        # Get transcript text
        transcript = self.transcriber.get_transcript_text(response)
        
        # Generate captions
        print(f"Generating {caption_format.upper()} captions...")
        caption_filename = f"{output_filename}.{caption_format}"
        caption_path = output_path / caption_filename
        self.formatter.response_to_captions(response, str(caption_path), caption_format)
        
        result = {
            "caption_file": str(caption_path),
            "transcript": transcript,
        }
        
        # Save transcript if requested
        if save_transcript:
            transcript_path = output_path / f"{output_filename}_transcript.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            result["transcript_file"] = str(transcript_path)
        
        print(f"Transcription complete!")
        print(f"  Caption file: {caption_path}")
        if save_transcript:
            print(f"  Transcript file: {result['transcript_file']}")
        
        return result

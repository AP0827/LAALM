"""Example usage of the DeepGram transcription module."""

import sys
from pathlib import Path

from DeepGram import AudioTranscriber
from DeepGram.caption_formatter import CaptionFormatter
from DeepGram.pipeline import TranscriptionPipeline


def example_file_transcription():
    """Example: Transcribe a local audio file and generate captions."""
    print("=" * 60)
    print("Example 1: Local File Transcription")
    print("=" * 60)
    
    # Using the full pipeline (recommended for most use cases)
    pipeline = TranscriptionPipeline()
    
    # Replace with your actual audio file path
    audio_file = "path/to/your/audio.mp3"
    
    result = pipeline.transcribe_and_caption_file(
        audio_file_path=audio_file,
        output_dir="./captions",
        caption_format="vtt",  # or "srt"
        save_transcript=True,
    )
    
    print("\nResult:")
    print(f"Caption file: {result['caption_file']}")
    print(f"Transcript: {result['transcript'][:200]}...")


def example_url_transcription():
    """Example: Transcribe audio from a URL and generate captions."""
    print("\n" + "=" * 60)
    print("Example 2: URL Transcription")
    print("=" * 60)
    
    pipeline = TranscriptionPipeline()
    
    # Using DeepGram's example audio
    audio_url = "https://static.deepgram.com/examples/deep-learning-podcast-clip.wav"
    
    result = pipeline.transcribe_and_caption_url(
        audio_url=audio_url,
        output_filename="podcast_clip",
        output_dir="./captions",
        caption_format="vtt",
        save_transcript=True,
    )
    
    print("\nResult:")
    print(f"Caption file: {result['caption_file']}")
    print(f"Transcript: {result['transcript'][:200]}...")


def example_manual_transcription():
    """Example: Using individual components for more control."""
    print("\n" + "=" * 60)
    print("Example 3: Manual Component Usage")
    print("=" * 60)
    
    # Initialize transcriber
    transcriber = AudioTranscriber()
    
    # Replace with your actual audio file path or URL
    audio_url = "https://static.deepgram.com/examples/deep-learning-podcast-clip.wav"
    
    # Transcribe
    print(f"Transcribing: {audio_url}")
    response = transcriber.transcribe_url(audio_url, include_utterances=True)
    
    # Extract transcript
    transcript = transcriber.get_transcript_text(response)
    print(f"\nTranscript: {transcript[:200]}...")
    
    # Generate captions in multiple formats
    formatter = CaptionFormatter()
    
    # Save as WebVTT
    formatter.response_to_captions(
        response,
        "./captions/output.vtt",
        format_type="vtt",
    )
    print("Saved WebVTT captions to ./captions/output.vtt")
    
    # Save as SRT
    formatter.response_to_captions(
        response,
        "./captions/output.srt",
        format_type="srt",
    )
    print("Saved SRT captions to ./captions/output.srt")


if __name__ == "__main__":
    """
    Run examples. Set DEEPGRAM_API_KEY environment variable before running:
    
    export DEEPGRAM_API_KEY="your-api-key-here"
    python examples.py
    """
    try:
        # Uncomment the example you want to run
        
        # example_file_transcription()
        example_url_transcription()
        # example_manual_transcription()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

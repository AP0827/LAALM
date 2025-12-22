"""Quick transcription function for one-liner usage."""

from .pipeline import TranscriptionPipeline


def transcribe(audio_file: str, format_type: str = "vtt", output_dir: str = "./captions"):
    """
    Quick one-liner transcription.
    
    Args:
        audio_file: Path to audio file
        format_type: 'vtt' or 'srt' (default: 'vtt')
        output_dir: Where to save output (default: './captions')
    
    Returns:
        Dictionary with 'caption_file', 'transcript_file', and 'transcript' keys
    
    Example:
        from DeepGram.quick import transcribe
        result = transcribe("audio.mp3")
        print(result['caption_file'])
    """
    pipeline = TranscriptionPipeline()
    return pipeline.transcribe_and_caption_file(
        audio_file_path=audio_file,
        output_dir=output_dir,
        caption_format=format_type,
        save_transcript=True,
    )

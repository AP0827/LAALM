"""
DeepGram CLI entry point - allows running as: python -m DeepGram <audio_file>
"""

import sys
import argparse
from pathlib import Path

from .pipeline import TranscriptionPipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files and generate captions using DeepGram API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m DeepGram audio.mp3
  python -m DeepGram audio.wav --format srt
  python -m DeepGram audio.mp3 --output-dir ./captions
        """
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to audio file to transcribe"
    )
    
    parser.add_argument(
        "--format",
        choices=["vtt", "srt"],
        default="vtt",
        help="Caption format (default: vtt)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./captions",
        help="Directory to save output files (default: ./captions)"
    )
    
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Don't save plain text transcript"
    )
    
    args = parser.parse_args()
    
    # Validate audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)
    
    try:
        print(f"Transcribing: {args.audio_file}")
        print(f"Format: {args.format.upper()}")
        print(f"Output directory: {args.output_dir}")
        print()
        
        pipeline = TranscriptionPipeline()
        result = pipeline.transcribe_and_caption_file(
            audio_file_path=str(audio_path),
            output_dir=args.output_dir,
            caption_format=args.format,
            save_transcript=not args.no_transcript,
        )
        
        print("\n✓ Success!")
        print(f"Caption file: {result['caption_file']}")
        if not args.no_transcript:
            print(f"Transcript file: {result['transcript_file']}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

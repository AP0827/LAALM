# DeepGram Audio Transcription Module

A modular, reusable Python library for transcribing audio files and generating WebVTT/SRT captions using the DeepGram API.

## Features

- **Easy-to-use API**: Simple interfaces for common transcription tasks
- **Modular Design**: Use individual components or the full pipeline
- **Multiple Caption Formats**: Support for WebVTT and SRT caption formats
- **Flexible Input**: Transcribe from local files or URLs
- **Comprehensive Error Handling**: Clear error messages and validation
- **Well-tested**: Includes unit tests for all components

## Installation

1. Install the required dependencies:

```bash
pip install deepgram-sdk deepgram-captions
```

2. Set your DeepGram API key as an environment variable:

```bash
export DEEPGRAM_API_KEY="your-api-key-here"
```

Or pass it directly to the components:

```python
from DeepGram.pipeline import TranscriptionPipeline

pipeline = TranscriptionPipeline(api_key="your-api-key")
```

## Quick Start

### Using the Full Pipeline (Recommended)

The `TranscriptionPipeline` handles everything end-to-end:

```python
from DeepGram.pipeline import TranscriptionPipeline

# Initialize pipeline
pipeline = TranscriptionPipeline()

# Transcribe local audio file
result = pipeline.transcribe_and_caption_file(
    audio_file_path="path/to/audio.mp3",
    output_dir="./output",
    caption_format="vtt",  # or "srt"
    save_transcript=True,
)

print(f"Captions saved to: {result['caption_file']}")
print(f"Transcript: {result['transcript']}")
```

### Using Individual Components

For more control, use components separately:

```python
from DeepGram.transcriber import AudioTranscriber
from DeepGram.caption_formatter import CaptionFormatter

# Transcribe audio
transcriber = AudioTranscriber()
response = transcriber.transcribe_url(
    "https://example.com/audio.wav",
    include_utterances=True
)

# Extract text
transcript = transcriber.get_transcript_text(response)

# Generate captions
formatter = CaptionFormatter()
formatter.response_to_captions(
    response,
    output_path="output.vtt",
    format_type="vtt"
)
```

## Module Structure

### `config.py`
Configuration management for DeepGram API settings.

**Main Class**: `DeepGramConfig`
- Handles API key configuration
- Validates required settings

### `transcriber.py`
Audio transcription using DeepGram API.

**Main Class**: `AudioTranscriber`
- `transcribe_url()`: Transcribe audio from a URL
- `transcribe_file()`: Transcribe a local audio file
- `get_transcript_text()`: Extract plain text transcript
- `get_utterances()`: Extract utterances for captioning

### `caption_formatter.py`
Caption format generation and file handling.

**Main Class**: `CaptionFormatter`
- `format_webvtt()`: Convert response to WebVTT format
- `format_srt()`: Convert response to SRT format
- `save_captions()`: Save captions to file
- `response_to_captions()`: End-to-end caption generation

### `pipeline.py`
High-level transcription pipeline.

**Main Class**: `TranscriptionPipeline`
- `transcribe_and_caption_file()`: Process local audio files
- `transcribe_and_caption_url()`: Process audio from URLs

## Examples

### Example 1: Transcribe Local File

```python
from DeepGram.pipeline import TranscriptionPipeline

pipeline = TranscriptionPipeline()

result = pipeline.transcribe_and_caption_file(
    audio_file_path="interview.mp3",
    output_dir="./captions",
    caption_format="srt",
    save_transcript=True,
)
```

### Example 2: Transcribe from URL

```python
from DeepGram.pipeline import TranscriptionPipeline

pipeline = TranscriptionPipeline()

result = pipeline.transcribe_and_caption_url(
    audio_url="https://example.com/podcast.wav",
    output_filename="podcast_episode",
    output_dir="./output",
    caption_format="vtt",
)
```

### Example 3: Custom Processing

```python
from DeepGram.transcriber import AudioTranscriber
from DeepGram.caption_formatter import CaptionFormatter

transcriber = AudioTranscriber()
formatter = CaptionFormatter()

# Transcribe with custom options
response = transcriber.transcribe_file("audio.mp3", include_utterances=True)

# Do custom processing
transcript = transcriber.get_transcript_text(response)
utterances = transcriber.get_utterances(response)

# Generate captions
formatter.response_to_captions(
    response,
    "captions.vtt",
    format_type="vtt"
)
```

## Running Tests

```bash
python -m pytest DeepGram/tests.py -v
```

Or using unittest:

```bash
python -m unittest DeepGram.tests
```

## API Response Structure

The DeepGram API returns responses in the following structure:

```python
{
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": "Full transcribed text...",
                        "utterances": [
                            {
                                "start": 0.0,
                                "end": 1.5,
                                "confidence": 0.95,
                                "transcript": "Hello"
                            },
                            ...
                        ]
                    }
                ]
            }
        ]
    }
}
```

## Caption File Formats

### WebVTT Format (`.vtt`)

```
WEBVTT

00:00:00.000 --> 00:00:02.000
Hello, this is a test

00:00:02.000 --> 00:00:04.500
of the caption system
```

### SRT Format (`.srt`)

```
1
00:00:00,000 --> 00:00:02,000
Hello, this is a test

2
00:00:02,000 --> 00:00:04,500
of the caption system
```

## Error Handling

The module provides clear error messages for common issues:

- Missing API key: "DeepGram API key not provided"
- File not found: "Audio file not found: /path/to/file"
- Unsupported format: "Unsupported format type: xyz"

## Performance Considerations

- **File Size**: Works with audio files up to DeepGram's limits
- **Processing Time**: Depends on audio duration and API response times
- **Rate Limiting**: Check DeepGram's rate limits for your API plan

## Dependencies

- `deepgram-sdk`: Official DeepGram Python SDK
- `deepgram-captions`: DeepGram's caption formatting library

## License

Same license as the parent LAALM project.

## Support

For DeepGram API documentation, visit: https://developers.deepgram.com/

For issues with this module, check the examples or review the test cases for usage patterns.

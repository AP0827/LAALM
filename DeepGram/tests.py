"""Unit tests for DeepGram transcription module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from DeepGram.config import DeepGramConfig
from DeepGram.transcriber import AudioTranscriber
from DeepGram.caption_formatter import CaptionFormatter
from DeepGram.pipeline import TranscriptionPipeline


class TestDeepGramConfig(unittest.TestCase):
    """Test DeepGramConfig class."""
    
    def test_config_with_api_key(self):
        """Test config initialization with explicit API key."""
        config = DeepGramConfig(api_key="test-key")
        self.assertEqual(config.api_key, "test-key")
    
    def test_config_missing_api_key(self):
        """Test config initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                DeepGramConfig()


class TestAudioTranscriber(unittest.TestCase):
    """Test AudioTranscriber class."""
    
    @patch("DeepGram.transcriber.DeepgramClient")
    def setUp(self, mock_client_class):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        mock_client_class.return_value = self.mock_client
        self.config = DeepGramConfig(api_key="test-key")
        self.transcriber = AudioTranscriber(config=self.config)
    
    @patch("DeepGram.transcriber.DeepgramClient")
    def test_transcribe_url(self, mock_client_class):
        """Test URL transcription."""
        mock_response = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "Test transcript",
                                "utterances": []
                            }
                        ]
                    }
                ]
            }
        }
        
        self.mock_client.listen.v1.media.transcribe_url.return_value = mock_response
        
        response = self.transcriber.transcribe_url("http://example.com/audio.wav")
        self.assertEqual(response, mock_response)
    
    def test_get_transcript_text(self):
        """Test transcript extraction from response."""
        response = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "Hello world",
                                "utterances": []
                            }
                        ]
                    }
                ]
            }
        }
        
        transcript = self.transcriber.get_transcript_text(response)
        self.assertEqual(transcript, "Hello world")
    
    def test_get_utterances(self):
        """Test utterances extraction from response."""
        utterances_data = [
            {
                "start": 0.0,
                "end": 1.0,
                "confidence": 0.95,
                "transcript": "Hello",
            }
        ]
        
        response = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "Hello world",
                                "utterances": utterances_data
                            }
                        ]
                    }
                ]
            }
        }
        
        utterances = self.transcriber.get_utterances(response)
        self.assertEqual(utterances, utterances_data)


class TestCaptionFormatter(unittest.TestCase):
    """Test CaptionFormatter class."""
    
    @patch("DeepGram.caption_formatter.DeepgramConverter")
    @patch("DeepGram.caption_formatter.webvtt")
    def test_format_webvtt(self, mock_webvtt, mock_converter_class):
        """Test WebVTT caption formatting."""
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        mock_webvtt.return_value = "WEBVTT\n00:00:00.000 --> 00:00:01.000\nHello"
        
        response = {"results": {}}
        captions = CaptionFormatter.format_webvtt(response)
        
        self.assertIn("WEBVTT", captions)
    
    @patch("DeepGram.caption_formatter.DeepgramConverter")
    @patch("DeepGram.caption_formatter.srt")
    def test_format_srt(self, mock_srt, mock_converter_class):
        """Test SRT caption formatting."""
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        mock_srt.return_value = "1\n00:00:00,000 --> 00:00:01,000\nHello"
        
        response = {"results": {}}
        captions = CaptionFormatter.format_srt(response)
        
        self.assertIn("00:00:00,000", captions)
    
    def test_save_captions(self, tmp_path):
        """Test caption file saving."""
        output_file = tmp_path / "test.vtt"
        caption_content = "WEBVTT\n00:00:00.000 --> 00:00:01.000\nHello"
        
        result_path = CaptionFormatter.save_captions(
            caption_content,
            str(output_file),
            format_type="vtt"
        )
        
        self.assertTrue(result_path.exists())
        with open(result_path) as f:
            content = f.read()
        self.assertEqual(content, caption_content)
    
    @patch("DeepGram.caption_formatter.CaptionFormatter.format_webvtt")
    def test_response_to_captions_vtt(self, mock_format_vtt, tmp_path):
        """Test converting response to VTT captions."""
        mock_format_vtt.return_value = "WEBVTT\n..."
        output_file = tmp_path / "test.vtt"
        
        response = {"results": {}}
        result = CaptionFormatter.response_to_captions(
            response,
            str(output_file),
            format_type="vtt"
        )
        
        self.assertTrue(result.exists())


if __name__ == "__main__":
    unittest.main()

"""DeepGram audio transcription module."""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from deepgram import DeepgramClient

from .config import DeepGramConfig


class AudioTranscriber:
    """Handles audio transcription using DeepGram API."""
    
    def __init__(self, config: Optional[DeepGramConfig] = None):
        """
        Initialize the audio transcriber.
        
        Args:
            config: DeepGramConfig instance. If None, creates with default settings.
        """
        if config is None:
            config = DeepGramConfig()
        
        self.config = config
        self.client = DeepgramClient(api_key=config.api_key)
    
    def transcribe_url(self, url: str, include_utterances: bool = True) -> Dict[str, Any]:
        """
        Transcribe audio from a URL.
        
        Args:
            url: URL of the audio file to transcribe.
            include_utterances: Whether to include utterances (for captioning).
            
        Returns:
            DeepGram API response with transcription data.
        """
        options = {
            "model": "nova-2",
            "smart_format": True,
            "utterances": include_utterances,
        }
        
        response = self.client.listen.v1.media.transcribe_url(url=url, **options)
        return response
    
    def transcribe_file(
        self,
        file_path: str,
        include_utterances: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribe audio from a local file.
        
        Args:
            file_path: Path to the local audio file.
            include_utterances: Whether to include utterances (for captioning).
            
        Returns:
            DeepGram API response with transcription data.
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        options = {
            "model": "nova-2",
            "smart_format": True,
            "utterances": include_utterances,
        }
        
        with open(file_path, "rb") as audio_file:
            response = self.client.listen.v1.media.transcribe_file(
                request=audio_file.read(),
                **options,
            )
        
        return response
    
    def get_transcript_text(self, response: Dict[str, Any]) -> str:
        """
        Extract plain text transcript from API response.
        
        Args:
            response: DeepGram API response (dict or ListenV1Response object).
            
        Returns:
            Plain text transcript.
        """
        try:
            # Convert response object to dict if needed
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
            elif hasattr(response, 'dict'):
                response_dict = response.dict()
            else:
                response_dict = response
            
            results = response_dict.get("results", {})
            channels = results.get("channels", [])
            
            if not channels:
                return ""
            
            alternatives = channels[0].get("alternatives", [])
            if not alternatives:
                return ""
            
            transcript = alternatives[0].get("transcript", "")
            return transcript
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise ValueError(f"Failed to extract transcript from response: {e}")
    
    def get_utterances(self, response: Dict[str, Any]) -> list:
        """
        Extract utterances from API response for captioning.
        
        Args:
            response: DeepGram API response (dict or ListenV1Response object).
            
        Returns:
            List of utterance dictionaries with timing information.
        """
        try:
            # Convert response object to dict if needed
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
            elif hasattr(response, 'dict'):
                response_dict = response.dict()
            else:
                response_dict = response
            
            results = response_dict.get("results", {})
            channels = results.get("channels", [])
            
            if not channels:
                return []
            
            alternatives = channels[0].get("alternatives", [])
            if not alternatives:
                return []
            
            utterances = alternatives[0].get("utterances", [])
            return utterances
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise ValueError(f"Failed to extract utterances from response: {e}")

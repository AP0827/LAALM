"""Configuration for DeepGram API."""

import os
from typing import Optional


class DeepGramConfig:
    """Configuration class for DeepGram API settings."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DeepGram configuration.
        
        Args:
            api_key: DeepGram API key. If not provided, will use DEEPGRAM_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepGram API key not provided. "
                "Set DEEPGRAM_API_KEY environment variable or pass api_key parameter."
            )
    
    def to_dict(self) -> dict:
        """Return configuration as dictionary."""
        return {
            "api_key": self.api_key,
        }

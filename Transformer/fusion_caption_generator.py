"""
Fusion Caption Generator

Generates timestamped captions (SRT/VTT) from the fused multi-modal transcript.
Combines the fusion layer's word choices with DeepGram's timestamp metadata.

Author: LAALM Project
Date: January 2026
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import re


class FusionCaptionGenerator:
    """
    Generates timestamped captions from fused transcripts.
    """
    
    def __init__(self, max_chars_per_line: int = 42, max_lines: int = 2):
        """
        Initialize caption generator.
        
        Args:
            max_chars_per_line: Maximum characters per caption line
            max_lines: Maximum number of lines per caption
        """
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
    
    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        """
        Format seconds as SRT timestamp (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            SRT timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        """
        Format seconds as WebVTT timestamp (HH:MM:SS.mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            WebVTT timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def _split_into_lines(self, text: str) -> List[str]:
        """
        Split text into caption-friendly lines.
        
        Args:
            text: Caption text
            
        Returns:
            List of lines
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + (1 if current_line else 0)  # +1 for space
            
            if current_length + word_length > self.max_chars_per_line:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    # Word is too long, split it
                    lines.append(word[:self.max_chars_per_line])
                    current_line = []
                    current_length = 0
            else:
                current_line.append(word)
                current_length += word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Limit to max_lines
        return lines[:self.max_lines]
    
    def generate_srt(
        self,
        final_transcript: str,
        word_data: List[Dict[str, Any]]
    ) -> str:
        """
        Generate SRT captions from fused transcript and word data.
        
        Args:
            final_transcript: Final corrected transcript (from Groq)
            word_data: List of word dicts with timing info from DeepGram
                       Each dict should have: word, start, end
            
        Returns:
            SRT formatted string
        """
        if not word_data:
            return ""
        
        # Group words into caption chunks (every ~5 words or punctuation-based)
        captions = []
        caption_id = 1
        
        # Split transcript by punctuation for natural breaks
        sentences = re.split(r'([.!?;:])', final_transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_text = ""
        start_time = None
        
        for i, word_info in enumerate(word_data):
            word = word_info.get('word', '')
            start = word_info.get('start', 0)
            end = word_info.get('end', 0)
            
            if start_time is None:
                start_time = start
            
            current_text += word + " "
            
            # Create caption at punctuation or every 5 words
            should_break = (
                i == len(word_data) - 1 or  # Last word
                len(current_text.split()) >= 5 or  # Max 5 words per caption
                any(p in word for p in ['.', '!', '?', ';'])  # Punctuation
            )
            
            if should_break and current_text.strip():
                caption_text = current_text.strip()
                lines = self._split_into_lines(caption_text)
                
                srt_entry = f"{caption_id}\n"
                srt_entry += f"{self._format_timestamp_srt(start_time)} --> {self._format_timestamp_srt(end)}\n"
                srt_entry += '\n'.join(lines) + "\n"
                
                captions.append(srt_entry)
                caption_id += 1
                current_text = ""
                start_time = None
        
        return '\n'.join(captions)
    
    def generate_vtt(
        self,
        final_transcript: str,
        word_data: List[Dict[str, Any]]
    ) -> str:
        """
        Generate WebVTT captions from fused transcript and word data.
        
        Args:
            final_transcript: Final corrected transcript (from Groq)
            word_data: List of word dicts with timing info from DeepGram
            
        Returns:
            WebVTT formatted string
        """
        if not word_data:
            return "WEBVTT\n\n"
        
        # Group words into caption chunks
        captions = []
        
        current_text = ""
        start_time = None
        
        for i, word_info in enumerate(word_data):
            word = word_info.get('word', '')
            start = word_info.get('start', 0)
            end = word_info.get('end', 0)
            
            if start_time is None:
                start_time = start
            
            current_text += word + " "
            
            # Create caption at punctuation or every 5 words
            should_break = (
                i == len(word_data) - 1 or
                len(current_text.split()) >= 5 or
                any(p in word for p in ['.', '!', '?', ';'])
            )
            
            if should_break and current_text.strip():
                caption_text = current_text.strip()
                lines = self._split_into_lines(caption_text)
                
                vtt_entry = f"{self._format_timestamp_vtt(start_time)} --> {self._format_timestamp_vtt(end)}\n"
                vtt_entry += '\n'.join(lines) + "\n"
                
                captions.append(vtt_entry)
                current_text = ""
                start_time = None
        
        return "WEBVTT\n\n" + '\n'.join(captions)
    
    def save_captions(
        self,
        final_transcript: str,
        word_data: List[Dict[str, Any]],
        output_path: str,
        format_type: str = 'vtt'
    ) -> Path:
        """
        Save captions to file.
        
        Args:
            final_transcript: Final corrected transcript
            word_data: Word timing data from DeepGram
            output_path: Path to save file
            format_type: 'vtt' or 'srt'
            
        Returns:
            Path to saved file
        """
        if format_type.lower() == 'vtt':
            content = self.generate_vtt(final_transcript, word_data)
        elif format_type.lower() == 'srt':
            content = self.generate_srt(final_transcript, word_data)
        else:
            raise ValueError(f"Unsupported format: {format_type}. Use 'vtt' or 'srt'.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return output_path


# Example usage
if __name__ == "__main__":
    # Sample data
    final_transcript = "Lay white with zed nine soon."
    word_data = [
        {'word': 'Lay', 'start': 0.8, 'end': 0.96},
        {'word': 'white', 'start': 0.96, 'end': 1.28},
        {'word': 'with', 'start': 1.28, 'end': 1.52},
        {'word': 'zed', 'start': 1.52, 'end': 1.76},
        {'word': 'nine', 'start': 1.76, 'end': 2.0},
        {'word': 'soon', 'start': 2.0, 'end': 2.5},
    ]
    
    generator = FusionCaptionGenerator()
    
    print("=== SRT ===")
    print(generator.generate_srt(final_transcript, word_data))
    
    print("\n=== VTT ===")
    print(generator.generate_vtt(final_transcript, word_data))

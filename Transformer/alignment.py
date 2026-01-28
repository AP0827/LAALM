"""
Needleman-Wunsch Alignment for Multi-Modal Fusion

This module implements robust sequence alignment to synchronize audio and visual
word lists. It uses the Needleman-Wunsch global alignment algorithm to insert
gaps where necessary, ensuring that corresponding words are matched even when
one modality misses a word (deletion) or adds an extra word (insertion).

Key Features:
- Standard Needleman-Wunsch implementation
- Custom scoring (match, mismatch, gap penalties)
- Support for detailed word objects (not just strings)

Author: LAALM Project
Date: January 2026
"""

import numpy as np
from typing import List, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class AlignedPair:
    audio_word: Optional[tuple]  # (word, conf, start, end)
    visual_word: Optional[tuple]  # (word, conf)
    score: float


class TranscriptAligner:
    """
    Aligns two transcripts using Needleman-Wunsch algorithm.
    """
    
    def __init__(
        self,
        match_score: float = 2.0,
        mismatch_penalty: float = -1.0,
        gap_penalty: float = -1.0,
        ignore_case: bool = True
    ):
        """
        Initialize aligner with scoring parameters.
        
        Args:
            match_score: Score for matching words
            mismatch_penalty: Penalty for mismatched words
            gap_penalty: Penalty for inserting a gap (insertion/deletion)
            ignore_case: Whether to ignore case when comparing words
        """
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
        self.ignore_case = ignore_case
    
    def _clean_word(self, word: str) -> str:
        """Clean word for comparison."""
        if not word:
            return ""
        w = word.strip()
        if self.ignore_case:
            w = w.lower()
        return w
    
    def _get_score(self, w1: str, w2: str) -> float:
        """Calculate alignment score between two words."""
        if self._clean_word(w1) == self._clean_word(w2):
            return self.match_score
        return self.mismatch_penalty
    
    def align(
        self, 
        audio_seq: List[Any], 
        visual_seq: List[Any]
    ) -> Tuple[List[Optional[Any]], List[Optional[Any]]]:
        """
        Perform global alignment on two sequences.
        
        Args:
            audio_seq: List of audio items. Each item must be a tuple where item[0] is the word string.
            visual_seq: List of visual items. Each item must be a tuple where item[0] is the word string.
            
        Returns:
            Tuple of (aligned_audio, aligned_visual) lists.
            Lists will have the same length. Gaps are represented by None.
        """
        n = len(audio_seq)
        m = len(visual_seq)
        
        # Initialize score matrix
        # Size is (n+1) x (m+1)
        score_matrix = np.zeros((n + 1, m + 1))
        
        # Initialize gap penalties for first row/column
        for i in range(n + 1):
            score_matrix[i][0] = i * self.gap_penalty
        for j in range(m + 1):
            score_matrix[0][j] = j * self.gap_penalty
            
        # Fill score matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Get words (0-indexed in sequence, but 1-indexed in matrix)
                w_audio = audio_seq[i-1][0]
                w_visual = visual_seq[j-1][0]
                
                match = score_matrix[i-1][j-1] + self._get_score(w_audio, w_visual)
                delete = score_matrix[i-1][j] + self.gap_penalty
                insert = score_matrix[i][j-1] + self.gap_penalty
                
                score_matrix[i][j] = max(match, delete, insert)
        
        # Traceback
        aligned_audio = []
        aligned_visual = []
        
        i, j = n, m
        
        while i > 0 and j > 0:
            w_audio = audio_seq[i-1][0]
            w_visual = visual_seq[j-1][0]
            
            score_current = score_matrix[i][j]
            score_diagonal = score_matrix[i-1][j-1]
            score_up = score_matrix[i-1][j]
            score_left = score_matrix[i][j-1]
            
            # Check if this cell came from diagonal (match/mismatch)
            if score_current == score_diagonal + self._get_score(w_audio, w_visual):
                aligned_audio.append(audio_seq[i-1])
                aligned_visual.append(visual_seq[j-1])
                i -= 1
                j -= 1
            # Check if left (gap in audio / insertion in visual)
            elif score_current == score_left + self.gap_penalty:
                aligned_audio.append(None)
                aligned_visual.append(visual_seq[j-1])
                j -= 1
            # Check if up (gap in visual / deletion in audio)
            elif score_current == score_up + self.gap_penalty:
                aligned_audio.append(audio_seq[i-1])
                aligned_visual.append(None)
                i -= 1
            else:
                # Should not happen if logic is correct, but safe fallback to diagonal
                aligned_audio.append(audio_seq[i-1])
                aligned_visual.append(visual_seq[j-1])
                i -= 1
                j -= 1
        
        # Handle remaining items (at the beginning of sequences)
        while i > 0:
            aligned_audio.append(audio_seq[i-1])
            aligned_visual.append(None)
            i -= 1
        
        while j > 0:
            aligned_audio.append(None)
            aligned_visual.append(visual_seq[j-1])
            j -= 1
        
        # Reverse to get correct order
        return aligned_audio[::-1], aligned_visual[::-1]

# Example usage
if __name__ == "__main__":
    print("Alignment Test")
    
    # Sample data
    audio = [("The", 0.9), ("quick", 0.8), ("brown", 0.9), ("fox", 0.9)]
    visual = [("The", 0.8), ("quick", 0.7), ("fox", 0.8)]  # Missed "brown"
    
    aligner = TranscriptAligner()
    a_aligned, v_aligned = aligner.align(audio, visual)
    
    for a, v in zip(a_aligned, v_aligned):
        a_str = a[0] if a else "---"
        v_str = v[0] if v else "---"
        print(f"{a_str:10} | {v_str:10}")

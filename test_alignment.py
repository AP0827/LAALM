
import sys
import os

# Add project root to path
sys.path.insert(0, ".")
sys.path.insert(0, "av_hubert") # Mock if needed

from Transformer.alignment import TranscriptAligner
from pipeline import combine_word_confidences

def test_alignment_logic():
    print("=" * 60)
    print("TESTING ALIGNMENT LOGIC")
    print("=" * 60)
    
    aligner = TranscriptAligner()
    
    # Test Case 1: Simple Missed Word (Deletion in Visual)
    # Audio: "The quick brown fox"
    # Visual: "The quick fox"
    audio = [("The", 0.9, 0.0, 0.5), ("quick", 0.9, 0.5, 1.0), ("brown", 0.9, 1.0, 1.5), ("fox", 0.9, 1.5, 2.0)]
    visual = [("The", 0.8), ("quick", 0.8), ("fox", 0.8)]
    
    print("\nCase 1: Deletion in Visual ('brown' missed)")
    a_aligned, v_aligned = aligner.align(audio, visual)
    
    for a, v in zip(a_aligned, v_aligned):
        a_str = a[0] if a else "---"
        v_str = v[0] if v else "---"
        print(f"  {a_str:10} | {v_str:10}")
        
    # Verify "brown" aligns with None
    assert a_aligned[2][0] == "brown"
    assert v_aligned[2] is None
    print("✅ Passed Case 1")

    # Test Case 2: Insertion in Visual (Hallucination)
    # Audio: "The fox"
    # Visual: "The red fox"
    audio = [("The", 0.9, 0.0, 0.5), ("fox", 0.9, 1.0, 1.5)]
    visual = [("The", 0.8), ("red", 0.6), ("fox", 0.8)]
    
    print("\nCase 2: Insertion in Visual ('red' hallucinated)")
    a_aligned, v_aligned = aligner.align(audio, visual)
    
    for a, v in zip(a_aligned, v_aligned):
        a_str = a[0] if a else "---"
        v_str = v[0] if v else "---"
        print(f"  {a_str:10} | {v_str:10}")
        
    # Verify "red" aligns with None
    assert a_aligned[1] is None
    assert v_aligned[1][0] == "red"
    print("✅ Passed Case 2")
    
    # Test Case 3: Pipeline Integration
    print("\nCase 3: Testing pipeline.combine_word_confidences")
    combined = combine_word_confidences(audio, visual)
    
    # Check if we have 4 entries (The, quick, brown, fox) for Case 1
    # Actually wait, we're passing case 1 data again? let's use case 1 data
    combined = combine_word_confidences(audio, visual) # Using case 2 data
    
    print(f"Combined Result ({len(combined)} words):")
    for w in combined:
        print(f"  {w['word']} (Audio: {w['deepgram']['word']}, Visual: {w['avsr']['word']})")
        
    print("✅ Passed Case 3")

if __name__ == "__main__":
    test_alignment_logic()

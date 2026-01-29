
import sys
import os

# Add project root to path
sys.path.insert(0, ".")

from Transformer.attention_fusion import AttentionFusion, PhoneticAnalyzer

def test_viseme_logic():
    print("=" * 60)
    print("TESTING VISEME-AWARE FUSION LOGIC")
    print("=" * 60)
    
    analyzer = PhoneticAnalyzer()
    
    # Test Case 1: Direct Conflict (Bat vs Pat)
    # Both are bilabials. Visual cannot distinguish. Audio says Bat.
    # Expected: Strong Audio Preference.
    print("\n🔍 Test 1: Bat (Audio) vs Pat (Visual)")
    conflict_bias = analyzer.check_viseme_conflict("bat", "pat")
    print(f"   Viseme Conflict Bias: {conflict_bias} (Expected: 2.0)")
    
    if conflict_bias > 1.0:
        print("   ✅ CORRECT: Strong Audio Bias detected.")
    else:
        print("   ❌ FAILED: System did not detect viseme conflict.")

    # Test Case 2: No Conflict (Cat vs Pat)
    # 'C' (Velar) vs 'P' (Bilabial). Visual CAN distinguish (Open mouth vs Closed lips).
    # Expected: No special override (0.0).
    print("\n🔍 Test 2: Cat (Audio) vs Pat (Visual)")
    conflict_bias = analyzer.check_viseme_conflict("cat", "pat")
    print(f"   Viseme Conflict Bias: {conflict_bias} (Expected: 0.0)")
    
    if conflict_bias == 0.0:
        print("   ✅ CORRECT: No invalid conflict detected.")
    else:
        print("   ❌ FAILED: System falsely flagged conflict.")
        
    # Test Case 3: Fusion Engine Integration
    print("\n🔍 Test 3: Full Fusion Engine")
    fusion = AttentionFusion()
    
    # Audio is LESS confident (0.6) than Visual (0.8)
    # But Audio says "Bat", Visual says "Pat" (Conflict!)
    # Normal logic might pick Visual. Viseme logic MUST pick Audio.
    audio_words = [("bat", 0.6)]
    visual_words = [("pat", 0.8)]
    
    result = fusion.fuse_transcripts(audio_words, visual_words)
    print(f"   Audio: 'bat' (0.6) | Visual: 'pat' (0.8)")
    print(f"   Fused Result: '{result.fused_transcript}'")
    
    if result.fused_transcript.lower() == "bat":
        print("   ✅ CORRECT: Fusion preferred Audio despite lower confidence.")
    else:
        print("   ❌ FAILED: Fusion preferred Visual (Viseme logic not applied).")

if __name__ == "__main__":
    test_viseme_logic()

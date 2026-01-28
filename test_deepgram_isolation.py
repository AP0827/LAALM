
import os
import sys
from load_env import load_env_file

# Add project root to path
sys.path.insert(0, ".")

from DeepGram.enhanced_transcriber import DeepGramWithConfidence

def test_deepgram_isolation():
    print("=" * 60)
    print("TESTING DEEPGRAM ENDPOINT ISOLATION")
    print("=" * 60)
    
    # Load env for API key
    load_env_file()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    
    if not api_key:
        print("❌ Error: DEEPGRAM_API_KEY not found in .env")
        return

    # Use the sample file we know exists
    audio_file = "samples/audio/lwwz9s.wav"
    
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file not found: {audio_file}")
        return
        
    print(f"📁 Input File: {audio_file}")
    
    try:
        dg = DeepGramWithConfidence(api_key=api_key)
        print("✅ DeepGram Client Initialized")
        
        print("🚀 Sending request to DeepGram API...")
        result = dg.transcribe_file_with_confidence(audio_file)
        
        print("\n📝 Result:")
        print(f"  Transcript: '{result['transcript']}'")
        print(f"  Confidence: {result['overall_confidence']:.3f}")
        print("\n🔍 Word-Level Details:")
        for w in result['word_confidences']:
            # Handle both formats (2-tuple or 4-tuple) just in case
            if len(w) >= 2:
                print(f"  - '{w[0]}': {w[1]:.3f}")
        
    except Exception as e:
        print(f"\n❌ DeepGram API Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deepgram_isolation()

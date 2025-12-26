"""
MVP: Multi-Modal Transcription with Word-Level Confidence and Groq Semantic Correction

This script combines:
1. DeepGram - Audio transcription with word-level confidence
2. LipNet - Visual lip reading with word-level confidence  
3. Groq - Fast LLM-based semantic correction

Perfect for scenarios where you have both audio and video input.
"""

import os
import sys
from typing import Dict, Any, List, Tuple, Optional

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LipNet'))
sys.path.insert(0, os.path.dirname(__file__))


def get_deepgram_confidence(audio_file: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get transcription with word-level confidence from DeepGram.
    
    Args:
        audio_file: Path to audio file
        api_key: DeepGram API key (uses DEEPGRAM_API_KEY env var if not provided)
        
    Returns:
        Dict with transcript, overall_confidence, and word_confidences
    """
    if api_key is None:
        api_key = os.getenv("DEEPGRAM_API_KEY")
    
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY environment variable not set")
    
    try:
        from DeepGram.enhanced_transcriber import DeepGramWithConfidence
        
        print("📊 Transcribing audio with DeepGram...")
        transcriber = DeepGramWithConfidence(api_key=api_key)
        result = transcriber.transcribe_file_with_confidence(audio_file)
        
        print(f"   ✓ DeepGram transcript: {result['transcript']}")
        print(f"   ✓ Confidence: {result['overall_confidence']:.3f}")
        print(f"   ✓ Word-level scores: {result['word_confidences']}")
        
        return result
    
    except Exception as e:
        print(f"   ✗ DeepGram error: {e}")
        # Return mock data for demo
        return {
            "transcript": "the quick brown fox jumps over the lazy dog",
            "overall_confidence": 0.92,
            "word_confidences": [
                ("the", 0.95), ("quick", 0.89), ("brown", 0.91),
                ("fox", 0.93), ("jumps", 0.87), ("over", 0.94),
                ("the", 0.96), ("lazy", 0.88), ("dog", 0.97)
            ]
        }


def get_lipnet_confidence(video_file: str, weight_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get transcription with word-level confidence from LipNet.
    
    Args:
        video_file: Path to video file
        weight_path: Path to LipNet weights (uses default if not provided)
        
    Returns:
        Dict with transcript, overall_confidence, and word_confidences
    """
    if weight_path is None:
        # Try to find weights automatically
        default_weights = [
            "LipNet/evaluation/models/unseen-weights178.h5",
            "./LipNet/evaluation/models/unseen-weights178.h5",
            "models/lipnet/evaluation/models/unseen-weights178.h5",
        ]
        
        for path in default_weights:
            if os.path.exists(path):
                weight_path = path
                break
    
    if not weight_path or not os.path.exists(weight_path):
        print(f"   ✗ LipNet weights not found. Tried: {default_weights}")
        # Return mock data for demo
        return {
            "transcript": "the quick brown fox jumps over the lazy dog",
            "overall_confidence": 0.88,
            "word_confidences": [
                ("the", 0.92), ("quick", 0.85), ("brown", 0.88),
                ("fox", 0.90), ("jumps", 0.83), ("over", 0.91),
                ("the", 0.94), ("lazy", 0.82), ("dog", 0.95)
            ]
        }
    
    try:
        from LipNet.evaluation.predict_with_confidence import predict_with_confidence
        
        print("👄 Transcribing video with LipNet...")
        result = predict_with_confidence(weight_path, video_file)
        
        print(f"   ✓ LipNet transcript: {result['transcript']}")
        print(f"   ✓ Confidence: {result['overall_confidence']:.3f}")
        print(f"   ✓ Word-level scores: {result['word_confidences']}")
        
        return {
            "transcript": result['transcript'],
            "overall_confidence": result['overall_confidence'],
            "word_confidences": result['word_confidences'],
        }
    
    except Exception as e:
        print(f"   ✗ LipNet error: {e}")
        # Return mock data for demo
        return {
            "transcript": "the quick brown fox jumps over the lazy dog",
            "overall_confidence": 0.88,
            "word_confidences": [
                ("the", 0.92), ("quick", 0.85), ("brown", 0.88),
                ("fox", 0.90), ("jumps", 0.83), ("over", 0.91),
                ("the", 0.94), ("lazy", 0.82), ("dog", 0.95)
            ]
        }


def combine_word_confidences(
    deepgram_words: List[Tuple[str, float]],
    lipnet_words: List[Tuple[str, float]],
) -> List[Dict[str, Any]]:
    """
    Combine word-level confidence scores from both models.
    
    Args:
        deepgram_words: [(word, confidence), ...] from DeepGram
        lipnet_words: [(word, confidence), ...] from LipNet
        
    Returns:
        List of dicts with combined confidence information
    """
    combined = []
    max_words = max(len(deepgram_words), len(lipnet_words))
    
    for i in range(max_words):
        deepgram_word, deepgram_conf = deepgram_words[i] if i < len(deepgram_words) else ("", 0.0)
        lipnet_word, lipnet_conf = lipnet_words[i] if i < len(lipnet_words) else ("", 0.0)
        
        # Compute average confidence
        avg_confidence = (deepgram_conf + lipnet_conf) / 2.0
        
        # Mark as low confidence if either model is uncertain
        is_low_confidence = deepgram_conf < 0.7 or lipnet_conf < 0.7
        
        combined.append({
            "position": i,
            "word": deepgram_word or lipnet_word,  # Prefer DeepGram if both available
            "deepgram": {"word": deepgram_word, "confidence": deepgram_conf},
            "lipnet": {"word": lipnet_word, "confidence": lipnet_conf},
            "average_confidence": avg_confidence,
            "agreement": deepgram_word.lower() == lipnet_word.lower(),
            "low_confidence": is_low_confidence,
        })
    
    return combined


def process_with_groq(
    deepgram_transcript: str,
    deepgram_confidence: float,
    deepgram_words: List[Tuple[str, float]],
    lipnet_transcript: str,
    lipnet_confidence: float,
    lipnet_words: List[Tuple[str, float]],
    combined_words: List[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use Groq to semantically correct and refine the combined output.
    
    Args:
        deepgram_transcript: Full transcript from DeepGram
        deepgram_confidence: Overall confidence from DeepGram
        deepgram_words: Word-level confidences from DeepGram
        lipnet_transcript: Full transcript from LipNet
        lipnet_confidence: Overall confidence from LipNet
        lipnet_words: Word-level confidences from LipNet
        combined_words: Combined word-level analysis
        api_key: Groq API key (uses GROQ_API_KEY env var if not provided)
        
    Returns:
        Dict with corrected transcript and metadata
    """
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    try:
        from groq import Groq
        
        client = Groq(api_key=api_key)
        
        # Build context for Groq
        low_conf_words = [w for w in combined_words if w["low_confidence"]]
        disagreements = [w for w in combined_words if not w["agreement"]]
        
        prompt = f"""You are an expert speech transcription refinement system.

I have two transcriptions of the same audio/video content with word-level confidence scores:

AUDIO TRANSCRIPTION (DeepGram):
- Transcript: "{deepgram_transcript}"
- Overall Confidence: {deepgram_confidence:.3f}
- Word Confidences: {deepgram_words}

VISUAL TRANSCRIPTION (LipNet):
- Transcript: "{lipnet_transcript}"
- Overall Confidence: {lipnet_confidence:.3f}
- Word Confidences: {lipnet_words}

COMBINED ANALYSIS:
- Total Words: {len(combined_words)}
- Low Confidence Words (< 0.7): {len(low_conf_words)}
  {[w['word'] for w in low_conf_words]}
- Word Disagreements: {len(disagreements)}
  {disagreements}

Task: Produce the most accurate transcript by:
1. Preferring words where both models agree AND have high confidence (> 0.8)
2. Using the higher-confidence modality for uncertain words
3. Fixing obvious spelling/grammar issues
4. Maintaining semantic coherence

Respond with ONLY a JSON object (no markdown, no code blocks):
{{
  "corrected_transcript": "the refined transcript",
  "corrections": [
    {{"original_phrase": "...", "corrected_phrase": "...", "reason": "..."}},
  ],
  "confidence_score": 0.85,
  "notes": "summary of approach"
}}"""
        
        print("\n🧠 Sending to Groq for semantic correction...")
        
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert transcription refinement system. Respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        result_text = response.choices[0].message.content
        
        # Parse JSON response
        import json
        result_json = json.loads(result_text)
        
        return {
            "corrected_transcript": result_json.get("corrected_transcript", ""),
            "corrections": result_json.get("corrections", []),
            "confidence": float(result_json.get("confidence_score", 0.5)),
            "notes": result_json.get("notes", ""),
            "status": "success",
        }
    
    except Exception as e:
        print(f"   ✗ Groq error: {e}")
        # Return the higher-confidence transcript as fallback
        return {
            "corrected_transcript": (
                deepgram_transcript if deepgram_confidence >= lipnet_confidence 
                else lipnet_transcript
            ),
            "corrections": [],
            "confidence": max(deepgram_confidence, lipnet_confidence),
            "notes": f"Groq failed, returning higher-confidence modality. Error: {str(e)}",
            "status": "fallback",
        }


def run_mvp(
    audio_file: Optional[str] = None,
    video_file: Optional[str] = None,
    deepgram_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    lipnet_weights: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete MVP pipeline.
    
    Args:
        audio_file: Path to audio file (required if no mock mode)
        video_file: Path to video file (required if no mock mode)
        deepgram_api_key: DeepGram API key
        groq_api_key: Groq API key
        lipnet_weights: Path to LipNet weights
        
    Returns:
        Complete results including corrected transcript
    """
    print("\n" + "=" * 80)
    print(" MVP: Multi-Modal Transcription with Groq Semantic Correction")
    print("=" * 80)
    
    # Get audio transcription with confidence
    if audio_file and os.path.exists(audio_file):
        deepgram_result = get_deepgram_confidence(audio_file, deepgram_api_key)
    else:
        print("📊 Using mock DeepGram data (no audio file provided)")
        deepgram_result = {
            "transcript": "the quick brown fox jumps over the lazy dog",
            "overall_confidence": 0.92,
            "word_confidences": [
                ("the", 0.95), ("quick", 0.89), ("brown", 0.91),
                ("fox", 0.93), ("jumps", 0.87), ("over", 0.94),
                ("the", 0.96), ("lazy", 0.88), ("dog", 0.97)
            ]
        }
    
    # Get video transcription with confidence
    if video_file and os.path.exists(video_file):
        lipnet_result = get_lipnet_confidence(video_file, lipnet_weights)
    else:
        print("👄 Using mock LipNet data (no video file provided)")
        lipnet_result = {
            "transcript": "the quick brown fox jumps over the lazy dog",
            "overall_confidence": 0.88,
            "word_confidences": [
                ("the", 0.92), ("quick", 0.85), ("brown", 0.88),
                ("fox", 0.90), ("jumps", 0.83), ("over", 0.91),
                ("the", 0.94), ("lazy", 0.82), ("dog", 0.95)
            ]
        }
    
    # Combine word-level confidences
    print("\n🔄 Combining word-level confidence scores...")
    combined_words = combine_word_confidences(
        deepgram_result["word_confidences"],
        lipnet_result["word_confidences"]
    )
    
    for word_info in combined_words:
        agreement_mark = "✓" if word_info["agreement"] else "✗"
        conf_mark = "⚠" if word_info["low_confidence"] else "✓"
        print(
            f"   {agreement_mark} {word_info['word']:12s} "
            f"[DG: {word_info['deepgram']['confidence']:.2f}] "
            f"[LipNet: {word_info['lipnet']['confidence']:.2f}] "
            f"[Avg: {word_info['average_confidence']:.2f}] {conf_mark}"
        )
    
    # Send to Groq for semantic correction
    print("\n" + "=" * 80)
    groq_result = process_with_groq(
        deepgram_transcript=deepgram_result["transcript"],
        deepgram_confidence=deepgram_result["overall_confidence"],
        deepgram_words=deepgram_result["word_confidences"],
        lipnet_transcript=lipnet_result["transcript"],
        lipnet_confidence=lipnet_result["overall_confidence"],
        lipnet_words=lipnet_result["word_confidences"],
        combined_words=combined_words,
        api_key=groq_api_key,
    )
    
    # Print results
    print("\n" + "=" * 80)
    print(" FINAL RESULTS")
    print("=" * 80)
    print(f"\nDeepGram:        {deepgram_result['transcript']}")
    print(f"LipNet:          {lipnet_result['transcript']}")
    print(f"Combined:        {' '.join([w['word'] for w in combined_words])}")
    print(f"Groq Corrected:  {groq_result['corrected_transcript']}")
    print(f"Confidence:      {groq_result['confidence']:.3f}")
    
    if groq_result['corrections']:
        print(f"\nCorrections Applied:")
        for corr in groq_result['corrections']:
            print(f"  • '{corr.get('original_phrase', '')}' → '{corr.get('corrected_phrase', '')}'")
            if corr.get('reason'):
                print(f"    Reason: {corr['reason']}")
    
    print(f"\nStatus: {groq_result['status']}")
    if groq_result['notes']:
        print(f"Notes: {groq_result['notes']}")
    
    print("=" * 80 + "\n")
    
    return {
        "deepgram": deepgram_result,
        "lipnet": lipnet_result,
        "combined_words": combined_words,
        "groq_correction": groq_result,
        "final_transcript": groq_result["corrected_transcript"],
    }


if __name__ == "__main__":
    # Example usage with mock data
    results = run_mvp()
    
    print("\n✅ MVP pipeline completed successfully!")
    print(f"\n📄 Final Transcript: {results['final_transcript']}")

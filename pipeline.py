"""
MVP: Multi-Modal Transcription with Word-Level Confidence and Groq Semantic Correction

This script combines:
1. DeepGram - Audio transcription with word-level confidence
2. auto_avsr - Visual speech recognition with word-level confidence  
3. Groq - Fast LLM-based semantic correction

Perfect for scenarios where you have both audio and video input.
"""

import os
import sys
from typing import Dict, Any, List, Tuple, Optional

# Load environment variables from .env file
from load_env import load_env_file
load_env_file()

# Import logger
from logger import get_logger

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'auto_avsr'))
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


def get_avsr_confidence(video_file: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get transcription with word-level confidence from auto_avsr.
    
    Args:
        video_file: Path to video file
        model_path: Path to auto_avsr model weights (uses default if not provided)
        
    Returns:
        Dict with transcript, overall_confidence, and word_confidences
    """
    if model_path is None:
        # Try to find model weights automatically
        default_models = [
            "auto_avsr/pretrained_models/vsr_trlrs2lrs3vox2avsp_base.pth",
            "./auto_avsr/pretrained_models/vsr_trlrs2lrs3vox2avsp_base.pth",
            os.path.join(os.path.dirname(__file__), "auto_avsr/pretrained_models/vsr_trlrs2lrs3vox2avsp_base.pth"),
        ]
        
        for path in default_models:
            if os.path.exists(path):
                model_path = path
                break
    
    if not model_path or not os.path.exists(model_path):
        print(f"   ✗ auto_avsr model not found. Tried: {default_models}")
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
        # Import from auto_avsr
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'auto_avsr'))
        from inference_wrapper import get_avsr_confidence as avsr_inference
        
        print("👁️👄 Transcribing video with auto_avsr (VSR)...")
        result = avsr_inference(video_file, model_path, detector="retinaface")
        
        print(f"   ✓ auto_avsr transcript: {result['transcript']}")
        print(f"   ✓ Confidence: {result['overall_confidence']:.3f}")
        print(f"   ✓ Word-level scores: {result['word_confidences']}")
        
        return {
            "transcript": result['transcript'],
            "overall_confidence": result['overall_confidence'],
            "word_confidences": result['word_confidences'],
        }
    
    except Exception as e:
        print(f"   ✗ auto_avsr error: {e}")
        import traceback
        traceback.print_exc()
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
    avsr_words: List[Tuple[str, float]],
) -> List[Dict[str, Any]]:
    """
    Combine word-level confidence scores from both models.
    
    Args:
        deepgram_words: [(word, confidence), ...] from DeepGram
        avsr_words: [(word, confidence), ...] from auto_avsr
        
    Returns:
        List of dicts with combined confidence information
    """
    combined = []
    max_words = max(len(deepgram_words), len(avsr_words))
    
    for i in range(max_words):
        deepgram_word, deepgram_conf = deepgram_words[i] if i < len(deepgram_words) else ("", 0.0)
        avsr_word, avsr_conf = avsr_words[i] if i < len(avsr_words) else ("", 0.0)
        
        # Compute average confidence
        avg_confidence = (deepgram_conf + avsr_conf) / 2.0
        
        # Mark as low confidence if either model is uncertain
        is_low_confidence = deepgram_conf < 0.7 or avsr_conf < 0.7
        
        combined.append({
            "position": i,
            "word": deepgram_word or avsr_word,  # Prefer DeepGram if both available
            "deepgram": {"word": deepgram_word, "confidence": deepgram_conf},
            "avsr": {"word": avsr_word, "confidence": avsr_conf},
            "average_confidence": avg_confidence,
            "agreement": deepgram_word.lower() == avsr_word.lower(),
            "low_confidence": is_low_confidence,
        })
    
    return combined


def process_with_groq(
    deepgram_transcript: str,
    deepgram_confidence: float,
    deepgram_words: List[Tuple[str, float]],
    avsr_transcript: str,
    avsr_confidence: float,
    avsr_words: List[Tuple[str, float]],
    combined_words: List[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use Groq to semantically correct and refine the combined output.
    
    Args:
        deepgram_transcript: Full transcript from DeepGram
        deepgram_confidence: Overall confidence from DeepGram
        deepgram_words: Word-level confidences from DeepGram
        avsr_transcript: Full transcript from auto_avsr
        avsr_confidence: Overall confidence from auto_avsr
        avsr_words: Word-level confidences from auto_avsr
        combined_words: Combined word-level analysis
        api_key: Groq API key (uses GROQ_API_KEY env var if not provided)
        
    Returns:
        Dict with corrected transcript and metadata
    """
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("   ⚠ GROQ_API_KEY not set - using fallback mode")
        # Return the higher-confidence transcript as fallback
        return {
            "corrected_transcript": (
                deepgram_transcript if deepgram_confidence >= avsr_confidence 
                else avsr_transcript
            ),
            "corrections": [],
            "confidence": max(deepgram_confidence, avsr_confidence),
            "notes": "Groq API key not set, returning higher-confidence modality",
            "status": "fallback",
        }
    
    try:
        from groq import Groq
        
        client = Groq(api_key=api_key)
        
        # Build context for Groq
        low_conf_words = [w for w in combined_words if w["low_confidence"]]
        disagreements = [w for w in combined_words if not w["agreement"]]
        
        prompt = f"""You are an expert speech transcription refinement system with deep understanding of linguistics, phonetics, and semantic coherence.

I have two transcriptions from different modalities (audio ASR and visual lip-reading) of the same speech:

AUDIO TRANSCRIPTION (DeepGram - reliable in clean conditions):
- Transcript: "{deepgram_transcript}"
- Overall Confidence: {deepgram_confidence:.3f}
- Word Confidences: {deepgram_words}

VISUAL TRANSCRIPTION (auto_avsr - lip reading, prone to homophones):
- Transcript: "{avsr_transcript}"
- Overall Confidence: {avsr_confidence:.3f}
- Word Confidences: {avsr_words}

COMBINED ANALYSIS:
- Total Words: {len(combined_words)}
- Low Confidence Words (< 0.7): {len(low_conf_words)}
  {[w['word'] for w in low_conf_words]}
- Word Disagreements: {len(disagreements)}
  {disagreements}

CRITICAL INSTRUCTIONS:
1. **SEMANTIC COHERENCE IS PARAMOUNT** - The final transcript MUST make grammatical and contextual sense
2. **Audio is more reliable for phonetically distinct words** - Trust audio for unique sounds
3. **Visual helps disambiguate similar-sounding words** - Use visual for homophones (their/there, two/to/too)
4. **Common VSR errors to watch for:**
   - Short words (at, a, in, is) often missed by lip-reading
   - Homophones confused (bin/been/bim, two/to, for/four)
   - Merged words (f two → F2, bin blue → binblue)
5. **VALIDATE semantic plausibility** - Reject transcripts that are grammatically nonsensical
6. **When both transcripts are implausible, prefer the one with:**
   - Better grammar
   - More common word usage
   - Higher overall confidence
   - Fewer unlikely word combinations

DECISION PROCESS:
Step 1: Check if EITHER transcript is semantically coherent as-is
Step 2: If audio transcript makes sense, prefer it (audio is generally more accurate)
Step 3: If visual transcript makes more sense, use it but validate each word
Step 4: For disagreements, choose based on:
   - Phonetic distinctiveness (audio wins for unique sounds)
   - Visual similarity (visual wins for homophones)
   - Word-level confidence
   - Context and grammar

EXAMPLES OF GOOD CORRECTIONS:

Example 1:
Audio: "bin blue at f two please"
Visual: "BIMBO F2 NOW"
Correct: "bin blue at f two please" (audio is coherent, visual has errors)

Example 2:
Audio: "set right with p four please"
Visual: "SET WHITE WITH B4 PLEASE"
Correct: "set right with p four please" (audio is coherent, visual confused homophones)

Example 3 (LRS3 grid commands - these are VERY COMMON):
- "lay white with zed nine soon"
- "bin blue at f two please"
- "place red at g five now"
- "set white with p four soon"
These are VALID English commands used in the LRS3 dataset. Colors, letters, and numbers are correct.

Respond with ONLY a JSON object (no markdown, no code blocks):
{{
  "corrected_transcript": "the refined transcript",
  "corrections": [
    {{"original_phrase": "...", "corrected_phrase": "...", "reason": "..."}},
  ],
  "confidence_score": 0.85,
  "status": "success",
  "notes": "brief explanation of correction strategy"
}}"""
        
        print("\n🧠 Sending to Groq for semantic correction...")
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert transcription refinement system with deep linguistic knowledge. Always prioritize semantic coherence and grammatical correctness. Respond with valid JSON only.",
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
                deepgram_transcript if deepgram_confidence >= avsr_confidence 
                else avsr_transcript
            ),
            "corrections": [],
            "confidence": max(deepgram_confidence, avsr_confidence),
            "notes": f"Groq failed, returning higher-confidence modality. Error: {str(e)}",
            "status": "fallback",
        }


def run_mvp(
    audio_file: Optional[str] = None,
    video_file: Optional[str] = None,
    deepgram_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    avsr_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete MVP pipeline.
    
    Args:
        audio_file: Path to audio file (required if no mock mode)
        video_file: Path to video file (required if no mock mode)
        deepgram_api_key: DeepGram API key
        groq_api_key: Groq API key
        avsr_model_path: Path to auto_avsr model weights
        
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
        avsr_result = get_avsr_confidence(video_file, avsr_model_path)
    else:
        print("👁️👄 Using mock auto_avsr data (no video file provided)")
        avsr_result = {
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
        avsr_result["word_confidences"]
    )
    
    for word_info in combined_words:
        agreement_mark = "✓" if word_info["agreement"] else "✗"
        conf_mark = "⚠" if word_info["low_confidence"] else "✓"
        print(
            f"   {agreement_mark} {word_info['word']:12s} "
            f"[DG: {word_info['deepgram']['confidence']:.2f}] "
            f"[AVSR: {word_info['avsr']['confidence']:.2f}] "
            f"[Avg: {word_info['average_confidence']:.2f}] {conf_mark}"
        )
    
    # Send to Groq for semantic correction
    print("\n" + "=" * 80)
    groq_result = process_with_groq(
        deepgram_transcript=deepgram_result["transcript"],
        deepgram_confidence=deepgram_result["overall_confidence"],
        deepgram_words=deepgram_result["word_confidences"],
        avsr_transcript=avsr_result["transcript"],
        avsr_confidence=avsr_result["overall_confidence"],
        avsr_words=avsr_result["word_confidences"],
        combined_words=combined_words,
        api_key=groq_api_key,
    )
    
    # Print results
    print("\n" + "=" * 80)
    print(" FINAL RESULTS")
    print("=" * 80)
    print(f"\nAudio (DG):      {deepgram_result['transcript']}")
    print(f"Video (VSR):     {avsr_result['transcript']}")
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
    
    # Prepare result dictionary
    result = {
        "deepgram": deepgram_result,
        "avsr": avsr_result,
        "combined_words": combined_words,
        "combined_transcript": ' '.join([w['word'] for w in combined_words]),
        "groq": groq_result,
        "final_transcript": groq_result["corrected_transcript"],
    }
    
    # Log everything
    logger = get_logger()
    logger.log_all(result, video_file=video_file or "mock_data")
    
    return result


if __name__ == "__main__":
    # Example usage with mock data
    results = run_mvp()
    
    print("\n✅ MVP pipeline completed successfully!")
    print(f"\n📄 Final Transcript: {results['final_transcript']}")


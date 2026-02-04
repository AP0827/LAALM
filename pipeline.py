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
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Load environment variables from .env file
from load_env import load_env_file
load_env_file()

# Import logger
from logger import get_logger

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'auto_avsr'))
sys.path.insert(0, os.path.dirname(__file__))

from video_utils import SafeVideoPreprocessor


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
        # Revert to RetinaFace as it performs better with this specific model
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
    Combine word-level confidence scores using ROBUST ALIGNMENT and Attention Fusion.
    
    IMPROVEMENT: Now uses Needleman-Wunsch alignment to align audio and visual
    streams before fusion, preventing cascading errors from missed words.
    
    Args:
        deepgram_words: List of tuples from DeepGram. Can be (word, conf) or (word, conf, start, end).
        avsr_words: List of tuples from auto_avsr. (word, conf).
        
    Returns:
        List of dicts with combined confidence information + attention weights
    """
    # Import our modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Transformer'))
    from attention_fusion import AttentionFusion
    from alignment import TranscriptAligner
    
    # 1. Initialize Aligner
    aligner = TranscriptAligner(
        match_score=2.0,
        mismatch_penalty=-1.0,
        gap_penalty=-1.0
    )
    
    # 2. Calibration (UNBIASING)
    # The user requested a "more unbiased" fusion.
    # Audio confidence is typically higher (0.8-0.9) than Visual (0.5-0.6).
    # We calibrate Visual scores to match Audio distribution.
    
    audio_confs = [w[1] for w in deepgram_words]
    video_confs = [w[1] for w in avsr_words]
    
    if audio_confs and video_confs:
        a_mean, a_std = np.mean(audio_confs), np.std(audio_confs)
        v_mean, v_std = np.mean(video_confs), np.std(video_confs)
        
        # Avoid division by zero
        v_std = v_std if v_std > 0.01 else 0.1
        a_std = a_std if a_std > 0.01 else 0.1
        
        print(f"⚖️  Calibration: Audio μ={a_mean:.2f} σ={a_std:.2f} | Visual μ={v_mean:.2f} σ={v_std:.2f}")
        
        # Scale visual words
        calibrated_avsr_words = []
        for w, conf in avsr_words:
            # Z-score normalization + re-scaling to Audio distribution
            z_score = (conf - v_mean) / v_std
            new_conf = (z_score * a_std) + a_mean
            # Clip to [0, 1]
            new_conf = max(0.01, min(0.99, new_conf))
            calibrated_avsr_words.append((w, new_conf))
        
        # Use calibrated words for alignment and fusion
        avsr_words_for_fusion = calibrated_avsr_words
        print("   ✓ Visual confidence calibrated to match audio distribution")
    else:
        avsr_words_for_fusion = avsr_words

    # 3. Perform Robust Alignment
    # Align the sequences (returns lists of same length with None for gaps)
    aligned_audio, aligned_visual = aligner.align(deepgram_words, avsr_words_for_fusion)
    
    # 4. Prepare for Fusion
    # We need to convert the aligned lists (which may have None) back into 
    # the format expected by AttentionFusion, but strictly paired.
    
    paired_audio = []
    paired_visual = []
    
    for a, v in zip(aligned_audio, aligned_visual):
        # Handle Audio Gap (Visual Only)
        if a is None:
            paired_audio.append(("", 0.0))
        else:
            # Handle both 2-tuple and 4-tuple formats
            paired_audio.append((a[0], a[1]))
            
        # Handle Visual Gap (Audio Only)
        if v is None:
            paired_visual.append(("", 0.0))
        else:
            paired_visual.append((v[0], v[1]))
            
    # 3. Perform Attention-Based Fusion
    fusion = AttentionFusion(
        temperature=2.0,           
        context_window=3,          
        switching_penalty=0.15,    
        min_confidence_threshold=0.6
    )
    
    fusion_result = fusion.fuse_transcripts(
        audio_words=paired_audio,
        visual_words=paired_visual
    )
    
    # 4. Format Output
    combined = []
    for i, detail in enumerate(fusion_result.word_details):
        # Recover original metadata if available (timestamps from audio)
        audio_source = aligned_audio[i]
        start_time = 0.0
        end_time = 0.0
        
        if audio_source and len(audio_source) >= 4:
            start_time = audio_source[2]
            end_time = audio_source[3]
            
        # Strip punctuation for more robust comparison
        a_word_clean = detail['audio_word'].lower().strip(".,?!:;")
        v_word_clean = detail['visual_word'].lower().strip(".,?!:;")
        
        combined.append({
            "position": detail['position'],
            "word": detail['word'],
            "deepgram": {
                "word": detail['audio_word'], 
                "confidence": detail['audio_conf']
            },
            "avsr": {
                "word": detail['visual_word'], 
                "confidence": detail['visual_conf']
            },
            "average_confidence": detail['confidence'],
            "agreed": a_word_clean == v_word_clean and a_word_clean != "",
            "low_confidence": detail['confidence'] < 0.6,
            "selected_modality": detail['selected_modality'],
            "audio_weight": detail['audio_weight'],
            "visual_weight": detail['visual_weight'],
            "start": start_time,
            "end": end_time
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
    nbest_transcripts: Optional[List[str]] = None,
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
        nbest_transcripts: List of alternative VSR transcripts (Optional)
        
    Returns:
        Dict with corrected transcript and metadata
    """
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    
    nbest_transcripts = nbest_transcripts or []
    
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
        disagreements = [w for w in combined_words if not w["agreed"]]
        
        # Format N-Best list
        nbest_text = "\n".join([f"   {i+1}. {txt}" for i, txt in enumerate(nbest_transcripts[:5])])

        prompt = f"""You are an advanced Audio-Visual Speech Recognition (AVSR) Fusion Engine.
Your task is to combine a noisy Audio Transcript (ASR) and a noisy Visual Transcript (Lip Reading/VSR) into a single, perfect Sentence.

CONTEXT: General conversational speech.
Goal: Produce the most accurate, semantically coherent English sentence possible.

INPUT DATA:

1. AUDIO TRANSCRIPTION (DeepGram - Usually Reliable for sound, bad for homophones):
   "{deepgram_transcript}"
   (Confidence: {deepgram_confidence:.3f})

2. VISUAL TRANSCRIPTION (VSR - Top-1 Hypothesis):
   "{avsr_transcript}"
   (Confidence: {avsr_confidence:.3f})

3. VISUAL ALTERNATIVES (N-Best Beam Search - Hidden possibilities):
{nbest_text}

4. FUSION ANALYSIS (Word-level Agreement):
   Disagreements: {len(disagreements)}
   Low Confidence Words: {[w['word'] for w in low_conf_words]}

INSTRUCTIONS:
1. **Prioritize Agreement**: If Audio and Video agree, trust them.
2. **Use N-Best List**: If the Top-1 Video is wrong (e.g. "Labor"), check the N-Best list. If "Late" is in the list and matches the Audio's sound, IT IS THE CORRECT WORD.
3. **Resolve Homophones**:
   - Audio: "Meat" vs "Meet" (Context dependent)
   - Visual: "Bat" vs "Pat" (Indistinguishable lips)
   - Use cross-referencing to solve these.
4. **Generalize**: Do NOT force any specific command structure (like Grid Corpus). The input could be anything (News, Commands, Casual Chat).
5. **Output Format**: plain text sentence with standard capitalization and punctuation.

Example 1 (Fusion):
Audio: "I went to the..." (cut off)
Video Top-1: "...store."
Video Alt: "...shore."
Context: "Buying milk."
Result: "I went to the store."

Example 2 (Correction):
Audio: "Lay white with zed..."
Video: "Lay white with red..."
Correction: "Lay white with red..." (Visual 'red' fits colors context better than 'zed' if color was expected, but trust audio for distinct sounds).

Your decision must be purely based on likelihood and semantic flow.

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
        
        # Parse JSON response with resilience
        import json
        import re
        import ast
        
        # Try to find a JSON block
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        result_json = None
        
        if json_match:
            candidate = json_match.group()
            try:
                result_json = json.loads(candidate)
            except json.JSONDecodeError:
                # Fallback 1: Try cleanup
                try:
                    clean = candidate.replace("```json", "").replace("```", "").strip()
                    result_json = json.loads(clean)
                except:
                    # Fallback 2: ast.literal_eval (handles trailing commas)
                    try:
                        result_json = ast.literal_eval(candidate)
                    except:
                        pass
        
        if not result_json:
             # Try parsing the whole text if regex failed or didn't contain valid JSON
             try:
                 result_json = json.loads(result_text)
             except:
                 try:
                     result_json = ast.literal_eval(result_text)
                 except:
                     pass

        if not result_json:
             raise ValueError(f"Failed to parse Groq response JSON. Raw: {result_text[:50]}...")
        
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
    # Advanced Preprocessing Configuration
    use_advanced_preprocessing: bool = True,
    video_denoise_strength: int = 0,
    use_temporal_smoothing: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete MVP pipeline.
    
    Args:
        audio_file: Path to audio file (required if no mock mode)
        video_file: Path to video file (required if no mock mode)
        deepgram_api_key: DeepGram API key
        groq_api_key: Groq API key
        avsr_model_path: Path to auto_avsr model weights
        use_advanced_preprocessing: Whether to use the advanced preprocessor
        video_denoise_strength: Strength of video denoising (0-10)
        use_temporal_smoothing: Whether to use temporal smoothing
        
    Returns:
        Complete results including corrected transcript
    """
    print("\n" + "=" * 80)
    print(" MVP: Multi-Modal Transcription with Groq Semantic Correction")
    print("=" * 80)
    
    # Get audio transcription with confidence
    if audio_file and os.path.exists(audio_file):
        # NOTE: Audio preprocessing is disabled by default to maintain DeepGram's native performance.
        # DeepGram usually handles noise better than simple bandpass filters.
        deepgram_result = get_deepgram_confidence(audio_file, deepgram_api_key)
    else:
        # Mock data/fallback
        processed_audio_path = None 
        # ... logic for mock data if needed, or get_deepgram_confidence handles it
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
    processed_video_path = None
    if video_file and os.path.exists(video_file):
        try:
            # Safe Preprocessing (FPS=25, 720p) is now MANDATORY for VSR stability.
            # "Advanced Preprocessing" toggle now controls optional enhancement (Sharpen/Denoise).
            
            print("🎥 Running Safe Video Preprocessing (Standardizing FPS/Res)...")
            processor = SafeVideoPreprocessor(output_dir=Path("uploads"))
            
            processed_video_path, success = processor.process(
                video_file, 
                Path(video_file).stem,
                apply_enhancement=use_advanced_preprocessing,
                denoise_strength=video_denoise_strength,
                temporal_smoothing=use_temporal_smoothing
            )
            
            if success:
                print(f"   ✓ Video optimized: {processed_video_path}")
                video_input = processed_video_path
            else:
                print("   ⚠ Preprocessing failed, falling back to original.")
                video_input = video_file

            avsr_result = get_avsr_confidence(video_input, avsr_model_path)
                
        except Exception as e:
            print(f"   ⚠ Video preprocessing failed, using original file: {e}")
            avsr_result = get_avsr_confidence(video_file, avsr_model_path)
        finally:
            # Clean up temp file
            if processed_video_path and processed_video_path != video_file and os.path.exists(processed_video_path):
                try:
                    os.remove(processed_video_path)
                except:
                    pass
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
        agreement_mark = "✓" if word_info["agreed"] else "✗"
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
        nbest_transcripts=avsr_result.get("nbest_transcripts", []),
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
    
    # Get logger first (needed for session_id)
    logger = get_logger()
    
    # Generate timestamped captions
    try:
        from Transformer.fusion_caption_generator import FusionCaptionGenerator
        
        caption_gen = FusionCaptionGenerator()
        
        # Prepare word data with timestamps from DeepGram
        word_data = [
            {
                'word': w[0],
                'start': w[2] if len(w) > 2 else 0,
                'end': w[3] if len(w) > 3 else 0
            }
            for w in deepgram_result["word_confidences"]
        ]
        
        # Save captions
        caption_dir = Path("captions")
        caption_dir.mkdir(exist_ok=True)
        
        session_id = logger.session_id if logger else "default"
        
        vtt_path = caption_gen.save_captions(
            final_transcript=groq_result["corrected_transcript"],
            word_data=word_data,
            output_path=f"captions/{session_id}.vtt",
            format_type='vtt'
        )
        
        srt_path = caption_gen.save_captions(
            final_transcript=groq_result["corrected_transcript"],
            word_data=word_data,
            output_path=f"captions/{session_id}.srt",
            format_type='srt'
        )
        
        print(f"\n📄 Captions saved:")
        print(f"   WebVTT: {vtt_path}")
        print(f"   SRT: {srt_path}")
        
        result["captions"] = {
            "vtt": str(vtt_path),
            "srt": str(srt_path)
        }
        
    except Exception as e:
        print(f"   ⚠ Caption generation failed: {e}")
    
    # Log everything
    logger.log_all(result, video_file=video_file or "mock_data")
    
    return result


if __name__ == "__main__":
    # Example usage with mock data
    results = run_mvp()
    
    print("\n✅ MVP pipeline completed successfully!")
    print(f"\n📄 Final Transcript: {results['final_transcript']}")


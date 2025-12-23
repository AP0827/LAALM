"""
Example usage of the complete Transformer pipeline combining DeepGram + LipNet with LLM refinement.

This script demonstrates:
1. Extracting word-level confidence from DeepGram
2. Getting visual predictions from LipNet with character probabilities
3. Fusing both modalities with confidence weighting
4. Applying LLM-based semantic correction
5. Generating comprehensive reports

Usage:
    # With real DeepGram API (requires API key)
    python example_usage.py --audio-file audio.wav --lipnet-video video.mp4 --llm-key your_openai_key
    
    # With mock data (for testing)
    python example_usage.py --mock-mode
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DeepGram.word_confidence import WordConfidenceExtractor
from Transformer import TransformerPipeline, LLMProvider


def create_mock_deepgram_response() -> Dict[str, Any]:
    """Create a mock DeepGram response for testing."""
    return {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "the quick brown fox jumps over the lazy dog",
                            "confidence": 0.92,
                            "words": [
                                {"word": "the", "start": 0.0, "end": 0.2, "confidence": 0.95},
                                {"word": "quick", "start": 0.2, "end": 0.5, "confidence": 0.89},
                                {"word": "brown", "start": 0.5, "end": 0.8, "confidence": 0.91},
                                {"word": "fox", "start": 0.8, "end": 1.1, "confidence": 0.94},
                                {"word": "jumps", "start": 1.1, "end": 1.5, "confidence": 0.87},
                                {"word": "over", "start": 1.5, "end": 1.8, "confidence": 0.92},
                                {"word": "the", "start": 1.8, "end": 2.0, "confidence": 0.95},
                                {"word": "lazy", "start": 2.0, "end": 2.3, "confidence": 0.88},
                                {"word": "dog", "start": 2.3, "end": 2.6, "confidence": 0.93},
                            ],
                        }
                    ]
                }
            ]
        },
        "metadata": {
            "duration": 2.6,
        },
    }


def create_mock_lipnet_output() -> Tuple[str, float, List[Tuple[str, float]]]:
    """Create mock LipNet output for testing."""
    # Transcript from visual model (might have slight differences)
    transcript = "the quick brown fox jumps over the lazy dog"
    overall_confidence = 0.85  # Generally lower than audio model
    
    # Word-level probabilities from character-level predictions
    word_confidences = [
        ("the", 0.91),
        ("quick", 0.82),  # Lower confidence
        ("brown", 0.88),
        ("fox", 0.90),
        ("jumps", 0.79),  # Lower confidence
        ("over", 0.89),
        ("the", 0.92),
        ("lazy", 0.84),
        ("dog", 0.91),
    ]
    
    return transcript, overall_confidence, word_confidences


def extract_deepgram_word_confidences(response: Dict[str, Any]) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Extract word-level confidences from DeepGram response.
    
    Args:
        response: DeepGram API response.
        
    Returns:
        (transcript, overall_confidence, [(word, confidence), ...])
    """
    extractor = WordConfidenceExtractor(low_confidence_threshold=0.75)
    
    # Extract words with confidence
    words = extractor.extract_word_confidences(response, alternative_index=0)
    
    # Get transcript
    results = response.get("results", {})
    channels = results.get("channels", [])
    alternatives = channels[0].get("alternatives", [])
    alternative = alternatives[0]
    transcript = alternative.get("transcript", "")
    overall_confidence = float(alternative.get("confidence", 0))
    
    # Convert to list of tuples
    word_confidences = [(w.word, w.confidence) for w in words]
    
    return transcript, overall_confidence, word_confidences


def run_complete_pipeline(
    deepgram_response: Dict[str, Any],
    lipnet_transcript: str,
    lipnet_confidence: float,
    lipnet_word_confidences: List[Tuple[str, float]],
    llm_provider: str = "openai",
    llm_api_key: Optional[str] = None,
    domain_context: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete Transformer pipeline.
    
    Args:
        deepgram_response: DeepGram API response.
        lipnet_transcript: LipNet transcript.
        lipnet_confidence: LipNet overall confidence.
        lipnet_word_confidences: LipNet word-level confidences.
        llm_provider: LLM provider to use.
        llm_api_key: API key for LLM provider.
        domain_context: Optional domain context.
        verbose: Whether to print progress.
        
    Returns:
        Complete pipeline result dictionary.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("INITIALIZING TRANSFORMER PIPELINE")
        print("=" * 80)
    
    # Initialize pipeline
    if llm_api_key:
        pipeline = TransformerPipeline(
            llm_provider=LLMProvider[llm_provider.upper()],
            llm_api_key=llm_api_key,
            use_confidence_weighting=True,
            llm_enabled=bool(llm_api_key),
        )
    else:
        pipeline = TransformerPipeline(
            use_confidence_weighting=True,
            llm_enabled=False,
        )
    
    # Extract DeepGram word confidences
    if verbose:
        print("\n[1/3] Extracting word-level confidence from DeepGram...")
    dg_transcript, dg_confidence, dg_word_confs = extract_deepgram_word_confidences(
        deepgram_response
    )
    
    if verbose:
        print(f"  ✓ Transcript: {dg_transcript}")
        print(f"  ✓ Confidence: {dg_confidence:.3f}")
        print(f"  ✓ Words with confidence: {len(dg_word_confs)}")
    
    # Process through pipeline
    if verbose:
        print("\n[2/3] Fusing multi-modal outputs...")
    
    result = pipeline.process(
        deepgram_transcript=dg_transcript,
        deepgram_confidence=dg_confidence,
        deepgram_word_confidences=dg_word_confs,
        lipnet_transcript=lipnet_transcript,
        lipnet_confidence=lipnet_confidence,
        lipnet_word_confidences=lipnet_word_confidences,
        domain_context=domain_context,
        audio_metadata={"duration": 2.6},
    )
    
    if verbose:
        print(f"  ✓ Alignment Score: {result['fusion_result']['alignment_score']:.3f}")
        print(f"  ✓ Fused Transcript: {result['fusion_result']['fused_transcript']}")
    
    if result["correction_result"]:
        if verbose:
            print("\n[3/3] Applying LLM semantic correction...")
            print(f"  ✓ Corrected Transcript: {result['correction_result']['corrected_transcript']}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Example Transformer pipeline combining DeepGram + LipNet with LLM refinement."
    )
    
    parser.add_argument(
        "--mock-mode",
        action="store_true",
        help="Run with mock data (for testing without real APIs)",
    )
    
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Path to audio file for DeepGram transcription",
    )
    
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "google", "ollama"],
        default="openai",
        help="LLM provider for semantic correction",
    )
    
    parser.add_argument(
        "--llm-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="API key for LLM provider (or set env var)",
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        help="Domain context (e.g., 'medical', 'legal', 'casual')",
    )
    
    parser.add_argument(
        "--output-json",
        type=str,
        help="Save full result as JSON to this file",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress",
    )
    
    args = parser.parse_args()
    
    print("\n╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "TRANSFORMER PIPELINE - DEEPGRAM + LIPNET + LLM" + " " * 17 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        if args.mock_mode:
            if args.verbose:
                print("\n[SETUP] Running in MOCK MODE with simulated data...")
            
            # Create mock data
            deepgram_response = create_mock_deepgram_response()
            lipnet_transcript, lipnet_confidence, lipnet_word_confidences = create_mock_lipnet_output()
            
            # Run pipeline
            result = run_complete_pipeline(
                deepgram_response=deepgram_response,
                lipnet_transcript=lipnet_transcript,
                lipnet_confidence=lipnet_confidence,
                lipnet_word_confidences=lipnet_word_confidences,
                llm_provider=args.llm_provider,
                llm_api_key=args.llm_key,
                domain_context=args.domain,
                verbose=args.verbose,
            )
        
        else:
            if not args.audio_file:
                print("\n❌ Error: --audio-file required (or use --mock-mode for testing)")
                sys.exit(1)
            
            print(f"\n[SETUP] Initializing real pipeline with audio: {args.audio_file}")
            print("⚠ This requires DeepGram API key and LipNet model setup")
            
            # TODO: Implement real API integration
            print("\n❌ Real mode not yet implemented in this example.")
            print("   Use --mock-mode to test the pipeline with simulated data.")
            sys.exit(1)
        
        # Print comprehensive report
        if args.verbose:
            report = TransformerPipeline().get_full_report(result)
            print(report)
        
        # Save results if requested
        if args.output_json:
            # Convert to JSON-serializable format
            json_result = {
                "deepgram_output": result["deepgram_output"],
                "lipnet_output": result["lipnet_output"],
                "fusion_result": {
                    "fused_transcript": result["fusion_result"]["fused_transcript"],
                    "alignment_score": result["fusion_result"]["alignment_score"],
                    "fusion_weights": result["fusion_result"]["fusion_weights"],
                    "flagged_discrepancies": result["fusion_result"]["flagged_discrepancies"],
                },
                "correction_result": result["correction_result"],
                "final_transcript": result["final_transcript"],
            }
            
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            
            if args.verbose:
                print(f"\n✓ Results saved to: {args.output_json}")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

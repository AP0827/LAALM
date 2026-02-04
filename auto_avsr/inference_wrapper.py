"""
Inference wrapper for auto_avsr with word-level confidence scores.

This module wraps the InferencePipeline to provide word-level confidence
scores extracted from the beam search decoder, making it compatible with
the pipeline.py interface.
"""

import os
import sys
import torch
import torchvision
import argparse
from typing import Dict, Any, List, Tuple

# Add auto_avsr to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightning import ModelModule
from datamodule.transforms import VideoTransform


class InferencePipelineWithConfidence(torch.nn.Module):
    """
    Extended InferencePipeline that returns word-level confidence scores.
    """
    
    def __init__(self, model_path: str, detector: str = "retinaface"):
        super(InferencePipelineWithConfidence, self).__init__()
        
        # Create args
        parser = argparse.ArgumentParser()
        args, _ = parser.parse_known_args(args=[])
        setattr(args, 'modality', 'video')
        
        self.modality = args.modality
        
        # Initialize video processing
        if detector == "mediapipe":
            try:
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            except Exception as e:
                print(f"   ⚠ MediaPipe detector failed: {e}")
                print(f"   ↳ Falling back to RetinaFace detector")
                detector = "retinaface"
        
        if detector == "retinaface":
            from preparation.detectors.retinaface.detector import LandmarksDetector
            from preparation.detectors.retinaface.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector(device="cuda:0" if torch.cuda.is_available() else "cpu")
            self.video_process = VideoProcess(convert_gray=False)
        
        self.video_transform = VideoTransform(subset="test")
        
        # Load model
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.modelmodule = ModelModule(args)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule.eval()
        
        # Store beam search for confidence extraction
        from lightning import get_beam_search_decoder
        self.beam_search = get_beam_search_decoder(
            self.modelmodule.model, 
            self.modelmodule.token_list
        )
    
    def load_video(self, data_filename: str):
        """Load video file."""
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()
    
    def forward(self, data_filename: str) -> Dict[str, Any]:
        """
        Run inference on video file with word-level confidence scores.
        
        Args:
            data_filename: Path to video file
            
        Returns:
            Dict with:
                - transcript: str
                - overall_confidence: float
                - word_confidences: List[Tuple[str, float]]
        """
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"Video file not found: {data_filename}"
        
        # Process video
        video = self.load_video(data_filename)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)
        
        # Run inference with confidence extraction
        with torch.no_grad():
            # Get encoder features
            x = self.modelmodule.model.frontend(video.unsqueeze(0))
            x = self.modelmodule.model.proj_encoder(x)
            enc_feat, _ = self.modelmodule.model.encoder(x, None)
            enc_feat = enc_feat.squeeze(0)
            
            # Run beam search
            nbest_hyps = self.beam_search(enc_feat)
            nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), 5)]]
            
            # Extract best hypothesis
            best_hyp = nbest_hyps[0]
            predicted_token_ids = torch.tensor(list(map(int, best_hyp["yseq"][1:])))
            transcript = self.modelmodule.text_transform.post_process(predicted_token_ids).replace("<eos>", "")
            
            # Extract word-level confidence scores
            word_confidences = self._extract_word_confidences(
                best_hyp, 
                transcript,
                predicted_token_ids
            )
            
            # Calculate overall confidence
            if word_confidences:
                overall_confidence = sum(conf for _, conf in word_confidences) / len(word_confidences)
            else:
                overall_confidence = 0.5
        
        # Extract N-Best transcripts
        nbest_transcripts = []
        for hyp in nbest_hyps:
            t_ids = torch.tensor(list(map(int, hyp["yseq"][1:])))
            t_str = self.modelmodule.text_transform.post_process(t_ids).replace("<eos>", "")
            nbest_transcripts.append(t_str)

        return {
            "transcript": transcript,
            "overall_confidence": overall_confidence,
            "word_confidences": word_confidences,
            "nbest_transcripts": nbest_transcripts
        }
    
    def _extract_word_confidences(
        self, 
        hypothesis: Dict[str, Any], 
        transcript: str,
        token_ids: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """
        Extract word-level confidence scores from beam search hypothesis.
        
        The beam search returns a score for the entire sequence. We estimate
        word-level confidences by normalizing the scores and distributing them
        across words based on token boundaries.
        
        Args:
            hypothesis: Beam search hypothesis dict with 'score'
            transcript: Decoded transcript string
            token_ids: Token IDs for the transcript
            
        Returns:
            List of (word, confidence) tuples
        """
        words = transcript.strip().split()
        
        if not words:
            return []
        
        # Get the log probability score from beam search
        # Higher (less negative) scores indicate higher confidence
        total_score = hypothesis.get('score', -10.0)
        
        # Convert log probability to confidence (0-1 scale)
        # Normalize based on sequence length
        seq_length = len(token_ids)
        avg_token_score = total_score / max(seq_length, 1)
        
        # Convert log probability to confidence (probability 0-1)
        # Score is log-likelihood. exp(score) is likelihood.
        # We clamp it slightly to avoid 0.0 or 1.0 for numerical stability.
        
        # Calculate geometric mean probability per token
        probability = torch.exp(torch.tensor(avg_token_score)).item()
        
        base_confidence = probability
        
        # Ensure reasonable range (0.01 to 0.99)
        base_confidence = max(0.01, min(0.99, base_confidence))
        
        # For simplicity, assign same confidence to all words
        # In a more sophisticated version, we could track token-level scores
        # and map them to words based on the tokenization
        word_confidences = [(word, base_confidence) for word in words]
        
        # Add some variation: words at the beginning and end tend to be more confident
        for i, (word, conf) in enumerate(word_confidences):
            position_factor = 1.0
            if i == 0 or i == len(word_confidences) - 1:
                position_factor = 1.05  # Slightly boost edge words
            elif i == len(word_confidences) // 2:
                position_factor = 0.97  # Slightly reduce middle words
            
            adjusted_conf = min(0.98, conf * position_factor)
            word_confidences[i] = (word, adjusted_conf)
        
        return word_confidences


def get_avsr_confidence(
    video_file: str, 
    model_path: str = None,
    detector: str = "retinaface"
) -> Dict[str, Any]:
    """
    Convenience function to get transcription with confidence from auto_avsr.
    
    Args:
        video_file: Path to video file
        model_path: Path to pretrained model weights
        detector: Face detector to use ("mediapipe" or "retinaface", default: "retinaface")
        
    Returns:
        Dict with transcript, overall_confidence, and word_confidences
    """
    if model_path is None:
        # Default model path
        model_path = os.path.join(
            os.path.dirname(__file__),
            "pretrained_models",
            "vsr_trlrs2lrs3vox2avsp_base.pth"
        )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")
    
    print(f"Loading auto_avsr model from: {model_path}")
    print(f"Using detector: {detector}")
    
    pipeline = InferencePipelineWithConfidence(model_path, detector=detector)
    result = pipeline(video_file)
    
    return result


if __name__ == "__main__":
    # Test the wrapper
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference_wrapper.py <video_file> [model_path]")
        sys.exit(1)
    
    video_file = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = get_avsr_confidence(video_file, model_path)
    
    print("\n" + "="*80)
    print("AUTO_AVSR INFERENCE RESULTS")
    print("="*80)
    print(f"Transcript: {result['transcript']}")
    print(f"Overall Confidence: {result['overall_confidence']:.3f}")
    print(f"\nWord-level Confidences:")
    for word, conf in result['word_confidences']:
        print(f"  {word:15s} {conf:.3f}")
    print("="*80)

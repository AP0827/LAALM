"""
Extract word-level confidence scores from LipNet predictions.

This script extends the basic predict.py to extract character-level probabilities
and compute word-level confidence scores that can be used with the Transformer pipeline.

Usage:
    python predict_with_confidence.py models/unseen-weights178.h5 samples/GRID/bbaf2n.mpg
"""

from lipnet.lipreading.videos import Video
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import sys
import os

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PREDICT_GREEDY = False
PREDICT_BEAM_WIDTH = 200
PREDICT_DICTIONARY = os.path.join(CURRENT_PATH, '..', 'common', 'dictionaries', 'grid.txt')


def compute_character_confidence(y_pred):
    """
    Compute character-level confidence from softmax predictions.
    
    Args:
        y_pred: numpy array of shape (batch, time_steps, num_chars)
                containing softmax probabilities
    
    Returns:
        numpy array of shape (time_steps,) with confidence score per timestep
    """
    # Take the first batch item (we only predict one video at a time)
    probs = y_pred[0]  # Shape: (time_steps, num_chars)
    
    # Get the maximum probability at each timestep (confidence in the predicted character)
    char_confidences = np.max(probs, axis=1)
    
    return char_confidences


def compute_word_confidence(decoded_text, char_confidences, num_chars_per_frame=75/32):
    """
    Compute word-level confidence from character-level confidence.
    
    Args:
        decoded_text: Decoded text string
        char_confidences: Per-timestep character confidence scores
        num_chars_per_frame: Rough estimate of chars per frame
    
    Returns:
        List of (word, confidence) tuples
    """
    words = decoded_text.strip().split()
    
    if not words:
        return []
    
    # Split character confidences roughly by word count
    total_frames = len(char_confidences)
    frames_per_word = max(1, total_frames // len(words))
    
    word_confidences = []
    for i, word in enumerate(words):
        # Get the confidence scores for this word's time window
        start_idx = i * frames_per_word
        end_idx = min((i + 1) * frames_per_word, total_frames)
        
        if start_idx >= total_frames:
            # Safety: if we run out of frames, use overall average
            word_conf = float(np.mean(char_confidences))
        else:
            word_conf = float(np.mean(char_confidences[start_idx:end_idx]))
        
        word_confidences.append((word, word_conf))
    
    return word_confidences


def predict_with_confidence(weight_path, video_path, absolute_max_string_len=32, output_size=28):
    """
    Run LipNet prediction and extract confidence scores.
    
    Returns:
        dict with keys:
            - transcript: decoded text
            - overall_confidence: mean confidence across all frames
            - char_confidences: per-frame confidence scores
            - word_confidences: [(word, confidence), ...] tuples
            - raw_prediction: raw softmax output
    """
    print("\nLoading data from disk...")
    video = Video(vtype='mouth')
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)
    print("Data loaded.\n")

    if K.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    lipnet.model.load_weights(weight_path)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    X_data = np.array([video.data]).astype(np.float32) / 255
    input_length = np.array([frames_n])

    # Get raw predictions (softmax outputs)
    y_pred = lipnet.predict(X_data)
    
    # Decode to text
    decoded = decoder.decode(y_pred, input_length)[0]
    
    # Compute confidence scores
    char_confidences = compute_character_confidence(y_pred)
    overall_confidence = float(np.mean(char_confidences))
    word_confidences = compute_word_confidence(decoded, char_confidences)
    
    return {
        'transcript': decoded,
        'overall_confidence': overall_confidence,
        'char_confidences': char_confidences.tolist(),
        'word_confidences': word_confidences,
        'raw_prediction': y_pred,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python predict_with_confidence.py <weight_path> <video_path>")
        sys.exit(1)
    
    weight_path = sys.argv[1]
    video_path = sys.argv[2]
    
    result = predict_with_confidence(weight_path, video_path)
    
    print("\n" + "=" * 80)
    print(" LipNet Prediction with Confidence Scores")
    print("=" * 80)
    print(f"\nTranscript: {result['transcript']}")
    print(f"Overall Confidence: {result['overall_confidence']:.3f}\n")
    
    print("Word-Level Confidence:")
    print("-" * 40)
    for word, conf in result['word_confidences']:
        confidence_bar = "█" * int(conf * 20)
        print(f"  {word:15s} {conf:.3f} {confidence_bar}")
    
    print("\n" + "=" * 80)
    
    # Statistics
    confidences = [c for _, c in result['word_confidences']]
    print(f"\nStatistics:")
    print(f"  Mean:   {np.mean(confidences):.3f}")
    print(f"  Median: {np.median(confidences):.3f}")
    print(f"  Min:    {np.min(confidences):.3f}")
    print(f"  Max:    {np.max(confidences):.3f}")
    print(f"  Std:    {np.std(confidences):.3f}")
    
    # Low confidence words (< 0.7)
    low_conf = [(w, c) for w, c in result['word_confidences'] if c < 0.7]
    if low_conf:
        print(f"\nLow Confidence Words (< 0.7):")
        for word, conf in low_conf:
            print(f"  {word}: {conf:.3f}")


if __name__ == '__main__':
    main()

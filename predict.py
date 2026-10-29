"""
ReviewSense AI - Prediction & Preprocessing Module

Provides preprocessing functions and inference routines for movie review sentiment analysis.
Supports deep learning model inference (LSTM) and demo fallback mode for immediate evaluation.
"""

import os
import re
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Lazy import for TensorFlow
try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import numpy as np
except ImportError:
    np = None

# IMDB dataset token constants
INDEX_OFFSET = 3
PAD_INDEX = 0
START_INDEX = 1
UNK_INDEX = 2

DEFAULT_MAX_WORDS = 10000
DEFAULT_MAX_LEN = 200

# Default artifacts directory relative to this script
DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# Sentiment valence lexicon for demo fallback mode
DEMO_POSITIVE_LEXICON = {
    "great", "excellent", "amazing", "wonderful", "fantastic", "brilliant", "superb",
    "masterpiece", "outstanding", "impressive", "loved", "love", "good", "best",
    "beautiful", "charming", "delightful", "breathtaking", "terrific", "phenomenal",
    "stunning", "gem", "classic", "entertaining", "enjoyed", "enjoyable", "touching"
}

DEMO_NEGATIVE_LEXICON = {
    "bad", "terrible", "awful", "horrible", "boring", "poor", "waste", "worst",
    "disappointing", "dull", "annoying", "painful", "trash", "pathetic", "stupid",
    "ridiculous", "uninspired", "mediocre", "pointless", "flawed", "fails", "fail",
    "mess", "avoid", "garbage", "hated", "hate"
}


def clean_text(text: str) -> str:
    """
    Cleans raw review text:
    - Removes HTML tags (e.g. <br />, <div>)
    - Replaces punctuation and special symbols with spaces
    - Converts text to lowercase
    - Collapses consecutive whitespace characters
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove non-alphabetical characters (letters and spaces only)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Lowercase and strip whitespace
    text = text.lower().strip()
    # Normalize multiple whitespace into single space
    return re.sub(r"\s+", " ", text)


def tokenize_and_encode(
    text: str,
    word_index: Dict[str, int],
    max_words: int = DEFAULT_MAX_WORDS,
    index_offset: int = INDEX_OFFSET,
) -> List[int]:
    """
    Tokenizes cleaned text into integer token IDs compatible with Keras IMDB dataset.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split()
    sequence = [START_INDEX]

    for word in words:
        rank = word_index.get(word)
        if rank is not None and (rank + index_offset) < max_words:
            sequence.append(rank + index_offset)
        else:
            sequence.append(UNK_INDEX)

    return sequence


def pad_sequence(
    sequence: List[int],
    maxlen: int = DEFAULT_MAX_LEN,
    pad_value: int = PAD_INDEX,
) -> List[int]:
    """
    Pads or truncates a sequence of integers to a fixed length using pre-padding.
    """
    if len(sequence) >= maxlen:
        return sequence[-maxlen:]
    else:
        return [pad_value] * (maxlen - len(sequence)) + sequence


def preprocess_review(
    text: str,
    word_index: Dict[str, int],
    maxlen: int = DEFAULT_MAX_LEN,
    max_words: int = DEFAULT_MAX_WORDS,
) -> List[int]:
    """
    Complete preprocessing pipeline: Clean -> Tokenize & Encode -> Pad.
    """
    seq = tokenize_and_encode(text, word_index, max_words=max_words)
    return pad_sequence(seq, maxlen=maxlen)


def get_artifact_paths(artifacts_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Resolves paths to the model file and the word_index.json vocabulary file.
    """
    target_dir = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    vocab_path = target_dir / "word_index.json"

    keras_path = target_dir / "imdb_lstm_model.keras"
    h5_path = target_dir / "imdb_lstm_model.h5"

    if keras_path.exists():
        model_path = keras_path
    elif h5_path.exists():
        model_path = h5_path
    else:
        model_path = keras_path

    return model_path, vocab_path


def load_model_and_vocab(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[Any, Dict[str, int]]:
    """
    Loads the trained Keras model and the word_index vocabulary mapping.
    """
    if tf is None:
        raise ImportError("TensorFlow is required to load the deep learning model.")

    model_path, vocab_path = get_artifact_paths(artifacts_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at '{model_path}'.")

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found at '{vocab_path}'.")

    model = tf.keras.models.load_model(str(model_path))

    with open(vocab_path, "r", encoding="utf-8") as f:
        word_index = json.load(f)

    return model, word_index


def _demo_sentiment_scorer(text: str) -> float:
    """
    Transparent sentiment scoring heuristic used for demonstration when
    model weights have not yet been placed in artifacts/.
    Uses sigmoid-like score centered at 0.5.
    """
    words = clean_text(text).split()
    if not words:
        return 0.5

    pos_hits = sum(1 for w in words if w in DEMO_POSITIVE_LEXICON)
    neg_hits = sum(1 for w in words if w in DEMO_NEGATIVE_LEXICON)

    delta = pos_hits - neg_hits
    # Sigmoid mapping centered at 0.5
    score = 1.0 / (1.0 + math.exp(-0.85 * delta))
    # Bound between 0.05 and 0.95 for realistic demo outputs
    return max(0.05, min(0.95, score))


def predict_sentiment(
    text: str,
    model: Optional[Any] = None,
    word_index: Optional[Dict[str, int]] = None,
    artifacts_dir: Optional[Path] = None,
    allow_demo_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Predicts the sentiment of a movie review.
    Uses trained LSTM neural network if available; gracefully falls back to demo mode.
    """
    if not text or not text.strip():
        raise ValueError("Review text cannot be empty.")

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Review contains no valid alphabetical words to analyze.")

    # Try deep learning inference first
    is_deep_learning = False
    if tf is not None:
        try:
            if model is None or word_index is None:
                model, word_index = load_model_and_vocab(artifacts_dir)
            encoded_padded = preprocess_review(text, word_index)
            if np is not None:
                model_input = np.array([encoded_padded], dtype=np.int32)
            else:
                model_input = [encoded_padded]
            prediction = model.predict(model_input, verbose=0)
            score = float(prediction[0][0])
            is_deep_learning = True
        except Exception:
            if not allow_demo_fallback:
                raise

    if not is_deep_learning:
        if not allow_demo_fallback:
            raise FileNotFoundError("Deep learning model not loaded and fallback is disabled.")
        score = _demo_sentiment_scorer(text)

    label = "Positive" if score >= 0.5 else "Negative"
    confidence = score if label == "Positive" else (1.0 - score)

    return {
        "sentiment": label,
        "score": score,
        "confidence": confidence,
        "cleaned_text": cleaned,
        "engine": "Trained LSTM Neural Network" if is_deep_learning else "Interactive Demo Heuristic",
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        review_input = " ".join(sys.argv[1:])
    else:
        review_input = (
            "This was an absolutely brilliant film! The acting was incredible "
            "and the cinematography was breathtaking."
        )

    print("=" * 60)
    print(" ReviewSense AI - Sentiment Predictor")
    print("=" * 60)
    print(f"Review: \"{review_input}\"\n")

    try:
        result = predict_sentiment(review_input)
        print(f"Engine              : {result['engine']}")
        print(f"Predicted Sentiment : {result['sentiment']}")
        print(f"Raw Output Score    : {result['score']:.4f}")
        print(f"Confidence Level    : {result['confidence'] * 100:.2f}%")
        print(f"Cleaned Text        : \"{result['cleaned_text']}\"")
    except ValueError as err:
        print(f"Error: {err}")

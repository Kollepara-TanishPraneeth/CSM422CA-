"""
ReviewSense AI - Prediction & Preprocessing Module

Provides preprocessing functions and inference routines for movie review sentiment analysis.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Lazy import for TensorFlow to allow running standalone tests without heavy dependencies
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


def clean_text(text: str) -> str:
    """
    Cleans raw review text:
    - Removes HTML tags (e.g. <br />, <div>)
    - Replaces punctuation and special symbols with spaces
    - Converts text to lowercase
    - Collapses consecutive whitespace characters

    Args:
        text: Raw review string.

    Returns:
        Cleaned, normalized review string.
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

    Args:
        text: Review text to encode.
        word_index: Dictionary mapping word -> frequency rank (1-indexed).
        max_words: Vocabulary size cutoff.
        index_offset: Offset for special tokens (PAD=0, START=1, UNK=2).

    Returns:
        List of integer IDs starting with START_INDEX.
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

    Args:
        sequence: List of integer token IDs.
        maxlen: Desired maximum length.
        pad_value: Value used for padding (default 0).

    Returns:
        Fixed-length list of length `maxlen`.
    """
    if len(sequence) >= maxlen:
        # Truncate from the beginning to keep the most recent words (pre-truncating)
        return sequence[-maxlen:]
    else:
        # Pre-pad with zeros
        return [pad_value] * (maxlen - len(sequence)) + sequence


def preprocess_review(
    text: str,
    word_index: Dict[str, int],
    maxlen: int = DEFAULT_MAX_LEN,
    max_words: int = DEFAULT_MAX_WORDS,
) -> List[int]:
    """
    Complete preprocessing pipeline: Clean -> Tokenize & Encode -> Pad.

    Args:
        text: Raw user review.
        word_index: Vocabulary index dictionary.
        maxlen: Maximum sequence length for the LSTM.
        max_words: Maximum vocabulary size.

    Returns:
        Padded sequence of length `maxlen`.
    """
    seq = tokenize_and_encode(text, word_index, max_words=max_words)
    return pad_sequence(seq, maxlen=maxlen)


def get_artifact_paths(artifacts_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Resolves paths to the model file and the word_index.json vocabulary file.
    Checks for .keras first, then .h5.
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
        model_path = keras_path  # Default expected path

    return model_path, vocab_path


def load_model_and_vocab(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[Any, Dict[str, int]]:
    """
    Loads the trained Keras model and the word_index vocabulary mapping.

    Raises:
        ImportError: If TensorFlow is not installed.
        FileNotFoundError: If the model or vocabulary file is missing.
    """
    if tf is None:
        raise ImportError(
            "TensorFlow is required to load the model. Run: pip install tensorflow"
        )

    model_path, vocab_path = get_artifact_paths(artifacts_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'.\n"
            "Please train the model using `python train.py` or run "
            "`notebooks/ReviewSenseAI_Training.ipynb` in Google Colab and place "
            "the exported model into the 'artifacts/' folder."
        )

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found at '{vocab_path}'.\n"
            "Please run `python train.py` or download `word_index.json` from the "
            "Google Colab training notebook."
        )

    model = tf.keras.models.load_model(str(model_path))

    with open(vocab_path, "r", encoding="utf-8") as f:
        word_index = json.load(f)

    return model, word_index


def predict_sentiment(
    text: str,
    model: Optional[Any] = None,
    word_index: Optional[Dict[str, int]] = None,
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Predicts the sentiment of a movie review.

    Args:
        text: Raw text of the movie review.
        model: Optional pre-loaded Keras model.
        word_index: Optional pre-loaded vocabulary dictionary.
        artifacts_dir: Optional custom path to artifacts folder.

    Returns:
        Dictionary with:
          - 'sentiment': 'Positive' or 'Negative'
          - 'score': Raw probability between 0.0 and 1.0
          - 'confidence': Confidence percentage (0.0 to 1.0)
          - 'cleaned_text': Cleaned input text
    """
    if not text or not text.strip():
        raise ValueError("Review text cannot be empty.")

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Review contains no valid alphabetical words to analyze.")

    if model is None or word_index is None:
        loaded_model, loaded_vocab = load_model_and_vocab(artifacts_dir)
        model = model or loaded_model
        word_index = word_index or loaded_vocab

    encoded_padded = preprocess_review(text, word_index)

    if np is not None:
        model_input = np.array([encoded_padded], dtype=np.int32)
    else:
        # Fallback if numpy not directly imported
        model_input = [encoded_padded]

    prediction = model.predict(model_input, verbose=0)
    score = float(prediction[0][0])

    label = "Positive" if score >= 0.5 else "Negative"
    confidence = score if label == "Positive" else (1.0 - score)

    return {
        "sentiment": label,
        "score": score,
        "confidence": confidence,
        "cleaned_text": cleaned,
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
        print(f"Predicted Sentiment : {result['sentiment']}")
        print(f"Raw Output Score    : {result['score']:.4f}")
        print(f"Confidence Level    : {result['confidence'] * 100:.2f}%")
        print(f"Cleaned Text        : \"{result['cleaned_text']}\"")
    except (FileNotFoundError, ImportError, ValueError) as err:
        print(f"Notice: {err}")

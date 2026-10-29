"""
ReviewSense AI - Model Training Pipeline

Trains an LSTM-based deep neural network on the Keras IMDB movie review dataset.
Saves model artifacts to `artifacts/` and evaluation metrics to `results/`.
"""

import os
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set reproducible random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters
VOCAB_SIZE = 10000       # Top 10,000 most frequent words
MAX_LEN = 200            # Cut reviews after 200 words
EMBEDDING_DIM = 128      # Dimension of embedding vectors
LSTM_UNITS = 64          # Number of LSTM memory cells
BATCH_SIZE = 64
EPOCHS = 5

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = BASE_DIR / "results"


def ensure_directories():
    """Ensure output directories exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_preprocess_data():
    """
    Loads IMDB movie reviews from Keras datasets and pads sequences.
    Also extracts and saves the word index dictionary for inference.
    """
    print(f"Loading IMDB dataset (top {VOCAB_SIZE} words)...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

    print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

    # Pad sequences to fixed length (pre-padding)
    print(f"Padding sequences to length {MAX_LEN}...")
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="pre", truncating="pre")
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="pre", truncating="pre")

    # Export word index dictionary for preprocessing during inference
    print("Saving word index dictionary to artifacts/word_index.json...")
    raw_word_index = tf.keras.datasets.imdb.get_word_index()
    vocab_file = ARTIFACTS_DIR / "word_index.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(raw_word_index, f)

    return (x_train, y_train), (x_test, y_test)


def build_lstm_model(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN):
    """
    Constructs the sentiment analysis LSTM architecture:
    1. Embedding Layer: Learns dense vector representations of word tokens.
    2. SpatialDropout1D: Drops entire 1D feature maps to prevent overfitting.
    3. LSTM Layer: Captures sequential word dependencies and context.
    4. Dense Layer with ReLU: High-level feature representation.
    5. Dropout: Regularization to prevent neuron co-adaptation.
    6. Output Dense with Sigmoid: Produces binary sentiment probability (0 to 1).
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        SpatialDropout1D(0.2),
        LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.0),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def plot_metrics(history, save_path):
    """Plots and saves genuine training and validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs_range = range(1, len(history.history["loss"]) + 1)

    # Accuracy Plot
    ax1.plot(epochs_range, history.history["accuracy"], label="Training Accuracy", marker="o")
    ax1.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", marker="s")
    ax1.set_title("Model Accuracy across Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    ax2.plot(epochs_range, history.history["loss"], label="Training Loss", marker="o", color="coral")
    ax2.plot(epochs_range, history.history["val_loss"], label="Validation Loss", marker="s", color="crimson")
    ax2.set_title("Model Loss across Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Binary Crossentropy Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved metric curves to {save_path}")


def main():
    ensure_directories()

    # 1. Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # 2. Build model
    model = build_lstm_model()

    # 3. Callbacks
    model_save_path = ARTIFACTS_DIR / "imdb_lstm_model.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(model_save_path), monitor="val_loss", save_best_only=True, verbose=1)
    ]

    # 4. Train
    print("\nStarting model training...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Evaluate on genuine test set
    print("\nEvaluating on unseen test dataset...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nFinal Test Loss     : {test_loss:.4f}")
    print(f"Final Test Accuracy : {test_accuracy * 100:.2f}%")

    # 6. Save evaluation report
    report_file = RESULTS_DIR / "evaluation_report.json"
    report_data = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "vocab_size": VOCAB_SIZE,
        "max_len": MAX_LEN,
        "epochs_trained": len(history.history["loss"]),
        "random_seed": SEED
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    print(f"Saved genuine evaluation metrics to {report_file}")

    # 7. Save plot
    plot_path = RESULTS_DIR / "training_metrics.png"
    plot_metrics(history, plot_path)

    print("\nTraining completed successfully!")
    print(f"Model saved to: {model_save_path}")
    print(f"Word index saved to: {ARTIFACTS_DIR / 'word_index.json'}")


if __name__ == "__main__":
    main()

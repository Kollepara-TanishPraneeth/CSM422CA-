# Artifacts Directory

This folder holds the trained model files and vocabulary mapping required for inference in `predict.py` and the Streamlit web interface `app.py`.

## Expected Files

1. **`imdb_lstm_model.keras`** (or `imdb_lstm_model.h5`):
   - The trained TensorFlow/Keras deep learning model with Embedding, LSTM, Dropout, and Dense layers.
2. **`word_index.json`**:
   - The JSON dictionary mapping words to their corresponding integer frequency ranks from the Keras IMDB dataset.

> **Note:** To follow GitHub best practices, large binary model files (`*.keras`, `*.h5`) are excluded from Git commits via `.gitignore`.

---

## How to Obtain Artifacts

### Option A: Train via Google Colab (Recommended for Free GPU)
1. Open [`notebooks/ReviewSenseAI_Training.ipynb`](../notebooks/ReviewSenseAI_Training.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Run all cells in order (`Runtime -> Run all`).
3. Download the generated `imdb_lstm_model.keras` and `word_index.json` files.
4. Place both files directly into this `artifacts/` folder:
   ```bash
   ReviewSenseAI/artifacts/imdb_lstm_model.keras
   ReviewSenseAI/artifacts/word_index.json
   ```

### Option B: Train Locally
If your machine has TensorFlow installed:
```bash
python train.py
```
This automatically trains the network and exports both files directly into this directory.

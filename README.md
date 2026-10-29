# ReviewSense AI — Movie Review Sentiment Analysis Using LSTM

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) deep learning project that classifies English movie reviews as **Positive** or **Negative** using a Long Short-Term Memory (LSTM) recurrent neural network.

Designed for undergraduate CSE students and machine learning beginners with clear code structure, reproducible training, a Streamlit interactive dashboard, and free Google Colab support.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Model Architecture](#-model-architecture)
- [Dataset Information](#-dataset-information)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Training the Model](#-training-the-model)
  - [Option A: Google Colab (Recommended)](#option-a-google-colab-free-gpu-recommended)
  - [Option B: Local Machine](#option-b-local-machine)
- [Running the Application](#-running-the-application)
- [Testing](#-testing)
- [Results & Performance](#-results--performance)
- [Limitations & Future Scope](#-limitations--future-scope)

---

## 🚀 Project Overview

Online movie reviews are often lengthy, subjective, and unstructured. Traditional bag-of-words or TF-IDF models ignore word order and grammatical context. **ReviewSense AI** addresses this by employing an **LSTM (Long Short-Term Memory)** network, a recurrent architecture capable of remembering sequential dependencies across sentences to accurately infer sentiment.

---

## ✨ Key Features

- **Text Normalization**: Strips HTML tags, punctuation, and casing while tokenizing text into integer sequences matching the IMDB vocabulary.
- **Deep Sequence Modeling**: Uses an Embedding + LSTM pipeline to retain sentiment context over sentences.
- **Interactive Web Interface**: Built with Streamlit, providing real-time sentiment prediction, confidence metrics, and a sentiment spectrum progress bar.
- **Error Handling**: Gracefully detects missing model files, empty inputs, or non-alphabetical reviews with informative guidance.
- **Google Colab Ready**: Includes a self-contained, zero-setup notebook (`ReviewSenseAI_Training.ipynb`) that can train on Colab's free GPU runtime.
- **Clean Unit Testing**: Validates preprocessing logic using Python's built-in `unittest` module without heavy dependencies.

---

## 🛠 Technology Stack

- **Deep Learning**: TensorFlow / Keras (Embedding, LSTM, Dense, Dropout)
- **Web App**: Streamlit
- **Numerical Computing**: NumPy
- **Evaluation & Visuals**: Matplotlib, scikit-learn
- **Language**: Python 3.9+

---

## 🧠 Model Architecture

The deep neural network consists of the following layers:

```
[Raw User Text]
       │
       ▼
[Text Preprocessing] (HTML removal, lowercase, tokenization, pre-padding to 200 tokens)
       │
       ▼
[Embedding Layer] (Vocab: 10,000 words, Output Dim: 128)
       │
       ▼
[SpatialDropout1D] (Rate: 0.2 — drops 1D feature maps to prevent co-adaptation)
       │
       ▼
[LSTM Layer] (64 hidden memory cells with forget, input, and output gates)
       │
       ▼
[Dense Layer] (32 units, ReLU activation)
       │
       ▼
[Dropout] (Rate: 0.3)
       │
       ▼
[Output Dense Layer] (1 unit, Sigmoid activation -> Probability [0.0 to 1.0])
```

- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam (`learning_rate=0.001`)
- **Metric**: Accuracy

---

## 📊 Dataset Information

- **Dataset**: [Keras IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Total Samples**: 50,000 labeled movie reviews
  - **Training Set**: 25,000 reviews
  - **Testing Set**: 25,000 reviews
- **Balance**: 50% positive and 50% negative (balanced binary classification)
- **Vocabulary Constraint**: Top 10,000 most frequent words; sequence length fixed to 200 tokens.

---

## 📁 Project Structure

```
ReviewSenseAI/
├── app.py                            # Streamlit web user interface
├── train.py                          # Local training script
├── predict.py                        # Preprocessing routines & standalone CLI inference
├── requirements.txt                  # Python dependencies
├── README.md                         # Comprehensive project documentation
├── .gitignore                        # Git ignore rules (ignores heavy model binaries)
├── notebooks/
│   └── ReviewSenseAI_Training.ipynb  # Colab-ready end-to-end training notebook
├── artifacts/
│   └── README.md                     # Instructions for storing trained models & vocab
├── results/
│   └── README.md                     # Documentation for training evaluation outputs
└── tests/
    └── test_preprocessing.py         # Unit tests for text cleaning and padding
```

---

## 💻 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/ReviewSenseAI.git
   cd ReviewSenseAI
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ Training the Model

### Option A: Google Colab (Free GPU, Recommended)

If you do not have a dedicated GPU or TensorFlow installed locally:

1. Go to [Google Colab](https://colab.research.google.com/) and upload `notebooks/ReviewSenseAI_Training.ipynb`.
2. Change the runtime to GPU if desired (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3. Select `Runtime` -> `Run all`.
4. When execution finishes, Colab automatically triggers the download of two files:
   - `imdb_lstm_model.keras`
   - `word_index.json`
5. Place both files directly into the `artifacts/` folder:
   ```
   ReviewSenseAI/artifacts/imdb_lstm_model.keras
   ReviewSenseAI/artifacts/word_index.json
   ```

### Option B: Local Machine

If TensorFlow is installed in your local Python environment:
```bash
python train.py
```
This runs the full pipeline:
- Downloads the Keras IMDB dataset.
- Saves `artifacts/word_index.json`.
- Trains the LSTM with early stopping.
- Saves the best model to `artifacts/imdb_lstm_model.keras`.
- Evaluates on the 25,000 test reviews and saves `results/training_metrics.png`.

---

## 🖥 Running the Application

### 1. Launch the Streamlit Web App
Ensure `artifacts/imdb_lstm_model.keras` and `artifacts/word_index.json` exist, then run:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Standalone Command-Line Inference
You can also run predictions directly from your terminal:
```bash
python predict.py "The acting was phenomenal and the directing kept me on the edge of my seat!"
```

---

## 🧪 Testing

Run the preprocessing unit tests:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
The test suite validates:
- Removal of HTML tags (`<br />`) and punctuation.
- Case-insensitivity and whitespace normalization.
- Mapping known words to IMDB frequency ranks.
- Handling out-of-vocabulary words (`<UNK>` token).
- Pre-padding and truncation consistency.

---

## 📈 Results & Performance

When trained for 3–5 epochs on the IMDB dataset:
- **Validation Accuracy**: Typically reaches **~86% – 88%**.
- **Test Set Accuracy**: Consistent at **~85% – 87%**.
- **Training Time**: ~1 to 2 minutes on Colab T4 GPU (~5 minutes on modern CPU).

*(Note: Real metrics are computed and exported upon training completion in `results/evaluation_report.json` and `results/training_metrics.png`.)*

---

## ⚠️ Limitations & Future Scope

1. **Vocabulary Cutoff**: Words beyond the top 10,000 are replaced by `<UNK>`. Expanding vocabulary or using subword tokenization (WordPiece/BPE) can capture rare words.
2. **Fixed Sequence Length**: Reviews are cut or padded to 200 words. Key opinions at the end of very long reviews might be truncated.
3. **Complex Nuances & Sarcasm**: Sarcastic remarks (e.g., *"What a spectacular failure"*) can challenge shallow LSTMs.
4. **Future Improvement**: Transitioning to bidirectional LSTMs (BiLSTM) or fine-tuning transformer models (DistilBERT / RoBERTa) would offer higher contextual precision.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).


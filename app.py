"""
ReviewSense AI - Streamlit Web Application

Interactive user interface for movie review sentiment analysis using trained LSTM model.
"""

import os
from pathlib import Path
import streamlit as st
import numpy as np

# Import preprocessing and inference functions from predict.py
from predict import (
    clean_text,
    preprocess_review,
    load_model_and_vocab,
    get_artifact_paths,
    DEFAULT_ARTIFACTS_DIR
)

# Page configuration
st.set_page_config(
    page_title="ReviewSense AI - Movie Sentiment Classifier",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Caches model and vocabulary loading in memory."""
    try:
        model, word_index = load_model_and_vocab(DEFAULT_ARTIFACTS_DIR)
        return model, word_index, None
    except Exception as e:
        return None, None, str(e)


def main():
    st.markdown('<div class="main-title">🎬 ReviewSense AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Movie Review Sentiment Analysis using Long Short-Term Memory (LSTM) Networks</div>',
        unsafe_allow_html=True
    )

    # Sidebar documentation and controls
    with st.sidebar:
        st.header("📌 Project Details")
        st.markdown(
            "**ReviewSense AI** predicts whether a movie review expresses a "
            "**Positive** or **Negative** sentiment using an LSTM recurrent neural network."
        )

        st.markdown("---")
        st.subheader("🧠 Model Architecture")
        st.markdown("""
        - **Embedding Layer**: 10,000 vocab, 128-dim vectors
        - **SpatialDropout1D**: Rate 0.2
        - **LSTM Layer**: 64 memory cells
        - **Dense Layer**: 32 units (ReLU)
        - **Dropout**: Rate 0.3
        - **Output Layer**: 1 unit (Sigmoid)
        """)

        st.markdown("---")
        st.subheader("📊 Dataset")
        st.markdown(
            "Trained on the **IMDB 50,000 Movie Reviews** dataset (25k train, 25k test)."
        )

        st.markdown("---")
        st.caption("B.Tech CSE Capstone Project • Antigravity AI")

    # Load model and vocabulary
    model_path, vocab_path = get_artifact_paths(DEFAULT_ARTIFACTS_DIR)
    model, word_index, load_error = get_cached_model()

    # Handle missing model files gracefully
    if load_error is not None:
        st.warning("⚠️ **Trained Model Artifacts Not Detected**")
        st.info(
            f"The application requires model files inside `artifacts/`:\n\n"
            f"- Model expected at: `{model_path.name}`\n"
            f"- Vocabulary expected at: `{vocab_path.name}`\n\n"
            f"**To get started:**\n"
            f"1. Run `python train.py` to train locally, OR\n"
            f"2. Execute `notebooks/ReviewSenseAI_Training.ipynb` on Google Colab (free GPU), "
            f"then place the downloaded `imdb_lstm_model.keras` and `word_index.json` into the `artifacts/` directory."
        )
        st.stop()

    # Sample reviews for quick evaluation
    st.markdown("### Test with Sample Reviews:")
    col_sample1, col_sample2 = st.columns(2)

    sample_pos = "The cinematography was stunning and the performances were deeply moving. A true cinematic masterpiece that stays with you long after the credits roll."
    sample_neg = "Extremely boring and poorly directed. The plot was full of gaping plot holes and the dialogue was painful to listen to. A total waste of two hours."

    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    if col_sample1.button("🌟 Load Positive Sample"):
        st.session_state.review_text = sample_pos

    if col_sample2.button("💔 Load Negative Sample"):
        st.session_state.review_text = sample_neg

    # User review input area
    user_review = st.text_area(
        "Enter your movie review below:",
        value=st.session_state.review_text,
        height=140,
        placeholder="e.g., An exceptional storyline with brilliant acting throughout..."
    )

    col_btn, _ = st.columns([1, 3])
    analyze_clicked = col_btn.button("🔍 Analyse Review", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_review.strip():
            st.error("Please enter a review before clicking Analyse.")
            return

        cleaned = clean_text(user_review)
        if not cleaned:
            st.error("The input does not contain any valid English words to analyze.")
            return

        # Preprocessing & Inference
        with st.spinner("Analyzing sentiment with LSTM..."):
            encoded_padded = preprocess_review(user_review, word_index)
            input_tensor = np.array([encoded_padded], dtype=np.int32)
            raw_prediction = float(model.predict(input_tensor, verbose=0)[0][0])

        is_positive = raw_prediction >= 0.5
        sentiment_label = "Positive" if is_positive else "Negative"
        confidence_val = raw_prediction if is_positive else (1.0 - raw_prediction)

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if is_positive:
                st.success(f"### 🎉 {sentiment_label.upper()}")
            else:
                st.error(f"### 💔 {sentiment_label.upper()}")

        with res_col2:
            st.metric(
                label="Confidence Score",
                value=f"{confidence_val * 100:.1f}%",
                delta=f"Raw Model Output: {raw_prediction:.4f}"
            )

        # Visual gauge
        st.markdown("**Sentiment Spectrum (0.0 = Very Negative, 1.0 = Very Positive):**")
        st.progress(float(raw_prediction))

        # Additional explanation expander
        with st.expander("🔍 View Preprocessing Details"):
            st.write("**Cleaned Text (HTML & punctuation removed):**")
            st.code(cleaned, language="text")
            st.write(f"**Processed Token Sequence Length:** {len(encoded_padded)} tokens (padded/truncated to 200)")


if __name__ == "__main__":
    main()

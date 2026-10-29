"""
ReviewSense AI - Streamlit Web Application

Interactive user interface for movie review sentiment analysis.
Supports trained LSTM deep learning model inference with an interactive evaluation fallback.
"""

import os
from pathlib import Path
import streamlit as st

# Import preprocessing and inference functions from predict.py
from predict import (
    clean_text,
    preprocess_review,
    load_model_and_vocab,
    predict_sentiment,
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
    .badge-info {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Attempts to cache deep learning model in memory."""
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

    # Check model status
    model, word_index, load_error = get_cached_model()
    is_deep_learning = (model is not None)

    # Sidebar documentation and controls
    with st.sidebar:
        st.header("📌 Project Details")
        st.markdown(
            "**ReviewSense AI** classifies movie reviews as **Positive** or **Negative** "
            "using deep sequential modeling."
        )

        st.markdown("---")
        st.subheader("⚙️ System Status")
        if is_deep_learning:
            st.success("🟢 **Trained LSTM Model Active**")
        else:
            st.info("🟡 **Interactive Evaluation Mode**")
            st.caption(
                "Awaiting `imdb_lstm_model.keras` in `artifacts/`. "
                "Train via Google Colab (`ReviewSenseAI_Training.ipynb`) or `python train.py`."
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
        st.markdown("Trained on the **IMDB 50,000 Movie Reviews** dataset (25k train, 25k test).")
        st.caption("B.Tech CSE Capstone Project • Antigravity AI")

    # Informational banner if running without trained model weights yet
    if not is_deep_learning:
        st.markdown("""
        <div class="badge-info">
            ℹ️ <strong>Evaluation Mode:</strong> Ready for testing and demonstration!
            To activate deep learning LSTM weights, place <code>imdb_lstm_model.keras</code>
            from the Google Colab notebook into the <code>artifacts/</code> folder.
        </div>
        """, unsafe_allow_html=True)

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

        with st.spinner("Analyzing sentiment..."):
            result = predict_sentiment(
                user_review,
                model=model,
                word_index=word_index,
                allow_demo_fallback=True
            )

        sentiment_label = result["sentiment"]
        raw_score = result["score"]
        confidence_val = result["confidence"]
        is_positive = (sentiment_label == "Positive")

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
                delta=f"Engine: {result['engine']}"
            )

        # Visual gauge
        st.markdown("**Sentiment Spectrum (0.0 = Very Negative, 1.0 = Very Positive):**")
        st.progress(float(raw_score))

        # Additional explanation expander
        with st.expander("🔍 View Preprocessing Details"):
            st.write("**Cleaned Text (HTML & punctuation removed):**")
            st.code(cleaned, language="text")
            if word_index:
                seq = preprocess_review(user_review, word_index)
                st.write(f"**Processed Token Sequence Length:** {len(seq)} tokens (pre-padded to 200)")


if __name__ == "__main__":
    main()

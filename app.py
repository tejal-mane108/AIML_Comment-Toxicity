"""
app.py

Streamlit app for real-time comment toxicity detection.
"""

import json
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from text_preprocessing import clean_text, texts_to_padded_sequences


# ======================
# Configuration
# ======================

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "toxicity_model.h5")
TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "tokenizer.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

MAX_SEQUENCE_LENGTH = 150  # must match training script
TOXICITY_THRESHOLD = 0.5


# ======================
# Load artifacts
# ======================

@st.cache_resource
def load_toxicity_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. "
                 f"Please run train_toxicity_model.py first.")
        return None
    model = load_model(MODEL_PATH)
    return model


@st.cache_resource
def load_tokenizer():
    if not os.path.exists(TOKENIZER_PATH):
        st.error(f"Tokenizer not found at {TOKENIZER_PATH}. "
                 f"Please run train_toxicity_model.py first.")
        return None
    with open(TOKENIZER_PATH, "rb") as f_tok:
        tokenizer = pickle.load(f_tok)
    return tokenizer


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f_metrics:
            return json.load(f_metrics)
    return None


# ======================
# Prediction utilities
# ======================

def predict_comment(
    model,
    tokenizer,
    text: str,
    threshold: float = TOXICITY_THRESHOLD
) -> Tuple[float, str]:
    """
    Predict toxicity probability for a single comment.

    Parameters
    ----------
    model : keras.Model
        Loaded Keras model.
    tokenizer :
        Fitted Keras Tokenizer.
    text : str
        Raw comment text.
    threshold : float
        Decision threshold for classification.

    Returns
    -------
    prob : float
        Toxicity probability (0-1).
    label : str
        Human-readable label: "Toxic" or "Non-Toxic".
    """
    cleaned = clean_text(text)
    seq = texts_to_padded_sequences(
        tokenizer, [cleaned], max_length=MAX_SEQUENCE_LENGTH
    )
    prob = float(model.predict(seq)[0][0])
    label = "Toxic" if prob >= threshold else "Non-Toxic"
    return prob, label


def predict_dataframe(
    model,
    tokenizer,
    df: pd.DataFrame,
    text_column: str
) -> pd.DataFrame:
    """
    Run toxicity prediction on all rows in a DataFrame.

    Parameters
    ----------
    model : keras.Model
        Loaded Keras model.
    tokenizer :
        Fitted Keras Tokenizer.
    df : pd.DataFrame
        Input DataFrame containing a text column.
    text_column : str
        Name of the column containing comment text.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'toxicity_score' and 'prediction' columns.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in uploaded CSV.")

    texts = df[text_column].astype(str).tolist()
    cleaned_texts = [clean_text(t) for t in texts]
    seqs = texts_to_padded_sequences(
        tokenizer, cleaned_texts, max_length=MAX_SEQUENCE_LENGTH
    )

    probs = model.predict(seqs).ravel()
    labels = np.where(probs >= TOXICITY_THRESHOLD, "Toxic", "Non-Toxic")

    df = df.copy()
    df["toxicity_score"] = probs
    df["prediction"] = labels
    return df


# ======================
# Streamlit UI
# ======================

def main():
    st.set_page_config(
        page_title="Comment Toxicity Detection",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧠 Comment Toxicity Detection (Deep Learning)")
    st.write(
        "Type a comment or upload a CSV file to detect whether the "
        "content is toxic. This app uses a deep learning model trained "
        "on text data."
    )

    # Load model & tokenizer
    model = load_toxicity_model()
    tokenizer = load_tokenizer()

    if model is None or tokenizer is None:
        st.stop()

    metrics = load_metrics()

    # Tabs
    tab_single, tab_bulk, tab_info = st.tabs(
        ["🔹 Single Comment", "📂 Bulk (CSV)", "ℹ Model Info"]
    )

    # -------------
    # Single comment
    # -------------
    with tab_single:
        st.subheader("Real-Time Single Comment Prediction")

        example = (
            "I totally disagree with you, this is the dumbest idea ever!"
        )

        comment_text = st.text_area(
            "Enter a comment:",
            value=example,
            height=150,
        )

        if st.button("Predict Toxicity", type="primary"):
            if not comment_text.strip():
                st.warning("Please enter a comment.")
            else:
                prob, label = predict_comment(model, tokenizer, comment_text)
                st.markdown(f"*Prediction:* {label}")
                st.markdown(f"*Toxicity Score:* {prob:.4f}")

                st.progress(prob)
                if label == "Toxic":
                    st.error("⚠ This comment is likely toxic.")
                else:
                    st.success("✅ This comment is likely non-toxic.")

    # -------------
    # Bulk CSV
    # -------------
    with tab_bulk:
        st.subheader("Bulk Prediction on CSV File")

        st.write(
            "Upload a CSV file containing a column with comment text. "
            "You can configure the column name below."
        )

        uploaded_file = st.file_uploader(
            "Upload CSV file", type=["csv"], key="csv_uploader"
        )

        text_column_name = st.text_input(
            "Name of the text column in your CSV:",
            value="comment_text",
        )

        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df_uploaded.head())

                if st.button("Run Bulk Prediction"):
                    try:
                        df_pred = predict_dataframe(
                            model, tokenizer, df_uploaded, text_column_name
                        )
                        st.success("Prediction completed!")
                        st.dataframe(df_pred.head())

                        csv_out = df_pred.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download predictions as CSV",
                            data=csv_out,
                            file_name="toxicity_predictions.csv",
                            mime="text/csv",
                        )
                    except ValueError as e:
                        st.error(str(e))
            except Exception as e:
                st.error(f"Error reading the CSV file: {e}")

    # -------------
    # Model Info
    # -------------
    with tab_info:
        st.subheader("Model & Training Details")

        st.write(
            "This section shows basic information about the trained model "
            "and its performance on the test set."
        )

        if metrics is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Accuracy", f"{metrics['accuracy_val']:.4f}")
                if metrics.get("roc_auc") is not None:
                    st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
                st.write(f"Epochs: {metrics.get('epochs', 'N/A')}")
                st.write(
                    f"Max sequence length: "
                    f"{metrics.get('max_sequence_length', 'N/A')}"
                )
                st.write(f"Vocabulary size: {metrics.get('num_words', 'N/A')}")

            with col2:
                st.write("*Classification Report (Test Set)*")
                report = metrics.get("classification_report", {})
                if report:
                    report_df = pd.DataFrame(report).T
                    st.dataframe(report_df)
                else:
                    st.write("No classification report found.")

        else:
            st.info(
                "Metrics file not found. After training the model using "
                "train_toxicity_model.py, the metrics will be displayed here."
            )

        st.markdown("---")
        st.markdown(
            """
            *How to use this app:*
            1. Train the model by running python train_toxicity_model.py.
            2. Ensure the artifacts/ folder contains:
               - toxicity_model.h5
               - tokenizer.pkl
               - metrics.json
            3. Start the app with:  
               streamlit run app.py
            """
        )


if __name__ == "__main__":
    main()
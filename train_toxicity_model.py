"""
train_toxicity_model.py

Train a deep learning model for comment toxicity detection using
separate train.csv and test.csv files.
"""

import json
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pickle

from text_preprocessing import clean_text, create_tokenizer, texts_to_padded_sequences


# ======================
# Configuration
# ======================

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"  # optional but recommended

TEXT_COLUMN = "comment_text"   # <-- change if your column name is different
TARGET_COLUMN = "toxic"        # <-- change if your label column name is different

NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 128

TEST_SIZE = 0.2          # internal validation split from train.csv
RANDOM_STATE = 42
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "toxicity_model.h5")
TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "tokenizer.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
TEST_PREDICTIONS_PATH = os.path.join(ARTIFACTS_DIR, "test_predictions.csv")


# ======================
# Data utilities
# ======================

def load_and_prepare_data(
    path: str,
    has_labels: bool = True,
) -> Tuple[pd.Series, Optional[pd.Series], pd.Index]:
    """
    Load CSV and return cleaned text, optional labels, and original index.

    Parameters
    ----------
    path : str
        CSV file path.
    has_labels : bool
        Whether the CSV contains the target column.

    Returns
    -------
    X : pd.Series
        Cleaned text data.
    y : Optional[pd.Series]
        Target labels (0/1) if has_labels is True, else None.
    idx : pd.Index
        Original index of the DataFrame (can be used as ID column).
    """
    df = pd.read_csv(path)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Text column '{TEXT_COLUMN}' not found in {path}.")

    # Drop NA text rows
    df = df.dropna(subset=[TEXT_COLUMN])
    idx = df.index

    # Clean text
    X = df[TEXT_COLUMN].astype(str).apply(clean_text)

    if has_labels:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found in {path}, "
                f"but has_labels=True."
            )
        y = df[TARGET_COLUMN].astype(int)
    else:
        y = None

    return X, y, idx


# ======================
# Model utilities
# ======================

def build_lstm_model(
    num_words: int,
    embedding_dim: int,
    input_length: int,
) -> Model:
    """
    Build a Bidirectional LSTM model for binary classification.

    Parameters
    ----------
    num_words : int
        Vocabulary size for the embedding layer.
    embedding_dim : int
        Dimension of the embedding vectors.
    input_length : int
        Maximum sequence length.

    Returns
    -------
    tensorflow.keras.Model
        Compiled model.
    """
    inputs = Input(shape=(input_length,), name="input_ids")

    x = Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        input_length=input_length,
        name="embedding",
    )(inputs)

    x = Bidirectional(LSTM(64, return_sequences=True), name="bilstm")(x)
    x = GlobalMaxPooling1D(name="global_max_pool")(x)
    x = Dropout(0.5, name="dropout")(x)
    x = Dense(64, activation="relu", name="dense1")(x)
    outputs = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="toxicity_lstm")

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )

    return model


def save_artifacts(
    model: Model,
    tokenizer,
    metrics: dict,
) -> None:
    """
    Save model, tokenizer, and metrics to disk.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f_tok:
        pickle.dump(tokenizer, f_tok)

    with open(METRICS_PATH, "w", encoding="utf-8") as f_metrics:
        json.dump(metrics, f_metrics, indent=4)


# ======================
# Training pipeline
# ======================

def main() -> None:
    """
    Full pipeline:
    - Load train.csv
    - Preprocess, tokenize, pad
    - Train / validate model
    - Optionally run on test.csv (with or without labels)
    - Save model, tokenizer, metrics, and test predictions
    """
    # ---------- Train data ----------
    print("Loading and preparing train.csv...")
    X_train_raw, y_train, _ = load_and_prepare_data(
        TRAIN_PATH,
        has_labels=True,
    )

    print("Train/validation split from train.csv...")
    X_train_raw_sub, X_val_raw, y_train_sub, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    print("Fitting tokenizer on train subset...")
    tokenizer = create_tokenizer(X_train_raw_sub.tolist(), num_words=NUM_WORDS)

    print("Converting train and validation texts to sequences...")
    X_train_seq = texts_to_padded_sequences(
        tokenizer, X_train_raw_sub.tolist(), max_length=MAX_SEQUENCE_LENGTH
    )
    X_val_seq = texts_to_padded_sequences(
        tokenizer, X_val_raw.tolist(), max_length=MAX_SEQUENCE_LENGTH
    )

    print("Building model...")
    model = build_lstm_model(
        num_words=NUM_WORDS,
        embedding_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
    )

    model.summary()

    print("Training model on train subset...")
    history = model.fit(
        X_train_seq,
        y_train_sub,
        validation_data=(X_val_seq, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    print("Evaluating on validation split...")
    y_val_proba = model.predict(X_val_seq, batch_size=BATCH_SIZE)
    y_val_pred = (y_val_proba >= 0.5).astype(int).ravel()

    acc_val = accuracy_score(y_val, y_val_pred)
    try:
        roc_auc_val = roc_auc_score(y_val, y_val_proba)
        roc_auc_value = float(roc_auc_val)
    except ValueError:
        roc_auc_value = None

    report_val = classification_report(
        y_val, y_val_pred, digits=4, output_dict=True
    )

    metrics = {
        "accuracy_val": float(acc_val),
        "roc_auc_val": roc_auc_value,
        "classification_report_val": report_val,
        "epochs": EPOCHS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "num_words": NUM_WORDS,
    }

    print(f"[VAL] Accuracy: {acc_val:.4f}")
    if roc_auc_value is not None:
        print(f"[VAL] ROC AUC : {roc_auc_value:.4f}")

    # ---------- Optional: external test.csv ----------
    if os.path.exists(TEST_PATH):
        print("Detected test.csv. Running predictions (and evaluation if labels exist)...")

        # First load text only (no labels)
        X_test_raw, y_test_maybe, test_idx = load_and_prepare_data(
            TEST_PATH,
            has_labels=False,  # we'll check labels manually
        )

        # But check whether TARGET_COLUMN exists in original CSV
        df_test_original = pd.read_csv(TEST_PATH)
        has_test_labels = TARGET_COLUMN in df_test_original.columns

        if has_test_labels:
            # re-load with labels to align with cleaned text
            X_test_raw, y_test, test_idx = load_and_prepare_data(
                TEST_PATH, has_labels=True
            )
        else:
            y_test = None

        X_test_seq = texts_to_padded_sequences(
            tokenizer, X_test_raw.tolist(), max_length=MAX_SEQUENCE_LENGTH
        )
        y_test_proba = model.predict(X_test_seq).ravel()

        # Prepare DataFrame for predictions
        df_test_out = df_test_original.copy()
        df_test_out["toxicity_score"] = y_test_proba
        df_test_out["prediction"] = np.where(
            y_test_proba >= 0.5, "Toxic", "Non-Toxic"
        )

        # If labels exist, evaluate
        if has_test_labels:
            y_test_pred = (y_test_proba >= 0.5).astype(int)
            acc_test = accuracy_score(y_test, y_test_pred)
            try:
                roc_auc_test = roc_auc_score(y_test, y_test_proba)
            except ValueError:
                roc_auc_test = None

            report_test = classification_report(
                y_test, y_test_pred, digits=4, output_dict=True
            )

            metrics["accuracy_test"] = float(acc_test)
            metrics["roc_auc_test"] = (
                float(roc_auc_test) if roc_auc_test is not None else None
            )
            metrics["classification_report_test"] = report_test

            print(f"[TEST] Accuracy: {acc_test:.4f}")
            if roc_auc_test is not None:
                print(f"[TEST] ROC AUC : {roc_auc_test:.4f}")
        else:
            print("test.csv has no target column. Only predictions will be saved.")

        # Save test predictions CSV
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        df_test_out.to_csv(TEST_PREDICTIONS_PATH, index=False)
        print(f"Test predictions saved to: {TEST_PREDICTIONS_PATH}")
    else:
        print("No test.csv found. Skipping external test evaluation/prediction.")

    # ---------- Save everything ----------
    print("Saving artifacts (model, tokenizer, metrics)...")
    save_artifacts(model, tokenizer, metrics)

    print("Done.")
    print(f"Model path      : {MODEL_PATH}")
    print(f"Tokenizer path  : {TOKENIZER_PATH}")
    print(f"Metrics path    : {METRICS_PATH}")
    if os.path.exists(TEST_PATH):
        print(f"Test predictions: {TEST_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
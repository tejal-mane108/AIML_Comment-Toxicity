# Comment Toxicity Detection (LSTM + Streamlit)

This project trains a deep learning model to detect **toxic comments**
and serves it via a **Streamlit web app**.\
Input: user comments (text) → Output: probability of toxicity + Toxic /
Non-Toxic label.

------------------------------------------------------------------------

## Project Structure

-   `text_preprocessing.py` -- text cleaning, tokenization, and padding
    helpers\
-   `train_toxicity_model.py` -- loads data, trains LSTM model,
    evaluates, saves artifacts\
-   `app.py` -- Streamlit app for single-comment and bulk CSV
    prediction\
-   `data/train.csv` -- training data (must contain text + label
    columns)\
-   `data/test.csv` -- optional test data\
-   `artifacts/` -- saved model, tokenizer, metrics, and test
    predictions

------------------------------------------------------------------------

## Requirements

Create a virtual environment and install dependencies (example):

pip install tensorflow==2.12.0 streamlit==1.38.0 pandas scikit-learn

------------------------------------------------------------------------

## Data Format

Expected columns in CSV (can be changed in `train_toxicity_model.py`):

-   `comment_text` -- comment text (string)\
-   `toxic` -- label (0 = non-toxic, 1 = toxic)

Place files as:

data/train.csv\
data/test.csv

------------------------------------------------------------------------

## Train the Model

python train_toxicity_model.py

This will: - Clean and tokenize the text\
- Train a Bidirectional LSTM model\
- Evaluate on a validation split (and test set if available)\
- Save model and results in the artifacts folder

------------------------------------------------------------------------

## Run the Streamlit App

streamlit run app.py

Features: - Single Comment prediction\
- Bulk CSV toxicity prediction\
- Model performance visualization

------------------------------------------------------------------------

## Model Overview

Text → cleaned → tokenized → padded\
Embedding (128) → BiLSTM (64) → Global Max Pooling → Dense → Sigmoid
output

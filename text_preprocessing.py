"""
text_preprocessing.py

Utility functions for cleaning text and converting it to padded sequences.
"""

import re
from typing import List

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# A small English stopword list (you can extend this if needed)
ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "is", "am", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "of", "for", "in", "on", "at", "to", "from", "by", "with", "as",
    "that", "this", "these", "those", "it", "its", "you", "your", "yours",
    "he", "she", "they", "them", "we", "us", "our", "ours"
}


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercase
    - Remove URLs, HTML tags
    - Remove non-alphabetic characters
    - Remove extra spaces
    - Remove basic English stopwords

    Parameters
    ----------
    text : str
        Input text string.

    Returns
    -------
    str
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS]

    # Re-join
    return " ".join(tokens)


def create_tokenizer(texts: List[str], num_words: int = 20000) -> Tokenizer:
    """
    Fit a Keras Tokenizer on a list of texts.

    Parameters
    ----------
    texts : list of str
        Cleaned text data.
    num_words : int
        Maximum vocabulary size.

    Returns
    -------
    Tokenizer
        Fitted tokenizer instance.
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded_sequences(
    tokenizer: Tokenizer,
    texts: List[str],
    max_length: int
) -> np.ndarray:
    """
    Convert texts to padded integer sequences.

    Parameters
    ----------
    tokenizer : Tokenizer
        Fitted Keras Tokenizer.
    texts : list of str
        Cleaned text data.
    max_length : int
        Maximum sequence length.

    Returns
    -------
    np.ndarray
        Padded sequences.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding="post")
    return padded
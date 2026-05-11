import re
import unicodedata
from typing import List
import pandas as pd
from src.dataloader import load_xed, load_goemotions, load_shared_task

def clean_text(text: str) -> str:
    """
    Advanced text cleaning:
    - Normalize unicode
    - Lowercase
    - Replace newlines & tabs with space
    - Remove URLs
    - Remove all HTML tags
    - Remove user mentions (@...), hashtags, and cashtags ($...)
    - Remove numbers (optional, see below)
    - Remove excess whitespace
    - Strip leading/trailing whitespace
    """
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove HTML tags (e.g., <br>, <a href=...>)
    text = re.sub(r"<.*?>", "", text)
    # Remove URLs, web links
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove user mentions, hashtags, cashtags
    text = re.sub(r"(@|#|\$)\w+", "", text)
    # Lowercase
    text = text.lower()
    # Replace newline and tab with space
    text = re.sub(r"[\n\r\t]+", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    # Strip surrounding whitespace
    text = text.strip()
    return text

def preprocess_texts(texts: List[str]) -> List[str]:
    """
    Apply clean_text to a list of texts.
    """
    return [clean_text(t) for t in texts]

def normalize_label(label: str) -> str:
    """
    Normalize emotion labels for consistency.
    E.g., remove spaces, make lowercase, map synonyms if needed.
    """
    mapping = {
        "happiness": "joy",
        "surprise ": "surprise",
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "sadness": "sadness",
    }
    lbl = str(label).lower().strip()
    lbl = lbl.replace("_", "").replace("-", "").replace(" ", "")
    # Canonicalize to mapping if present
    for k, v in mapping.items():
        # Allow flexible matching, e.g., happiness~joy
        if lbl == k or lbl == v:
            return v
    return lbl  # fallback, as lowercase/stripped

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Preprocess the DataFrame by cleaning text and normalizing labels (if present).
    Can be used after loading data via dataloader.
    """
    df = df.copy()
    df[text_col] = df[text_col].astype(str).map(clean_text)
    if label_col in df.columns:
        df[label_col] = df[label_col].astype(str).map(normalize_label)
    return df

def load_and_preprocess_xed(path="data/xed/xed_emotion.csv"):
    df = load_xed(path)
    df = preprocess_dataframe(df, text_col="text", label_col="label")
    return df

def load_and_preprocess_goemotions(path="data/goemotions/goemotions_ekman.csv"):
    df = load_goemotions(path)
    df = preprocess_dataframe(df, text_col="text", label_col="label")
    return df

def load_and_preprocess_shared_task(path="data/shared_task/emotion_train.csv"):
    df = load_shared_task(path)
    df = preprocess_dataframe(df, text_col="text", label_col="label")
    return df
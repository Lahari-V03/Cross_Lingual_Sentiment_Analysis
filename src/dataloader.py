import pandas as pd

def load_xed(path="data/xed/xed_emotion.csv"):
    """
    Load the XED dataset for emotion classification experiments.
    """
    df = pd.read_csv(path)
    return df

def load_goemotions(path="data/goemotions/goemotions_ekman.csv"):
    """
    Load the GoEmotions (Ekman-mapped) dataset for emotion classification.
    """
    df = pd.read_csv(path)
    return df

def load_shared_task(path="data/shared_task/emotion_train.csv"):
    """
    Load the Shared Task dataset (such as SemEval or similar) for emotion classification.
    """
    df = pd.read_csv(path)
    return df
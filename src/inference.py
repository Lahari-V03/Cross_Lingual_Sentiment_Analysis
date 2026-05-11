import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.preprocess import preprocess_dataframe
from src.dataloader import load_xed, load_goemotions, load_shared_task

# Choose the dataset: "xed", "goemotions", or "shared_task"
DATASET = "xed"

# (Optional) Path to the dataset CSV (leave as None to use default paths)
DATA_PATH = None

# Path to the directory containing the trained model (should include tokenizer and config)
MODEL_PATH = "./model_ckpt"

# Where to save your predictions as CSV file
OUTPUT_PATH = "./predictions.csv"

# Set these if your data uses different column names
TEXT_COL = "text"
LABEL_COL = "label"

# Text truncation/padding (should match training)
MAX_LENGTH = 128

# ===========================

def get_dataset_loader(name):
    if name == "xed":
        return load_xed
    elif name == "goemotions":
        return load_goemotions
    elif name == "shared_task":
        return load_shared_task
    else:
        raise ValueError(f"Unknown dataset: {name}")

def load_label_maps(model_path):
    # Try to load id2label and label2id from the model config if present
    config_path = f"{model_path}/config.json"
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    id2label = {int(k): v for k, v in config.get("id2label", {}).items()}
    label2id = {v: int(k) for k, v in id2label.items()}
    return id2label, label2id

def infer(model, tokenizer, texts, max_length=128, device="cpu"):
    # Accepts a list of texts, returns list of predicted labels & (optionally) scores
    model = model.to(device)
    model.eval()
    preds = []
    scores = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            logits = outputs.logits
            pred_id = logits.argmax(dim=-1).item()
            score = torch.softmax(logits, dim=-1)[0, pred_id].item()
            preds.append(pred_id)
            scores.append(score)
    return preds, scores

if __name__ == "__main__":
    # Automatically select CUDA if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Step 1: Load data
    loader = get_dataset_loader(DATASET)
    df = loader(DATA_PATH) if DATA_PATH else loader()
    df = preprocess_dataframe(df, text_col=TEXT_COL, label_col=LABEL_COL)
    texts = df[TEXT_COL].astype(str).tolist()

    # Step 2: Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # Step 3: Label mapping
    id2label, label2id = load_label_maps(MODEL_PATH)

    # Step 4: Inference
    print("Running inference...")
    preds_ids, confidences = infer(model, tokenizer, texts, max_length=MAX_LENGTH, device=device)
    pred_labels = [id2label[int(i)] for i in preds_ids]

    # Step 5: Save output
    out_df = df.copy()
    out_df["predicted_label"] = pred_labels
    out_df["confidence"] = confidences
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH}")
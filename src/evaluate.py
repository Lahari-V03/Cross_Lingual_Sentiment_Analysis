import pandas as pd
from src.dataloader import load_xed, load_goemotions, load_shared_task
from src.preprocess import preprocess_dataframe
from src.inference import load_finetuned_model, predict_finetuned_batch

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Which dataset to evaluate: "xed", "goemotions", or "shared_task"
DATASET = "xed"

# Path to your model checkpoint directory (should contain your fine-tuned model)
MODEL_PATH = "./model_ckpt"

# Path to your data CSV (set to None to use the default file for the given dataset)
DATA_PATH = None

# Name of the text and label columns in your CSV
TEXT_COL = "text"
LABEL_COL = "label"

# Inference settings (should match your training)
MAX_LENGTH = 128
BATCH_SIZE = 16
# ================================================

def get_dataset_loader(name):
    if name == "xed":
        return load_xed
    elif name == "goemotions":
        return load_goemotions
    elif name == "shared_task":
        return load_shared_task
    else:
        raise ValueError(f"Unknown dataset loader: {name}")

def evaluate_model(
    model_path,
    dataset,
    data_path,
    text_col="text",
    label_col="label",
    max_length=128,
    batch_size=16,
):
    # Load and preprocess data
    loader = get_dataset_loader(dataset)
    df = loader(data_path) if data_path else loader()
    df = preprocess_dataframe(df, text_col=text_col, label_col=label_col)
    
    # Prepare inputs
    texts = df[text_col].tolist()
    true_labels = df[label_col].tolist()

    # Load model & tokenizer
    tokenizer, model, device = load_finetuned_model(model_path)

    # Predict
    pred_labels = predict_finetuned_batch(
        texts,
        tokenizer,
        model,
        device,
        batch_size=batch_size,
        max_length=max_length,
        show_progress=True,
    )

    # Compute metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    report = classification_report(true_labels, pred_labels, digits=4)
    conf_mat = confusion_matrix(true_labels, pred_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_mat)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": conf_mat,
    }

if __name__ == "__main__":
    print("Evaluating model...")
    evaluate_model(
        MODEL_PATH,
        DATASET,
        DATA_PATH,
        TEXT_COL,
        LABEL_COL,
        MAX_LENGTH,
        BATCH_SIZE
    )
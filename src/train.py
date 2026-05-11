import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from src.dataloader import load_xed, load_goemotions, load_shared_task
from src.preprocess import preprocess_dataframe

# Simple PyTorch Dataset for text/label pairs
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the text, map label to integer
        text = str(self.texts[idx])
        label = self.label2id[self.labels[idx]]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        return item

def load_dataset(name, path=None):
    if name == "xed":
        return load_xed(path) if path else load_xed()
    elif name == "goemotions":
        return load_goemotions(path) if path else load_goemotions()
    elif name == "shared_task":
        return load_shared_task(path) if path else load_shared_task()
    else:
        raise ValueError(f"Unknown dataset: {name}")

if __name__ == "__main__":
    # ---- User-friendly config ----
    DATASET = "xed"                  # Choose: "xed", "goemotions", or "shared_task"
    MODEL_NAME = "distilbert-base-uncased"  # Any Hugging Face model name
    OUTPUT_DIR = "./model_ckpt"      # Where to save the trained model
    NUM_EPOCHS = 3
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    VAL_SIZE = 0.1                   # Fraction of data for validation
    TEST_SIZE = 0.1                  # Fraction of data for test
    TEXT_COL = "text"
    LABEL_COL = "label"
    DATA_PATH = None                 # Optionally override with a custom CSV path

    # ---- Load and preprocess data ----
    df = load_dataset(DATASET, DATA_PATH)
    df = preprocess_dataframe(df, text_col=TEXT_COL, label_col=LABEL_COL)

    train_df, temp_df = train_test_split(
        df, test_size=VAL_SIZE + TEST_SIZE, stratify=df[LABEL_COL], random_state=42
    )
    val_rel = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_rel, stratify=temp_df[LABEL_COL], random_state=42
    )

    label_list = sorted(df[LABEL_COL].unique())
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = EmotionDataset(train_df[TEXT_COL].tolist(), train_df[LABEL_COL].tolist(), tokenizer, label2id, max_length=MAX_LENGTH)
    val_dataset = EmotionDataset(val_df[TEXT_COL].tolist(), val_df[LABEL_COL].tolist(), tokenizer, label2id, max_length=MAX_LENGTH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": accuracy, "macro_f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f"Training model '{MODEL_NAME}' on {DATASET} dataset...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Model saved in '{OUTPUT_DIR}'.")
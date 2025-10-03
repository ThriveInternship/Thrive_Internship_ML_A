
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import os

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/customer_support_tickets_clean_500.csv"
OUTPUT_DIR = "models/distilbert-ticket-classifier"

def load_data(path):
    df = pd.read_csv(path)
    # normalize columns
    if 'text' not in df.columns:
        for c in df.columns:
            if c.lower() in ('message','ticket_text','body'):
                df = df.rename(columns={c:'text'})
                break
    if 'label' not in df.columns:
        for c in df.columns:
            if c.lower() in ('category','class'):
                df = df.rename(columns={c:'label'})
                break
    return df[['text','label']].dropna()

def encode_labels(series):
    classes = sorted(series.unique())
    label2id = {c:i for i,c in enumerate(classes)}
    id2label = {i:c for c,i in label2id.items()}
    return series.map(label2id), label2id, id2label

class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], truncation=True)
        item['labels'] = self.labels[idx]
        return item

def main():
    df = load_data(DATA_PATH)
    y_enc, label2id, id2label = encode_labels(df['label'])

    X_train, X_val, y_train, y_val = train_test_split(df['text'], y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TicketDataset(X_train, y_train, tok)
    val_ds   = TicketDataset(X_val, y_val, tok)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label2id), id2label={i:k for k,i in label2id.items()}, label2id=label2id)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        logging_steps=50,
        save_strategy="epoch"
    )

    collator = DataCollatorWithPadding(tokenizer=tok)
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        acc = (preds == labels).mean()
        return {"accuracy": float(acc)}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()

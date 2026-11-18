import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert/distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MAX_LENGTH = 64


def build_tfidf(ngram_range: tuple[int, int] = (1, 2)):
    return make_pipeline(
        TfidfVectorizer(min_df=2, ngram_range=ngram_range),
        LogisticRegression(max_iter=1000, random_state=42),
    )


def classification_metrics(labels, predictions) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, label_to_id):
        self.encodings = tokenizer(
            frame["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(frame["category"].map(label_to_id).tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            **{key: values[index] for key, values in self.encodings.items()},
            "labels": self.labels[index],
        }


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, labels, predictions = 0.0, [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(**batch)
        if optimizer is not None:
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += output.loss.item() * len(batch["labels"])
        labels.extend(batch["labels"].cpu().tolist())
        predictions.extend(output.logits.argmax(dim=1).detach().cpu().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        **classification_metrics(labels, predictions),
    }


def train_distilbert(train, tune, output_dir: Path, device: str):
    np.random.seed(42)
    torch.manual_seed(42)
    categories = sorted(train["category"].unique())
    label_to_id = {label: index for index, label in enumerate(categories)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        num_labels=len(categories),
        label2id=label_to_id,
        id2label={index: label for label, index in label_to_id.items()},
    ).to(device)
    train_loader = DataLoader(
        TextDataset(train, tokenizer, label_to_id), batch_size=16, shuffle=True
    )
    tune_loader = DataLoader(TextDataset(tune, tokenizer, label_to_id), batch_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    history, best_f1 = [], -np.inf
    started = time.perf_counter()
    for epoch in range(1, 4):
        train_scores = run_epoch(model, train_loader, device, optimizer)
        tune_scores = run_epoch(model, tune_loader, device)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_scores.items()},
            **{f"tune_{key}": value for key, value in tune_scores.items()},
        }
        history.append(row)
        print(row, flush=True)
        if tune_scores["macro_f1"] <= best_f1 + 1e-4:
            break
        best_f1 = tune_scores["macro_f1"]
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    elapsed = time.perf_counter() - started
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    return model, tokenizer, pd.DataFrame(history), elapsed


@torch.inference_mode()
def predict_distilbert(model, tokenizer, texts, batch_size=32):
    model.eval()
    results = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            list(texts[start : start + batch_size]),
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(model.device)
        results.append(model(**batch).logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(results)

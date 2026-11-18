"""Run from the repository root: PYTHONPATH=src python -m banking_intent.experiment."""

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from banking_intent.data import prepare_splits
from banking_intent.models import (
    build_tfidf,
    classification_metrics,
    predict_distilbert,
    train_distilbert,
)
from banking_intent.routing import routing_curve, routing_metrics, select_threshold


def prediction_frame(frame, probabilities, categories):
    return frame.assign(
        predicted_category=np.asarray(categories)[probabilities.argmax(axis=1)],
        confidence=probabilities.max(axis=1),
    )


def benchmark(predict, texts, batch_size, repeats=50):
    batch = texts[:batch_size]
    for _ in range(5):
        predict(batch)
    durations = []
    for repeat in range(repeats):
        start = (repeat * batch_size) % (len(texts) - batch_size + 1)
        batch = texts[start : start + batch_size]
        before = time.perf_counter()
        predict(batch)
        durations.append((time.perf_counter() - before) * 1000)
    return {
        "batch_size": batch_size,
        "repeats": repeats,
        "median_ms": np.median(durations),
        "p95_ms": np.quantile(durations, 0.95),
        "median_ms_per_text": np.median(durations) / batch_size,
    }


def main():
    root = Path.cwd()
    report = root / "reports/portfolio"
    model_dir = root / "models/portfolio"
    report.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    splits = prepare_splits(root / "data/raw")
    pd.DataFrame(
        [
            {"split": name, "rows": len(frame), "classes": frame["category"].nunique()}
            for name, frame in splits.items()
        ]
    ).to_csv(report / "split_summary.csv", index=False)
    candidates, scores = {}, []
    for ngrams in [(1, 1), (1, 2)]:
        pipeline = build_tfidf(ngrams)
        started = time.perf_counter()
        pipeline.fit(splits["train"]["text"], splits["train"]["category"])
        candidates[str(ngrams)] = pipeline
        scores.append(
            {
                "ngrams": str(ngrams),
                "training_seconds": time.perf_counter() - started,
                **classification_metrics(
                    splits["tune"]["category"], pipeline.predict(splits["tune"]["text"])
                ),
            }
        )
    scores = pd.DataFrame(scores).sort_values("macro_f1", ascending=False)
    scores.to_csv(report / "tfidf_selection.csv", index=False)
    pipeline = candidates[scores.iloc[0]["ngrams"]]
    joblib.dump(pipeline, model_dir / "tfidf.joblib")
    model, tokenizer, history, training_seconds = train_distilbert(
        splits["train"],
        splits["tune"],
        model_dir / "distilbert",
        device,
    )
    history.to_csv(report / "distilbert_history.csv", index=False)
    categories = [model.config.id2label[i] for i in range(model.config.num_labels)]
    prediction_functions = {
        "tfidf": (pipeline.predict_proba, pipeline.classes_),
        "distilbert": (
            lambda texts: predict_distilbert(model, tokenizer, texts),
            categories,
        ),
    }
    metrics, routing, thresholds = [], [], {}
    for name, (predict, labels) in prediction_functions.items():
        predictions = {}
        for split in ["tune", "policy", "test"]:
            frame = splits[split]
            predictions[split] = prediction_frame(
                frame, predict(frame["text"].tolist()), labels
            )
            predictions[split].to_csv(report / f"{name}_{split}.csv", index=False)
            metrics.append(
                {
                    "model": name,
                    "split": split,
                    **classification_metrics(
                        frame["category"],
                        predictions[split]["predicted_category"],
                    ),
                }
            )
        curve = routing_curve(predictions["policy"])
        curve.to_csv(report / f"{name}_routing_curve.csv", index=False)
        threshold = select_threshold(curve)
        thresholds[name] = threshold if np.isfinite(threshold) else None
        for split in ["policy", "test"]:
            routing.append(
                {
                    "model": name,
                    "split": split,
                    **routing_metrics(predictions[split], threshold),
                }
            )
    pd.DataFrame(metrics).to_csv(report / "metrics.csv", index=False)
    pd.DataFrame(routing).to_csv(report / "routing.csv", index=False)
    (report / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    # CPU on both models makes the latency comparison useful without a GPU server.
    model.to("cpu")
    latency = []
    texts = splits["test"]["text"].tolist()
    for name, (predict, _) in prediction_functions.items():
        for batch_size in [1, 32]:
            latency.append({"model": name, **benchmark(predict, texts, batch_size)})
    pd.DataFrame(latency).to_csv(report / "latency.csv", index=False)
    environment = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "training_device": device,
        "latency_device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "distilbert_training_seconds": training_seconds,
        "seed": 42,
        "policy_target_accuracy": 0.95,
        "raw_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "data/raw").glob("*.csv")
        },
        "model_sizes_bytes": {
            "tfidf": (model_dir / "tfidf.joblib").stat().st_size,
            "distilbert": sum(
                p.stat().st_size for p in (model_dir / "distilbert").glob("*")
            ),
        },
    }
    (report / "environment.json").write_text(json.dumps(environment, indent=2))
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(pd.DataFrame(routing).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

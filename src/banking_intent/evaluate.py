import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from banking_intent.data import prepare_splits
from banking_intent.experiment import benchmark
from banking_intent.models import predict_distilbert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-latency", action="store_true")
    arguments = parser.parse_args()
    root = Path.cwd()
    torch.set_num_threads(4)
    report = root / "reports/evaluation"
    splits = prepare_splits(root / "data/raw")
    pd.DataFrame(
        [
            {"split": name, "rows": len(frame), "classes": frame.category.nunique()}
            for name, frame in splits.items()
        ]
    ).to_csv(report / "split_summary.csv", index=False)

    directory = root / "models/trained"
    model = AutoModelForSequenceClassification.from_pretrained(directory / "distilbert")
    tokenizer = AutoTokenizer.from_pretrained(directory / "distilbert")
    pipeline = joblib.load(directory / "tfidf.joblib")
    functions = {
        "tfidf": (pipeline.predict_proba, pipeline.classes_),
        "distilbert": (
            lambda texts: predict_distilbert(model, tokenizer, texts),
            [model.config.id2label[i] for i in range(77)],
        ),
    }
    thresholds = json.loads((report / "thresholds.json").read_text())
    challenge = pd.read_csv(report / "ood_challenge.csv")
    ood_rows, latency = [], []
    for name, (predict, labels) in functions.items():
        probabilities = predict(challenge.text.tolist())
        threshold = thresholds[name]
        predictions = challenge.assign(
            model=name,
            predicted_category=[labels[i] for i in probabilities.argmax(1)],
            confidence=probabilities.max(1),
            accepted=probabilities.max(1)
            >= (threshold if threshold is not None else float("inf")),
        )
        ood_rows.append(predictions)
        if not arguments.skip_latency:
            for batch_size in [1, 32]:
                latency.append(
                    {
                        "model": name,
                        **benchmark(predict, splits["test"].text.tolist(), batch_size),
                    }
                )
    pd.concat(ood_rows).to_csv(report / "ood_predictions.csv", index=False)
    if latency:
        pd.DataFrame(latency).to_csv(report / "latency.csv", index=False)
    environment = json.loads((report / "environment.json").read_text())
    environment.update(
        {
            "latency_isolated": not arguments.skip_latency,
            "hardware": environment.get("hardware", environment["platform"]),
            "benchmark_includes": (
                "tokenization/vectorization + forward + CPU probabilities; "
                "excludes model loading"
            ),
        }
    )
    (report / "environment.json").write_text(json.dumps(environment, indent=2))
    errors = pd.read_csv(report / "distilbert_test.csv").query(
        "category != predicted_category"
    )
    errors.sample(n=60, random_state=42).reset_index().rename(
        columns={"index": "test_row"}
    ).to_csv(report / "error_candidates.csv", index=False)
    print(pd.DataFrame(latency).to_string(index=False))
    print(
        pd.concat(ood_rows).groupby(["model", "group"]).accepted.agg(["count", "sum"])
    )


if __name__ == "__main__":
    main()

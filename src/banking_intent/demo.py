import argparse
import json
from pathlib import Path

import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from banking_intent.models import predict_distilbert


def route_message(text: str, model_name: str = "distilbert") -> dict:
    directory = Path("models/portfolio")
    thresholds = json.loads(Path("reports/portfolio/thresholds.json").read_text())
    if model_name == "tfidf":
        model = joblib.load(directory / "tfidf.joblib")
        probabilities = model.predict_proba([text])[0]
        categories = model.classes_
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            directory / "distilbert"
        )
        tokenizer = AutoTokenizer.from_pretrained(directory / "distilbert")
        probabilities = predict_distilbert(model, tokenizer, [text])[0]
        categories = model.config.id2label
    predicted_id = int(probabilities.argmax())
    confidence = float(probabilities[predicted_id])
    threshold = thresholds[model_name]
    accepted = threshold is not None and confidence >= threshold
    return {
        "route": "intent" if accepted else "operator",
        "intent": categories[predicted_id] if accepted else None,
        "suggested_intent": categories[predicted_id],
        "confidence": round(confidence, 4),
        "threshold": threshold,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument(
        "--model", choices=["tfidf", "distilbert"], default="distilbert"
    )
    arguments = parser.parse_args()
    print(json.dumps(route_message(arguments.text, arguments.model), indent=2))

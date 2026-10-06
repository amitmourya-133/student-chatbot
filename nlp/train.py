"""Train the offline intent classifier (TF-IDF + Multinomial Naive Bayes).

Run from the project root:

    python nlp/train.py

The trained artifacts are saved to ``models/vectorizer.pkl`` and
``models/intent_model.pkl`` so ``main.py`` never has to retrain on startup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from nlp.preprocessing import preprocess  # noqa: E402


def load_intents(path: Path | None = None) -> dict:
    """Load and validate the intents dataset."""
    path = path or config.INTENTS_FILE
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    intents = data.get("intents")
    if not intents or not isinstance(intents, list):
        raise ValueError(f"No intents found in {path}")
    return data


def build_dataset(data: dict) -> tuple[list[str], list[str]]:
    """Flatten the intents file into parallel text/label lists."""
    texts: list[str] = []
    labels: list[str] = []
    for intent in data["intents"]:
        tag = intent.get("tag")
        patterns = intent.get("patterns", [])
        if not tag or not patterns:
            continue
        for pattern in patterns:
            cleaned = preprocess(pattern)
            if cleaned:
                texts.append(cleaned)
                labels.append(tag)
    return texts, labels


def train(
    intents_path: Path | None = None,
    vectorizer_path: Path | None = None,
    model_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Train and persist the classifier. Returns training metrics."""
    vectorizer_path = vectorizer_path or config.VECTORIZER_FILE
    model_path = model_path or config.INTENT_MODEL_FILE

    data = load_intents(intents_path)
    texts, labels = build_dataset(data)
    if len(set(labels)) < 2:
        raise ValueError("Need at least two intents to train a classifier")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    features = vectorizer.fit_transform(texts)

    model = MultinomialNB(alpha=0.1)
    model.fit(features, labels)

    # simple resubstitution accuracy - useful as a sanity check only
    predictions = model.predict(features)
    accuracy = float(np.mean(np.asarray(predictions) == np.asarray(labels)))

    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)

    report = {
        "samples": len(texts),
        "intents": len(set(labels)),
        "features": int(features.shape[1]),
        "accuracy": accuracy,
        "vectorizer_path": str(vectorizer_path),
        "model_path": str(model_path),
    }

    if verbose:
        print("Model trained successfully.")
        print(
            f"  samples={report['samples']} intents={report['intents']} "
            f"features={report['features']} train_accuracy={accuracy:.2%}"
        )
        print(f"  saved: {vectorizer_path.name}, {model_path.name}")
    return report


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:  # pragma: no cover - surfaced to the operator
        print(f"Training failed: {exc}")
        sys.exit(1)

"""Load and use the trained TF-IDF + Naive Bayes intent model."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import joblib

import config
from nlp.preprocessing import preprocess

logger = logging.getLogger(__name__)

# below this model confidence we treat the prediction as uncertain
DEFAULT_CONFIDENCE_THRESHOLD = 0.35

_bootstrap_lock = threading.Lock()


class IntentClassifier:
    """Wraps the persisted vectorizer + MultinomialNB model."""

    def __init__(
        self,
        vectorizer_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        auto_train: bool = True,
    ) -> None:
        self.vectorizer_path = vectorizer_path or config.VECTORIZER_FILE
        self.model_path = model_path or config.INTENT_MODEL_FILE
        self.confidence_threshold = confidence_threshold
        self.vectorizer = None
        self.model = None
        self.loaded = False
        self._auto_train = auto_train

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    @property
    def model_files_exist(self) -> bool:
        return self.vectorizer_path.exists() and self.model_path.exists()

    def load(self) -> bool:
        """Load the persisted model. Optionally trains it when missing.

        Returns True when a usable model is available.
        """
        if not self.model_files_exist:
            if not self._auto_train:
                logger.warning("Intent model files missing at %s", self.model_path)
                return False
            if not self._train_now():
                return False

        try:
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.model = joblib.load(self.model_path)
            self.loaded = True
            logger.info(
                "Loaded intent model (%d classes) from %s",
                len(getattr(self.model, "classes_", [])),
                self.model_path,
            )
            return True
        except Exception:
            logger.exception("Failed to load intent model")
            self.loaded = False
            return False

    def _train_now(self) -> bool:
        with _bootstrap_lock:
            if self.model_files_exist:
                return True
            try:
                from nlp.train import train

                logger.warning("Intent model missing - training now (bootstrap)")
                print("⚠️ Intent model not found. Training it now, please wait...")
                train(verbose=False)
                print("✅ Model trained.\n")
                return True
            except Exception:
                logger.exception("Automatic model training failed")
                return False

    def ensure_ready(self) -> bool:
        """Load the model, retraining automatically if necessary."""
        if self.loaded:
            return True
        return self.load()

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def predict(self, text: str) -> dict:
        """Classify ``text``.

        Returns a dict with ``intent``, ``confidence`` and ``fallback``.
        Never raises for normal input.
        """
        result = {"intent": "fallback", "confidence": 0.0, "fallback": True}
        if not self.ensure_ready():
            return result

        cleaned = preprocess(text)
        if not cleaned:
            return result

        try:
            features = self.vectorizer.transform([cleaned])
            proba = self.model.predict_proba(features)[0]
            classes = list(self.model.classes_)
            best_index = int(proba.argmax())
            confidence = float(proba[best_index])
            intent = str(classes[best_index])

            result["confidence"] = confidence
            result["intent"] = intent if confidence >= self.confidence_threshold else "fallback"
            result["fallback"] = result["intent"] == "fallback"
            return result
        except Exception:
            logger.exception("Intent prediction failed for input")
            return result

    def known_intents(self) -> list[str]:
        if not self.ensure_ready():
            return []
        return [str(label) for label in self.model.classes_]

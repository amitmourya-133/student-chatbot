"""Offline NLP pipeline: preprocessing, training and intent classification."""

from nlp.preprocessing import preprocess
from nlp.classifier import IntentClassifier

__all__ = ["preprocess", "IntentClassifier"]

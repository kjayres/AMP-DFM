"""Base class for AMP classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseClassifier(ABC):
    """Abstract base for peptide property prediction models"""

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the model"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for positive class"""
        pass

    @abstractmethod
    def evaluate(
        self, X: np.ndarray, y: np.ndarray, split_name: str = "Test"
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model, returns (auc, probabilities, predictions)"""
        pass

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save model to disk"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> BaseClassifier:
        """Load model from disk"""
        pass


#!/usr/bin/env python3
"""Base class for AMP judges."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseJudge(ABC):
    """Abstract base for peptide property prediction models."""

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the model. Returns dict with training metrics and history."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for the positive class."""
        pass

    @abstractmethod
    def evaluate(
        self, X: np.ndarray, y: np.ndarray, split_name: str = "Test"
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model. Returns (auc, probabilities, predictions)."""
        pass

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> BaseJudge:
        """Load model from disk."""
        pass


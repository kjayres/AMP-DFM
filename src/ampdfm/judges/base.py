#!/usr/bin/env python3
"""Base class for AMP judges.

Provides a common interface for all judge models (potency, haemolysis, cytotoxicity).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseJudge(ABC):
    """Abstract base class for peptide property prediction models (judges)."""

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the judge model.

        Args:
            X_train: Training feature matrix (N_train, feature_dim)
            y_train: Training labels (N_train,)
            X_val: Validation feature matrix (N_val, feature_dim)
            y_val: Validation labels (N_val,)
            **kwargs: Additional training parameters (e.g., sample_weight, params_override)

        Returns:
            Dictionary containing training metrics and history
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for input features.

        Args:
            X: Feature matrix (N, feature_dim)

        Returns:
            Probability array (N,) for the positive class
        """
        pass

    @abstractmethod
    def evaluate(
        self, X: np.ndarray, y: np.ndarray, split_name: str = "Test"
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model performance on a dataset.

        Args:
            X: Feature matrix (N, feature_dim)
            y: True labels (N,)
            split_name: Name of the split for logging

        Returns:
            Tuple of (auc_score, predicted_probabilities, predicted_labels)
        """
        pass

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save model to disk.

        Args:
            path: Directory or file path to save the model
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> BaseJudge:
        """Load model from disk.

        Args:
            path: Directory or file path containing the saved model

        Returns:
            Loaded judge instance
        """
        pass


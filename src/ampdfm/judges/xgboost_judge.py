#!/usr/bin/env python3
"""XGBoost judge base class with shared training/inference logic.

Provides common XGBoost training, prediction, evaluation, and persistence
methods for all task-specific judges.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .base import BaseJudge

logger = logging.getLogger(__name__)


class XGBoostJudge(BaseJudge):
    """Base class for XGBoost-based judges with shared training/inference logic."""

    def __init__(self, decision_threshold: float = 0.5):
        """Initialize XGBoost judge.

        Args:
            decision_threshold: Probability threshold for binary classification
        """
        self.decision_threshold = decision_threshold
        self.model: Optional[xgb.Booster] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        params_override: Optional[dict[str, Any]] = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 100,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train XGBoost model with validation.

        Args:
            X_train: Training features (N_train, feature_dim)
            y_train: Training labels (N_train,)
            X_val: Validation features (N_val, feature_dim)
            y_val: Validation labels (N_val,)
            sample_weight: Optional training sample weights (N_train,) - only used by subclasses
            params_override: Optional XGBoost parameters to override defaults
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping patience
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'model' (trained Booster) and 'evals_result' (training history)
        """
        logger.info(f"Training XGBoost {self.__class__.__name__}...")

        # Note: sample_weight is accepted but not used by default
        # Subclasses can override this method to use sample_weight
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Minimal defaults; hyperparameters should come from YAML/Optuna via params_override
        params = {
            "verbosity": 1,
        }
        if params_override:
            params.update(params_override)

        evals_result: dict[str, dict[str, list[float]]] = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100,
            evals_result=evals_result,
        )

        return {"model": self.model, "evals_result": evals_result}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for the positive class.

        Args:
            X: Feature matrix (N, feature_dim)

        Returns:
            Probability array (N,)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        dmat = xgb.DMatrix(X)
        return self.model.predict(dmat)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, split_name: str = "Test"
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model on a dataset.

        Args:
            X: Feature matrix (N, feature_dim)
            y: True labels (N,)
            split_name: Name of the split for logging

        Returns:
            Tuple of (auc_score, predicted_probabilities, predicted_labels)
        """
        logger.info(f"Evaluating {self.__class__.__name__} on {split_name} set...")

        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba > self.decision_threshold).astype(int)

        auc_score = roc_auc_score(y, y_pred_proba)

        logger.info(f"{split_name} AUC: {auc_score:.4f}")
        logger.info(f"{split_name} Classification Report:")
        target_names = self._get_target_names()
        logger.info(
            "\n%s",
            classification_report(
                y, y_pred, target_names=target_names, zero_division=0
            ),
        )

        logger.info(f"{split_name} Confusion Matrix:")
        logger.info("\n%s", confusion_matrix(y, y_pred))

        return auc_score, y_pred_proba, y_pred

    def _get_target_names(self) -> list[str]:
        """Get target class names for classification report.

        Override in subclasses for task-specific naming.

        Returns:
            List of [negative_class_name, positive_class_name]
        """
        return ["Negative", "Positive"]

    def save(self, path: Path | str) -> None:
        """Save model to disk.

        Args:
            path: File path to save the XGBoost model
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")

        path = Path(path)
        self.model.save_model(str(path))
        logger.info(f"{self.__class__.__name__} saved to {path}")

    @classmethod
    def load(cls, path: Path | str, **kwargs: Any) -> XGBoostJudge:
        """Load model from disk.

        Args:
            path: File path to the saved XGBoost model
            **kwargs: Additional arguments to pass to constructor

        Returns:
            Loaded judge instance
        """
        judge = cls(**kwargs)
        judge.model = xgb.Booster()
        judge.model.load_model(str(path))
        logger.info(f"{cls.__name__} loaded from {path}")
        return judge


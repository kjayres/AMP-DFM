#!/usr/bin/env python3
"""Sklearn model wrapper implementing BaseJudge interface.

Provides a unified interface for scikit-learn models (Random Forest, Logistic Regression)
to work seamlessly with the judge training pipeline.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

from .base import BaseJudge

logger = logging.getLogger(__name__)


class SklearnJudge(BaseJudge):
    """Wrapper for scikit-learn models implementing BaseJudge interface.
    
    Supports any sklearn classifier with fit/predict_proba methods,
    particularly RandomForestClassifier and LogisticRegression.
    """

    def __init__(self, model_class: type, model_params: Optional[dict[str, Any]] = None, decision_threshold: Optional[float] = None):
        """Initialize sklearn judge wrapper.

        Args:
            model_class: Sklearn model class (e.g., RandomForestClassifier)
            model_params: Dict of parameters to pass to model constructor
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.model: Optional[Any] = None
        self.decision_threshold: float = float(decision_threshold) if decision_threshold is not None else 0.5

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train sklearn model.

        Args:
            X_train: Training features (N_train, feature_dim)
            y_train: Training labels (N_train,)
            X_val: Validation features (N_val, feature_dim)
            y_val: Validation labels (N_val,)
            sample_weight: Optional training sample weights (N_train,)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'model' and 'evals_result' (single-point baseline)
        """
        logger.info(f"Training {self.model_class.__name__}...")

        self.model = self.model_class(**self.model_params)
        
        # Fit with sample weights if provided
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)

        # Compute single-point AUCs for downstream persistence
        train_proba = self.model.predict_proba(X_train)[:, 1]
        val_proba = self.model.predict_proba(X_val)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)
        val_auc = roc_auc_score(y_val, val_proba)

        return {'model': self.model, 'train_auc': float(train_auc), 'val_auc': float(val_auc)}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for the positive class.

        Args:
            X: Feature matrix (N, feature_dim)

        Returns:
            Probability array (N,)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)[:, 1]

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
        logger.info(f"Evaluating {self.model_class.__name__} on {split_name} set...")

        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba > self.decision_threshold).astype(int)

        auc_score = roc_auc_score(y, y_pred_proba)

        logger.info(f"{split_name} AUC: {auc_score:.4f}")
        logger.info(f"{split_name} Classification Report:")
        print(
            classification_report(
                y, y_pred, target_names=["Negative", "Positive"], zero_division=0
            )
        )

        logger.info(f"{split_name} Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

        return auc_score, y_pred_proba, y_pred

    def save(self, path: Path | str) -> None:
        """Save model to disk.

        Args:
            path: File path to save the sklearn model (will use pickle)
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")

        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"{self.model_class.__name__} saved to {path}")

    @classmethod
    def load(cls, path: Path | str, model_class: type) -> SklearnJudge:
        """Load model from disk.

        Args:
            path: File path to the saved sklearn model
            model_class: The sklearn class type (for initialization)

        Returns:
            Loaded SklearnJudge instance
        """
        judge = cls(model_class=model_class)
        
        with open(path, 'rb') as f:
            judge.model = pickle.load(f)
        
        logger.info(f"{model_class.__name__} loaded from {path}")
        return judge


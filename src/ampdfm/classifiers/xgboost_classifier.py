#!/usr/bin/env python3
"""XGBoost classifier base class with shared training/inference logic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .base import BaseClassifier

logger = logging.getLogger(__name__)


class XGBoostClassifier(BaseClassifier):
    """Base class for XGBoost-based classifiers"""

    def __init__(self, decision_threshold: float = 0.5):
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
        logger.info(f"Training XGBoost {self.__class__.__name__}...")
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {"verbosity": 1}
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
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(xgb.DMatrix(X))

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, split_name: str = "Test"
    ) -> tuple[float, np.ndarray, np.ndarray]:
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba > self.decision_threshold).astype(int)
        auc_score = roc_auc_score(y, y_pred_proba)

        logger.info(f"{split_name} AUC: {auc_score:.4f}")
        logger.info(f"{split_name} Classification Report:")
        logger.info(
            "\n%s",
            classification_report(
                y, y_pred, target_names=self._get_target_names(), zero_division=0
            ),
        )
        logger.info(f"{split_name} Confusion Matrix:")
        logger.info("\n%s", confusion_matrix(y, y_pred))

        return auc_score, y_pred_proba, y_pred

    def _get_target_names(self) -> list[str]:
        return ["Negative", "Positive"]

    def save(self, path: Path | str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        path = Path(path)
        self.model.save_model(str(path))
        logger.info(f"{self.__class__.__name__} saved to {path}")

    @classmethod
    def load(cls, path: Path | str, **kwargs: Any) -> XGBoostClassifier:
        classifier = cls(**kwargs)
        classifier.model = xgb.Booster()
        classifier.model.load_model(str(path))
        logger.info(f"{cls.__name__} loaded from {path}")
        return classifier


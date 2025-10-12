#!/usr/bin/env python3
"""Cytotoxicity judge for cell-line toxicity prediction.

Classifies peptides as safe (ICx50 > 50 µM) or toxic (ICx50 ≤ 50 µM)
based on ESM-2 embeddings using XGBoost.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .base import BaseJudge

logger = logging.getLogger(__name__)

# Default threshold
DEFAULT_TOX_THRESHOLD_UM = 50.0  # µM (ICx50 threshold for defining toxicity)
DEFAULT_DECISION_THRESHOLD = 0.5  # Probability threshold for classification

FIFTY_PERCENT_ENDPOINTS = {"IC50", "CC50", "EC50", "LC50", "LD50"}


def label_cytotoxicity_sequences(
    df: pd.DataFrame,
    tox_threshold_um: float = DEFAULT_TOX_THRESHOLD_UM,
) -> pd.DataFrame:
    """Label sequences using the ICx50 ≤ 50 µM toxicity rule.

    Labelling rule:
        - If ANY measurement has ICx ≤ tox_threshold_um µM ⇒ label 0 (cytotoxic / toxic)
        - Else, if peptide has at least one measurement at > tox_threshold_um µM
          (not upper-bound "<") ⇒ label 1 (non-cytotoxic / safe)
        - Otherwise excluded (unlabelled)

    Args:
        df: DataFrame with columns 'sequence', 'value' (log10 µM), 'unit',
            'measure_type', 'qualifier', 'sequence_id', 'cluster_id', 'split'
        tox_threshold_um: ICx50 threshold for defining toxicity (µM)

    Returns:
        DataFrame with columns: sequence, sequence_id, cluster_id, split, label (0/1)
    """
    logger.info(
        f"Labelling cytotoxicity sequences: ICx ≤ %.0f µM ⇒ toxic; "
        f"ICx > %.0f µM ⇒ safe",
        tox_threshold_um,
        tox_threshold_um,
    )

    df = df.copy()

    # Filter to 50% endpoints and µM units
    endpoint_mask = df["measure_type"].str.upper().isin(FIFTY_PERCENT_ENDPOINTS)
    unit_mask = df["unit"].str.lower() == "um"
    df = df[endpoint_mask & unit_mask]

    # Compute µM concentration from stored log10(µM)
    df["conc_uM"] = 10 ** df["value"]

    # Per-peptide labelling
    records = []
    for seq, grp in df.groupby("sequence", sort=False):
        q = grp["qualifier"].fillna("")
        conc = grp["conc_uM"]

        toxic_mask = (conc <= tox_threshold_um) & (q != ">")  # '<' or exact ⇒ toxic
        safe_mask = (conc > tox_threshold_um) & (q != "<")  # allow '>' if > threshold

        if toxic_mask.any():
            label = 0  # toxic / cytotoxic
        elif safe_mask.any():
            label = 1  # safe / non-cytotoxic
        else:
            continue  # cannot assign label confidently

        first = grp.iloc[0]
        rec = {
            "sequence": seq,
            "sequence_id": first["sequence_id"],
            "cluster_id": first["cluster_id"],
            "split": first["split"],
            "label": label,
        }
        records.append(rec)

    labelled_df = pd.DataFrame(records)
    logger.info(
        "Cytotoxicity sequences labelled: %d  (safe: %d, toxic: %d)",
        len(labelled_df),
        (labelled_df["label"] == 1).sum(),
        (labelled_df["label"] == 0).sum(),
    )
    return labelled_df


class CytotoxicityJudge(BaseJudge):
    """XGBoost-based classifier for cell-line cytotoxicity prediction."""

    def __init__(
        self,
        tox_threshold_um: float = DEFAULT_TOX_THRESHOLD_UM,
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    ):
        """Initialize cytotoxicity judge.

        Args:
            tox_threshold_um: ICx50 threshold for defining toxicity (µM)
            decision_threshold: Probability threshold for classifying as safe
        """
        self.tox_threshold_um = tox_threshold_um
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
            sample_weight: Optional training sample weights (N_train,)
            params_override: Optional XGBoost parameters to override defaults
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping patience
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'model' (trained Booster) and 'evals_result' (training history)
        """
        logger.info("Training XGBoost cytotoxicity judge...")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Use params from YAML/Optuna via params_override. Keep minimal defaults only.
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
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
        """Predict probabilities for the positive (safe) class.

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
        logger.info(f"Evaluating cytotoxicity judge on {split_name} set...")

        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba > self.decision_threshold).astype(int)

        auc_score = roc_auc_score(y, y_pred_proba)

        logger.info(f"{split_name} AUC: {auc_score:.3f}")
        logger.info(f"{split_name} Classification Report:")
        print(
            classification_report(
                y, y_pred, target_names=["Toxic", "Safe"], zero_division=0
            )
        )

        logger.info(f"{split_name} Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

        return auc_score, y_pred_proba, y_pred

    def save(self, path: Path | str) -> None:
        """Save model to disk.

        Args:
            path: File path to save the XGBoost model
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")

        path = Path(path)
        self.model.save_model(path)
        logger.info(f"Cytotoxicity judge saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> CytotoxicityJudge:
        """Load model from disk.

        Args:
            path: File path to the saved XGBoost model

        Returns:
            Loaded CytotoxicityJudge instance
        """
        judge = cls()
        judge.model = xgb.Booster()
        judge.model.load_model(str(path))
        logger.info(f"Cytotoxicity judge loaded from {path}")
        return judge


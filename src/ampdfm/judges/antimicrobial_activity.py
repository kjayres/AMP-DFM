#!/usr/bin/env python3
"""Antimicrobial activity judge for peptide potency prediction.

Classifies peptides as potent (MIC ≤32 μg/mL) or not potent (MIC ≥128 μg/mL)
based on ESM-2 embeddings using XGBoost.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from .xgboost_judge import XGBoostJudge

logger = logging.getLogger(__name__)

# Default MIC thresholds (μg/mL)
DEFAULT_POS_THRESHOLD_UGML = 32.0
DEFAULT_NEG_THRESHOLD_UGML = 128.0
DEFAULT_GAMMA_SYNTHETIC = 0.25


def convert_ugml_to_um(ugml_value: float, mw_da: float) -> float:
    """Convert μg/mL to μM using molecular weight.

    Args:
        ugml_value: Concentration in μg/mL
        mw_da: Molecular weight in Daltons

    Returns:
        Concentration in μM
    """
    return (ugml_value / mw_da) * 1000.0


def label_antimicrobial_activity_sequences(
    activity_df: pd.DataFrame,
    negatives_dfs: Optional[list[pd.DataFrame]] = None,
    pos_threshold_ugml: float = DEFAULT_POS_THRESHOLD_UGML,
    neg_threshold_ugml: float = DEFAULT_NEG_THRESHOLD_UGML,
    gamma_synthetic: float = DEFAULT_GAMMA_SYNTHETIC,
    organism_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Process activity data and negatives to create labeled sequences with weights.

    Args:
        activity_df: DataFrame with columns 'sequence', 'value' (log10 MIC in μM),
            'mw_da', 'qualifier', 'organism', 'sequence_id', 'cluster_id', 'split'
        negatives_dfs: Optional list of DataFrames with negative sequences
        pos_threshold_ugml: Positive threshold in μg/mL (≤ this value)
        neg_threshold_ugml: Negative threshold in μg/mL (≥ this value)
        gamma_synthetic: Down-weight factor for synthetic negatives (0-1)
        organism_filter: Optional organism name to filter activity data (e.g., "Escherichia coli")

    Returns:
        DataFrame with columns: sequence, sequence_id, cluster_id, split, label (0/1),
        quality ('curated'/'synthetic'), weight (sample weight for training)
    """
    logger.info(
        f"Labelling sequences with thresholds: positive ≤{pos_threshold_ugml} μg/mL, "
        f"negative ≥{neg_threshold_ugml} μg/mL"
    )

    df = activity_df.copy()

    # Restrict to MIC-family endpoints if available, mirroring legacy logic
    # (GRAMPA sometimes aliases MIC under a combined group label)
    if "measure_group" in df.columns:
        allowed_groups = {"MIC", "MIC50", "MIC90", "MIC,IC50,LC50,LD50"}
        before_rows = len(df)
        df = df[df["measure_group"].isin(allowed_groups)].copy()
        logger.info(
            "Filtered activity data to MIC family endpoints – %d → %d rows (%.1f%%)",
            before_rows,
            len(df),
            (len(df) / before_rows * 100.0) if before_rows else 0.0,
        )
    else:
        logger.info("No 'measure_group' column found – skipping MIC endpoint filter")

    # Optional organism filtering for species-specific models
    if organism_filter:
        logger.info(f"Filtering to organism: {organism_filter}")
        df = df[
            df["organism"].str.lower().str.contains(organism_filter.lower(), na=False)
        ]
        logger.info(f"Retained {len(df)} rows after organism filter")

    # Convert log MIC values to linear μM
    df["linear_value_um"] = 10 ** df["value"]

    # Convert thresholds from μg/mL to μM for each sequence based on MW
    df["pos_threshold_um"] = convert_ugml_to_um(pos_threshold_ugml, df["mw_da"])
    df["neg_threshold_um"] = convert_ugml_to_um(neg_threshold_ugml, df["mw_da"])

    # Create binary indicators for each measurement
    df["is_potent"] = False  # ≤ positive threshold
    df["is_not_potent"] = False  # ≥ negative threshold

    # Handle different qualifiers
    for idx, row in df.iterrows():
        qualifier = row["qualifier"]
        value_um = row["linear_value_um"]
        pos_thresh = row["pos_threshold_um"]
        neg_thresh = row["neg_threshold_um"]

        if pd.isna(qualifier) or qualifier in {"±", "range"}:
            # Exact or approximate value
            df.loc[idx, "is_potent"] = value_um <= pos_thresh
            df.loc[idx, "is_not_potent"] = value_um >= neg_thresh
        elif qualifier == "<":
            # Value is less than reported, so definitely potent if reported < pos_thresh
            df.loc[idx, "is_potent"] = value_um <= pos_thresh
            df.loc[idx, "is_not_potent"] = False
        elif qualifier == ">":
            # Value is greater than reported, so definitely not potent if reported > neg_thresh
            df.loc[idx, "is_potent"] = False
            df.loc[idx, "is_not_potent"] = value_um >= neg_thresh

    # Aggregate by sequence to get sequence-level labels
    # Positive: potent against at least one strain
    # Negative: not potent against all strains
    agg_dict = {
        "is_potent": ("is_potent", "any"),
        "is_not_potent": ("is_not_potent", "all"),
        "sequence_id": ("sequence_id", "first"),
        "mw_da": ("mw_da", "first"),
        "cluster_id": ("cluster_id", "first"),
        "split": ("split", "first"),
    }

    seq_labels = df.groupby("sequence").agg(**agg_dict).reset_index()

    # Create final binary labels
    seq_labels["label"] = None
    seq_labels.loc[seq_labels["is_potent"], "label"] = 1  # Positive
    seq_labels.loc[seq_labels["is_not_potent"], "label"] = 0  # Negative

    # Remove ambiguous sequences
    labeled_sequences = seq_labels.dropna(subset=["label"]).copy()
    labeled_sequences["label"] = labeled_sequences["label"].astype(int)

    logger.info(f"Activity-derived sequences with clear labels: {len(labeled_sequences)}")
    logger.info(f"Positive (potent): {sum(labeled_sequences['label'] == 1)}")
    logger.info(f"Negative (not potent): {sum(labeled_sequences['label'] == 0)}")

    # Tag as curated quality
    labeled_sequences["quality"] = "curated"

    # Append additional negative sequences
    if negatives_dfs:
        for neg_df in negatives_dfs:
            required_cols = ["sequence", "sequence_id", "cluster_id", "split"]
            missing_cols = [c for c in required_cols if c not in neg_df.columns]
            if missing_cols:
                raise ValueError(f"Negatives DataFrame missing columns: {missing_cols}")

            # Preserve provided quality if available; otherwise default to curated
            keep_cols = required_cols + (["quality"] if "quality" in neg_df.columns else [])
            neg_sequences = neg_df[keep_cols].drop_duplicates().copy()
            neg_sequences["label"] = 0
            if "quality" not in neg_sequences.columns:
                neg_sequences["quality"] = "curated"

            before = len(labeled_sequences)
            labeled_sequences = pd.concat(
                [labeled_sequences, neg_sequences], ignore_index=True
            )
            after = len(labeled_sequences)
            logger.info(f"Appended {after - before} negative sequences (total now {after})")

    # Compute sample weights using balanced class weighting with synthetic down-weighting
    gamma = gamma_synthetic

    P_total = (labeled_sequences["label"] == 1).sum()
    N_curated_total = (
        (labeled_sequences["label"] == 0) & (labeled_sequences["quality"] == "curated")
    ).sum()
    N_syn_total = (
        (labeled_sequences["label"] == 0) & (labeled_sequences["quality"] == "synthetic")
    ).sum()

    N_total = P_total + N_curated_total + N_syn_total

    if P_total == 0 or (N_curated_total + gamma * N_syn_total) == 0:
        logger.error(
            f"Invalid class counts – cannot compute sample weights "
            f"(P={P_total}, N_curated={N_curated_total}, N_syn={N_syn_total})"
        )
        sys.exit(1)

    omega_pos = N_total / (2 * P_total)
    omega_neg_cur = N_total / (2 * (N_curated_total + gamma * N_syn_total))

    def _assign_weight(row):
        if row["label"] == 1:
            return omega_pos
        elif row["quality"] == "synthetic":
            return gamma * omega_neg_cur
        else:
            return omega_neg_cur

    labeled_sequences["weight"] = labeled_sequences.apply(_assign_weight, axis=1)

    logger.info(
        f"Sample weighting: ω_pos={omega_pos:.3f}, ω_neg_cur={omega_neg_cur:.3f}, "
        f"γ={gamma:.3f} (ω_neg_syn={gamma * omega_neg_cur:.3f})"
    )

    return labeled_sequences


def compute_sample_weights(
    train_sequences: pd.DataFrame,
    gamma_synthetic: float = DEFAULT_GAMMA_SYNTHETIC,
) -> pd.Series:
    """Compute per-sequence sample weights for antimicrobial activity.

    Expects a DataFrame containing at least the columns:
    'sequence', 'label' (0/1), and optionally 'quality' ('curated'/'synthetic').
    If 'quality' is missing, negatives are treated as 'curated' by default.

    Returns a pandas Series mapping sequence → weight, with one entry per
    unique sequence (first occurrence per sequence defines its weight).
    """
    df = train_sequences.copy()

    if "quality" not in df.columns:
        # Default to curated if quality not provided
        df["quality"] = np.where(df["label"] == 0, "curated", "curated")

    gamma = gamma_synthetic

    num_positive = (df["label"] == 1).sum()
    num_negative_curated = ((df["label"] == 0) & (df["quality"] == "curated")).sum()
    num_negative_synthetic = ((df["label"] == 0) & (df["quality"] == "synthetic")).sum()

    total = num_positive + num_negative_curated + num_negative_synthetic

    if num_positive == 0 or (num_negative_curated + gamma * num_negative_synthetic) == 0:
        logger.error(
            "Invalid class counts – cannot compute sample weights (P=%d, N_curated=%d, N_syn=%d)",
            num_positive,
            num_negative_curated,
            num_negative_synthetic,
        )
        raise ValueError("Invalid class counts for sample weighting")

    omega_pos = total / (2 * num_positive)
    omega_neg_cur = total / (2 * (num_negative_curated + gamma * num_negative_synthetic))

    def _weight_row(row: pd.Series) -> float:
        if row["label"] == 1:
            return float(omega_pos)
        if row.get("quality", "curated") == "synthetic":
            return float(gamma * omega_neg_cur)
        return float(omega_neg_cur)

    df["_w"] = df.apply(_weight_row, axis=1)

    # Reduce to a single weight per unique sequence (first occurrence)
    weight_map = df.groupby("sequence")["_w"].first()
    weight_map.name = "weight"
    return weight_map


class AntimicrobialActivityJudge(XGBoostJudge):
    """XGBoost-based classifier for antimicrobial activity prediction."""

    def __init__(
        self,
        pos_threshold_ugml: float = DEFAULT_POS_THRESHOLD_UGML,
        neg_threshold_ugml: float = DEFAULT_NEG_THRESHOLD_UGML,
        gamma_synthetic: float = DEFAULT_GAMMA_SYNTHETIC,
        decision_threshold: float = 0.5,
    ):
        """Initialize antimicrobial activity judge.

        Args:
            pos_threshold_ugml: Positive threshold in μg/mL
            neg_threshold_ugml: Negative threshold in μg/mL
            gamma_synthetic: Down-weight factor for synthetic negatives
            decision_threshold: Probability threshold for binary classification
        """
        super().__init__(decision_threshold=decision_threshold)
        self.pos_threshold_ugml = pos_threshold_ugml
        self.neg_threshold_ugml = neg_threshold_ugml
        self.gamma_synthetic = gamma_synthetic

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
        """Train XGBoost model with validation and sample weighting.

        Args:
            X_train: Training features (N_train, feature_dim)
            y_train: Training labels (N_train,)
            X_val: Validation features (N_val, feature_dim)
            y_val: Validation labels (N_val,)
            sample_weight: Training sample weights (N_train,)
            params_override: Optional XGBoost parameters to override defaults
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping patience
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'model' (trained Booster) and 'evals_result' (training history)
        """
        logger.info(f"Training XGBoost {self.__class__.__name__}...")

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        dval = xgb.DMatrix(X_val, label=y_val)

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

    def _get_target_names(self) -> list[str]:
        """Get target class names for classification report."""
        return ["Not Potent", "Potent"]


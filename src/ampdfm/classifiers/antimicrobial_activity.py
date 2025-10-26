"""Antimicrobial activity classifier for peptide antimicrobial activity prediction."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .xgboost_classifier import XGBoostClassifier

logger = logging.getLogger(__name__)

DEFAULT_POS_THRESHOLD_UGML = 32.0
DEFAULT_NEG_THRESHOLD_UGML = 128.0
DEFAULT_GAMMA_SYNTHETIC = 0.25


def convert_ugml_to_um(ugml_value: float, mw_da: float) -> float:
    """Convert ug/mL to uM"""
    return (ugml_value / mw_da) * 1000.0


def label_antimicrobial_activity_sequences(
    activity_df: pd.DataFrame,
    negatives_dfs: Optional[list[pd.DataFrame]] = None,
    pos_threshold_ugml: float = DEFAULT_POS_THRESHOLD_UGML,
    neg_threshold_ugml: float = DEFAULT_NEG_THRESHOLD_UGML,
    gamma_synthetic: float = DEFAULT_GAMMA_SYNTHETIC,
    organism_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Label sequences for antimicrobial activity classification"""
    logger.info(f"Labelling: positive ≤{pos_threshold_ugml} μg/mL, negative ≥{neg_threshold_ugml} μg/mL")

    df = activity_df.copy()

    if "measure_group" in df.columns:
        allowed = {"MIC", "MIC50", "MIC90", "MIC,IC50,LC50,LD50"}
        df = df[df["measure_group"].isin(allowed)].copy()
        logger.info(f"Filtered to MIC endpoints: {len(df)} rows")

    if organism_filter:
        df = df[df["organism"].str.lower().str.contains(organism_filter.lower(), na=False)]
        logger.info(f"Filtered to {organism_filter}: {len(df)} rows")

    df["linear_value_um"] = 10 ** df["value"]
    df["pos_threshold_um"] = convert_ugml_to_um(pos_threshold_ugml, df["mw_da"])
    df["neg_threshold_um"] = convert_ugml_to_um(neg_threshold_ugml, df["mw_da"])

    df["is_active"] = False
    df["is_not_active"] = False

    for idx, row in df.iterrows():
        qualifier = row["qualifier"]
        value_um = row["linear_value_um"]
        pos_thresh = row["pos_threshold_um"]
        neg_thresh = row["neg_threshold_um"]

        if pd.isna(qualifier) or qualifier in {"±", "range"}:
            df.loc[idx, "is_active"] = value_um <= pos_thresh
            df.loc[idx, "is_not_active"] = value_um >= neg_thresh
        elif qualifier == "<":
            df.loc[idx, "is_active"] = value_um <= pos_thresh
        elif qualifier == ">":
            df.loc[idx, "is_not_active"] = value_um >= neg_thresh

    agg_dict = {
        "is_active": ("is_active", "any"),
        "is_not_active": ("is_not_active", "all"),
        "sequence_id": ("sequence_id", "first"),
        "mw_da": ("mw_da", "first"),
        "cluster_id": ("cluster_id", "first"),
        "split": ("split", "first"),
    }
    seq_labels = df.groupby("sequence").agg(**agg_dict).reset_index()

    seq_labels["label"] = None
    seq_labels.loc[seq_labels["is_active"], "label"] = 1
    seq_labels.loc[seq_labels["is_not_active"], "label"] = 0

    labeled_sequences = seq_labels.dropna(subset=["label"]).copy()
    labeled_sequences["label"] = labeled_sequences["label"].astype(int)
    labeled_sequences["quality"] = "curated"

    pos = sum(labeled_sequences['label'] == 1)
    neg = sum(labeled_sequences['label'] == 0)
    logger.info(f"Activity-derived: {len(labeled_sequences)} ({pos} pos, {neg} neg)")

    if negatives_dfs:
        for neg_df in negatives_dfs:
            required = ["sequence", "sequence_id", "cluster_id", "split"]
            missing = [c for c in required if c not in neg_df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            keep = required + (["quality"] if "quality" in neg_df.columns else [])
            neg_seqs = neg_df[keep].drop_duplicates().copy()
            neg_seqs["label"] = 0
            if "quality" not in neg_seqs.columns:
                neg_seqs["quality"] = "curated"

            before = len(labeled_sequences)
            labeled_sequences = pd.concat([labeled_sequences, neg_seqs], ignore_index=True)
            logger.info(f"Added {len(labeled_sequences) - before} negatives")

    return labeled_sequences


def compute_sample_weights(
    train_sequences: pd.DataFrame,
    gamma_synthetic: float = DEFAULT_GAMMA_SYNTHETIC,
) -> pd.Series:
    """Compute per-sequence sample weights"""
    df = train_sequences.copy()

    if "quality" not in df.columns:
        df["quality"] = "curated"

    num_pos = (df["label"] == 1).sum()
    num_neg_cur = ((df["label"] == 0) & (df["quality"] == "curated")).sum()
    num_neg_syn = ((df["label"] == 0) & (df["quality"] == "synthetic")).sum()
    total = num_pos + num_neg_cur + num_neg_syn

    if num_pos == 0 or (num_neg_cur + gamma_synthetic * num_neg_syn) == 0:
        raise ValueError("Invalid class counts for sample weighting")

    omega_pos = total / (2 * num_pos)
    omega_neg = total / (2 * (num_neg_cur + gamma_synthetic * num_neg_syn))

    def _weight_row(row: pd.Series) -> float:
        if row["label"] == 1:
            return omega_pos
        if row.get("quality", "curated") == "synthetic":
            return gamma_synthetic * omega_neg
        return omega_neg

    df["_w"] = df.apply(_weight_row, axis=1)
    weight_map = df.groupby("sequence")["_w"].first()
    weight_map.name = "weight"
    return weight_map


class AntimicrobialActivityClassifier(XGBoostClassifier):
    """XGBoost classifier for antimicrobial activity"""

    def __init__(
        self,
        pos_threshold_ugml: float = DEFAULT_POS_THRESHOLD_UGML,
        neg_threshold_ugml: float = DEFAULT_NEG_THRESHOLD_UGML,
        gamma_synthetic: float = DEFAULT_GAMMA_SYNTHETIC,
        decision_threshold: float = 0.5,
    ):
        super().__init__(decision_threshold=decision_threshold)
        self.pos_threshold_ugml = pos_threshold_ugml
        self.neg_threshold_ugml = neg_threshold_ugml
        self.gamma_synthetic = gamma_synthetic

    def _get_target_names(self) -> list[str]:
        return ["Not Active", "Active"]


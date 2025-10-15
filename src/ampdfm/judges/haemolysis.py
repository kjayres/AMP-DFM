#!/usr/bin/env python3
"""Haemolysis judge for erythrocyte toxicity prediction.

Classifies peptides as safe (≤20% haemolysis at ≥50 µM) or toxic (>20% at any conc)
based on ESM-2 embeddings using XGBoost.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .xgboost_judge import XGBoostJudge

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_CONC_THRESHOLD_UM = 50.0  # µM (used for SAFE criterion)
DEFAULT_PCT_THRESHOLD = 20.0  # % haemolysis (threshold for toxicity)


def label_haemolysis_sequences(
    df: pd.DataFrame,
    conc_threshold_um: float = DEFAULT_CONC_THRESHOLD_UM,
    pct_threshold: float = DEFAULT_PCT_THRESHOLD,
) -> pd.DataFrame:
    """Label sequences using the haemolysis 20% / 50 µM rule.

    Labelling rule:
        - If any measurement shows > pct_threshold % haemolysis ⇒ label 0 (toxic)
        - Else, if peptide has at least one measurement at ≥ conc_threshold_um µM
          (not upper-bound "<") showing ≤ pct_threshold % haemolysis ⇒ label 1 (safe)
        - Otherwise excluded (unlabelled)

    Args:
        df: DataFrame with columns 'sequence', 'value' (log10 µM), 'unit',
            'percentage' (haemolysis %), 'conc_qual', 'sequence_id', 'cluster_id', 'split'
        conc_threshold_um: Concentration threshold for SAFE criterion (µM)
        pct_threshold: Percentage haemolysis threshold for toxicity

    Returns:
        DataFrame with columns: sequence, sequence_id, cluster_id, split, label (0/1)
    """
    logger.info(
        f"Labelling haemolysis sequences: >%.0f %% haemolysis at any conc ⇒ toxic; "
        f"≤%.0f %% at ≥%.0f µM ⇒ safe",
        pct_threshold,
        pct_threshold,
        conc_threshold_um,
    )

    df = df.copy()

    # Compute µM concentration from stored log10(µM)
    df["conc_uM"] = 10 ** df["value"]

    # Keep only rows with units in µM
    df = df[df["unit"].str.lower() == "um"].copy()

    # Drop lower-bound rows (conc_qual == '>') – not informative
    df = df[df["conc_qual"].fillna("") != ">"].copy()

    # Cap impossible values (>100 %)
    if (df["percentage"] > 100).any():
        n_clip = (df["percentage"] > 100).sum()
        logger.warning("Clipping %d haemolysis values >100%% to 100%%", n_clip)
        df.loc[df["percentage"] > 100, "percentage"] = 100.0

    # Per-peptide labelling
    records = []
    for seq, grp in df.groupby("sequence", sort=False):
        if (grp["percentage"] > pct_threshold).any():
            label = 0  # toxic
        elif (
            (grp["conc_uM"] >= conc_threshold_um)
            & (grp["percentage"] <= pct_threshold)
            & (grp["conc_qual"].fillna("") != "<")
        ).any():
            label = 1  # safe
        else:
            continue  # cannot assign label

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
        "Haemolysis sequences labelled: %d  (safe: %d, toxic: %d)",
        len(labelled_df),
        (labelled_df["label"] == 1).sum(),
        (labelled_df["label"] == 0).sum(),
    )
    return labelled_df


class HaemolysisJudge(XGBoostJudge):
    """XGBoost-based classifier for erythrocyte haemolysis prediction."""

    def __init__(
        self,
        conc_threshold_um: float = DEFAULT_CONC_THRESHOLD_UM,
        pct_threshold: float = DEFAULT_PCT_THRESHOLD,
        decision_threshold: float = 0.5,
    ):
        """Initialize haemolysis judge.

        Args:
            conc_threshold_um: Concentration threshold for SAFE criterion (µM)
            pct_threshold: Percentage haemolysis threshold for toxicity
            decision_threshold: Probability threshold for binary classification
        """
        super().__init__(decision_threshold=decision_threshold)
        self.conc_threshold_um = conc_threshold_um
        self.pct_threshold = pct_threshold

    def _get_target_names(self) -> list[str]:
        """Get target class names for classification report."""
        return ["Toxic", "Safe"]


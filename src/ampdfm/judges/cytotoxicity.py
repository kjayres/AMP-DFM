#!/usr/bin/env python3
"""Cytotoxicity judge for cell-line toxicity prediction.

Classifies peptides as safe (ICx50 > 50 µM) or toxic (ICx50 ≤ 50 µM)
based on ESM-2 embeddings using XGBoost.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .xgboost_judge import XGBoostJudge

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


class CytotoxicityJudge(XGBoostJudge):
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
        super().__init__(decision_threshold=decision_threshold)
        self.tox_threshold_um = tox_threshold_um

    def _get_target_names(self) -> list[str]:
        """Get target class names for classification report."""
        return ["Toxic", "Safe"]


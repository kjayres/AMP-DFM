#!/usr/bin/env python3
"""Cytotoxicity classifier for cell-line toxicity prediction."""

from __future__ import annotations

import logging

import pandas as pd

from .xgboost_classifier import XGBoostClassifier

logger = logging.getLogger(__name__)

DEFAULT_TOX_THRESHOLD_UM = 50.0
DEFAULT_DECISION_THRESHOLD = 0.5

FIFTY_PERCENT_ENDPOINTS = {"IC50", "CC50", "EC50", "LC50", "LD50"}


def label_cytotoxicity_sequences(
    df: pd.DataFrame,
    tox_threshold_um: float = DEFAULT_TOX_THRESHOLD_UM,
) -> pd.DataFrame:
    """Label sequences using ICx50 ≤ 50µM toxicity rule."""
    logger.info(f"Labelling cytotoxicity: ICx ≤{tox_threshold_um}µM toxic, >{tox_threshold_um}µM safe")

    df = df.copy()
    endpoint_mask = df["measure_type"].str.upper().isin(FIFTY_PERCENT_ENDPOINTS)
    unit_mask = df["unit"].str.lower() == "um"
    df = df[endpoint_mask & unit_mask]
    df["conc_uM"] = 10 ** df["value"]

    records = []
    for seq, grp in df.groupby("sequence", sort=False):
        q = grp["qualifier"].fillna("")
        conc = grp["conc_uM"]

        toxic_mask = (conc <= tox_threshold_um) & (q != ">")
        safe_mask = (conc > tox_threshold_um) & (q != "<")

        if toxic_mask.any():
            label = 0
        elif safe_mask.any():
            label = 1
        else:
            continue

        first = grp.iloc[0]
        records.append({
            "sequence": seq,
            "sequence_id": first["sequence_id"],
            "cluster_id": first["cluster_id"],
            "split": first["split"],
            "label": label,
        })

    labelled_df = pd.DataFrame(records)
    safe = (labelled_df["label"] == 1).sum()
    toxic = (labelled_df["label"] == 0).sum()
    logger.info(f"Cytotoxicity: {len(labelled_df)} ({safe} safe, {toxic} toxic)")
    return labelled_df


class CytotoxicityClassifier(XGBoostClassifier):
    """XGBoost classifier for cytotoxicity prediction."""

    def __init__(
        self,
        tox_threshold_um: float = DEFAULT_TOX_THRESHOLD_UM,
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    ):
        super().__init__(decision_threshold=decision_threshold)
        self.tox_threshold_um = tox_threshold_um

    def _get_target_names(self) -> list[str]:
        return ["Toxic", "Safe"]


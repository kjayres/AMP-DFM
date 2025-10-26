"""Haemolysis classifier for erythrocyte toxicity prediction."""

from __future__ import annotations

import logging

import pandas as pd

from .xgboost_classifier import XGBoostClassifier

logger = logging.getLogger(__name__)

DEFAULT_CONC_THRESHOLD_UM = 50.0
DEFAULT_PCT_THRESHOLD = 20.0


def label_haemolysis_sequences(
    df: pd.DataFrame,
    conc_threshold_um: float = DEFAULT_CONC_THRESHOLD_UM,
    pct_threshold: float = DEFAULT_PCT_THRESHOLD,
) -> pd.DataFrame:
    """Label sequences using haemolysis 20%/50uM rule"""
    logger.info(f"Labelling haemolysis: >{pct_threshold}% toxic, ≤{pct_threshold}% at ≥{conc_threshold_um}µM safe")

    df = df.copy()
    df["conc_uM"] = 10 ** df["value"]
    df = df[df["unit"].str.lower() == "um"].copy()
    df = df[df["conc_qual"].fillna("") != ">"].copy()

    if (df["percentage"] > 100).any():
        logger.warning(f"Clipping {(df['percentage'] > 100).sum()} values >100%")
        df.loc[df["percentage"] > 100, "percentage"] = 100.0

    records = []
    for seq, grp in df.groupby("sequence", sort=False):
        if (grp["percentage"] > pct_threshold).any():
            label = 0
        elif (
            (grp["conc_uM"] >= conc_threshold_um)
            & (grp["percentage"] <= pct_threshold)
            & (grp["conc_qual"].fillna("") != "<")
        ).any():
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
    logger.info(f"Haemolysis: {len(labelled_df)} ({safe} safe, {toxic} toxic)")
    return labelled_df


class HaemolysisClassifier(XGBoostClassifier):
    """XGBoost classifier for haemolysis prediction"""

    def __init__(
        self,
        conc_threshold_um: float = DEFAULT_CONC_THRESHOLD_UM,
        pct_threshold: float = DEFAULT_PCT_THRESHOLD,
        decision_threshold: float = 0.5,
    ):
        super().__init__(decision_threshold=decision_threshold)
        self.conc_threshold_um = conc_threshold_um
        self.pct_threshold = pct_threshold

    def _get_target_names(self) -> list[str]:
        return ["Toxic", "Safe"]


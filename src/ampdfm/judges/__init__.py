#!/usr/bin/env python3
"""XGBoost judge models for antimicrobial peptide property prediction.

Provides judges for antimicrobial activity, haemolysis, and cytotoxicity.
"""

from .antimicrobial_activity import (
    AntimicrobialActivityJudge,
    label_antimicrobial_activity_sequences,
    compute_sample_weights,
)
from .base import BaseJudge
from .cytotoxicity import CytotoxicityJudge, label_cytotoxicity_sequences
from .haemolysis import HaemolysisJudge, label_haemolysis_sequences
from .xgboost_judge import XGBoostJudge
from .utils import create_train_val_test_splits, load_embeddings, prepare_features
from .optuna_helpers import tune_xgboost

__all__ = [
    # Base classes
    "BaseJudge",
    "XGBoostJudge",
    # Judge classes
    "AntimicrobialActivityJudge",
    "HaemolysisJudge",
    "CytotoxicityJudge",
    # Labelling helpers
    "label_antimicrobial_activity_sequences",
    "label_haemolysis_sequences",
    "label_cytotoxicity_sequences",
    "compute_sample_weights",
    # Utility functions
    "load_embeddings",
    "prepare_features",
    "create_train_val_test_splits",
    # Optuna tuning
    "tune_xgboost",
]


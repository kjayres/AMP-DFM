#!/usr/bin/env python3
"""XGBoost classifier models for antimicrobial peptide property prediction.

Provides classifiers for antimicrobial activity, haemolysis, and cytotoxicity.
"""

from .antimicrobial_activity import (
    AntimicrobialActivityClassifier,
    label_antimicrobial_activity_sequences,
    compute_sample_weights,
)
from .base import BaseClassifier
from .cytotoxicity import CytotoxicityClassifier, label_cytotoxicity_sequences
from .haemolysis import HaemolysisClassifier, label_haemolysis_sequences
from .xgboost_classifier import XGBoostClassifier
from .utils import create_train_val_test_splits, load_embeddings, prepare_features
from .optuna_helpers import tune_xgboost

__all__ = [
    # Base classes
    "BaseClassifier",
    "XGBoostClassifier",
    # Classifier classes
    "AntimicrobialActivityClassifier",
    "HaemolysisClassifier",
    "CytotoxicityClassifier",
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


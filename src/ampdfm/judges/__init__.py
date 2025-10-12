#!/usr/bin/env python3
"""Judge models for antimicrobial peptide property prediction.

This module provides judge models for:
- Antimicrobial activity: potency classification (MIC thresholds)
- Haemolysis: erythrocyte toxicity prediction
- Cytotoxicity: cell-line toxicity prediction

Each judge can be trained, evaluated, saved, and loaded for inference.

Example:
    >>> from ampdfm.judges import (
    ...     AntimicrobialActivityJudge,
    ...     label_antimicrobial_activity_sequences,
    ...     load_embeddings,
    ...     prepare_features,
    ... )
    >>> 
    >>> # Label sequences (activity + negatives)
    >>> labeled_df = label_antimicrobial_activity_sequences(activity_df, negatives_dfs=[neg_df])
    >>> 
    >>> # Load embeddings and extract features
    >>> embeddings, seq_index = load_embeddings()
    >>> X_train, y_train, _ = prepare_features(train_df, embeddings, seq_index)
    >>> X_val, y_val, _ = prepare_features(val_df, embeddings, seq_index)
    >>> 
    >>> # Train judge
    >>> judge = AntimicrobialActivityJudge(pos_threshold_ugml=32, neg_threshold_ugml=128)
    >>> judge.train(X_train, y_train, X_val, y_val)
    >>> 
    >>> # Evaluate
    >>> auc, proba, pred = judge.evaluate(X_test, y_test, split_name="Test")
    >>> 
    >>> # Save for later use
    >>> judge.save("models/antimicrobial_activity/model.json")
"""

from .antimicrobial_activity import (
    AntimicrobialActivityJudge,
    label_antimicrobial_activity_sequences,
    compute_sample_weights,
)
from .base import BaseJudge
from .cytotoxicity import CytotoxicityJudge, label_cytotoxicity_sequences
from .haemolysis import HaemolysisJudge, label_haemolysis_sequences
from .sklearn_judge import SklearnJudge
from .utils import create_train_val_test_splits, load_embeddings, prepare_features
from .optuna_helpers import (
    tune_xgboost,
    tune_random_forest,
    tune_logistic_regression,
)

__all__ = [
    # Base class
    "BaseJudge",
    # Judge classes
    "AntimicrobialActivityJudge",
    "HaemolysisJudge",
    "CytotoxicityJudge",
    "SklearnJudge",
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
    "tune_random_forest",
    "tune_logistic_regression",
]


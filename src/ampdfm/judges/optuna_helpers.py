#!/usr/bin/env python3
"""Centralised Optuna hyper-parameter tuning for judge models.

Provides model-specific search spaces and objective functions for XGBoost,
Random Forest, and Logistic Regression judges.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    import optuna
except ImportError:
    optuna = None

logger = logging.getLogger(__name__)


def get_xgboost_search_space(trial: Any) -> dict[str, Any]:
    """Define XGBoost hyper-parameter search space.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of XGBoost parameters
    """
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'max_leaves': trial.suggest_int('max_leaves', 16, 128),
        'learning_rate': trial.suggest_float('learning_rate', 3e-3, 3e-2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
        'random_state': 42,
    }


def get_random_forest_search_space(trial: Any) -> dict[str, Any]:
    """Define Random Forest hyper-parameter search space.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of RandomForest parameters
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),  # Reduced from 1000
        'max_depth': trial.suggest_int('max_depth', 10, 30),  # Narrowed from 5-50
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),  # Removed None (all features)
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),  # Reduced from 20
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),  # Reduced from 10
        'random_state': 42,
        'n_jobs': -1,  # Use all CPU cores for parallel tree building
    }


def get_logistic_regression_search_space(trial: Any) -> dict[str, Any]:
    """Define Logistic Regression hyper-parameter search space.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of LogisticRegression parameters
    """
    return {
        'C': trial.suggest_float('C', 0.001, 100.0, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l2']),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
        'max_iter': trial.suggest_int('max_iter', 500, 5000, step=500),
        'random_state': 42,
    }


def tune_xgboost(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    groups_dev: np.ndarray,
    sample_weight_dev: Optional[np.ndarray] = None,
    n_trials: int = 20,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Run Optuna tuning for XGBoost model.

    Args:
        X_dev: Development set features (train+val combined)
        y_dev: Development set labels
        groups_dev: Cluster IDs for GroupKFold
        sample_weight_dev: Optional sample weights
        n_trials: Number of Optuna trials
        cv_folds: Number of cross-validation folds

    Returns:
        Best parameters dictionary
    """
    if optuna is None:
        logger.error("Optuna not installed – cannot run tuning.")
        return {}

    logger.info(f"Running {n_trials} Optuna trials for XGBoost tuning...")

    gkf = GroupKFold(n_splits=cv_folds)

    def objective(trial):
        params = get_xgboost_search_space(trial)
        aucs = []

        for tr_idx, va_idx in gkf.split(X_dev, groups=groups_dev):
            dtr_weight = sample_weight_dev[tr_idx] if sample_weight_dev is not None else None
            dtr = xgb.DMatrix(X_dev[tr_idx], label=y_dev[tr_idx], weight=dtr_weight)
            dva = xgb.DMatrix(X_dev[va_idx], label=y_dev[va_idx])

            bst = xgb.train(
                params,
                dtr,
                num_boost_round=2000,
                evals=[(dva, 'val')],
                early_stopping_rounds=100,
                verbose_eval=False,
            )
            aucs.append(bst.best_score)

        return float(np.mean(aucs))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best XGBoost params: {study.best_params}")
    return study.best_params


def tune_random_forest(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    groups_dev: np.ndarray,
    sample_weight_dev: Optional[np.ndarray] = None,
    n_trials: int = 20,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Run Optuna tuning for Random Forest model.

    Args:
        X_dev: Development set features (train+val combined)
        y_dev: Development set labels
        groups_dev: Cluster IDs for GroupKFold
        sample_weight_dev: Optional sample weights
        n_trials: Number of Optuna trials
        cv_folds: Number of cross-validation folds

    Returns:
        Best parameters dictionary
    """
    if optuna is None:
        logger.error("Optuna not installed – cannot run tuning.")
        return {}

    logger.info(f"Running {n_trials} Optuna trials for Random Forest tuning...")

    gkf = GroupKFold(n_splits=cv_folds)

    def objective(trial):
        params = get_random_forest_search_space(trial)
        aucs = []

        for tr_idx, va_idx in gkf.split(X_dev, groups=groups_dev):
            rf = RandomForestClassifier(**params)
            
            if sample_weight_dev is not None:
                rf.fit(X_dev[tr_idx], y_dev[tr_idx], sample_weight=sample_weight_dev[tr_idx])
            else:
                rf.fit(X_dev[tr_idx], y_dev[tr_idx])
            
            proba = rf.predict_proba(X_dev[va_idx])[:, 1]
            aucs.append(roc_auc_score(y_dev[va_idx], proba))

        return float(np.mean(aucs))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best Random Forest params: {study.best_params}")
    return study.best_params


def tune_logistic_regression(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    groups_dev: np.ndarray,
    sample_weight_dev: Optional[np.ndarray] = None,
    n_trials: int = 20,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Run Optuna tuning for Logistic Regression model.

    Args:
        X_dev: Development set features (train+val combined)
        y_dev: Development set labels
        groups_dev: Cluster IDs for GroupKFold
        sample_weight_dev: Optional sample weights
        n_trials: Number of Optuna trials
        cv_folds: Number of cross-validation folds

    Returns:
        Best parameters dictionary
    """
    if optuna is None:
        logger.error("Optuna not installed – cannot run tuning.")
        return {}

    logger.info(f"Running {n_trials} Optuna trials for Logistic Regression tuning...")

    gkf = GroupKFold(n_splits=cv_folds)

    def objective(trial):
        params = get_logistic_regression_search_space(trial)
        aucs = []

        for tr_idx, va_idx in gkf.split(X_dev, groups=groups_dev):
            lr = LogisticRegression(**params)
            
            if sample_weight_dev is not None:
                lr.fit(X_dev[tr_idx], y_dev[tr_idx], sample_weight=sample_weight_dev[tr_idx])
            else:
                lr.fit(X_dev[tr_idx], y_dev[tr_idx])
            
            proba = lr.predict_proba(X_dev[va_idx])[:, 1]
            aucs.append(roc_auc_score(y_dev[va_idx], proba))

        return float(np.mean(aucs))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best Logistic Regression params: {study.best_params}")
    return study.best_params


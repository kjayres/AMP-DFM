#!/usr/bin/env python3
"""Optuna hyper-parameter tuning for XGBoost judges."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

try:
    import optuna
except ImportError:
    optuna = None

logger = logging.getLogger(__name__)


def get_xgboost_search_space(
    trial: Any,
    search_space_config: Optional[dict[str, Any]] = None,
    base_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build XGBoost parameter dict from Optuna trial and base params."""
    search_space_config = search_space_config or {}
    params = dict(base_params) if base_params else {}

    for param in ['max_depth', 'max_leaves']:
        if param in search_space_config:
            low, high = search_space_config[param]
            params[param] = trial.suggest_int(param, low, high)

    if 'learning_rate' in search_space_config:
        low, high = search_space_config['learning_rate']
        params['learning_rate'] = trial.suggest_float('learning_rate', low, high, log=True)

    for param in ['subsample', 'colsample_bytree', 'min_child_weight', 'gamma', 'reg_lambda', 'scale_pos_weight']:
        if param in search_space_config:
            low, high = search_space_config[param]
            params[param] = trial.suggest_float(param, low, high)

    return params


def tune_xgboost(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    groups_dev: np.ndarray,
    sample_weight_dev: Optional[np.ndarray] = None,
    val_weight_dev: Optional[np.ndarray] = None,
    n_trials: int = 20,
    cv_folds: int = 5,
    search_space_config: Optional[dict[str, Any]] = None,
    base_params: Optional[dict[str, Any]] = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
) -> dict[str, Any]:
    """Run Optuna tuning for XGBoost. Returns best parameters."""
    if optuna is None:
        logger.error("Optuna not installed")
        return {}

    logger.info(f"Running {n_trials} Optuna trials...")
    gkf = GroupKFold(n_splits=cv_folds)

    def objective(trial):
        params = get_xgboost_search_space(trial, search_space_config, base_params)
        aucs = []

        for tr_idx, va_idx in gkf.split(X_dev, groups=groups_dev):
            dtr_weight = sample_weight_dev[tr_idx] if sample_weight_dev is not None else None
            dva_weight = val_weight_dev[va_idx] if val_weight_dev is not None else None
            dtr = xgb.DMatrix(X_dev[tr_idx], label=y_dev[tr_idx], weight=dtr_weight)
            dva = xgb.DMatrix(X_dev[va_idx], label=y_dev[va_idx], weight=dva_weight)

            bst = xgb.train(
                params,
                dtr,
                num_boost_round=num_boost_round,
                evals=[(dva, 'val')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
            aucs.append(bst.best_score)

        return float(np.mean(aucs))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best params: {study.best_params}")
    return study.best_params




#!/usr/bin/env python3
"""Optuna hyper-parameter tuning for XGBoost classifiers."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
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
    n_trials: int = 20,
    cv_folds: int = 5,
    search_space_config: Optional[dict[str, Any]] = None,
    base_params: Optional[dict[str, Any]] = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    # Optional: leakage-free per-fold weighting for antimicrobial activity
    dev_df: Optional[pd.DataFrame] = None,
    dev_sequences: Optional[list[str]] = None,
    gamma_synthetic: Optional[float] = None,
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
            # Prefer leakage-free per-fold weights when dev_df/dev_sequences/gamma are provided
            if (
                dev_df is not None
                and dev_sequences is not None
                and gamma_synthetic is not None
                and "quality" in dev_df.columns
            ):
                # Build sequence→(label, quality) map once from dev_df
                # dev_df is expected to have one row per sequence
                seq_meta = (
                    dev_df.drop_duplicates("sequence")[["sequence", "label", "quality"]]
                    .set_index("sequence")
                )

                # Get sequence order for current fold indices
                tr_seqs = [dev_sequences[i] for i in tr_idx]
                va_seqs = [dev_sequences[i] for i in va_idx]

                # Compute omegas from TRAIN FOLD ONLY (no leakage)
                tr_rows_df = dev_df[dev_df["sequence"].isin(tr_seqs)].copy()
                num_pos = int((tr_rows_df["label"] == 1).sum())
                num_neg_cur = int(
                    ((tr_rows_df["label"] == 0) & (tr_rows_df["quality"] == "curated")).sum()
                )
                num_neg_syn = int(
                    ((tr_rows_df["label"] == 0) & (tr_rows_df["quality"] == "synthetic")).sum()
                )
                total = num_pos + num_neg_cur + num_neg_syn

                # Guard against degenerate folds; if invalid, fall back to unweighted for this fold
                if num_pos == 0 or (num_neg_cur + gamma_synthetic * num_neg_syn) == 0 or total == 0:
                    dtr = xgb.DMatrix(X_dev[tr_idx], label=y_dev[tr_idx])
                    dva = xgb.DMatrix(X_dev[va_idx], label=y_dev[va_idx])
                else:
                    omega_pos = total / (2.0 * num_pos)
                    omega_neg = total / (
                        2.0 * (num_neg_cur + float(gamma_synthetic) * num_neg_syn)
                    )

                    def _weight_for(seq: str) -> float:
                        row = seq_meta.loc[seq]
                        if int(row["label"]) == 1:
                            return float(omega_pos)
                        return (
                            float(gamma_synthetic) * float(omega_neg)
                            if str(row["quality"]).lower() == "synthetic"
                            else float(omega_neg)
                        )

                    tr_w = np.asarray([_weight_for(s) for s in tr_seqs], dtype=float)
                    # Keep validation UNWEIGHTED to match final training/evaluation
                    dtr = xgb.DMatrix(X_dev[tr_idx], label=y_dev[tr_idx], weight=tr_w)
                    dva = xgb.DMatrix(X_dev[va_idx], label=y_dev[va_idx])
            else:
                # Unweighted for tasks without synthetic negatives (haemolysis, cytotoxicity)
                dtr = xgb.DMatrix(X_dev[tr_idx], label=y_dev[tr_idx])
                dva = xgb.DMatrix(X_dev[va_idx], label=y_dev[va_idx])

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




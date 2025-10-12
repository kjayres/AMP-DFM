#!/usr/bin/env python3
"""Unified judge training script for all AMP-DFM judge models.

This script handles training for all combinations of:
- Tasks: antimicrobial_activity, haemolysis, cytotoxicity
- Models: xgboost, random_forest, logistic_regression
- Variants: all organisms, species-specific (for antimicrobial_activity)

Configuration is loaded from YAML files to keep the script minimal and flexible.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay

# Ensure local package imports work when running from PBS wrappers
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../amp_dfm
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ampdfm.judges import (
    AntimicrobialActivityJudge,
    CytotoxicityJudge,
    HaemolysisJudge,
    SklearnJudge,
    compute_sample_weights,
    create_train_val_test_splits,
    label_antimicrobial_activity_sequences,
    label_cytotoxicity_sequences,
    label_haemolysis_sequences,
    load_embeddings,
    prepare_features,
    tune_logistic_regression,
    tune_random_forest,
    tune_xgboost,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train AMP-DFM judge models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Extract configuration
    task = config['task']
    model_type = config['model']
    data_dir = Path(config['data_dir'])
    checkpoint_dir = Path(config['checkpoint_dir'])
    output_dir = Path(config['output_dir'])

    # Task-specific configuration
    if task == 'antimicrobial_activity':
        organism = config.get('organism', None)
        pos_threshold = config['thresholds']['pos_threshold_ugml']
        neg_threshold = config['thresholds']['neg_threshold_ugml']
        gamma_synthetic = config.get('gamma_synthetic', 0.3)
        negatives = config.get('negatives', [
            'negatives_with_splits.csv',
            'negatives_synth_with_splits.csv'
        ])
        xgboost_params_cfg = config.get('xgboost_params', {})
        model_params_cfg = config.get('model_params', {})
    elif task == 'haemolysis':
        conc_threshold = config['thresholds']['conc_threshold_um']
        pct_threshold = config['thresholds']['pct_threshold']
        xgboost_params_cfg = config.get('xgboost_params', {})
        model_params_cfg = config.get('model_params', {})
    elif task == 'cytotoxicity':
        tox_threshold = config['thresholds']['tox_threshold_um']
        decision_threshold_cfg = config.get('thresholds', {}).get('decision_threshold', None)
        xgboost_params_cfg = config.get('xgboost_params', {})
        model_params_cfg = config.get('model_params', {})
    else:
        raise ValueError(f"Unknown task: {task}")

    # Optuna configuration
    optuna_config = config.get('optuna', {})
    n_trials = optuna_config.get('trials', 50)
    cv_folds = optuna_config.get('cv_folds', 5)

    # Training configuration
    training_cfg = config.get('training', {})
    num_boost_round_cfg = int(training_cfg.get('num_boost_round', 2000))
    early_stopping_rounds_cfg = int(training_cfg.get('early_stopping_rounds', 100))

    # Evaluation configuration (optional cluster-aware CV after training)
    evaluation_cfg = config.get('evaluation', {})
    eval_cv_folds = int(evaluation_cfg.get('cv_folds', 0))

    # Create output directories (no "_all" suffix; nest organism if provided)
    base_run_name = f"{task}_{model_type}"
    organism_subdir = (
        organism.replace(' ', '_').lower()
        if task == 'antimicrobial_activity' and organism
        else None
    )
    checkpoint_path = (
        checkpoint_dir / task / organism_subdir / base_run_name
        if organism_subdir else checkpoint_dir / task / base_run_name
    )
    output_path = (
        output_dir / task / organism_subdir / base_run_name
        if organism_subdir else output_dir / task / base_run_name
    )
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    organism_label = f" ({organism_subdir})" if organism_subdir else ""
    logger.info(f"Training {model_type} for {task}{organism_label}")
    logger.info(f"Checkpoints: {checkpoint_path}")
    logger.info(f"Outputs: {output_path}")

    # Load and label data based on task
    if task == 'antimicrobial_activity':
        # Expect data_dir to be the dataset root (…/data). Read from clustered subfolder.
        clustered = data_dir / 'clustered'
        activity_df = pd.read_csv(clustered / 'activities_with_splits.csv')
        # Load negatives and tag quality based on filename (synthetic vs curated)
        negatives_dfs = []
        for neg_file in negatives:
            neg_path = clustered / neg_file
            df_neg = pd.read_csv(neg_path)
            df_neg = df_neg.copy()
            df_neg['quality'] = 'synthetic' if 'synth' in neg_file.lower() else 'curated'
            negatives_dfs.append(df_neg)

        df_labelled = label_antimicrobial_activity_sequences(
            activity_df=activity_df,
            negatives_dfs=negatives_dfs,
            pos_threshold_ugml=pos_threshold,
            neg_threshold_ugml=neg_threshold,
            organism_filter=organism,
        )
    elif task == 'haemolysis':
        raw_df = pd.read_csv(data_dir / 'clustered/haemolysis_with_splits.csv')
        df_labelled = label_haemolysis_sequences(
            df=raw_df,
            conc_threshold_um=conc_threshold,
            pct_threshold=pct_threshold,
        )
    elif task == 'cytotoxicity':
        raw_df = pd.read_csv(data_dir / 'clustered/cytotoxicity_with_splits.csv')
        df_labelled = label_cytotoxicity_sequences(
            df=raw_df,
            tox_threshold_um=tox_threshold,
        )

    logger.info(f"Labelled data: {len(df_labelled)} sequences")
    logger.info(f"Label distribution:\n{df_labelled['label'].value_counts()}")

    # Load embeddings (ESM-2) using library helper. Expect at data_dir/embeddings.
    embeddings, seq_index = load_embeddings(data_dir / 'embeddings')

    # Create train/val/test splits as DataFrames
    df_train, df_val, df_test = create_train_val_test_splits(df_labelled)

    # Prepare features per split
    X_train, y_train, train_sequences = prepare_features(df_train, embeddings, seq_index)
    X_val, y_val, val_sequences = prepare_features(df_val, embeddings, seq_index)
    X_test, y_test, test_sequences = prepare_features(df_test, embeddings, seq_index)

    # Extract cluster IDs per split for CV/metadata
    # De-duplicate sequence index to avoid length inflation when mapping cluster IDs
    cluster_map_train = df_train.drop_duplicates('sequence').set_index('sequence')['cluster_id']
    clusters_train = cluster_map_train.loc[train_sequences].to_numpy()
    
    cluster_map_val = df_val.drop_duplicates('sequence').set_index('sequence')['cluster_id']
    clusters_val = cluster_map_val.loc[val_sequences].to_numpy()
    
    cluster_map_test = df_test.drop_duplicates('sequence').set_index('sequence')['cluster_id']
    clusters_test = cluster_map_test.loc[test_sequences].to_numpy()

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Compute sample weights (only for antimicrobial_activity)
    sample_weight = None
    if task == 'antimicrobial_activity':
        # Compute per-sequence weights on the training split, then align to feature order
        weight_map = compute_sample_weights(
            df_train,
            gamma_synthetic=gamma_synthetic,
        )
        sample_weight = weight_map.loc[train_sequences].to_numpy()

    # Run Optuna tuning on dev set (train + val)
    X_dev = np.vstack([X_train, X_val])
    y_dev = np.concatenate([y_train, y_val])
    clusters_dev = np.concatenate([clusters_train, clusters_val])
    
    # Sanity check: ensure lengths match for GroupKFold
    assert X_dev.shape[0] == y_dev.shape[0] == clusters_dev.shape[0], \
        f"Length mismatch: X_dev={X_dev.shape[0]}, y_dev={y_dev.shape[0]}, clusters_dev={clusters_dev.shape[0]}"
    
    if sample_weight is not None:
        # Create sample weight for dev set (train only, val gets weight=1)
        sample_weight_dev = np.concatenate([sample_weight, np.ones(len(y_val))])
    else:
        sample_weight_dev = None

    if model_type == 'xgboost':
        best_params = tune_xgboost(
            X_dev, y_dev, clusters_dev,
            sample_weight_dev=sample_weight_dev,
            n_trials=n_trials,
            cv_folds=cv_folds,
        )
        # Merge YAML-provided xgboost_params as defaults (tuned values override)
        if best_params is None:
            best_params = {}
        best_params = {**xgboost_params_cfg, **best_params}
        
        # Create and train final XGBoost judge
        if task == 'antimicrobial_activity':
            judge = AntimicrobialActivityJudge()
        elif task == 'haemolysis':
            judge = HaemolysisJudge()
        elif task == 'cytotoxicity':
            # Allow optional decision threshold override from config
            judge = CytotoxicityJudge(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg is not None else CytotoxicityJudge()
            
    elif model_type == 'random_forest':
        best_params = tune_random_forest(
            X_dev, y_dev, clusters_dev,
            sample_weight_dev=sample_weight_dev,
            n_trials=n_trials,
            cv_folds=cv_folds,
        )
        # Merge YAML model_params defaults with tuned params (tuned override)
        if best_params is None:
            best_params = {}
        combined_params = {**model_params_cfg, **best_params}
        # For cytotoxicity, pass decision threshold to the wrapper
        if task == 'cytotoxicity' and decision_threshold_cfg is not None:
            judge = SklearnJudge(RandomForestClassifier, combined_params, decision_threshold=decision_threshold_cfg)
        else:
            judge = SklearnJudge(RandomForestClassifier, combined_params)
        
    elif model_type == 'logistic_regression':
        best_params = tune_logistic_regression(
            X_dev, y_dev, clusters_dev,
            sample_weight_dev=sample_weight_dev,
            n_trials=n_trials,
            cv_folds=cv_folds,
        )
        if best_params is None:
            best_params = {}
        combined_params = {**model_params_cfg, **best_params}
        if task == 'cytotoxicity' and decision_threshold_cfg is not None:
            judge = SklearnJudge(LogisticRegression, combined_params, decision_threshold=decision_threshold_cfg)
        else:
            judge = SklearnJudge(LogisticRegression, combined_params)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train final model
    if model_type == 'xgboost':
        result = judge.train(
            X_train, y_train, X_val, y_val,
            sample_weight=sample_weight,
            params_override=best_params,
            num_boost_round=num_boost_round_cfg,
            early_stopping_rounds=early_stopping_rounds_cfg,
        )
    else:
        result = judge.train(
            X_train, y_train, X_val, y_val,
            sample_weight=sample_weight,
        )

    # Evaluate on all splits
    train_auc, train_proba, train_pred = judge.evaluate(X_train, y_train, "Train")
    val_auc, val_proba, val_pred = judge.evaluate(X_val, y_val, "Val")
    test_auc, test_proba, test_pred = judge.evaluate(X_test, y_test, "Test")

    # Save results (aligned to the feature order used for X_test)
    test_results = pd.DataFrame({
        'sequence': test_sequences,
        'true_label': y_test,
        'predicted_proba': test_proba,
        'predicted_label': test_pred,
        'cluster_id': clusters_test,
    })
    test_results.to_csv(output_path / 'test_results.csv', index=False)
    logger.info(f"Test results saved to {output_path / 'test_results.csv'}")

    # Save ROC data (FPR/TPR) for later plotting
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, test_proba)
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds})
    roc_df.to_csv(output_path / 'roc_curve_data.csv', index=False)
    logger.info(f"ROC curve data saved to {output_path / 'roc_curve_data.csv'}")

    # Save learning curve data (not plots)
    if model_type == 'xgboost' and 'evals_result' in result:
        evals = result['evals_result']
        pd.DataFrame({
            'round': np.arange(len(evals['train']['auc'])),
            'train_auc': evals['train']['auc'],
            'val_auc': evals['val']['auc'],
        }).to_csv(output_path / 'learning_curve.csv', index=False)
        logger.info(f"Learning curve data saved to {output_path / 'learning_curve.csv'}")
    elif model_type != 'xgboost':
        # Save a simple CSV with single-point AUCs for sklearn models
        pd.DataFrame({'train_auc': [train_auc], 'val_auc': [val_auc]}).to_csv(output_path / 'learning_curve.csv', index=False)
        logger.info(f"Learning curve (single-point) saved to {output_path / 'learning_curve.csv'}")

    # Optional cluster-aware cross-validation for robustness estimates
    if eval_cv_folds and eval_cv_folds > 1:
        logger.info(f"Running {eval_cv_folds}-fold cluster-aware cross-validation…")
        from sklearn.model_selection import GroupKFold

        groups_full = df_labelled['cluster_id'].to_numpy()
        gkf = GroupKFold(n_splits=eval_cv_folds)
        aucs: list[float] = []
        fold_dir = output_path / 'cv'
        fold_dir.mkdir(exist_ok=True)

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(df_labelled, groups=groups_full), start=1):
            tr_rows = df_labelled.iloc[tr_idx]
            va_rows = df_labelled.iloc[va_idx]

            X_tr_cv, y_tr_cv, tr_seqs_cv = prepare_features(tr_rows, embeddings, seq_index)
            X_val_cv, y_val_cv, _ = prepare_features(va_rows, embeddings, seq_index)

            # Build judge per task/model
            if model_type == 'xgboost':
                if task == 'antimicrobial_activity':
                    judge_cv = AntimicrobialActivityJudge()
                elif task == 'haemolysis':
                    judge_cv = HaemolysisJudge()
                else:
                    judge_cv = CytotoxicityJudge(decision_threshold=decision_threshold_cfg) if 'decision_threshold_cfg' in locals() and decision_threshold_cfg is not None else CytotoxicityJudge()

                # Sample weights for antimicrobial activity
                if task == 'antimicrobial_activity':
                    w_map_cv = compute_sample_weights(tr_rows, gamma_synthetic=gamma_synthetic)
                    w_tr_cv = w_map_cv.loc[tr_seqs_cv].to_numpy()
                else:
                    w_tr_cv = None

                res_cv = judge_cv.train(
                    X_tr_cv, y_tr_cv, X_val_cv, y_val_cv,
                    sample_weight=w_tr_cv,
                    params_override=best_params,
                    num_boost_round=num_boost_round_cfg,
                    early_stopping_rounds=early_stopping_rounds_cfg,
                )
                auc_cv, _, _ = judge_cv.evaluate(X_val_cv, y_val_cv, split_name=f"CV{fold}")
            else:
                # Sklearn models via wrapper
                if model_type == 'random_forest':
                    judge_cv = SklearnJudge(RandomForestClassifier, model_params_cfg, decision_threshold=decision_threshold_cfg if task == 'cytotoxicity' else None)
                else:
                    judge_cv = SklearnJudge(LogisticRegression, model_params_cfg, decision_threshold=decision_threshold_cfg if task == 'cytotoxicity' else None)

                # Sklearn fit; antimicrobial sample weights ignored unless supported
                res_cv = judge_cv.train(X_tr_cv, y_tr_cv, X_val_cv, y_val_cv)
                auc_cv, _, _ = judge_cv.evaluate(X_val_cv, y_val_cv, split_name=f"CV{fold}")

            aucs.append(float(auc_cv))

        pd.DataFrame({'fold': np.arange(1, len(aucs)+1), 'auc': aucs}).to_csv(fold_dir / 'cv_metrics.csv', index=False)
        logger.info(f"CV metrics saved to {fold_dir / 'cv_metrics.csv'}")

    # Inference timing on test set (store per-run and aggregate)
    start = time.perf_counter()
    _ = judge.predict_proba(X_test)
    elapsed = time.perf_counter() - start
    ms_per_pep = (elapsed / len(X_test)) * 1000.0 if len(X_test) else float('nan')

    timing_df = pd.DataFrame({
        'task': [task],
        'model_type': [model_type],
        'organism': [organism_subdir if organism_subdir else 'all'],
        'n_peptides': [int(len(X_test))],
        'seconds': [float(elapsed)],
        'ms_per_peptide': [float(ms_per_pep)],
    })
    timing_df.to_csv(output_path / 'inference_timing.csv', index=False)
    # Also append/aggregate under outputs/model_timings
    timings_root = output_dir.parent / 'model_timings'
    timings_root.mkdir(parents=True, exist_ok=True)
    aggregate_csv = timings_root / 'inference_timing.csv'
    if aggregate_csv.exists():
        prev = pd.read_csv(aggregate_csv)
        pd.concat([prev, timing_df], ignore_index=True).to_csv(aggregate_csv, index=False)
    else:
        timing_df.to_csv(aggregate_csv, index=False)
    logger.info(f"Inference timing saved to {output_path / 'inference_timing.csv'} and aggregated at {aggregate_csv}")

    # Save model checkpoint
    if model_type == 'xgboost':
        model_path = checkpoint_path / 'model.json'
    else:
        model_path = checkpoint_path / 'model.pkl'
    
    judge.save(model_path)

    # Save metadata
    metadata = {
        'task': task,
        'model_type': model_type,
        'organism': organism_subdir if organism_subdir else 'all',
        'best_params': best_params,
        'train_auc': float(train_auc),
        'val_auc': float(val_auc),
        'test_auc': float(test_auc),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'feature_dim': X_train.shape[1],
    }
    
    if task == 'antimicrobial_activity':
        metadata.update({
            'organism': organism,
            'pos_threshold_ugml': pos_threshold,
            'neg_threshold_ugml': neg_threshold,
            'gamma_synthetic': gamma_synthetic,
        })
    elif task == 'haemolysis':
        metadata.update({
            'conc_threshold_um': conc_threshold,
            'pct_threshold': pct_threshold,
        })
    elif task == 'cytotoxicity':
        metadata.update({
            'tox_threshold_um': tox_threshold,
        })
    
    with open(checkpoint_path / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"Metadata saved to {checkpoint_path / 'metadata.pkl'}")
    logger.info("Training complete!")


if __name__ == '__main__':
    main()


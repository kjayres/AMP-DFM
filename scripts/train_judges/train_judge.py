#!/usr/bin/env python3
"""Unified judge training script for all AMP-DFM judge models.

This script handles training for all combinations of:
- Tasks: antimicrobial_activity, haemolysis, cytotoxicity
- Variants: all organisms, species-specific (for antimicrobial_activity)

All models use XGBoost. Configuration is loaded from YAML files to keep the script minimal and flexible.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

# Ensure local package imports work when running from PBS wrappers
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../amp_dfm
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ampdfm.judges import (
    AntimicrobialActivityJudge,
    CytotoxicityJudge,
    HaemolysisJudge,
    compute_sample_weights,
    create_train_val_test_splits,
    label_antimicrobial_activity_sequences,
    label_cytotoxicity_sequences,
    label_haemolysis_sequences,
    load_embeddings,
    prepare_features,
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
    data_dir = Path(config['data_dir'])
    checkpoint_dir = Path(config['checkpoint_dir'])
    output_dir = Path(config['output_dir'])

    # Task-specific configuration
    # Extract decision_threshold for all tasks (optional)
    decision_threshold_cfg = config.get('thresholds', {}).get('decision_threshold', None)
    
    if task == 'antimicrobial_activity':
        organism = config.get('organism', None)
        pos_threshold = config['thresholds']['pos_threshold_ugml']
        neg_threshold = config['thresholds']['neg_threshold_ugml']
        gamma_synthetic = config.get('gamma_synthetic', 0.25)
        # Negatives are fixed; not configurable via YAML
        negatives = [
            'negatives_swissprot_with_splits.csv',  # curated negatives (SwissProt)
            'negatives_synth_with_splits.csv'       # synthetic negatives
        ]
        xgboost_params_cfg = config.get('xgboost_params', {})
    elif task == 'haemolysis':
        conc_threshold = config['thresholds']['conc_threshold_um']
        pct_threshold = config['thresholds']['pct_threshold']
        xgboost_params_cfg = config.get('xgboost_params', {})
    elif task == 'cytotoxicity':
        tox_threshold = config['thresholds']['tox_threshold_um']
        xgboost_params_cfg = config.get('xgboost_params', {})
    else:
        raise ValueError(f"Unknown task: {task}")

    # Optuna configuration
    optuna_config = config.get('optuna', {})
    n_trials = optuna_config.get('trials', 50)
    cv_folds = optuna_config.get('cv_folds', 5)
    search_space = optuna_config.get('search_space', {})

    # Training configuration
    training_cfg = config.get('training', {})
    num_boost_round_cfg = int(training_cfg.get('num_boost_round', 2000))
    early_stopping_rounds_cfg = int(training_cfg.get('early_stopping_rounds', 100))

    # Read seed and random_state directly from YAML
    seed_cfg = int(config['seed'])
    random_state_cfg = int(config.get('xgboost_params', {}).get('random_state', seed_cfg))

    # Evaluation configuration (optional cluster-aware CV after training)
    evaluation_cfg = config.get('evaluation', {})
    eval_cv_folds = int(evaluation_cfg.get('cv_folds', 0))

    # Create output directories with correct structure
    # For antimicrobial_activity: outputs/judges/antimicrobial_activity/{generic|escherichia_coli|...}
    # For others: outputs/judges/{cytotoxicity|haemolysis}
    if task == 'antimicrobial_activity':
        if organism:
            # Map organism name to folder name
            organism_folder = organism.replace(' ', '_').lower()
        else:
            organism_folder = 'generic'
        
        checkpoint_path = checkpoint_dir / task / organism_folder
        output_path = output_dir / task / organism_folder
        organism_label = f" ({organism_folder})"
    else:
        # cytotoxicity or haemolysis - no subfolders
        checkpoint_path = checkpoint_dir / task
        output_path = output_dir / task
        organism_label = ""
    
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training XGBoost for {task}{organism_label}")
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
            gamma_synthetic=gamma_synthetic,
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
    dev_sequences = train_sequences + val_sequences
    
    # Sanity check: ensure lengths match for GroupKFold
    assert X_dev.shape[0] == y_dev.shape[0] == clusters_dev.shape[0], \
        f"Length mismatch: X_dev={X_dev.shape[0]}, y_dev={y_dev.shape[0]}, clusters_dev={clusters_dev.shape[0]}"
    
    # For antimicrobial activity, use per-sequence weights for BOTH train and val during tuning
    if task == 'antimicrobial_activity':
        w_map_dev = compute_sample_weights(
            pd.concat([df_train, df_val], ignore_index=True),
            gamma_synthetic=gamma_synthetic,
        )
        # Align to dev feature order
        w_dev = w_map_dev.loc[dev_sequences].to_numpy()
        sample_weight_dev = w_dev
        val_weight_dev = w_dev
    else:
        sample_weight_dev = None
        val_weight_dev = None

    # Enforce fixed binary classification setup with AUC everywhere
    if not isinstance(xgboost_params_cfg, dict):
        xgboost_params_cfg = {}
    # Always use binary logistic objective and AUC eval metric
    xgboost_params_cfg['objective'] = 'binary:logistic'
    xgboost_params_cfg['eval_metric'] = 'auc'
    # Set both keys directly (no remapping logic)
    xgboost_params_cfg['seed'] = seed_cfg
    xgboost_params_cfg['random_state'] = random_state_cfg

    # Run Optuna tuning for XGBoost
    best_params = tune_xgboost(
        X_dev, y_dev, clusters_dev,
        sample_weight_dev=sample_weight_dev,
        val_weight_dev=val_weight_dev,
        n_trials=n_trials,
        cv_folds=cv_folds,
        search_space_config=search_space,
        base_params=xgboost_params_cfg,
        num_boost_round=num_boost_round_cfg,
        early_stopping_rounds=early_stopping_rounds_cfg,
    )
    # Merge YAML-provided xgboost_params as defaults (tuned values override)
    if best_params is None:
        best_params = {}
    best_params = {**xgboost_params_cfg, **best_params}
    
    # Create task-specific XGBoost judge
    if task == 'antimicrobial_activity':
        judge = AntimicrobialActivityJudge(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg is not None else AntimicrobialActivityJudge()
    elif task == 'haemolysis':
        judge = HaemolysisJudge(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg is not None else HaemolysisJudge()
    elif task == 'cytotoxicity':
        judge = CytotoxicityJudge(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg is not None else CytotoxicityJudge()
    else:
        raise ValueError(f"Unknown task: {task}")

    # Train final XGBoost model
    result = judge.train(
        X_train, y_train, X_val, y_val,
        sample_weight=sample_weight,
        params_override=best_params,
        num_boost_round=num_boost_round_cfg,
        early_stopping_rounds=early_stopping_rounds_cfg,
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

    # Save learning curve data (AUC history should always exist now)
    if 'evals_result' in result:
        evals = result['evals_result']
        pd.DataFrame({
            'round': np.arange(len(evals['train']['auc'])),
            'train_auc': evals['train']['auc'],
            'val_auc': evals['val']['auc'],
        }).to_csv(output_path / 'learning_curve.csv', index=False)
        logger.info(f"Learning curve data saved to {output_path / 'learning_curve.csv'}")

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

            # Build task-specific XGBoost judge
            if task == 'antimicrobial_activity':
                judge_cv = AntimicrobialActivityJudge(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg is not None else AntimicrobialActivityJudge()
            elif task == 'haemolysis':
                judge_cv = HaemolysisJudge(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg is not None else HaemolysisJudge()
            else:
                judge_cv = CytotoxicityJudge(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg is not None else CytotoxicityJudge()

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

            aucs.append(float(auc_cv))

        pd.DataFrame({'fold': np.arange(1, len(aucs)+1), 'auc': aucs}).to_csv(fold_dir / 'cv_metrics.csv', index=False)
        logger.info(f"CV metrics saved to {fold_dir / 'cv_metrics.csv'}")

    # Save XGBoost model checkpoint
    model_path = checkpoint_path / 'model.json'
    judge.save(model_path)

    # Save metadata
    metadata = {
        'task': task,
        'model_type': 'xgboost',
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
            'organism': organism if organism else 'all',
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


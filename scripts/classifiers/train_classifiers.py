"""Unified classifier training script for AMP-DFM classifiers."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ampdfm.classifiers import (
    AntimicrobialActivityClassifier,
    CytotoxicityClassifier,
    HaemolysisClassifier,
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
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train AMP-DFM classifier models")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Loaded config: {args.config}")

    task = config['task']
    data_dir = Path(config['data_dir'])
    checkpoint_dir = Path(config['checkpoint_dir'])
    output_dir = Path(config['output_dir'])

    decision_threshold_cfg = config.get('thresholds', {}).get('decision_threshold', None)
    
    if task == 'antimicrobial_activity':
        organism = config.get('organism', None)
        pos_threshold = config['thresholds']['pos_threshold_ugml']
        neg_threshold = config['thresholds']['neg_threshold_ugml']
        gamma_synthetic = config.get('gamma_synthetic', 0.25)
        negatives = [
            'negatives_swissprot_with_splits.csv',
            'negatives_synth_with_splits.csv'
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

    optuna_config = config.get('optuna', {})
    n_trials = optuna_config.get('trials', 50)
    cv_folds = optuna_config.get('cv_folds', 5)
    search_space = optuna_config.get('search_space', {})

    training_cfg = config.get('training', {})
    num_boost_round_cfg = int(training_cfg.get('num_boost_round', 2000))
    early_stopping_rounds_cfg = int(training_cfg.get('early_stopping_rounds', 100))

    seed_cfg = int(config['seed'])
    random_state_cfg = int(config.get('xgboost_params', {}).get('random_state', seed_cfg))

    evaluation_cfg = config.get('evaluation', {})
    eval_cv_folds = int(evaluation_cfg.get('cv_folds', 0))

    if task == 'antimicrobial_activity':
        organism_folder = organism.replace(' ', '_').lower() if organism else 'generic'
        checkpoint_path = checkpoint_dir / task / organism_folder
        output_path = output_dir / task / organism_folder
        organism_label = f" ({organism_folder})"
    else:
        checkpoint_path = checkpoint_dir / task
        output_path = output_dir / task
        organism_label = ""
    
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {task}{organism_label}")
    logger.info(f"Checkpoints: {checkpoint_path}, Outputs: {output_path}")

    if task == 'antimicrobial_activity':
        clustered = data_dir / 'clustered'
        activity_df = pd.read_csv(clustered / 'activities_with_splits.csv', low_memory=False)
        negatives_dfs = []
        for neg_file in negatives:
            df_neg = pd.read_csv(clustered / neg_file)
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
        df_labelled = label_haemolysis_sequences(raw_df, conc_threshold, pct_threshold)
    elif task == 'cytotoxicity':
        raw_df = pd.read_csv(data_dir / 'clustered/cytotoxicity_with_splits.csv')
        df_labelled = label_cytotoxicity_sequences(raw_df, tox_threshold)

    logger.info(f"Labelled: {len(df_labelled)} sequences")
    logger.info(f"Distribution:\n{df_labelled['label'].value_counts()}")

    embeddings, seq_index = load_embeddings(data_dir / 'embeddings')
    df_train, df_val, df_test = create_train_val_test_splits(df_labelled)

    X_train, y_train, train_sequences = prepare_features(df_train, embeddings, seq_index)
    X_val, y_val, val_sequences = prepare_features(df_val, embeddings, seq_index)
    X_test, y_test, test_sequences = prepare_features(df_test, embeddings, seq_index)

    cluster_map_train = df_train.drop_duplicates('sequence').set_index('sequence')['cluster_id']
    clusters_train = cluster_map_train.loc[train_sequences].to_numpy()
    cluster_map_val = df_val.drop_duplicates('sequence').set_index('sequence')['cluster_id']
    clusters_val = cluster_map_val.loc[val_sequences].to_numpy()
    cluster_map_test = df_test.drop_duplicates('sequence').set_index('sequence')['cluster_id']
    clusters_test = cluster_map_test.loc[test_sequences].to_numpy()

    sample_weight = None
    if task == 'antimicrobial_activity':
        weight_map = compute_sample_weights(df_train, gamma_synthetic)
        sample_weight = weight_map.loc[train_sequences].to_numpy()

    X_dev = np.vstack([X_train, X_val])
    y_dev = np.concatenate([y_train, y_val])
    clusters_dev = np.concatenate([clusters_train, clusters_val])
    dev_sequences = train_sequences + val_sequences
    
    assert X_dev.shape[0] == y_dev.shape[0] == clusters_dev.shape[0]

    if not isinstance(xgboost_params_cfg, dict):
        xgboost_params_cfg = {}
    xgboost_params_cfg['objective'] = 'binary:logistic'
    xgboost_params_cfg['eval_metric'] = 'auc'
    xgboost_params_cfg['seed'] = seed_cfg
    xgboost_params_cfg['random_state'] = random_state_cfg

    # Prepare tuning kwargs and enable leakage-free per-fold weighting for antimicrobial activity
    tune_kwargs = dict(
        n_trials=n_trials,
        cv_folds=cv_folds,
        search_space_config=search_space,
        base_params=xgboost_params_cfg,
        num_boost_round=num_boost_round_cfg,
        early_stopping_rounds=early_stopping_rounds_cfg,
    )

    if task == 'antimicrobial_activity':
        dev_df = pd.concat([df_train, df_val], ignore_index=True)
        tune_kwargs.update({
            'dev_df': dev_df,
            'dev_sequences': dev_sequences,
            'gamma_synthetic': gamma_synthetic,
        })

    best_params = tune_xgboost(
        X_dev, y_dev, clusters_dev,
        **tune_kwargs,
    )
    best_params = {**xgboost_params_cfg, **(best_params or {})}
    
    if task == 'antimicrobial_activity':
        classifier = AntimicrobialActivityClassifier(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg else AntimicrobialActivityClassifier()
    elif task == 'haemolysis':
        classifier = HaemolysisClassifier(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg else HaemolysisClassifier()
    else:
        classifier = CytotoxicityClassifier(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg else CytotoxicityClassifier()

    result = classifier.train(
        X_train, y_train, X_val, y_val,
        sample_weight=sample_weight,
        params_override=best_params,
        num_boost_round=num_boost_round_cfg,
        early_stopping_rounds=early_stopping_rounds_cfg,
    )

    train_auc, train_proba, train_pred = classifier.evaluate(X_train, y_train, "Train")
    val_auc, val_proba, val_pred = classifier.evaluate(X_val, y_val, "Val")
    test_auc, test_proba, test_pred = classifier.evaluate(X_test, y_test, "Test")

    test_results = pd.DataFrame({
        'sequence': test_sequences,
        'true_label': y_test,
        'predicted_proba': test_proba,
        'predicted_label': test_pred,
        'cluster_id': clusters_test,
    })
    test_results.to_csv(output_path / 'test_results.csv', index=False)

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, test_proba)
    pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds}).to_csv(
        output_path / 'roc_curve_data.csv', index=False
    )

    if 'evals_result' in result:
        evals = result['evals_result']
        pd.DataFrame({
            'round': np.arange(len(evals['train']['auc'])),
            'train_auc': evals['train']['auc'],
            'val_auc': evals['val']['auc'],
        }).to_csv(output_path / 'learning_curve.csv', index=False)

    if eval_cv_folds and eval_cv_folds > 1:
        logger.info(f"Running {eval_cv_folds}-fold CV")
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=eval_cv_folds)
        aucs: list[float] = []
        fold_dir = output_path / 'cv'
        fold_dir.mkdir(exist_ok=True)

        for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(df_labelled, groups=df_labelled['cluster_id']), start=1
        ):
            tr_rows = df_labelled.iloc[tr_idx]
            va_rows = df_labelled.iloc[va_idx]

            X_tr_cv, y_tr_cv, tr_seqs_cv = prepare_features(tr_rows, embeddings, seq_index)
            X_val_cv, y_val_cv, _ = prepare_features(va_rows, embeddings, seq_index)

            if task == 'antimicrobial_activity':
                classifier_cv = AntimicrobialActivityClassifier(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg else AntimicrobialActivityClassifier()
            elif task == 'haemolysis':
                classifier_cv = HaemolysisClassifier(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg else HaemolysisClassifier()
            else:
                classifier_cv = CytotoxicityClassifier(decision_threshold=decision_threshold_cfg) if decision_threshold_cfg else CytotoxicityClassifier()

            w_tr_cv = None
            if task == 'antimicrobial_activity':
                w_map_cv = compute_sample_weights(tr_rows, gamma_synthetic)
                w_tr_cv = w_map_cv.loc[tr_seqs_cv].to_numpy()

            classifier_cv.train(
                X_tr_cv, y_tr_cv, X_val_cv, y_val_cv,
                sample_weight=w_tr_cv,
                params_override=best_params,
                num_boost_round=num_boost_round_cfg,
                early_stopping_rounds=early_stopping_rounds_cfg,
            )
            auc_cv, _, _ = classifier_cv.evaluate(X_val_cv, y_val_cv, split_name=f"CV{fold}")
            aucs.append(float(auc_cv))

        pd.DataFrame({'fold': np.arange(1, len(aucs)+1), 'auc': aucs}).to_csv(
            fold_dir / 'cv_metrics.csv', index=False
        )

    classifier.save(checkpoint_path / 'model.json')

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
            'organism': organism or 'all',
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
        metadata.update({'tox_threshold_um': tox_threshold})
    
    with open(checkpoint_path / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()


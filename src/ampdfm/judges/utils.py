#!/usr/bin/env python3
"""Shared utilities for judge training.

Common helpers for embedding loading, feature extraction, and data splitting.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_embeddings(base_dir: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, dict[str, int]]:
    """Load ESM-2 mean-pooled embeddings and sequence→row index mapping.

    If base_dir is provided, look under `<base_dir>/esm2/` first, then fallback
    to legacy flat files under `<base_dir>/`. If base_dir is None, use
    project-relative defaults ("embeddings/esm2/" then legacy flat).

    Returns:
        Tuple of (embeddings array, sequence_index dict)
    """
    if base_dir is not None:
        base = Path(base_dir)
        emb_path = base / "esm2/esm2_all.npy"
        index_path = base / "esm2/sequence_index.pkl"
        if not emb_path.exists():
            emb_path = base / "esm2_all.npy"
            index_path = base / "sequence_index.pkl"
    else:
        emb_path = Path("embeddings/esm2/esm2_all.npy")
        index_path = Path("embeddings/esm2/sequence_index.pkl")
        if not emb_path.exists():
            emb_path = Path("embeddings/esm2_all.npy")
            index_path = Path("embeddings/sequence_index.pkl")

    logger.info(f"Loading embeddings from {emb_path}")
    embeddings = np.load(emb_path)

    with open(index_path, "rb") as f:
        sequence_index: dict[str, int] = pickle.load(f)

    logger.info(f"Loaded embeddings: {embeddings.shape[0]} × {embeddings.shape[1]}")
    return embeddings, sequence_index


def prepare_features(
    sequences_df: pd.DataFrame,
    embeddings: np.ndarray,
    sequence_index: dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract embedding features for sequences in a DataFrame.

    Args:
        sequences_df: DataFrame with 'sequence' and 'label' columns
        embeddings: Pre-loaded embedding array
        sequence_index: Mapping from sequence string to embedding row index

    Returns:
        Tuple of (feature_matrix, labels, sequence_list)
    """
    feats, labels, seqs = [], [], []

    for _, row in sequences_df.iterrows():
        seq = row["sequence"]
        if seq not in sequence_index:
            logger.warning(f"Sequence {seq} not found in embedding index")
            continue
        idx = sequence_index[seq]
        feats.append(embeddings[idx])
        labels.append(row["label"])
        seqs.append(seq)

    if not feats:
        logger.error("No sequences with embeddings found – aborting.")
        sys.exit(1)

    X = np.asarray(feats)
    y = np.asarray(labels)
    return X, y, seqs


def create_train_val_test_splits(
    labeled_sequences: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split sequences into train/val/test based on pre-assigned 'split' column.

    Args:
        labeled_sequences: DataFrame with 'split' column ('train', 'val', 'test')

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Using pre-assigned train/val/test splits...")

    train = labeled_sequences[labeled_sequences["split"] == "train"].copy()
    val = labeled_sequences[labeled_sequences["split"] == "val"].copy()
    test = labeled_sequences[labeled_sequences["split"] == "test"].copy()

    logger.info(f"Train: {len(train)} sequences")
    logger.info(f"Val: {len(val)} sequences")
    logger.info(f"Test: {len(test)} sequences")

    # Check label distribution in each split
    for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        if len(split_df) > 0:
            pos_count = sum(split_df["label"] == 1)
            neg_count = sum(split_df["label"] == 0)
            logger.info(
                f"{name} split: {pos_count} positive, {neg_count} negative "
                f"({pos_count/(pos_count+neg_count)*100:.1f}% positive)"
            )
        else:
            logger.warning(f"{name} split is empty!")

    return train, val, test


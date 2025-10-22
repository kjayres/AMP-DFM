#!/usr/bin/env python3
"""Shared utilities for classifier training."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_embeddings(base_dir: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, dict[str, int]]:
    """Load ESM-2 embeddings and sequence index"""
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
    """Extract embedding features for sequences"""
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
        raise ValueError("No sequences with embeddings found")

    return np.asarray(feats), np.asarray(labels), seqs


def create_train_val_test_splits(
    labeled_sequences: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split sequences into train/val/test based on 'split' column"""
    train = labeled_sequences[labeled_sequences["split"] == "train"].copy()
    val = labeled_sequences[labeled_sequences["split"] == "val"].copy()
    test = labeled_sequences[labeled_sequences["split"] == "test"].copy()

    for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        pos = sum(split_df["label"] == 1)
        neg = sum(split_df["label"] == 0)
        logger.info(f"{name}: {len(split_df)} ({pos} pos, {neg} neg)")

    return train, val, test


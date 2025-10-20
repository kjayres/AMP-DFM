#!/usr/bin/env python3
"""Inference helpers for XGBoost classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import xgboost as xgb
import torch
from torch import nn

from ampdfm.utils.esm_embed import get_esm_embeddings
from ampdfm.utils.tokenization import detokenise


class EmbeddedBooster:
    """XGBoost Booster + ESM-2 embedding pipeline for inference."""

    def __init__(self, booster_path: str | Path, device: str | torch.device | None = None):
        self.booster = xgb.Booster()
        self.booster.load_model(str(booster_path))
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    def _features(self, sequences: List[str]) -> xgb.DMatrix:
        embs = get_esm_embeddings(sequences, device=self.device, batch_size=128)
        return xgb.DMatrix(embs)

    def predict_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict from precomputed embeddings."""
        return self.booster.predict(xgb.DMatrix(embeddings))

    def predict(self, sequences: Iterable[str]) -> np.ndarray:
        """Predict from sequences."""
        seqs = list(sequences)
        if len(seqs) == 0:
            return np.zeros((0,), dtype=np.float32)
        return self.booster.predict(self._features(seqs))


class TokenBoosterAdapter:
    """Adapter to score token tensors by detokenising to sequences."""

    def __init__(self, booster_path: str | Path, device: str | torch.device | None = None):
        self._booster = EmbeddedBooster(booster_path, device=device)

    def forward(self, tokens):
        seqs = [detokenise(row) for row in tokens.cpu().tolist()]
        return np.asarray(self._booster.predict(seqs))

    def predict_sequences(self, sequences: Iterable[str]) -> np.ndarray:
        return self._booster.predict(list(sequences))

    def predict_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        return self._booster.predict_from_embeddings(embeddings)


class TorchBoosterAdapter(nn.Module):
    """PyTorch wrapper for token-based inference."""

    def __init__(self, booster_path: str | Path, device: str | torch.device | None = None):
        super().__init__()
        self._adapter = TokenBoosterAdapter(booster_path, device=device)

    def forward(self, tokens: torch.Tensor):  # type: ignore[override]
        proba = self._adapter.forward(tokens)
        return torch.from_numpy(proba).to(tokens.device)

    def predict_sequences(self, sequences: Iterable[str]) -> np.ndarray:
        return self._adapter.predict_sequences(sequences)

    def predict_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        return self._adapter.predict_from_embeddings(embeddings)


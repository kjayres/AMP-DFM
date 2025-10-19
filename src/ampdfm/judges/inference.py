#!/usr/bin/env python3
"""Lightweight inference helpers for XGBoost judges.

Provides an embedded-booster wrapper that computes ESM-2 embeddings for
peptide strings using utils.esm_embed and feeds them into a loaded XGBoost
Booster to obtain probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import xgboost as xgb
import torch
from torch import nn

from ampdfm.utils.esm_embed import get_esm_embeddings
from ampdfm.utils.tokenization import detokenise


class EmbeddedBooster:
    """XGBoost Booster + ESM-2 embedding pipeline for inference from sequences.

    Usage:
        booster = EmbeddedBooster(model_path)
        probs = booster.predict(["GIGKFLKKAKKFGKAFVKILKK"])  # list[str] -> np.ndarray (N,)
    """

    def __init__(self, booster_path: str | Path, device: str | torch.device | None = None):
        self.booster = xgb.Booster()
        self.booster.load_model(str(booster_path))
        # Default to GPU if available, else CPU
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    def _features(self, sequences: List[str]) -> xgb.DMatrix:
        embs = get_esm_embeddings(sequences, device=self.device)
        return xgb.DMatrix(embs)

    def predict_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict from precomputed ESM-2 embeddings.

        Args:
            embeddings: NumPy array of shape (N, D)

        Returns:
            Probability array (N,)
        """
        dmat = xgb.DMatrix(embeddings)
        proba = self.booster.predict(dmat)
        return proba

    def predict(self, sequences: Iterable[str]) -> np.ndarray:
        seqs = list(sequences)
        if len(seqs) == 0:
            return np.zeros((0,), dtype=np.float32)
        dmat = self._features(seqs)
        proba = self.booster.predict(dmat)
        return proba



class TokenBoosterAdapter:
    """Adapter to score token tensors by first detokenising to sequences.

    Exposes a PyTorch-like ``forward(tokens)`` so it can be plugged into
    guidance utilities expecting a callable. Internally wraps ``EmbeddedBooster``.
    """

    def __init__(self, booster_path: str | Path, device: str | torch.device | None = None):
        self._booster = EmbeddedBooster(booster_path, device=device)

    def forward(self, tokens):
        seqs = [detokenise(row) for row in tokens.cpu().tolist()]
        proba = self._booster.predict(seqs)
        return np.asarray(proba)

    def predict_sequences(self, sequences: Iterable[str]) -> np.ndarray:
        """Predict probabilities directly from sequence strings.

        This forwards to the shared EmbeddedBooster instance, avoiding any
        duplicate booster loads elsewhere.
        """
        return self._booster.predict(list(sequences))

    def predict_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict probabilities from precomputed embeddings.

        Returns a NumPy array, matching other inference paths.
        """
        return self._booster.predict_from_embeddings(embeddings)


class TorchBoosterAdapter(nn.Module):
    """PyTorch nn.Module wrapper around TokenBoosterAdapter.

    Accepts token tensors and returns probabilities as a torch.Tensor on the
    same device, enabling seamless use inside guided sampling code that expects
    torch callables.
    """

    def __init__(self, booster_path: str | Path, device: str | torch.device | None = None):
        super().__init__()
        self._adapter = TokenBoosterAdapter(booster_path, device=device)

    def forward(self, tokens: torch.Tensor):  # type: ignore[override]
        proba = self._adapter.forward(tokens)
        return torch.from_numpy(proba).to(tokens.device)

    def predict_sequences(self, sequences: Iterable[str]) -> np.ndarray:
        """Predict probabilities from sequence strings using the same booster.

        Returns a NumPy array, matching the non-torch inference use case.
        """
        return self._adapter.predict_sequences(sequences)

    def predict_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict probabilities from precomputed embeddings using the same booster."""
        return self._adapter.predict_from_embeddings(embeddings)


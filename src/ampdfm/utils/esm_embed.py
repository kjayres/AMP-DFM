"""ampflow.py_scripts.esm_embed
Utility functions to generate fixed-length embeddings for peptide sequences
using Facebook AI's ESM-2 (650 M) model.  We follow the same mean-pooling
strategy used in Pranam Chatterjee's MOG-DFM classifiers so that the
resulting vectors can be consumed interchangeably by the XGBoost judge
models and by guidance functions at sampling time.

Typical usage
-------------
>>> from esm_embed import get_esm_embeddings
>>> seqs = ["MKTIIALSYIFCLVFADYKDDDDK", "GIGKFLKKAKKFGKAFVKILKK" ]
>>> embs = get_esm_embeddings(seqs, batch_size=16, device="cuda")  # (2, 1280)

Embeddings are returned as a NumPy array of shape (N, 1280).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

_logger = logging.getLogger(__name__)

# Simple per-device cache so we load ESM-2 only once per process/device
_TOKENIZER_CACHE: dict[tuple[str, str], AutoTokenizer] = {}
_MODEL_CACHE: dict[tuple[str, str], EsmModel] = {}


def _load_esm(model_name: str = ESM2_MODEL_NAME, device: str | torch.device = "cpu"):
    """Load ESM-2 model + tokenizer once and move model to *device*.

    Uses a lightweight in-process cache keyed by (model_name, str(device)).
    """
    key = (model_name, str(device))
    tok = _TOKENIZER_CACHE.get(key)
    mdl = _MODEL_CACHE.get(key)
    if tok is None or mdl is None:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = EsmModel.from_pretrained(model_name)
        mdl.eval()
        mdl.to(device)
        _TOKENIZER_CACHE[key] = tok
        _MODEL_CACHE[key] = mdl
        _logger.info("Loaded ESM-2 '%s' on device %s", model_name, device)
    return tok, mdl


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over the sequence length dimension, ignoring padding."""
    # last_hidden_state: [B, L, D], attention_mask: [B, L]
    mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)  # [B, D]
    lengths = mask.sum(dim=1)  # [B, 1]
    return summed / lengths.clamp(min=1e-8)


def get_esm_embeddings(
    sequences: List[str],
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """Compute 1 280-d mean-pooled ESM-2 embeddings for *sequences*.

    Parameters
    ----------
    sequences : list[str]
        List of peptide sequences using one-letter amino-acid codes.
    batch_size : int, default 32
        GPU batch size.
    device : str or torch.device, default "cpu"
        Where to run the model ("cuda", "cpu", "cuda:0", etc.).
    dtype : torch.dtype, default torch.float32
        Output precision – set to *bfloat16* or *float16* to save memory.
    """
    tokenizer, model = _load_esm(device=device)

    all_embs: list[torch.Tensor] = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        inputs = tokenizer(batch_seqs, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False)
            pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = pooled.to("cpu", dtype=dtype)
        all_embs.append(pooled)

        if (i // batch_size) % 50 == 0:
            _logger.info("Embedded %d / %d sequences", min(i + batch_size, len(sequences)), len(sequences))

    return torch.cat(all_embs, dim=0).numpy()


# -----------------------------------------------------------------------------
# Convenience CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="Generate ESM-2 embeddings for a FASTA or plain-text file.")
    parser.add_argument("input", type=str, help="Path to a .txt (one seq per line) or .fasta file.")
    parser.add_argument("output", type=str, help="Destination .npy file for embeddings.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    args = parser.parse_args()

    # Read sequences
    in_path = Path(args.input)
    if in_path.suffix.lower() in {".fa", ".fasta"}:
        seqs: list[str] = []
        with open(in_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(">"):
                    seqs.append(line)
    else:
        # Plain text – one sequence per line
        seqs = [line.strip() for line in open(in_path) if line.strip()]

    embs = get_esm_embeddings(seqs, batch_size=args.batch, device=args.device)
    np.save(args.output, embs)

    # Also write an index file so we can map rows back to sequences
    with open(Path(args.output).with_suffix(".txt"), "w") as f:
        f.write("\n".join(seqs))

    print(f"Wrote {embs.shape[0]} embeddings to {args.output}") 
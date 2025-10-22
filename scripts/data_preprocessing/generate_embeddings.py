"""generate_embeddings.py

Collect all unique peptide sequences from the filtered data tables
and cache mean-pooled ESM-2 (650 M) embeddings.
Run via: qsub amp_dfm/scripts/data_preprocessing/generate_embeddings.sh
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch

# Import from amp_dfm package
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "amp_dfm" / "src"))
from ampdfm.utils.esm_embed import get_esm_embeddings

AMP_DFM_ROOT = ROOT / "amp_dfm"
DATA_DIR = AMP_DFM_ROOT / "data" / "filtered"
EMB_DIR = AMP_DFM_ROOT / "data" / "embeddings"
EMB_DIR.mkdir(exist_ok=True)

ESM2_DIR = EMB_DIR / "esm2"
ESM2_DIR.mkdir(exist_ok=True)
CSV_FILES = [
    DATA_DIR / "activities_final_long.csv",
    DATA_DIR / "haemolysis_final_long.csv",
    DATA_DIR / "cytotoxicity_final_long.csv",
    DATA_DIR / "negatives_swissprot_long.csv",
    DATA_DIR / "negatives_general_peptides_long.csv",
    DATA_DIR / "negatives_uniprot_long.csv",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("generate_embeddings")


def load_unique_sequences(files: List[Path]) -> List[str]:
    """Read sequence column from each CSV and return a deduplicated list"""
    seq_set: set[str] = set()

    for csv in files:
        logger.info("Loading sequences from %s", csv.name)
        df = pd.read_csv(csv, usecols=["sequence"])
        seq_set.update(df["sequence"].dropna().astype(str).tolist())
        logger.info("  cumulative unique seqs: %d", len(seq_set))

    return sorted(seq_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()

    seq_txt = EMB_DIR / "seqs.txt"
    emb_npy = ESM2_DIR / "esm2_all.npy"
    index_pkl = ESM2_DIR / "sequence_index.pkl"

    if emb_npy.exists() and not args.overwrite:
        logger.info("%s already exists – nothing to do. Use --overwrite to regenerate.", emb_npy)
        return

    sequences = load_unique_sequences(CSV_FILES)
    seq_txt.write_text("\n".join(sequences))
    logger.info("Wrote %s", seq_txt.relative_to(ROOT))

    logger.info("Embedding %d sequences on device %s", len(sequences), args.device)
    embs = get_esm_embeddings(sequences, batch_size=args.batch, device=args.device, dtype=torch.float32)
    np.save(emb_npy, embs)
    logger.info("Saved embeddings to %s (shape %s)", emb_npy.relative_to(ROOT), embs.shape)

    seq_index = {seq: i for i, seq in enumerate(sequences)}
    with open(index_pkl, "wb") as f:
        pickle.dump(seq_index, f)
    logger.info("Saved sequence index to %s", index_pkl.relative_to(ROOT))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""score_samples.py

Score a FASTA file of PepDFM sequences with the three XGBoost judges
(potency, haemolysis, cytotoxicity) and write the CSV that
`pepdfm_mog.py` would normally produce.

Usage example (inside `mog-dfm` environment):

    python score_samples.py \
        --fasta ampflow/results/mog/mog_samples.fa \
        --out   ampflow/results/mog/mog_samples_scores.csv \
        --device cuda:0
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root on PYTHONPATH
# ---------------------------------------------------------------------------
import sys
from pathlib import Path as _P
PROJECT_ROOT = _P(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Re-use judges and token mapping from pepdfm_mog
from ampflow.ampdfm_scripts.pepdfm_mog import (
    PotencyJudge, HemolysisJudge, CytotoxicityJudge, _IDX_TO_AA
)

AA_TO_IDX = {v: k for k, v in _IDX_TO_AA.items()}


def encode(seq: str) -> List[int]:
    """Encode AA string → PepDFM token list incl. <cls>/<eos>."""
    toks = [0]  # <cls>
    toks += [AA_TO_IDX.get(res.upper(), 3) for res in seq]
    toks.append(2)  # <eos>
    return toks


def load_fasta(path: Path) -> List[str]:
    seqs = []
    with path.open() as fh:
        for line in fh:
            if not line.startswith(">"):
                seqs.append(line.strip())
    return seqs


def main():
    p = argparse.ArgumentParser(description="Score PepDFM FASTA samples with potency/hemolysis/cytotox judges")
    p.add_argument("--fasta", required=True, help="Path to FASTA file of sequences")
    p.add_argument("--out", required=True, help="CSV output path (same layout as pepdfm_mog)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    fasta_path = Path(args.fasta)
    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seqs = load_fasta(fasta_path)
    print(f"Loaded {len(seqs):,} sequences from {fasta_path}")

    device = args.device
    judges = [PotencyJudge(device), HemolysisJudge(device), CytotoxicityJudge(device)]

    batch = 256
    results = []
    for i in tqdm(range(0, len(seqs), batch), desc="Scoring"):
        chunk_seqs = seqs[i:i + batch]
        tok_batch  = [encode(s) for s in chunk_seqs]
        max_len    = max(len(t) for t in tok_batch)
        padded     = [t + [1] * (max_len - len(t)) for t in tok_batch]  # <pad>=1
        x          = torch.tensor(padded, device=device)

        p_score = judges[0](x).cpu().tolist()
        h_score = judges[1](x).cpu().tolist()
        c_score = judges[2](x).cpu().tolist()

        for s, p, h, c in zip(chunk_seqs, p_score, h_score, c_score):
            results.append((s, p, h, c))

    with out_path.open("w", newline="") as csvf:
        w = csv.writer(csvf)
        w.writerow(["sample_id", "sequence", "potency", "hemolysis", "cytotox"])
        for idx, (seq, p, h, c) in enumerate(results, start=1):
            w.writerow([idx, seq, p, h, c])

    print("Written", out_path)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

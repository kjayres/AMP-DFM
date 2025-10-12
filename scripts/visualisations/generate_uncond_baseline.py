#!/usr/bin/env python3
"""generate_uncond_baseline.py

Sample an unconditional PepDFM model, score the sequences with the three
XGBoost judges (potency, haemolysis, cytotoxicity), and persist both the
FASTA and the CSV.  Intended to be run *once* (or whenever you wish to
refresh the baseline) so that downstream notebooks / visualisation
scripts can load the pre-computed baseline instantly.

Example (inside mog-dfm conda env, GPU available)::

    python generate_uncond_baseline.py \
        --ckpt ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt \
        --out_dir ampflow/results/mog/baseline \
        --n 2500 \
        --device cuda:0 \
        --seed 42

Outputs::
    <out_dir>/baseline_samples.fa
    <out_dir>/baseline_scores.csv
"""
from __future__ import annotations

import argparse, math, random
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on PYTHONPATH (handles invocation from sub-dirs)
# ---------------------------------------------------------------------------
import sys as _sys
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

# PepDFM helpers -------------------------------------------------------------
from models.peptide_classifiers import load_solver
from flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver
from ampflow.ampdfm_scripts.pepdfm_mog import (
    PotencyJudge,
    HemolysisJudge,
    CytotoxicityJudge,
    _IDX_TO_AA,
    detokenise,
)

AA_TO_IDX = {v: k for k, v in _IDX_TO_AA.items()}

# ---------------------------------------------------------------------------
# Sampling helper (copied from analysis_guidance.py with minimal tweaks)
# ---------------------------------------------------------------------------

def unconditional_sample(ckpt: Path, n: int, batches: int, length: int, device: str) -> List[str]:
    """Sample *n* unconditional sequences of fixed *length* from checkpoint."""
    vocab_size = 24
    step = 1.0 / 100  # 100 Euler steps – matches guided runs
    solver: MixtureDiscreteEulerSolver = load_solver(str(ckpt), vocab_size, device)

    per = math.ceil(n / batches)
    seqs: List[str] = []
    for _ in tqdm(range(batches), desc="Sampling"):
        core = torch.randint(4, vocab_size, (per, length), device=device)
        x0 = torch.cat([
            torch.zeros((per, 1), dtype=torch.long, device=device),
            core,
            torch.full((per, 1), 2, dtype=torch.long, device=device),
        ], 1)
        x_fin = solver.sample(x0, step_size=step)
        seqs.extend(detokenise(row) for row in x_fin.cpu().tolist())
    return seqs[:n]

# ---------------------------------------------------------------------------
# Scoring helper (lightweight wrapper around the three judges)
# ---------------------------------------------------------------------------

def encode(seq: str) -> List[int]:
    toks = [0]
    toks += [AA_TO_IDX.get(res.upper(), 3) for res in seq]
    toks.append(2)
    return toks


def score_sequences(seqs: List[str], device: str):
    judges = [PotencyJudge(device), HemolysisJudge(device), CytotoxicityJudge(device)]
    chunk = 256
    pot, hml, cyt = [], [], []
    for i in tqdm(range(0, len(seqs), chunk), desc="Scoring"):
        batch_seqs = seqs[i:i + chunk]
        tok_batch = [encode(s) for s in batch_seqs]
        max_len = max(len(t) for t in tok_batch)
        padded = [t + [1] * (max_len - len(t)) for t in tok_batch]
        x = torch.tensor(padded, device=device)
        pot.extend(judges[0](x).cpu().tolist())
        hml.extend(judges[1](x).cpu().tolist())
        cyt.extend(judges[2](x).cpu().tolist())
    return pot, hml, cyt

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Generate unconditional baseline FASTA + scores")
    p.add_argument("--ckpt", required=True, help="PepDFM unconditional checkpoint")
    p.add_argument("--out_dir", default="ampflow/results/mog/baseline", help="Output directory for FASTA + CSV")
    p.add_argument("--n", type=int, default=2500, help="Total number of sequences to sample (default: 2500)")
    p.add_argument("--length", type=int, help="Fixed peptide core length (overrides --len_min/max)")
    p.add_argument("--len_min", type=int, default=10, help="Minimum core length when sampling variable lengths")
    p.add_argument("--len_max", type=int, default=40, help="Maximum core length when sampling variable lengths")
    p.add_argument("--device", default="cuda:0", help="Device for sampling & scoring (cuda:0 / cpu)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = p.parse_args()

    # ------------------------------------------------------------------
    # Reproducibility – set global RNG seed
    # ------------------------------------------------------------------
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.length is not None and (args.length < 1):
        raise ValueError("--length must be positive")
    if args.len_min > args.len_max:
        raise ValueError("--len_min must be ≤ --len_max")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = out_dir / "baseline_samples.fa"
    csv_path   = out_dir / "baseline_scores.csv"

    # ------------------------------------------------------------------
    # Sampling (variable length support) ---------------------------------
    # ------------------------------------------------------------------
    if fasta_path.exists():
        print(f"Using existing FASTA {fasta_path}")
        seqs = [line.strip() for line in fasta_path.open() if not line.startswith(">")]
    else:
        print("Sampling unconditional baseline …")

        # Initialise solver once
        vocab_size = 24
        step = 1.0 / 100
        solver: MixtureDiscreteEulerSolver = load_solver(str(args.ckpt), vocab_size, args.device)

        batches = 10
        per_batch = math.ceil(args.n / batches)
        seqs: List[str] = []
        for _ in tqdm(range(batches), desc="Sampling"):
            if args.length is not None:
                L = args.length
            else:
                L = random.randint(args.len_min, args.len_max)
            core = torch.randint(4, vocab_size, (per_batch, L), device=args.device)
            x0 = torch.cat([
                torch.zeros((per_batch, 1), dtype=torch.long, device=args.device),
                core,
                torch.full((per_batch, 1), 2, dtype=torch.long, device=args.device),
            ], 1)
            x_fin = solver.sample(x0, step_size=step)
            seqs.extend(detokenise(row) for row in x_fin.cpu().tolist())
        seqs = seqs[: args.n]

        with fasta_path.open("w") as fh:
            for i, s in enumerate(seqs, 1):
                fh.write(f">uncond_{i}\n{s}\n")
        print(f"Written {fasta_path}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    if csv_path.exists():
        print(f"Scores already exist at {csv_path} – nothing to do.")
    else:
        pot, hml, cyt = score_sequences(seqs, args.device)
        import csv as _csv
        with csv_path.open("w", newline="") as csvf:
            w = _csv.writer(csvf)
            w.writerow(["sample_id", "sequence", "potency", "hemolysis", "cytotox"])
            for idx, (seq, p, h, c) in enumerate(zip(seqs, pot, hml, cyt), start=1):
                w.writerow([idx, seq, p, h, c])
        print(f"Written {csv_path}")

    print("Baseline generation complete →", out_dir)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

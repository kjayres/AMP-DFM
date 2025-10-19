#!/usr/bin/env python3
"""ampdfm_mog.py

Multi-objective ampdfm sampling guided by three XGBoost judges
(potency, haemolysis, cytotoxicity).

Usage (defaults shown):
    $ python ampdfm_mog.py \
        --ckpt ampflow/ampdfm_ckpt/ampdfm_unconditional_epoch200.ckpt \
        --out  ampflow/results/mog/mog_samples.fa \
        --n_samples 2000 --n_batches 10 \
        --importance 1,1,1                  # potency,hml,cyto
        # additional guidance flags (lambda_, beta, etc.) are forwarded

The script writes 2 files:
    <out>.fa            – FASTA peptides
    <out>_scores.csv    – per-peptide judge scores
"""
# fmt: off
import argparse
import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import random
import torch
from tqdm import tqdm

# fmt: on

# ---------------------------------------------------------------------------
# Project-internal imports (refactored to ampdfm package) --------------------
# ---------------------------------------------------------------------------
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ampdfm.utils.parsing import parse_guidance_args
from ampdfm.dfm.models.model_utils import load_solver
from ampdfm.dfm.flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver
from ampdfm.utils.tokenization import detokenise, CLS_IDX, EOS_IDX, AA_START_IDX, AA_END_IDX
from ampdfm.judges.inference import TokenBoosterAdapter, TorchBoosterAdapter, EmbeddedBooster
from ampdfm.utils.esm_embed import get_esm_embeddings


def _resolve_judge_paths(task: str, variant: Optional[str] = None) -> tuple[Path, Path]:
    """Find judge model.json and metadata.pkl under outputs/judges/… layout.

    - antimicrobial_activity: outputs/judges/antimicrobial_activity/<variant>/
    - haemolysis, cytotoxicity: outputs/judges/<task>/
    """
    project_root = Path(__file__).resolve().parents[2]
    base = project_root / "outputs" / "judges"
    if task == "antimicrobial_activity":
        org_folder = (variant or "generic").replace(" ", "_").lower()
        base = base / task / org_folder
    else:
        base = base / task
    model_path = base / "model.json"
    meta_path = base / "metadata.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Judge model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Judge metadata not found: {meta_path}")
    return model_path, meta_path

# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ampdfm multi-objective generator (potency, haemolysis, cytotox)")
    parser.add_argument("--config", help="Optional YAML config for MOG sampling")
    parser.add_argument("--ckpt", help="Path to ampdfm CNN checkpoint")
    parser.add_argument("--out", help="Optional explicit FASTA output path")
    parser.add_argument("--out_dir", help="Optional output directory (used if --out not given)")
    parser.add_argument("--run_tag", help="Sub-directory under out_dir (ignored if --out is given). Defaults to potency_variant.")
    parser.add_argument("--potency_variant", help="Which potency judge to use: generic|ecoli|saureus|paeruginosa")
    parser.add_argument("--n_samples", type=int, help="Total peptides to sample")
    parser.add_argument("--n_batches", type=int, help="Number of sampler batches")
    parser.add_argument("--seq_length", type=int, help="Fixed core length; overrides range flags")
    parser.add_argument("--len_min", type=int, help="Minimum AA length if --length not provided")
    parser.add_argument("--len_max", type=int, help="Maximum AA length if --length not provided")
    parser.add_argument("--device")
    parser.add_argument("--importance", help="Comma-sep importance weights potency,haemolysis,cytotox")

    # Let parse_guidance_args consume its own flags (lambda_, beta, Phi …)
    args, unknown = parser.parse_known_args()
    g_args = parse_guidance_args(unknown)  # Namespace with guidance params

    # ---------------- YAML config (optional) ----------------
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    def _cfg(key, default=None):
        return cfg.get(key, default)

    # Merge precedence: CLI > YAML > defaults
    ckpt = args.ckpt or _cfg("ckpt")
    if not ckpt:
        raise ValueError("--ckpt or config['ckpt'] must be provided")
    out_cli = args.out
    out_dir = args.out_dir or _cfg("out_dir")
    run_tag = args.run_tag or _cfg("run_tag", None)
    potency_variant = (args.potency_variant or _cfg("potency_variant", "generic")).lower()
    n_samples = int(args.n_samples or _cfg("n_samples", 2500))
    n_batches = int(args.n_batches or _cfg("n_batches", 10))
    length = args.seq_length or _cfg("seq_length", None)
    len_min = int(args.len_min or _cfg("len_min", 10))
    len_max = int(args.len_max or _cfg("len_max", 30))
    device = args.device or _cfg("device", ("cuda:0" if torch.cuda.is_available() else "cpu"))
    importance = (args.importance or _cfg("importance", "1,1,1"))

    # ---------------- Reproducibility ----------------
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional guidance hyperparameters from YAML (e.g., homopolymer penalty)
    homo_gamma = _cfg("homopolymer_gamma", None)

    # ------------------------------------------------------------------
    # device already resolved above
    vocab_size = 24
    step_size = 1.0 / g_args.T

    # Importance weights ------------------------------------------------
    importance = [float(x) for x in importance.split(",")]
    if len(importance) != 3:
        raise ValueError("--importance must have exactly 3 comma-sep values (potency,haemolysis,cytotoxicity)")

    # Judges ------------------------------------------------------------
    variant = potency_variant
    pot_model, _pot_meta = _resolve_judge_paths("antimicrobial_activity", variant)
    hml_model, _hml_meta = _resolve_judge_paths("haemolysis")
    cyt_model, _cyt_meta = _resolve_judge_paths("cytotoxicity")

    # Torch adapters for guided sampling (operate on token tensors)
    # Also reuse these adapters for sequence-level scoring to avoid double-loading boosters
    potency_j = TorchBoosterAdapter(pot_model, device=device)
    haemolysis_j = TorchBoosterAdapter(hml_model, device=device)
    cytotoxicity_j = TorchBoosterAdapter(cyt_model, device=device)
    score_models = [potency_j, haemolysis_j, cytotoxicity_j]

    # ampdfm solver -----------------------------------------------------
    solver: MixtureDiscreteEulerSolver = load_solver(ckpt, vocab_size, device)

    # If homopolymer_gamma provided in YAML, inject into guidance args
    if homo_gamma is not None:
        try:
            val = float(homo_gamma)
        except Exception:
            val = None
        if val is not None:
            setattr(g_args, 'homopolymer_gamma', val)

    # Configure guided transition scoring for AMP-DFM tokenisation
    setattr(g_args, 'pos_low', 1)        # allow edits starting after <cls>=0
    setattr(g_args, 'aa_start_idx', AA_START_IDX)   # amino-acids occupy AA_START_IDX..AA_END_IDX

    n_base = n_samples // n_batches
    remainder = n_samples % n_batches
    batch_sizes = [n_base + (1 if i < remainder else 0) for i in range(n_batches)]
    results = []  # (seq, scores)

    # Reuse the same Torch adapters for sequence-level prediction to avoid duplicate booster loads

    for b, n_this in enumerate(batch_sizes):
        if n_this <= 0:
            continue
        # ---------- init tokens ----------------
        if length:
            L = length
        else:
            L = random.randint(len_min, len_max)
        core = torch.randint(low=AA_START_IDX, high=vocab_size, size=(n_this, L), device=device)
        x_init = torch.cat([
            torch.full((n_this,1), CLS_IDX, dtype=torch.long, device=device),  # <cls>
            core,
            torch.full((n_this,1), EOS_IDX, dtype=torch.long, device=device)  # <eos>
        ], dim=1)

        x_final = solver.multi_guidance_sample(
            args=g_args,
            x_init=x_init,
            step_size=step_size,
            time_grid=torch.tensor([0.0, 1.0-1e-3]),
            verbose=True,
            score_models=score_models,
            importance=importance,
        )

        # ---------- decode + score (batch) --------------
        # Decode all sequences in this batch
        batch_seqs = [detokenise(row) for row in x_final.cpu().tolist()]
        
        # Compute ESM embeddings once per batch on the chosen device, reuse across judges
        batch_embs = get_esm_embeddings(batch_seqs, device=device)
        p_scores = potency_j.predict_from_embeddings(batch_embs)
        h_scores = haemolysis_j.predict_from_embeddings(batch_embs)
        c_scores = cytotoxicity_j.predict_from_embeddings(batch_embs)
        
        # Collect results
        for seq, p, h, c in zip(batch_seqs, p_scores, h_scores, c_scores):
            results.append((seq, float(p), float(h), float(c)))

    # ------------------------------------------------------------------
    if out_cli:
        out_path = Path(out_cli)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Determine output directory
        if out_dir:
            out_dir = Path(out_dir)
        else:
            # fallback to project-root relative path
            PROJECT_ROOT = Path(__file__).resolve().parents[2]
            out_dir = PROJECT_ROOT / "ampflow" / "results" / "mog" / (run_tag if run_tag else variant)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "mog_samples.fa"

    # FASTA
    with open(out_path, "w") as fh:
        for i, (seq, *_scores) in enumerate(results, start=1):
            fh.write(f">sample_{i}\n{seq}\n")

    # CSV with scores
    csv_path = out_path.with_name(out_path.stem + "_scores.csv")
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["sample_id", "sequence", "potency", "haemolysis", "cytotoxicity"])
        for i, (seq, p, h, c) in enumerate(results, start=1):
            writer.writerow([i, seq, p, h, c])

    print("Written", out_path, "and", csv_path)


if __name__ == "__main__":
    main()
# --------------------------------------------------------------------------- 
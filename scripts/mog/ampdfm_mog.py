#!/usr/bin/env python3
"""Multi-objective ampdfm sampling guided by XGBoost classifiers"""
# fmt: off
import argparse
import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import random
import torch
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ampdfm.utils.parsing import parse_guidance_args
from ampdfm.dfm.models.model_utils import load_solver
from ampdfm.dfm.flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver
from ampdfm.utils.tokenization import detokenise, CLS_IDX, EOS_IDX, AA_START_IDX, AA_END_IDX
from ampdfm.classifiers.inference import TokenBoosterAdapter, TorchBoosterAdapter, EmbeddedBooster
from ampdfm.utils.esm_embed import get_esm_embeddings


def _resolve_classifier_paths(task: str, variant: Optional[str] = None) -> tuple[Path, Path]:
    """Locate classifier model.json and metadata.pkl"""
    project_root = Path(__file__).resolve().parents[2]

    candidate_bases = [
        project_root / "outputs" / "classifiers",
        project_root / "checkpoints" / "classifiers",
    ]

    def _normalise_variant(name: Optional[str]) -> List[str]:
        if not name:
            return ["generic"]
        norm = name.replace(" ", "_").lower()
        variants = [norm]
        if norm in {"ecoli", "e_coli"}:
            variants.append("escherichia_coli")
        elif norm == "paeruginosa":
            variants.append("pseudomonas_aeruginosa")
        elif norm == "saureus":
            variants.append("staphylococcus_aureus")
        return variants

    tried_paths = []

    if task == "antimicrobial_activity":
        for base in candidate_bases:
            for vn in _normalise_variant(variant):
                root = base / task / vn
                model_path = root / "model.json"
                meta_path = root / "metadata.pkl"
                tried_paths.append(root)
                if model_path.exists() and meta_path.exists():
                    return model_path, meta_path
    else:
        for base in candidate_bases:
            root = base / task
            model_path = root / "model.json"
            meta_path = root / "metadata.pkl"
            tried_paths.append(root)
            if model_path.exists() and meta_path.exists():
                return model_path, meta_path

    tried_str = ", ".join(str(p) for p in tried_paths)
    raise FileNotFoundError(
        f"Classifier files not found for task '{task}'. Tried: {tried_str}"
    )


def main():
    parser = argparse.ArgumentParser(description="ampdfm multi-objective generator (antimicrobial activity, haemolysis, cytotox)")
    parser.add_argument("--config", required=True, help="YAML config for MOG sampling")
    args, unknown = parser.parse_known_args()
    g_args = parse_guidance_args(unknown)

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt = cfg["ckpt"]
    out_dir = cfg["out_dir"]
    run_name = cfg["run_name"]
    amp_variant = cfg["amp_variant"]
    n_samples = cfg.get("n_samples", 2500)
    n_batches = cfg.get("n_batches", 10)
    seq_length = cfg.get("seq_length")
    len_min = cfg.get("len_min", 10)
    len_max = cfg.get("len_max", 30)
    device = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    importance = cfg.get("importance", [1, 1, 1])

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    homo_gamma = cfg.get("homopolymer_gamma")

    vocab_size = 24
    step_size = 1.0 / g_args.T

    importance = [float(x) for x in importance]
    if len(importance) != 3:
        raise ValueError("importance must have exactly 3 values (antimicrobial activity, haemolysis, cytotoxicity)")

    variant = amp_variant
    amp_model, _amp_meta = _resolve_classifier_paths("antimicrobial_activity", variant)
    hml_model, _hml_meta = _resolve_classifier_paths("haemolysis")
    cyt_model, _cyt_meta = _resolve_classifier_paths("cytotoxicity")

    antimicrobial_activity_j = TorchBoosterAdapter(amp_model, device=device)
    haemolysis_j = TorchBoosterAdapter(hml_model, device=device)
    cytotoxicity_j = TorchBoosterAdapter(cyt_model, device=device)
    score_models = [antimicrobial_activity_j, haemolysis_j, cytotoxicity_j]

    solver: MixtureDiscreteEulerSolver = load_solver(ckpt, vocab_size, device)

    if homo_gamma is not None:
        try:
            val = float(homo_gamma)
        except Exception:
            val = None
        if val is not None:
            setattr(g_args, 'homopolymer_gamma', val)

    setattr(g_args, 'pos_low', 1)
    setattr(g_args, 'aa_start_idx', AA_START_IDX)
    setattr(g_args, 'aa_end_idx', AA_END_IDX)

    n_base = n_samples // n_batches
    remainder = n_samples % n_batches
    batch_sizes = [n_base + (1 if i < remainder else 0) for i in range(n_batches)]
    results = []

    for b, n_this in enumerate(batch_sizes):
        if n_this <= 0:
            continue
        if seq_length:
            L = seq_length
        else:
            L = random.randint(len_min, len_max)
        core = torch.randint(low=AA_START_IDX, high=vocab_size, size=(n_this, L), device=device)
        x_init = torch.cat([
            torch.full((n_this,1), CLS_IDX, dtype=torch.long, device=device),
            core,
            torch.full((n_this,1), EOS_IDX, dtype=torch.long, device=device)
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

        batch_seqs = [detokenise(row) for row in x_final.cpu().tolist()]
        batch_embs = get_esm_embeddings(batch_seqs, device=device, batch_size=128)
        a_scores = antimicrobial_activity_j.predict_from_embeddings(batch_embs)
        h_scores = haemolysis_j.predict_from_embeddings(batch_embs)
        c_scores = cytotoxicity_j.predict_from_embeddings(batch_embs)
        
        # Collect results
        for seq, a, h, c in zip(batch_seqs, a_scores, h_scores, c_scores):
            results.append((seq, float(a), float(h), float(c)))

    parent_dir = Path(out_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = parent_dir / f"{run_name}.fa"
    scores_path = parent_dir / f"{run_name}_scores.csv"

    with open(fasta_path, "w") as fh:
        for i, (seq, *_scores) in enumerate(results, start=1):
            fh.write(f">sample_{i}\n{seq}\n")

    with open(scores_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["sample_id", "sequence", "antimicrobial_activity", "non_haemolysis", "non_cytotoxicity"])
        for i, (seq, p, h, c) in enumerate(results, start=1):
            writer.writerow([i, seq, p, h, c])

    print(f"Written {len(results)} sequences to {fasta_path} and {scores_path}")


if __name__ == "__main__":
    main()
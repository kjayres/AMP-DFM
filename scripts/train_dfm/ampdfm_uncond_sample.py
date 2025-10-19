#!/usr/bin/env python3
"""ampdfm_uncond_sample.py

Sample peptides from a trained unconditional AMP-DFM model and score with judges.

Usage:
    python ampdfm_uncond_sample.py --config configs/flow_matching/ampdfm_uncond_sample.yaml
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
import argparse
import yaml

import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ampdfm.dfm.models.model_utils import load_solver
from ampdfm.utils.tokenization import detokenise, CLS_IDX, EOS_IDX, AA_START_IDX
from ampdfm.judges.inference import TorchBoosterAdapter
from ampdfm.utils.esm_embed import get_esm_embeddings

parser = argparse.ArgumentParser(description="Sample from unconditional AMP-DFM")
parser.add_argument("--config", required=True, help="Path to YAML config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

device = cfg["device"]
ckpt_path = cfg["ckpt"]
fasta_path = cfg["output"]["fasta_path"]
scores_path = cfg["output"]["scores_path"]
n_samples = cfg["sampling"]["n_samples"]
n_steps = cfg["sampling"]["n_steps"]
len_min = cfg["sampling"]["len_min"]
len_max = cfg["sampling"]["len_max"]
amp_variant = cfg["amp_variant"]

seed = cfg["sampling"]["seed"]
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

vocab_size = 24
epsilon = 1e-3

Path(fasta_path).parent.mkdir(parents=True, exist_ok=True)
Path(scores_path).parent.mkdir(parents=True, exist_ok=True)

solver = load_solver(ckpt_path, vocab_size, device)

def _resolve_judge_paths(task: str, variant: str = None):
    base = PROJECT_ROOT / "checkpoints" / "judges"
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

amp_model, _ = _resolve_judge_paths("antimicrobial_activity", amp_variant)
hml_model, _ = _resolve_judge_paths("haemolysis")
cyt_model, _ = _resolve_judge_paths("cytotoxicity")

antimicrobial_activity_j = TorchBoosterAdapter(amp_model, device=device)
haemolysis_j = TorchBoosterAdapter(hml_model, device=device)
cytotoxicity_j = TorchBoosterAdapter(cyt_model, device=device)

print(f"Sampling {n_samples} peptides from {ckpt_path}")
print(f"Length range: {len_min}-{len_max}, Steps: {n_steps}, Device: {device}")

results = []
batch_size = 100
remaining = n_samples

while remaining > 0:
    n_this = min(batch_size, remaining)
    L = random.randint(len_min, len_max)
    
    core = torch.randint(low=AA_START_IDX, high=vocab_size, size=(n_this, L), device=device)
    x_init = torch.cat([
        torch.full((n_this, 1), CLS_IDX, dtype=torch.long, device=device),
        core,
        torch.full((n_this, 1), EOS_IDX, dtype=torch.long, device=device)
    ], dim=1)
    
    step_size = (1.0 - epsilon) / n_steps
    x_final = solver.sample(
        x_init=x_init,
        step_size=step_size,
        time_grid=torch.tensor([0.0, 1.0 - epsilon], device=device)
    )
    
    batch_seqs = [detokenise(row) for row in x_final.cpu().tolist()]
    
    batch_embs = get_esm_embeddings(batch_seqs, device=device)
    a_scores = antimicrobial_activity_j.predict_from_embeddings(batch_embs)
    h_scores = haemolysis_j.predict_from_embeddings(batch_embs)
    c_scores = cytotoxicity_j.predict_from_embeddings(batch_embs)
    
    for seq, a, h, c in zip(batch_seqs, a_scores, h_scores, c_scores):
        results.append((seq, float(a), float(h), float(c)))
    
    remaining -= n_this
    print(f"  Generated {len(results)}/{n_samples}")

with open(fasta_path, "w") as fh:
    for i, (seq, *_) in enumerate(results, start=1):
        fh.write(f">sample_{i}\n{seq}\n")

with open(scores_path, "w", newline="") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["sample_id", "sequence", "antimicrobial_activity", "non_haemolysis", "non_cytotoxicity"])
    for i, (seq, p, h, c) in enumerate(results, start=1):
        writer.writerow([i, seq, p, h, c])

print(f"Written {len(results)} sequences to {fasta_path} and {scores_path}")


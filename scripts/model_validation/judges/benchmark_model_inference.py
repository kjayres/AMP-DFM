#!/usr/bin/env python3
"""benchmark_model_inference.py

Measure per-peptide inference time for all trained potency / haemolysis /
cytotoxicity classifiers (XGBoost, Random-Forest, Logistic Regression), the
three species-specific potency judges, and the descriptor-augmented potency
judge.

Results are written to:
    ampflow/models/model_timings/inference_timing.csv
The script never modifies existing model artefacts or embeddings.
"""
from __future__ import annotations

import csv
import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ---------------------------------------------------------------------------
# Helper: load ESM-2 embeddings + test sequences (potency task)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # mog_dfm/ampflow
DATA_DIR = PROJECT_ROOT / "data"
EMB_DIR_LEGACY = PROJECT_ROOT / "embeddings"
EMB_DIR_NESTED = EMB_DIR_LEGACY / "esm2"

# Embedding paths (new nested structure preferred, fallback to flat)
_emb_path = EMB_DIR_NESTED / "esm2_all.npy"
_idx_path = EMB_DIR_NESTED / "sequence_index.pkl"
if not _emb_path.exists():
    _emb_path = EMB_DIR_LEGACY / "esm2_all.npy"
    _idx_path = EMB_DIR_LEGACY / "sequence_index.pkl"

print(f"Loading embeddings from {_emb_path}", file=sys.stderr)
embeddings: np.ndarray = np.load(_emb_path)
with open(_idx_path, "rb") as fp:
    seq_index: dict[str, int] = pickle.load(fp)
print(f"Embeddings loaded: {embeddings.shape}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Obtain a representative test set (potency task, cluster-aware split)
# ---------------------------------------------------------------------------

act_path = DATA_DIR / "activities_with_splits.csv"
allowed_groups = {"MIC", "MIC50", "MIC90", "MIC,IC50,LC50,LD50"}

act_df = pd.read_csv(act_path)
act_df = act_df[act_df["measure_group"].isin(allowed_groups)].copy()

test_df = act_df[act_df["split"] == "test"].copy()
seqs_test = test_df["sequence"].unique().tolist()
print(f"Test sequences (potency task): {len(seqs_test)}", file=sys.stderr)

# Filter to sequences that have an embedding entry
seqs_emb: List[str] = [s for s in seqs_test if s in seq_index]
missing = len(seqs_test) - len(seqs_emb)
if missing:
    print(f"Warning: {missing} sequences had no embedding – excluded", file=sys.stderr)
seqs_test = seqs_emb

X_emb = np.vstack([embeddings[seq_index[s]] for s in seqs_test])  # (N, 1280)
print(f"Feature matrix shape (embeddings only): {X_emb.shape}", file=sys.stderr)

# Descriptor helper ----------------------------------------------------------

def _compute_descriptors(sequences: List[str]) -> np.ndarray:
    """Return N×4 array – net_charge, hydrophobicity (gravy), length, pI"""
    feats: list[list[float]] = []
    for s in sequences:
        pa = ProteinAnalysis(s if s else "A")
        net_c = pa.charge_at_pH(7.0)  # includes sign
        gravy = pa.gravy()
        length = len(s)
        pi = pa.isoelectric_point()
        feats.append([net_c, gravy, length, pi])
    return np.asarray(feats, dtype=np.float32)

DESC_mat = _compute_descriptors(seqs_test)

# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------
MODELS: list[Tuple[str, Path, str, bool]] = [
    # Generic potency
    ("potency_xgb",   PROJECT_ROOT / "models/potency_judge/potency_judge.model", "xgb", False),
    ("potency_rf",    PROJECT_ROOT / "models/potency_rf/potency_rf.model.pkl",    "skl", False),
    ("potency_logreg",PROJECT_ROOT / "models/potency_logreg/potency_logreg.model.pkl", "skl", False),
    # Haemolysis
    ("hemolysis_xgb",   PROJECT_ROOT / "models/hemolysis_judge/hemolysis_judge.model", "xgb", False),
    ("hemolysis_rf",    PROJECT_ROOT / "models/hemolysis_rf/hemolysis_rf.model.pkl",    "skl", False),
    ("hemolysis_logreg",PROJECT_ROOT / "models/hemolysis_logreg/hemolysis_logreg.model.pkl", "skl", False),
    # Cytotoxicity
    ("cytotox_xgb",   PROJECT_ROOT / "models/cytotoxicity_judge/cytotoxicity_judge.model", "xgb", False),
    ("cytotox_rf",    PROJECT_ROOT / "models/cytotoxicity_rf/cytotoxicity_rf.model.pkl",    "skl", False),
    ("cytotox_logreg",PROJECT_ROOT / "models/cytotoxicity_logreg/cytotoxicity_logreg.model.pkl", "skl", False),
    # Species-specific potency judges (XGB only)
    ("potency_ecoli_xgb",        PROJECT_ROOT / "models/potency_judge_ecoli/potency_judge_ecoli.model", "xgb", False),
    ("potency_saureus_xgb",      PROJECT_ROOT / "models/potency_judge_saureus/potency_judge_saureus.model", "xgb", False),
    ("potency_paeruginosa_xgb",  PROJECT_ROOT / "models/potency_judge_paeruginosa/potency_judge_paeruginosa.model", "xgb", False),
    # Descriptor-augmented potency judge
    ("potency_desc_xgb", PROJECT_ROOT / "models/potency_judge_descriptors/potency_judge/potency_judge.model", "xgb", True),
]

# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------
results: list[tuple[str, int, float, float]] = []  # name, N, sec, ms/pep

for name, path, kind, require_desc in MODELS:
    if not path.exists():
        print(f"[SKIP] {name}: model file not found ({path})", file=sys.stderr)
        continue

    print(f"Timing {name} …", file=sys.stderr)

    # Load model
    if kind == "xgb":
        booster = xgb.Booster()
        booster.load_model(str(path))
        model = booster
    else:
        with open(path, "rb") as fp:
            model = pickle.load(fp)

    # Prepare feature matrix (copy to avoid accidental mutation)
    X = X_emb if not require_desc else np.hstack([X_emb, DESC_mat])

    # Time prediction
    start = time.perf_counter()
    if kind == "xgb":
        _ = model.predict(xgb.DMatrix(X))
    else:
        _ = model.predict_proba(X)[:, 1]
    elapsed = time.perf_counter() - start
    ms_per_pep = elapsed / X.shape[0] * 1000.0
    results.append((name, X.shape[0], elapsed, ms_per_pep))

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "models/model_timings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_csv = OUT_DIR / "inference_timing.csv"

with open(out_csv, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["model", "n_peptides", "seconds", "ms_per_peptide"])
    writer.writerows(results)

print("Saved timing results to", out_csv, file=sys.stderr)

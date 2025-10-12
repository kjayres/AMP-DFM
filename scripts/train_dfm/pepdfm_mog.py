#!/usr/bin/env python3
"""pepdfm_mog.py

Multi-objective PepDFM sampling guided by three XGBoost judges
(potency, haemolysis, cytotoxicity).

Usage (defaults shown):
    $ python pepdfm_mog.py \
        --ckpt ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt \
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
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import random
import torch
from torch import nn
from transformers import AutoTokenizer, EsmModel
import xgboost as xgb
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so that `import utils.*` works even
# when this script is executed from a sub-directory.
# ---------------------------------------------------------------------------
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# fmt: on

# ---------------------------------------------------------------------------
# Project-internal imports ---------------------------------------------------
# ---------------------------------------------------------------------------
from utils.parsing import parse_guidance_args
from models.peptide_classifiers import load_solver  # unconditional PepDFM denoiser
from flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver  # type: ignore
# ---------------------------------------------------------------------------
# Runtime patch: guided_transition_scoring for PepDFM (AA tokens 4–23)
# ---------------------------------------------------------------------------
from flow_matching.utils import multi_guidance as _mg


def _pep_guided_transition_scoring(x_t, u_t, w, s_models, t, importance, args):
    """Copy of original guided_transition_scoring but enumerates AA tokens 4–23
    so that A (4) and C (5) are included for PepDFM sampling."""
    B, L, vocab_size = u_t.shape
    device = x_t.device
    guided_u_t = u_t.clone()

    # 1. Randomly select one mutable position per sequence (skip <cls>=0).
    #    We must guarantee that the token at the chosen position is an amino-acid
    #    (indices 4…23) so that it is present in `aa_tokens` below; otherwise the
    #    mask would contain `num_aa` (not `num_aa-1`) true values, breaking the
    #    fixed-shape `view` operation later on.
    pos_indices = torch.randint(low=1, high=L - 1, size=(B,), device=device)
    batch_idx = torch.arange(B, device=device)

    current_tokens = x_t[batch_idx, pos_indices]

    # Resample any positions whose token is not an amino-acid.
    invalid = (current_tokens < 4) | (current_tokens > 23)
    max_resample = 10  # absolute safety cap → should very rarely loop
    tries = 0
    while invalid.any() and tries < max_resample:
        # Draw fresh indices only for invalid rows
        new_pos = torch.randint(low=1, high=L - 1, size=(invalid.sum().item(),), device=device)
        pos_indices[invalid] = new_pos
        current_tokens = x_t[batch_idx, pos_indices]
        invalid = (current_tokens < 4) | (current_tokens > 23)
        tries += 1
    # Should the unlikely case remain where a non-AA token persists, we simply
    # proceed – the subsequent mask will then keep `num_aa` true values and the
    # fallback slice below will trim the extra column.

    # 2. Candidate amino-acid tokens: 4…vocab_size-1 (24-token PepDFM)
    aa_tokens = torch.arange(4, vocab_size, device=device)
    num_aa = aa_tokens.numel()

    full_cand_tokens = aa_tokens.unsqueeze(0).expand(B, num_aa)
    mask = full_cand_tokens != current_tokens.unsqueeze(1)

    cand_flat = torch.masked_select(full_cand_tokens, mask)
    # If every row satisfied the AA-token condition above, `cand_flat` will have
    # exactly `B*(num_aa-1)` elements.  Otherwise (extremely rare) some rows will
    # have one extra element; we trim the excess so that `.view` succeeds.
    expected = B * (num_aa - 1)
    if cand_flat.numel() > expected:
        cand_flat = cand_flat[:expected]
    cand_tokens = cand_flat.view(B, num_aa - 1)

    num_candidates = cand_tokens.size(1)

    new_x = x_t.unsqueeze(1).expand(B, num_aa, L).clone()
    new_x = new_x[mask].view(B, num_candidates, L)
    new_x[batch_idx, :, pos_indices] = cand_tokens

    new_x_flat = new_x.view(B * num_candidates, L)
    improvements_list = []
    with torch.no_grad():
        count = 0
        for i, s in enumerate(s_models):
            import inspect
            sig = inspect.signature(s.forward) if hasattr(s, 'forward') else inspect.signature(s)
            if 't' in sig.parameters:
                candidate_scores = s(new_x_flat, t)
                base_score = s(x_t, t)
            else:
                candidate_scores = s(new_x_flat)
                base_score = s(x_t)

            if isinstance(candidate_scores, tuple):
                for k, score in enumerate(candidate_scores):
                    improvement = candidate_scores[k].view(B, num_candidates) - base_score[k].unsqueeze(1)
                    improvement = improvement.float() * importance[count]
                    improvements_list.append(improvement.unsqueeze(2))
                    count += 1
            else:
                improvement = candidate_scores.view(B, num_candidates) - base_score.unsqueeze(1)
                improvement = improvement.float() * importance[count]
                improvements_list.append(improvement.unsqueeze(2))
                count += 1

    improvement_values = torch.cat(improvements_list, dim=2)
    ranks = torch.argsort(torch.argsort(improvement_values, dim=1), dim=1).float() + 1
    I_n = ranks / float(num_candidates)
    avg_I = I_n.mean(dim=2)
    norm_avg_I = (avg_I - avg_I.mean(dim=-1, keepdim=True)) / (avg_I.std(dim=-1, unbiased=False, keepdim=True) + 1e-8)

    D = (improvement_values * w.view(1, 1, -1)).sum(dim=2)
    norm_D = (D - D.mean(dim=-1, keepdim=True)) / (D.std(dim=-1, unbiased=False, keepdim=True) + 1e-8)

    delta_S = norm_avg_I + args.lambda_ * norm_D
    factor = torch.exp(args.beta * delta_S).clamp(min=-100, max=100)

    # Optional homopolymer penalty: down-weight candidates that create runs ≥ 3
    if getattr(args, 'homopolymer_gamma', 0.0) and args.homopolymer_gamma > 0:
        B, num_candidates, L = new_x.shape
        penalty = torch.ones((B, num_candidates), device=device, dtype=guided_u_t.dtype)
        hp_scalar = torch.exp(torch.tensor(-float(args.homopolymer_gamma), device=device, dtype=guided_u_t.dtype))
        # Check the edited position and its neighbours for runs >= 3
        for b in range(B):
            pos = pos_indices[b].item()
            for c in range(num_candidates):
                seq = new_x[b, c]
                tok = seq[pos]
                run = 1
                # extend left
                i = pos - 1
                while i >= 0 and seq[i] == tok:
                    run += 1; i -= 1
                # extend right
                i = pos + 1
                while i < L and seq[i] == tok:
                    run += 1; i += 1
                if run >= 3:
                    penalty[b, c] = hp_scalar
        factor = factor * penalty

    guided_u_t[batch_idx.unsqueeze(1), pos_indices.unsqueeze(1), cand_tokens] = \
        u_t[batch_idx.unsqueeze(1), pos_indices.unsqueeze(1), cand_tokens] * factor

    updated_vals = guided_u_t[batch_idx, pos_indices, :]
    sum_off_diag = updated_vals.sum(dim=1) - updated_vals[batch_idx, current_tokens]
    guided_u_t[batch_idx, pos_indices, current_tokens] = -sum_off_diag

    # Filtering and Euler update are performed by the solver – no extra call here.
    return guided_u_t, pos_indices, cand_tokens, improvement_values, delta_S

# Patch it in
_mg.guided_transition_scoring = _pep_guided_transition_scoring
# Ensure MixtureDiscreteEulerSolver, which imported the original function
# into its own module namespace, also uses our PepDFM-aware version.
import flow_matching.solver.discrete_solver as _ds
_ds.guided_transition_scoring = _pep_guided_transition_scoring

# ---------------------------------------------------------------------------
# Constant vocabulary mapping (PepDFM 24-token)
# ---------------------------------------------------------------------------
_IDX_TO_AA = {
    4: "A", 5: "C", 6: "D", 7: "E", 8: "F", 9: "G", 10: "H",
    11: "I", 12: "K", 13: "L", 14: "M", 15: "N", 16: "P", 17: "Q",
    18: "R", 19: "S", 20: "T", 21: "V", 22: "W", 23: "Y",
}

# ---------------------------------------------------------------------------
# Judge wrapper --------------------------------------------------------------
# ---------------------------------------------------------------------------
class _BaseXGBJudge(nn.Module):
    """Lightweight inference wrapper for an XGBoost classifier trained on
    1280-d ESM-2 embeddings plus optional hand-crafted descriptors."""

    _ESM_CACHE = {}
    _TOK_CACHE = {}

    def __init__(self, booster_path: Path, metadata_path: Optional[Path], *, device: str):
        super().__init__()
        self.device = device
        # XGBoost booster
        self.booster = xgb.Booster()
        self.booster.load_model(str(booster_path))

        # All new judges are embedding-only – no hand-crafted descriptors
        self.has_desc = False
        self.desc_mean = None
        self.desc_std = None

        # ESM-2 model (shared per device)
        if device not in _BaseXGBJudge._ESM_CACHE:
            model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
            model.eval()
            _BaseXGBJudge._ESM_CACHE[device] = model
            _BaseXGBJudge._TOK_CACHE[device] = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm_model: EsmModel = _BaseXGBJudge._ESM_CACHE[device]
        self.tokenizer: AutoTokenizer = _BaseXGBJudge._TOK_CACHE[device]

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def _tensor_to_strings(self, tokens: torch.Tensor) -> List[str]:
        """Decode PepDFM token tensor → list[str] amino-acid sequences."""
        seqs: List[str] = []
        for row in tokens.cpu().tolist():
            aa_chars = [ _IDX_TO_AA.get(tok, "") for tok in row if 4 <= tok <= 23 ]
            seqs.append("".join(aa_chars))
        return seqs

    def _esm_embeddings(self, tokens: torch.Tensor) -> np.ndarray:
        seqs = self._tensor_to_strings(tokens)
        enc = self.tokenizer(seqs, return_tensors="pt", padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.esm_model(**enc).last_hidden_state  # (B, L, 1280)
            mask = enc["attention_mask"].unsqueeze(-1)  # (B, L, 1)
            summed = (out * mask).sum(dim=1)
            lengths = mask.sum(dim=1)
            emb = summed / lengths  # mean-pool
        return emb.cpu().numpy()  # (B, 1280)

    @staticmethod
    def _descriptors(seq_strs: List[str]) -> np.ndarray:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        feats = []
        for s in seq_strs:
            pa = ProteinAnalysis(s if s else "A")  # avoid zero-len
            net_c = abs(pa.charge_at_pH(7.0))
            length = len(s)
            gravy = pa.gravy()
            # crude hydrophobic-moment proxy (same as training)
            hydro_moment = sum(pa.get_amino_acids_percent().values()) / 20.0
            feats.append([net_c, length, gravy, hydro_moment])
        return np.asarray(feats, dtype=np.float32)

    def _features(self, tokens: torch.Tensor):
        embs = self._esm_embeddings(tokens)
        return xgb.DMatrix(embs)

    def forward(self, tokens: torch.Tensor):  # type: ignore[override]
        dmat = self._features(tokens)
        proba = self.booster.predict(dmat)  # (N,)
        return torch.from_numpy(proba).to(tokens.device)

# Specific judges ------------------------------------------------------------
class PotencyJudge(_BaseXGBJudge):
    def __init__(self, device: str):
        root = Path(__file__).resolve().parents[1] / "models" / "potency_judge"
        super().__init__(root / "potency_judge.model", root / "potency_judge_metadata.pkl", device=device)

class HemolysisJudge(_BaseXGBJudge):
    def __init__(self, device: str):
        root = Path(__file__).resolve().parents[1] / "models" / "hemolysis_judge"
        super().__init__(root / "hemolysis_judge.model", root / "hemolysis_judge_metadata.pkl", device=device)

class CytotoxicityJudge(_BaseXGBJudge):
    def __init__(self, device: str):
        root = Path(__file__).resolve().parents[1] / "models" / "cytotoxicity_judge"
        super().__init__(root / "cytotoxicity_judge.model", root / "cytotoxicity_judge_metadata.pkl", device=device)

# ---------------------------------------------------------------------------
# Utility: decode tokens → peptide string
# ---------------------------------------------------------------------------

def detokenise(tokens: List[int]) -> str:
    return "".join(_IDX_TO_AA.get(t, "") for t in tokens if 4 <= t <= 23)

# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PepDFM multi-objective generator (potency, haemolysis, cytotox)")
    parser.add_argument("--ckpt", required=True, help="Path to PepDFM CNN checkpoint")
    parser.add_argument("--out", help="Optional explicit FASTA output path")
    parser.add_argument("--run_tag", help="Sub-directory under ampflow/results/mog/ (ignored if --out is given). Defaults to potency_variant.")
    parser.add_argument("--potency_variant", default="generic",
                        help="Which potency judge to use: generic|ecoli|saureus|paeruginosa")
    parser.add_argument("--n_samples", type=int, default=2500, help="Total peptides to sample")
    parser.add_argument("--n_batches", type=int, default=10, help="Number of sampler batches")
    parser.add_argument("--length", type=int, help="Fixed core length; overrides range flags")
    parser.add_argument("--len_min", type=int, default=10, help="Minimum AA length if --length not provided")
    parser.add_argument("--len_max", type=int, default=30, help="Maximum AA length if --length not provided")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--importance", default="1,1,1",
                        help="Comma-sep importance weights potency,hemolysis,cytotox")

    # Let parse_guidance_args consume its own flags (lambda_, beta, Phi …)
    args, unknown = parser.parse_known_args()
    g_args = parse_guidance_args(unknown)  # Namespace with guidance params

    # ------------------------------------------------------------------
    device = args.device
    vocab_size = 24
    step_size = 1.0 / g_args.T

    # Importance weights ------------------------------------------------
    importance = [float(x) for x in args.importance.split(",")]
    if len(importance) != 3:
        raise ValueError("--importance must have exactly 3 comma-sep values (potency,hemolysis,cytotox)")

    # Judges ------------------------------------------------------------
    variant = args.potency_variant.lower()
    root_dir = Path(__file__).resolve().parents[1] / "models"
    if variant == "generic":
        pot_dir = root_dir / "potency_judge"
        model_path = pot_dir / "potency_judge.model"
        meta_path  = pot_dir / "potency_judge_metadata.pkl"
    else:
        pot_dir = root_dir / f"potency_judge_{variant}"
        model_path = pot_dir / f"potency_judge_{variant}.model"
        meta_path  = pot_dir / f"potency_judge_{variant}_metadata.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Potency model not found: {model_path}")
    potency_j = _BaseXGBJudge(model_path, meta_path if meta_path.exists() else None, device=device)

    hemolysis_j = HemolysisJudge(device)
    cytotox_j   = CytotoxicityJudge(device)
    score_models = [potency_j, hemolysis_j, cytotox_j]

    # PepDFM solver -----------------------------------------------------
    solver: MixtureDiscreteEulerSolver = load_solver(args.ckpt, vocab_size, device)

    n_per_batch = args.n_samples // args.n_batches
    results = []  # (seq, scores)
    sample_id = 0

    for b in range(args.n_batches):
        # ---------- init tokens ----------------
        if args.length:
            L = args.length
        else:
            L = random.randint(args.len_min, args.len_max)
        core = torch.randint(low=4, high=vocab_size, size=(n_per_batch, L), device=device)
        x_init = torch.cat([
            torch.zeros((n_per_batch,1), dtype=torch.long, device=device),  # <cls>
            core,
            torch.full((n_per_batch,1), 2, dtype=torch.long, device=device)  # <eos>
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

        # ---------- decode + score --------------
        for row in x_final.cpu().tolist():
            seq = detokenise(row)
            tokens_tensor = torch.tensor([row], device=device)
            p_score = potency_j(tokens_tensor).item()
            h_score = hemolysis_j(tokens_tensor).item()
            c_score = cytotox_j(tokens_tensor).item()
            results.append((seq, p_score, h_score, c_score))

    # ------------------------------------------------------------------
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        run_tag = args.run_tag if args.run_tag else variant
        out_dir = PROJECT_ROOT / "ampflow" / "results" / "mog" / run_tag
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
        writer.writerow(["sample_id", "sequence", "potency", "hemolysis", "cytotox"])
        for i, (seq, p, h, c) in enumerate(results, start=1):
            writer.writerow([i, seq, p, h, c])

    print("Written", out_path, "and", csv_path)


if __name__ == "__main__":
    main()
# --------------------------------------------------------------------------- 

from __future__ import annotations

import argparse, math, random, json
import os, tempfile, subprocess, gzip
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Collate helper that leaves variable-length sequences untouched.  This
# prevents the default PyTorch collate_fn from crashing on unequal-length
# lists coming from HF Arrow rows.
# ---------------------------------------------------------------------------


def raw_hf_collate(batch):
    out = {}
    for key in batch[0]:
        out[key] = [item[key] for item in batch]
    return out

from models.peptide_models import CNNModel
from models.peptide_models import CNNModelPep
from models.peptide_models import CNNModelOriginal
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper

# --- Potency judges (XGBoost-based) ----------------------------------------
# Self-contained judge implementation to avoid vocabulary inconsistencies
import math
import pickle
import xgboost as xgb
from transformers import EsmModel, AutoTokenizer
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ---------------------------- Arguments ------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="Space-separated list of TAG:CKPT[:cond] specs")
    p.add_argument("--n", type=int, default=10000, help="Samples per model")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp_val_path", type=str,
                   default="ampflow/ampdfm_data/tokenized_amp/val")
    p.add_argument("--pep_val_path", type=str,
                   default="ampflow/ampdfm_data/tokenized_pep/val")
    # Optional per-TAG validation dataset override
    p.add_argument("--val_map", nargs="*", default=[],
                   help="Zero or more TAG:/abs/path/to/val_dataset entries")
    # Optional training data per TAG for diagnostics (AA/length KL, overlap)
    p.add_argument("--train_fasta_map", nargs="*", default=[],
                   help="Zero or more TAG:/abs/path/to/train.fasta entries")
    p.add_argument("--train_hf_map", nargs="*", default=[],
                   help="Zero or more TAG:/abs/path/to/hf_dataset entries (load_from_disk)")
    p.add_argument("--lev_samples", type=int, default=0,
                   help="Pairs and novelty samples for Levenshtein-based metrics; 0 to skip")
    p.add_argument("--mmseqs", action="store_true",
                   help="Compute MMseqs ≥80% overlap vs training if provided")
    return p.parse_args()

# --------------------------- Helpers ---------------------------------------

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 6 for i, aa in enumerate(AA_ORDER)}
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}

# 26-token vocabulary (AMP-DFM)
SPECIALS = {"<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3, "<AMP>": 4, "<NEG>": 5}


def encode_tokens(seq: str) -> List[int]:
    """Return AMP-DFM token list for a peptide string (assumes AMP class)."""
    toks = [SPECIALS["<cls>"], SPECIALS["<AMP>"]]
    toks.extend(AA_TO_IDX.get(ch, SPECIALS["<unk>"]) for ch in seq.upper())
    toks.append(SPECIALS["<eos>"])
    return toks


def _decode_amp_tokens(tokens: List[int]) -> str:
    """Convert AMP-DFM token list → peptide string (skips specials)."""
    return "".join(IDX_TO_AA.get(t, "X") for t in tokens if t >= 6)


def sample_ampdfm(ckpt: Path, _cond_unused: Optional[str], n_samples: int, device: str, seed: int) -> List[str]:
    """Sample *n_samples* strings from an AMP-DFM checkpoint (token-conditioned)."""
    random.seed(seed); torch.manual_seed(seed)
    vocab_size = 26

    # Infer dims from checkpoint
    state = torch.load(ckpt, map_location="cpu")
    def _infer_dims_cnn(state_dict):
        # token_embedding.weight: (vocab, embed_dim)
        embed_dim = state_dict.get("token_embedding.weight").shape[1]
        # linear.weight: (hidden_dim, embed_dim, k)
        hidden_dim = state_dict.get("linear.weight").shape[0]
        return int(embed_dim), int(hidden_dim)
    embed_dim, hidden_dim = _infer_dims_cnn(state)
    model = CNNModel(alphabet_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    class Wrapped(ModelWrapper):
        def forward(self, x, t, **kw):
            return torch.softmax(self.model(x, t), dim=-1)

    wrapped = Wrapped(model)
    sched = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=sched)
    solver = MixtureDiscreteEulerSolver(model=wrapped, path=path, vocabulary_size=vocab_size)

    step_size = 1 / 100
    batch = 256
    collected: List[str] = []
    lengths = list(range(6, 51))
    p_len = np.ones(len(lengths)) / len(lengths)

    with torch.no_grad():
        while len(collected) < n_samples:
            cur = min(batch, n_samples - len(collected))
            length = random.choices(lengths, weights=p_len)[0]
            core = torch.randint(6, vocab_size, (cur, length), device=device)
            cls  = torch.zeros((cur, 1), dtype=core.dtype, device=device)
            amp  = torch.full((cur, 1), 4, dtype=core.dtype, device=device)
            eos  = torch.full((cur, 1), 2, dtype=core.dtype, device=device)
            x_init = torch.cat([cls, amp, core, eos], dim=1)
            sol = solver.sample(x_init=x_init, step_size=step_size,
                                time_grid=torch.tensor([0.0, 1.0-1e-3], device=device))
            collected.extend(_decode_amp_tokens(seq) for seq in sol.tolist())
    return collected[:n_samples]


PEP_IDX_TO_AA = {**{i: aa for i, aa in enumerate(AA_ORDER, 4)}, **{0: "<cls>", 2: "<eos>"}}


def decode_pep_tokens(tokens: List[int]) -> str:
    return "".join(PEP_IDX_TO_AA.get(t, "X") for t in tokens if t >= 4)


def sample_pepdfm(ckpt: Path, n_samples: int, device: str, seed: int) -> List[str]:
    random.seed(seed); torch.manual_seed(seed)
    vocab_size = 24
    # Infer dims from checkpoint
    state = torch.load(ckpt, map_location="cpu")
    def _infer_dims_pep(state_dict):
        embed_dim = state_dict.get("token_embedding.weight").shape[1]
        hidden_dim = state_dict.get("linear.weight").shape[0]
        # infer number of conv blocks (count entries under 'blocks.N.')
        block_indices = []
        for k in state_dict.keys():
            if k.startswith("blocks."):
                try:
                    idx = int(k.split(".")[1])
                    block_indices.append(idx)
                except Exception:
                    pass
        num_blocks = (max(block_indices) + 1) if block_indices else 6
        return int(embed_dim), int(hidden_dim), int(num_blocks)
    embed_dim, hidden_dim, num_blocks = _infer_dims_pep(state)
    # Use 5-block original architecture when detected
    if num_blocks <= 5:
        model = CNNModelOriginal(alphabet_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    else:
        model = CNNModelPep(alphabet_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    class Wrapped(ModelWrapper):
        def forward(self, x, t, **kw):
            return torch.softmax(self.model(x, t), dim=-1)

    wrapped = Wrapped(model)
    sched = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=sched)
    solver = MixtureDiscreteEulerSolver(model=wrapped, path=path, vocabulary_size=vocab_size)

    step_size = 1 / 100
    batch = 256
    collected: List[str] = []
    lengths = list(range(6, 51))
    p_len = np.ones(len(lengths)) / len(lengths)

    with torch.no_grad():
        while len(collected) < n_samples:
            cur = min(batch, n_samples - len(collected))
            length = random.choices(lengths, weights=p_len)[0]
            core = torch.randint(4, vocab_size, (cur, length), device=device)
            cls  = torch.zeros((cur, 1), dtype=core.dtype, device=device)
            eos  = torch.full((cur, 1), 2, dtype=core.dtype, device=device)
            x_init = torch.cat([cls, core, eos], dim=1)
            sol = solver.sample(x_init=x_init, step_size=step_size,
                                time_grid=torch.tensor([0.0, 1.0-1e-3], device=device))
            collected.extend(decode_pep_tokens(seq) for seq in sol.tolist())
    return collected[:n_samples]


def sample_conditional_pepdfm(ckpt: Path, cond: str, n_samples: int, device: str, seed: int) -> List[str]:
    """Sample from conditional PepDFM (24-token) with a 4-bit cond_vec.

    cond values: generic|ec|pa|sa|all
    Maps to [AMP, EC, PA, SA] bits.
    """
    random.seed(seed); torch.manual_seed(seed)
    vocab_size = 24
    cond_map = {
        "generic": [1, 0, 0, 0],
        "ec":      [1, 1, 0, 0],
        "pa":      [1, 0, 1, 0],
        "sa":      [1, 0, 0, 1],
        "all":     [1, 1, 1, 1],
    }
    key = cond.lower()
    if key not in cond_map:
        raise ValueError(f"Unknown conditional target '{cond}'. Expected one of {list(cond_map)}")
    cond_vec = torch.tensor(cond_map[key], dtype=torch.float32, device=device).unsqueeze(0)

    state = torch.load(ckpt, map_location="cpu")
    def _infer_dims_cnn_cond(state_dict):
        embed_dim = state_dict.get("token_embedding.weight").shape[1]
        hidden_dim = state_dict.get("linear.weight").shape[0]
        # cond_proj.weight: (embed_dim, cond_dim)
        cond_w = state_dict.get("cond_proj.weight", None)
        cond_dim = int(cond_w.shape[1]) if cond_w is not None else 4
        return int(embed_dim), int(hidden_dim), cond_dim
    embed_dim, hidden_dim, cond_dim = _infer_dims_cnn_cond(state)
    model = CNNModel(alphabet_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, cond_dim=cond_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    class Wrapped(ModelWrapper):
        def forward(self, x, t, **kw):
            B = x.shape[0]
            return torch.softmax(self.model(x, t, cond_vec=cond_vec.expand(B, -1)), dim=-1)

    wrapped = Wrapped(model)
    sched = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=sched)
    solver = MixtureDiscreteEulerSolver(model=wrapped, path=path, vocabulary_size=vocab_size)

    step_size = 1 / 100
    batch = 256
    collected: List[str] = []
    # Match unconditional PepDFM sampling length range for fair comparison
    lengths = list(range(6, 51))
    p_len = np.ones(len(lengths)) / len(lengths)

    with torch.no_grad():
        while len(collected) < n_samples:
            cur = min(batch, n_samples - len(collected))
            length = random.choices(lengths, weights=p_len)[0]
            core = torch.randint(4, vocab_size, (cur, length), device=device)
            cls  = torch.zeros((cur, 1), dtype=core.dtype, device=device)
            eos  = torch.full((cur, 1), 2, dtype=core.dtype, device=device)
            x_init = torch.cat([cls, core, eos], dim=1)
            sol = solver.sample(x_init=x_init, step_size=step_size,
                                time_grid=torch.tensor([0.0, 1.0-1e-3], device=device))
            collected.extend(decode_pep_tokens(seq) for seq in sol.tolist())
    return collected[:n_samples]


# ---------------- Diversity metrics ----------------------------------------

from collections import Counter
try:
    import Levenshtein
    _HAS_LEV = True
except ImportError:
    _HAS_LEV = False


def positional_entropy(seqs: List[str]) -> float:
    max_len = max(len(s) for s in seqs)
    entropies = []
    for pos in range(max_len):
        counts = Counter(s[pos] if pos < len(s) else "-" for s in seqs)
        total = float(sum(counts.values()))
        probs = np.array([c/total for c in counts.values()])
        ent = -(probs * np.log2(probs + 1e-9)).sum()
        entropies.append(ent)
    return float(np.mean(entropies))


def mean_levenshtein(seqs: List[str], n_pairs: int = 2000) -> float:
    if not _HAS_LEV:
        return float("nan")
    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(seqs), size=(n_pairs, 2))
    dists = [Levenshtein.distance(seqs[i], seqs[j]) for i, j in idxs]
    return float(np.mean(dists))

# ---------------- Training diagnostics helpers ------------------------------

def aa_distribution(seqs: List[str]) -> np.ndarray:
    counts = {}
    total = 0
    for s in seqs:
        for ch in s:
            counts[ch] = counts.get(ch, 0) + 1
            total += 1
    freq = np.array([counts.get(aa, 0) / max(total, 1) for aa in AA_ORDER], dtype=float)
    return freq


def length_distribution(seqs: List[str]) -> np.ndarray:
    lens = [len(s) for s in seqs]
    if not lens:
        return np.array([1.0], dtype=float)
    max_len = max(lens)
    counts = np.bincount(lens, minlength=max_len + 1)
    return counts / counts.sum()


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(np.sum(p * np.log(p / q)))


def shannon_entropy(freq: np.ndarray) -> float:
    eps = 1e-12
    f = np.clip(freq, eps, 1)
    return float(-np.sum(f * np.log2(f)))


def fasta_iter(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        seq = []
        for line in f:
            if line.startswith(">"):
                if seq:
                    yield "".join(seq)
                    seq = []
            else:
                seq.append(line.strip())
        if seq:
            yield "".join(seq)


def _load_train_from_fasta(fp: Path) -> List[str]:
    return list(fasta_iter(fp))


def _load_train_from_hf(hf_path: str) -> List[str]:
    ds_loaded = load_from_disk(hf_path)
    # If a DatasetDict is passed, prefer the 'train' split; otherwise merge all splits
    try:
        # DatasetDict-like
        keys = list(ds_loaded.keys())  # type: ignore[attr-defined]
        if "train" in keys:
            datasets = [ds_loaded["train"]]
        else:
            datasets = [ds_loaded[k] for k in keys]
    except Exception:
        # Single Dataset
        datasets = [ds_loaded]

    seqs: List[str] = []
    for ds in datasets:
        for rec in ds:
            ids = rec.get("input_ids")
            if ids is None:
                continue
            # Convert to flat list of ints
            if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
                # nested -> flatten
                flat = [int(x) for sub in ids for x in sub]
            else:
                flat = [int(x) for x in ids]
            max_tok = max(flat) if flat else 0
            # Heuristic: 26-token AMP if tokens can reach >= 24; else 24-token PepDFM
            if max_tok >= 24:
                # Use 26-token decoding: amino acids at indices >=6
                seqs.append("".join(IDX_TO_AA.get(t, "") for t in flat if t >= 6))
            else:
                # Use 24-token decoding: amino acids at indices >=4
                seqs.append("".join(PEP_IDX_TO_AA.get(t, "") for t in flat if t >= 4))
    return seqs


# Parse TAG:PATH lists into dicts
def _parse_tag_map(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if ":" not in it:
            continue
        tag, path = it.split(":", 1)
        out[tag] = path
    return out

# ---------------- Validation loss (perplexity) -----------------------------

from flow_matching.loss import MixturePathGeneralizedKL
from datasets import load_from_disk

def validation_perplexity(ckpt: Path, val_path: str, vocab_size: int, cond_vec: Optional[List[int]], device: str) -> float:
    val_ds = load_from_disk(val_path)

    loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4,
                        collate_fn=raw_hf_collate)

    # Infer model dims from checkpoint
    state = torch.load(ckpt, map_location="cpu")
    def _infer_dims(state_dict):
        embed_dim = state_dict.get("token_embedding.weight").shape[1]
        hidden_dim = state_dict.get("linear.weight").shape[0]
        cond_w = state_dict.get("cond_proj.weight", None)
        cond_dim = int(cond_w.shape[1]) if cond_w is not None else None
        return int(embed_dim), int(hidden_dim), cond_dim
    embed_dim, hidden_dim, cond_dim_infer = _infer_dims(state)

    model_kwargs = dict(alphabet_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
    # Set conditioning dimensionality to match the length of the provided
    # condition vector (e.g. 4 for AMP-DFM).  Hard-coding this to 3 causes
    # state-dict shape mismatches when loading checkpoints that were trained
    # with a different number of conditioning dimensions.
    if cond_vec is not None:
        model_kwargs["cond_dim"] = len(cond_vec)
    elif cond_dim_infer is not None and cond_dim_infer > 0:
        model_kwargs["cond_dim"] = cond_dim_infer
    # Choose model class: conditional → CNNModel; unconditional pepdfm → CNNModelPep or CNNModelOriginal (5-block)
    if cond_vec is None:
        # Detect number of blocks for unconditional PepDFM checkpoints
        block_indices = []
        for k in state.keys():
            if k.startswith("blocks."):
                try:
                    idx = int(k.split(".")[1])
                    block_indices.append(idx)
                except Exception:
                    pass
        num_blocks = (max(block_indices) + 1) if block_indices else 6
        ModelCls = CNNModelOriginal if num_blocks <= 5 else CNNModelPep
    else:
        ModelCls = CNNModel
    model = ModelCls(**model_kwargs).to(device)
    # Fail fast if checkpoint weights do not match the current architecture
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---------------- Debug FiLM stats on first call -------------------
    if cond_vec is not None and getattr(model, "film", None) is not None:
        with torch.no_grad():
            gb_dbg = model.film[0](torch.tensor(cond_vec, dtype=torch.float32, device=device).unsqueeze(0))
            g_dbg, b_dbg = gb_dbg.chunk(2, dim=1)
            print("[DEBUG] FiLM γ mean:", g_dbg.abs().mean().item(), "β mean:", b_dbg.abs().mean().item())

    sched = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=sched)
    loss_fn = MixturePathGeneralizedKL(path=path)

    # ------------------------------------------------------------------
    # Accumulate *token*-level loss to compute true perplexity.
    # `MixturePathGeneralizedKL` returns a mean over the batch *sequence*
    # length, so we need to weight by the number of *non-pad tokens*.
    # ------------------------------------------------------------------
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        # Helper to flatten & tensorise nested ragged lists
        import torch.nn.functional as F

        def _to_long_tensor(raw):
            if torch.is_tensor(raw):
                return raw.to(device)
            if raw and isinstance(raw[0], (list, tuple)) and raw[0] and isinstance(raw[0][0], (list, tuple)):
                flat = [seq for group in raw for seq in group]
            else:
                flat = raw
            if flat and torch.is_tensor(flat[0]):
                return torch.stack([t.to(device) for t in flat])
            if flat and isinstance(flat[0], (list, tuple)):
                # variable-length sequences → pad then stack
                max_len = max(len(seq) for seq in flat)
                padded = [seq + [0]*(max_len - len(seq)) for seq in flat]
                return torch.tensor(padded, dtype=torch.long, device=device)
            return torch.tensor(flat, dtype=torch.long, device=device)

        for batch in loader:
            x_1 = _to_long_tensor(batch["input_ids"])

            cond = _to_long_tensor(batch["cond_vec"]) if cond_vec is not None else None
            x_0 = torch.randint_like(x_1, high=vocab_size)
            if cond_vec is None:
                x_0[:, 0] = x_1[:, 0]
            else:
                x_0[:, :2] = x_1[:, :2]
            t = torch.rand(x_1.shape[0], device=device) * (1 - 1e-3)
            sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            if cond_vec is not None:
                # `cond` already has shape (B, cond_dim); cast to float before
                # passing to the model rather than reshaping to a hard-coded
                # size.
                logits = model(sample.x_t, sample.t, cond_vec=cond.float())
            else:
                logits = model(sample.x_t, sample.t)
            loss = loss_fn(logits=logits, x_1=x_1, x_t=sample.x_t, t=sample.t)
            # Number of *valid* tokens in this batch
            # Mask out *both* common pad IDs (0 or 1) to cope with datasets that
            # encode padding differently.  Only non-pad tokens contribute to the
            # accumulated loss used for perplexity.
            tok_count = ((x_1 != 0) & (x_1 != 1)).sum().item()
            total_loss += loss.item() * tok_count
            total_tokens += tok_count
    mean_loss = total_loss / max(total_tokens, 1)
    return float(math.exp(mean_loss))

# ---------------- Potency Judge Implementation ------------------------------

# Hydrophobic moment descriptor (Eisenberg scale)
_EISENBERG = {
    'A':  0.25, 'C': 0.04, 'D': -2.64, 'E': -2.62, 'F': 0.93,
    'G':  0.16, 'H': -0.40, 'I':  0.73, 'K': -1.50, 'L': 0.53,
    'M':  0.26, 'N': -0.78, 'P': -1.23, 'Q': -0.85, 'R': -2.53,
    'S': -0.26, 'T': -0.18, 'V':  0.54, 'W': 0.37, 'Y': 0.02,
}
_THETA_RAD = math.radians(100)  # 100° per residue in α-helix


def _hydrophobic_moment(seq: str) -> float:
    """Eisenberg α-helix hydrophobic moment, normalised by length."""
    x_sum = y_sum = 0.0
    for i, aa in enumerate(seq):
        h_i = _EISENBERG.get(aa, 0.0)
        angle = i * _THETA_RAD
        x_sum += h_i * math.cos(angle)
        y_sum += h_i * math.sin(angle)
    n = len(seq)
    return math.hypot(x_sum, y_sum) / n if n else 0.0


class _BaseXGBJudge:
    """Common utilities for wrapping an XGBoost Booster that works on ESM
    embeddings + (optionally) 5 simple descriptors.
    """

    def __init__(self, booster_path: Path, metadata_pkl: Optional[Path], *, device: str):
        self.device = device

        # ------------- XGBoost model ------------------
        self.booster = xgb.Booster()
        self.booster.load_model(str(booster_path))

        # All new judges are embedding-only – skip descriptor logic
        self.desc_mean = None
        self.desc_std = None
        self.has_desc = False

        # ------------- ESM model (shared singleton per device) --------------
        if not hasattr(_BaseXGBJudge, "_ESM_CACHE"):
            _BaseXGBJudge._ESM_CACHE = {}
        if not hasattr(_BaseXGBJudge, "_TOK_CACHE"):
            _BaseXGBJudge._TOK_CACHE = {}

        if device not in _BaseXGBJudge._ESM_CACHE:
            model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
            model.eval()
            _BaseXGBJudge._ESM_CACHE[device] = model
            _BaseXGBJudge._TOK_CACHE[device] = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

        self.esm_model = _BaseXGBJudge._ESM_CACHE[device]
        self.tokenizer = _BaseXGBJudge._TOK_CACHE[device]

    def _esm_embeddings(self, seq_tensor: torch.Tensor) -> np.ndarray:
        """Return mean-pooled ESM embeddings (N, 1280) using tokenizer-based
        encoding of amino-acid strings decoded from AMP-DFM tokens.

        Decoding: tokens >= 6 map to amino acids; specials are dropped.
        """
        # Decode AMP-DFM tokens → peptide strings, then tokenize with HF
        seq_strs = [_decode_amp_tokens(seq.tolist()) for seq in seq_tensor]
        enc = self.tokenizer(seq_strs, return_tensors="pt", padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.esm_model(**enc).last_hidden_state  # (B, L, 1280)
            mask = enc["attention_mask"].unsqueeze(-1)  # (B, L, 1)
            summed = (out * mask).sum(dim=1)
            lengths = mask.sum(dim=1)
            emb = summed / lengths
        return emb.cpu().numpy()  # (N, 1280)

    @staticmethod
    def _descriptors(seq_strs: List[str]) -> np.ndarray:
        """Compute 4 hand-crafted descriptors for each peptide string."""
        feats = []
        for s in seq_strs:
            pa = ProteinAnalysis(s) if s else ProteinAnalysis("A")  # avoid zero-len
            net_charge = abs(pa.charge_at_pH(7.0))
            length = len(s)
            gravy = pa.gravy()
            norm_hydro_moment = _hydrophobic_moment(s)
            feats.append([net_charge, length, gravy, norm_hydro_moment])
        return np.asarray(feats, dtype=float)

    def _features(self, tokens: torch.Tensor) -> xgb.DMatrix:
        """Return DMatrix ready for the Booster."""
        emb = self._esm_embeddings(tokens)  # (N, 1280)
        if self.has_desc:
            seqs = [_decode_amp_tokens(seq.tolist()) for seq in tokens]
            desc = self._descriptors(seqs)
            if self.desc_mean is not None:
                desc = (desc - self.desc_mean) / self.desc_std
            features = np.hstack([emb, desc])
        else:
            features = emb
        return xgb.DMatrix(features)

    def __call__(self, tokens: torch.Tensor):
        """Return potency probability scores as a torch tensor."""
        dmat = self._features(tokens)
        probs = self.booster.predict(dmat)  # (N,)
        return torch.from_numpy(probs).to(self.device)


class PotencyJudge(_BaseXGBJudge):
    """XGBoost-based potency judge for different species."""
    
    def __init__(self, species: str, device: str):
        script_dir = Path(__file__).resolve().parent
        root = script_dir.parent.parent / "ampflow" / "models"
        
        species_map = {
            "generic": "potency_judge",
            "sa":      "potency_judge_saureus", 
            "pa":      "potency_judge_paeruginosa",
            "ec":      "potency_judge_ecoli",
        }
        folder = root / species_map[species]
        super().__init__(folder / f"{species_map[species]}.model",
                         folder / f"{species_map[species]}_metadata.pkl",
                         device=device)

# ---------------- Main ------------------------------------------------------


def main():
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Build judges once
    generic_judge = PotencyJudge("generic", device)
    judge_sa = PotencyJudge("sa", device)
    judge_pa = PotencyJudge("pa", device)
    judge_ec = PotencyJudge("ec", device)

    # ------------------------------------------------------------------
    # Evaluation threshold for a sequence to count as a potency "hit".
    # Adjust here if you want a stricter or looser cut-off.
    # ------------------------------------------------------------------
    thr = 0.8

    summaries = []

    # Training data maps (optional)
    fasta_map = _parse_tag_map(args.train_fasta_map)
    hf_map = _parse_tag_map(args.train_hf_map)
    val_map = _parse_tag_map(args.val_map)
    train_cache: Dict[str, List[str]] = {}

    def _display_label(tag_raw: str, ckpt_path: Path, cond: Optional[str]) -> str:
        t = tag_raw.lower()
        # Explicit tag names used in our scripts
        if t == "pep_ours":
            return "AMP-DFM (Unconditional)"
        if t == "pep_orig":
            return "Pep-DFM"
        if t in ("condpep", "pepdfm_cond"):
            return "AMP-DFM (Conditional)"
        if t in ("pep", "pepdfm") and cond is None:
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                block_indices = []
                for k in state.keys():
                    if k.startswith("blocks."):
                        try:
                            idx = int(k.split(".")[1])
                            block_indices.append(idx)
                        except Exception:
                            pass
                num_blocks = (max(block_indices) + 1) if block_indices else 6
                return "Pep-DFM" if num_blocks <= 5 else "AMP-DFM (Unconditional)"
            except Exception:
                return "AMP-DFM (Unconditional)"
        return tag_raw

    def _slug(s: str) -> str:
        return (
            s.replace(" ", "_")
             .replace("(", "_")
             .replace(")", "_")
             .replace("/", "_")
             .replace(",", "_")
        )

    for spec in args.models:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Model spec '{spec}' must be TAG:CKPT[:cond]")
        tag, ckpt_path = parts[0], Path(parts[1])
        cond = parts[2] if len(parts) == 3 else None

        print(f"\n=== Sampling {tag} ===")
        if tag.lower() in ("pep", "pepdfm") and cond is None:
            seqs = sample_pepdfm(ckpt_path, args.n, device, args.seed)
            val_path = args.pep_val_path
            vocab_size = 24
        elif tag.lower() in ("condpep", "pepdfm_cond") and cond is not None:
            seqs = sample_conditional_pepdfm(ckpt_path, cond, args.n, device, args.seed)
            # Use tokenized_amp validation (includes cond_vec) for perplexity
            val_path = args.amp_val_path
            vocab_size = 24
        elif tag.lower() in ("amp", "ampdfm"):
            seqs = sample_ampdfm(ckpt_path, None, args.n, device, args.seed)
            val_path = args.amp_val_path
            vocab_size = 26
        else:
            # Fallback: treat any unknown tag without a conditioning suffix as PepDFM (unconditional)
            if cond is None:
                seqs = sample_pepdfm(ckpt_path, args.n, device, args.seed)
                val_path = args.pep_val_path
                vocab_size = 24
            else:
                raise ValueError(f"Unsupported model spec '{spec}'. Use pep:CKPT or condpep:CKPT:generic|ec|pa|sa|all or ampdfm:CKPT[:generic]")

        label = _display_label(tag, ckpt_path, cond)

        # Score sequences ---------------------------------------------------
        print("Scoring …")
        enc_seqs = [encode_tokens(s) for s in seqs]
        max_len = max(len(t) for t in enc_seqs)
        pad_tok = SPECIALS["<pad>"]
        enc_padded = [t + [pad_tok]*(max_len - len(t)) for t in enc_seqs]
        toks_tensor = torch.tensor(enc_padded, dtype=torch.long, device=device)

        # -------- batched scoring to avoid GPU OOM --------------------------------
        def _batched_scores(judge, toks, batch_size=256):
            out_chunks = []
            for i in range(0, toks.size(0), batch_size):
                out_chunks.append(judge(toks[i:i+batch_size]).cpu())
            return torch.cat(out_chunks).numpy()

        pot_generic = _batched_scores(generic_judge, toks_tensor)
        pot_sa = _batched_scores(judge_sa,      toks_tensor)
        pot_pa = _batched_scores(judge_pa,      toks_tensor)
        pot_ec = _batched_scores(judge_ec,      toks_tensor)

        df = pd.DataFrame({
            "sequence": seqs,
            "pot_generic": pot_generic,
            "pot_sa": pot_sa,
            "pot_pa": pot_pa,
            "pot_ec": pot_ec,
        })
        df.to_csv(out_dir / f"{_slug(label)}_scores.csv", index=False)

        # Diversity ---------------------------------------------------------
        pct_unique = len(set(seqs)) / len(seqs) * 100
        dup_rate = 1.0 - (len(set(seqs)) / max(len(seqs), 1))
        shannon = positional_entropy(seqs)
        lev_mean = mean_levenshtein(seqs)

        # Perplexity --------------------------------------------------------
        print("Validation perplexity …")
        # Perplexity: pass a 4-bit cond_vec for conditional PepDFM so the
        # correct model architecture (CNNModel with cond_dim=4) is constructed.
        if tag.lower() in ("condpep", "pepdfm_cond") and cond is not None:
            _map = {"generic": [1,0,0,0], "ec": [1,1,0,0], "pa": [1,0,1,0], "sa": [1,0,0,1], "all": [1,1,1,1]}
            cond_key = cond.lower()
            if cond_key not in _map:
                raise ValueError(f"Unknown cond '{cond}' for condpep; expected one of {list(_map)}")
            cond_vec_for_ppl = _map[cond_key]
        else:
            cond_vec_for_ppl = None
        # Per-tag validation override (useful to evaluate original PepDFM on its own val split)
        val_override = val_map.get(tag, None)
        vp = val_override if val_override else val_path
        ppl = validation_perplexity(ckpt_path, vp, vocab_size, cond_vec_for_ppl, device)

        # Training-based diagnostics (optional) -----------------------------
        kl_aa = None
        kl_len = None
        exact_overlap = None
        mmseqs80_hits = None
        lev_to_train_avg = None
        lev_to_train_median = None
        lev_to_train_prop_lt5 = None
        aa_ent = None

        train_tag = tag  # use the provided tag to look up training data
        train_seqs: Optional[List[str]] = None
        if train_tag in train_cache:
            train_seqs = train_cache[train_tag]
        elif train_tag in fasta_map:
            try:
                train_seqs = _load_train_from_fasta(Path(fasta_map[train_tag]))
                train_cache[train_tag] = train_seqs
            except Exception:
                train_seqs = None
        elif train_tag in hf_map:
            try:
                path_str = hf_map[train_tag]
                train_seqs = _load_train_from_hf(path_str)
                train_cache[train_tag] = train_seqs
            except Exception as e:
                print(f"[WARN] Failed to load HF training dataset for {train_tag}: {e}")
                train_seqs = None

        if train_seqs:
            p_gen = aa_distribution(seqs)
            p_train = aa_distribution(train_seqs)
            kl_aa = kl_div(p_gen, p_train)
            aa_ent = shannon_entropy(p_gen)

            len_gen = length_distribution(seqs)
            len_train = length_distribution(train_seqs)
            if len_gen.size != len_train.size:
                max_len_arr = max(len_gen.size, len_train.size)
                len_gen = np.pad(len_gen, (0, max_len_arr - len_gen.size))
                len_train = np.pad(len_train, (0, max_len_arr - len_train.size))
            kl_len = kl_div(len_gen, len_train)

            exact_overlap = len(set(seqs) & set(train_seqs))
        else:
            print(f"[WARN] No training sequences available for {tag}; skipping KL/overlap/novelty diagnostics")

            # Skip Levenshtein-based novelty metrics when training sequences are missing.
            if train_seqs is not None and args.lev_samples and args.lev_samples > 0:
                rng = np.random.default_rng(0)
                gen_sample = seqs if len(seqs) <= args.lev_samples else list(rng.choice(seqs, size=args.lev_samples, replace=False))
                train_sample = train_seqs if len(train_seqs) <= 2000 else list(rng.choice(train_seqs, size=2000, replace=False))
                import Levenshtein  # type: ignore
                dists = [min(Levenshtein.distance(g, t) for t in train_sample) for g in gen_sample]
                if dists:
                    lev_to_train_avg = float(np.mean(dists))
                    lev_to_train_median = float(np.median(dists))
                    lev_to_train_prop_lt5 = float(np.mean(np.array(dists) < 5))

            if args.mmseqs:
                try:
                    tmp = Path(tempfile.mkdtemp())
                    gen_fa = tmp / "gen.fa"
                    with gen_fa.open("w") as f:
                        for i, s in enumerate(seqs, 1):
                            f.write(f">g{i}\n{s}\n")
                    db_gen = tmp / "gen_db"
                    db_train = tmp / "train_db"
                    # If we loaded from HF, dump a temporary training FASTA
                    if train_tag in fasta_map:
                        train_path = fasta_map[train_tag]
                    else:
                        train_tmp = tmp / "train.fa"
                        with train_tmp.open("w") as f:
                            for i, s in enumerate(train_seqs, 1):
                                f.write(f">t{i}\n{s}\n")
                        train_path = str(train_tmp)
                    subprocess.run(["mmseqs", "createdb", str(gen_fa), str(db_gen)], check=True)
                    subprocess.run(["mmseqs", "createdb", str(train_path), str(db_train)], check=True)
                    res = tmp / "res"
                    subprocess.run(["mmseqs", "easy-search", str(gen_fa), str(train_path), str(res), str(tmp/"tmp"),
                                    "--min-seq-id", "0.8", "-e", "1e-4"], check=True)
                    mmseqs80_hits = sum(1 for _ in Path(res).open())
                except Exception:
                    mmseqs80_hits = None

        # Hit counts --------------------------------------------------------
        hits_generic = int((pot_generic >= thr).sum())
        hits_sa = int((pot_sa >= thr).sum())
        hits_pa = int((pot_pa >= thr).sum())
        hits_ec = int((pot_ec >= thr).sum())

        summaries.append({
            "tag": tag,
            "label": label,
            "hits_generic": hits_generic,
            "hits_sa": hits_sa,
            "hits_pa": hits_pa,
            "hits_ec": hits_ec,
            "%unique": pct_unique,
            "duplicate_rate": dup_rate,
            "shannon": shannon,
            "positional_entropy": shannon,
            "lev_mean": lev_mean,
            "perplexity": ppl,
            "kl_aa": kl_aa,
            "kl_len": kl_len,
            "exact_overlap": exact_overlap,
            "mmseqs80_hits": mmseqs80_hits,
            "lev_to_train_avg": lev_to_train_avg,
            "lev_to_train_median": lev_to_train_median,
            "lev_to_train_prop_lt5": lev_to_train_prop_lt5,
            "aa_entropy": aa_ent,
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    # ------------------- Plot grouped bar ----------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = summary_df.get("label", summary_df["tag"]).tolist()
    idx = np.arange(len(tags))

    generic = summary_df["hits_generic"].values
    sa = summary_df["hits_sa"].values
    pa = summary_df["hits_pa"].values
    ec = summary_df["hits_ec"].values

    bar_w = 0.18
    plt.figure(figsize=(12, 6))

    plt.bar(idx - 1.5*bar_w, generic, width=bar_w, label="Generic", color="#4daf4a")
    plt.bar(idx - 0.5*bar_w, sa,      width=bar_w, label="SA",      color="#377eb8")
    plt.bar(idx + 0.5*bar_w, pa,      width=bar_w, label="PA",      color="#984ea3")
    plt.bar(idx + 1.5*bar_w, ec,      width=bar_w, label="EC",      color="#ff7f00")

    plt.xticks(idx, tags, rotation=45, ha="right")
    plt.ylabel(f"Hits (prob ≥{thr}) out of {args.n}")
    plt.title("AMP antimicrobial activity hits by model (grouped)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "antimicrobial_activity_hits.png", dpi=300)

    # Also dump JSON for quick inspection
    (out_dir / "summary_metrics.json").write_text(json.dumps(summaries, indent=2))
    print("Saved summary →", out_dir)


if __name__ == "__main__":
    main() 
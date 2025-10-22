from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    import Levenshtein  # type: ignore
    _HAS_LEV = True
except Exception:
    _HAS_LEV = False


# Canonical amino-acid order used across the project for distributions
AA_ORDER: str = "ACDEFGHIKLMNPQRSTVWY"


# Eisenberg hydrophobicity scale for hydrophobic moment
_EISENBERG: Dict[str, float] = {
    'A': 0.25, 'C': 0.04, 'D': -2.64, 'E': -2.62, 'F': 0.93,
    'G': 0.16, 'H': -0.40, 'I': 0.73, 'K': -1.50, 'L': 0.53,
    'M': 0.26, 'N': -0.78, 'P': -1.23, 'Q': -0.85, 'R': -2.53,
    'S': -0.26, 'T': -0.18, 'V': 0.54, 'W': 0.37, 'Y': 0.02,
}


# Kyte–Doolittle hydropathy index (for GRAVY)
_KYTEDOOLITTLE: Dict[str, float] = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3,
}


def aa_distribution(sequences: Iterable[str], aa_order: str = AA_ORDER) -> np.ndarray:
    counts: Dict[str, int] = {aa: 0 for aa in aa_order}
    total = 0
    for s in sequences:
        for ch in s:
            if ch in counts:
                counts[ch] += 1
                total += 1
    if total == 0:
        arr = np.zeros(len(aa_order), dtype=float)
    else:
        arr = np.asarray([counts[aa] / total for aa in aa_order], dtype=float)
    return arr


def length_distribution(sequences: Iterable[str]) -> np.ndarray:
    lengths = [len(s) for s in sequences]
    if not lengths:
        return np.array([1.0], dtype=float)
    max_len = max(lengths)
    counts = np.bincount(lengths, minlength=max_len + 1)
    return counts / counts.sum()


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p_c = np.clip(p, eps, 1)
    q_c = np.clip(q, eps, 1)
    return float(np.sum(p_c * np.log(p_c / q_c)))


def shannon_entropy(freq: np.ndarray, eps: float = 1e-12) -> float:
    f = np.clip(freq, eps, 1)
    return float(-np.sum(f * np.log2(f)))


def hydrophobic_moment(sequence: str, theta_degrees: float = 100.0) -> float:
    x_sum = 0.0
    y_sum = 0.0
    theta_rad = math.radians(theta_degrees)
    for i, aa in enumerate(sequence):
        h = _EISENBERG.get(aa, 0.0)
        angle = i * theta_rad
        x_sum += h * math.cos(angle)
        y_sum += h * math.sin(angle)
    n = len(sequence)
    return math.hypot(x_sum, y_sum) / n if n else 0.0


def gravy(sequence: str) -> float:
    if not sequence:
        return 0.0
    vals = [_KYTEDOOLITTLE.get(aa, 0.0) for aa in sequence]
    return float(np.mean(vals)) if vals else 0.0


def mean_levenshtein(sequences: List[str], n_pairs: int = 2000) -> float:
    if not _HAS_LEV:
        return float("nan")
    if not sequences:
        return float("nan")
    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(sequences), size=(n_pairs, 2))
    dists = [Levenshtein.distance(sequences[i], sequences[j]) for i, j in idxs]
    return float(np.mean(dists))


def compute_diversity_metrics(
    sequences: List[str],
    lev_pairs: int = 2000,
) -> Dict[str, Optional[float]]:
    if not sequences:
        return {
            "pct_unique": 0.0,
            "duplicate_rate": 0.0,
            "lev_mean": float("nan"),
        }

    pct_unique = len(set(sequences)) / len(sequences) * 100.0
    duplicate_rate = 1.0 - (len(set(sequences)) / max(len(sequences), 1))
    lev_mean = mean_levenshtein(sequences, n_pairs=lev_pairs) if lev_pairs and lev_pairs > 0 else float("nan")

    return {
        "pct_unique": float(pct_unique),
        "duplicate_rate": float(duplicate_rate),
        "lev_mean": float(lev_mean),
    }


__all__ = [
    "AA_ORDER",
    "aa_distribution",
    "length_distribution",
    "kl_div",
    "shannon_entropy",
    "hydrophobic_moment",
    "gravy",
    "mean_levenshtein",
    "compute_diversity_metrics",
]
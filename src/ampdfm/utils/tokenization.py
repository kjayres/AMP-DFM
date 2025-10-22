#!/usr/bin/env python3
"""Token mappings for AMP-DFM 24-token vocabulary"""

from __future__ import annotations

from typing import Dict, List

IDX_TO_AA: Dict[int, str] = {
    4: "A",
    5: "C",
    6: "D",
    7: "E",
    8: "F",
    9: "G",
    10: "H",
    11: "I",
    12: "K",
    13: "L",
    14: "M",
    15: "N",
    16: "P",
    17: "Q",
    18: "R",
    19: "S",
    20: "T",
    21: "V",
    22: "W",
    23: "Y",
}

AA_TO_IDX: Dict[str, int] = {aa: idx for idx, aa in IDX_TO_AA.items()}

CLS_IDX: int = 0
EOS_IDX: int = 2
AA_START_IDX: int = 4
AA_END_IDX: int = 23


def detokenise(tokens: List[int]) -> str:
    """Convert token ids to amino-acid string"""
    return "".join(IDX_TO_AA.get(t, "") for t in tokens if AA_START_IDX <= t <= AA_END_IDX)
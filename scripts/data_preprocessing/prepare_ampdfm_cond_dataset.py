"""Build the conditional AMP-DFM tokenised dataset with 4-bit conditioning vectors."""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
AMP_DFM_ROOT = PROJECT_ROOT / "amp_dfm"
DATA_DIR = AMP_DFM_ROOT / "data" / "clustered"

# Input tables
ACTIVITY_CSV = DATA_DIR / "activities_with_splits.csv"
NEGATIVE_CSVS = [
    DATA_DIR / "negatives_swissprot_with_splits.csv",
    DATA_DIR / "negatives_general_peptides_with_splits.csv",
    DATA_DIR / "negatives_uniprot_with_splits.csv",
]

CSV_PATHS = [ACTIVITY_CSV] + NEGATIVE_CSVS

# Output: conditional dataset (with cond_vec)
OUT_ROOT = AMP_DFM_ROOT / "data" / "dfm" / "tokenised_ampdfm_cond"

GROUP_SIZE = 12  # sequences per Arrow record

# Vocabulary: fixed 24-token scheme
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: Dict[str, int] = {aa: i + 4 for i, aa in enumerate(AA_ORDER)}
SPECIAL_TOKENS = {"<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3}

# Activity thresholds
POS_THRESHOLD_UGML = 32
NEG_THRESHOLD_UGML = 128


def _convert_ugml_to_um(ugml: float, mw_da: float) -> float:
    """Convert ug/mL to uM given molecular weight (Da)"""
    return (ugml / mw_da) * 1000


def label_activity_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return subset with definitive AMP/non-AMP labels using MIC thresholds"""
    df = df.copy()

    df["linear_value_um"] = 10 ** df["value"]
    df["pos_thresh_um"] = _convert_ugml_to_um(POS_THRESHOLD_UGML, df["mw_da"])
    df["neg_thresh_um"] = _convert_ugml_to_um(NEG_THRESHOLD_UGML, df["mw_da"])

    df["is_active"] = df["linear_value_um"] <= df["pos_thresh_um"]
    df["is_not_active"] = df["linear_value_um"] >= df["neg_thresh_um"]

    agg = (
        df.groupby("sequence")[["is_active", "is_not_active", "split"]]
        .agg({"is_active": "any", "is_not_active": "all", "split": "first"})
        .reset_index()
    )

    agg["is_amp"] = None
    agg.loc[agg["is_active"], "is_amp"] = 1
    agg.loc[agg["is_not_active"], "is_amp"] = 0
    labelled = agg.dropna(subset=["is_amp"]).copy()
    labelled["is_amp"] = labelled["is_amp"].astype(int)
    return labelled


def encode(seq: str) -> List[int]:
    """Convert AA string to token IDs"""
    toks = [SPECIAL_TOKENS["<cls>"]]
    toks.extend(AA_TO_IDX.get(res, SPECIAL_TOKENS["<unk>"]) for res in seq.upper())
    toks.append(SPECIAL_TOKENS["<eos>"])
    return toks


def pad(seq: List[int], max_len: int, pad_tok: int = SPECIAL_TOKENS["<pad>"]) -> List[int]:
    """Right-pad seq to max_len"""
    return seq + [pad_tok] * (max_len - len(seq))


# Activities
activity_rows = pd.read_csv(ACTIVITY_CSV)
df_activity = label_activity_rows(activity_rows)
print("Activity sequences kept:", len(df_activity))

# Curated negatives
neg_dfs = []
for neg_path in NEGATIVE_CSVS:
    df_neg = pd.read_csv(neg_path, usecols=["sequence", "split"])
    df_neg["is_amp"] = 0
    neg_dfs.append(df_neg)

# Build negatives
df_negatives = pd.concat(neg_dfs, ignore_index=True)
df_negatives = df_negatives.drop_duplicates(subset="sequence").reset_index(drop=True)
df_negatives["is_amp"] = 0

print("Unique negatives:", len(df_negatives))

df = pd.concat([df_activity, df_negatives], ignore_index=True)
df = df.drop_duplicates(subset=["sequence"]).copy()
print("Total unique sequences after merging:", len(df))

def _derive_cond_vec(is_amp: int, organisms: List[str]) -> List[int]:
    """Derive 4-bit conditioning vector from AMP flag and organism list"""
    cond = [0, 0, 0, 0]
    if not is_amp:
        return cond
    cond[0] = 1
    orgs_l = [str(o).lower() for o in organisms if isinstance(o, str)]
    cond[1] = int(any(("e. coli" in o) or ("escherichia coli" in o) for o in orgs_l))
    cond[2] = int(any(("p. aeruginosa" in o) or ("pseudomonas aeruginosa" in o) for o in orgs_l))
    cond[3] = int(any(("s. aureus" in o) or ("staphylococcus aureus" in o) for o in orgs_l))
    return cond
_tmp = activity_rows.copy()
_tmp["linear_value_um"] = 10 ** _tmp["value"]
_tmp["pos_thresh_um"] = _convert_ugml_to_um(POS_THRESHOLD_UGML, _tmp["mw_da"])
_tmp["is_active"] = _tmp["linear_value_um"] <= _tmp["pos_thresh_um"]

active_orgs_map = (
    _tmp[_tmp["is_active"]]
    .groupby("sequence")["organism"]
    .apply(lambda s: s.dropna().astype(str).unique().tolist())
    .to_dict()
)

df["cond_vec"] = [
    _derive_cond_vec(is_amp, active_orgs_map.get(seq, []))
    for seq, is_amp in zip(df["sequence"], df["is_amp"])
]

encoded: List[List[int]] = [
    encode(seq) for seq in tqdm(df["sequence"], total=len(df), ncols=80)
]
max_len = max(len(toks) for toks in encoded)
if max_len > 52:
    print(f"Note: Found sequence length {max_len-2} aa > 50. Padding to {max_len} tokens.")
print(f"Longest sequence (tokens incl. specials): {max_len}")

padded = [pad(toks, max_len) for toks in encoded]
mask = [[1] * len(toks) + [0] * (max_len - len(toks)) for toks in encoded]

df["input_ids"] = padded
df["attention_mask"] = mask
df["labels"] = padded
features = Features({
    "input_ids": Sequence(feature=Sequence(Value("int32"))),
    "attention_mask": Sequence(feature=Sequence(Value("int8"))),
    "labels": Sequence(feature=Sequence(Value("int32"))),
    "cond_vec": Sequence(feature=Sequence(Value("int8"))),
})

OUT_ROOT.mkdir(parents=True, exist_ok=True)

for split_name in ("train", "val", "test"):
    split_df = df[df["split"] == split_name]
    if split_df.empty:
        print(f"Note: No rows for split '{split_name}' - skipping.")
        continue
    records = []
    rows = split_df.to_dict("records")
    for i in range(0, len(rows), GROUP_SIZE):
        chunk = rows[i:i + GROUP_SIZE]
        records.append({
            "input_ids": [r["input_ids"] for r in chunk],
            "attention_mask": [r["attention_mask"] for r in chunk],
            "labels": [r["labels"] for r in chunk],
            "cond_vec": [r["cond_vec"] for r in chunk],
        })

    ds = Dataset.from_list(records, features=features)
    out_dir = OUT_ROOT / split_name
    print(f"Saving {len(ds):,} records -> {out_dir.relative_to(AMP_DFM_ROOT)}")
    ds.save_to_disk(out_dir.as_posix())
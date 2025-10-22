#!/usr/bin/env python3
"""prepare_ampdfm_uncond_dataset.py

Build an unconditional AMP-DFM dataset (no conditioning vectors).
Run via: qsub amp_dfm/scripts/hpc_cluster/prepare_ampdfm_uncond_dataset.sh
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm

POS_THRESHOLD_UGML = 32
NEG_THRESHOLD_UGML = 128


def _convert_ugml_to_um(ugml: float, mw_da: float) -> float:
    return (ugml / mw_da) * 1000


def label_activity_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return sequences with definitive AMP/non-AMP labels"""
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

    keep_mask = agg["is_active"] | agg["is_not_active"]
    return agg.loc[keep_mask, ["sequence", "split"]]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
AMP_DFM_ROOT = PROJECT_ROOT / "amp_dfm"
DATA_DIR = AMP_DFM_ROOT / "data" / "clustered"

# Source CSVs
ACTIVITY_CSV = DATA_DIR / "activities_with_splits.csv"
NEG_CSVS = [
    DATA_DIR / "negatives_swissprot_with_splits.csv",
    DATA_DIR / "negatives_general_peptides_with_splits.csv",
    DATA_DIR / "negatives_uniprot_with_splits.csv",
]
CSV_PATHS = [ACTIVITY_CSV] + NEG_CSVS

OUT_ROOT = AMP_DFM_ROOT / "data" / "dfm" / "tokenised_ampdfm_uncond"
GROUP_SIZE = 12
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: Dict[str, int] = {aa: i + 4 for i, aa in enumerate(AA_ORDER)}
SPECIALS = {"<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3}

def encode(seq: str) -> List[int]:
    """Encode amino-acid string to token IDs"""
    toks = [SPECIALS["<cls>"]]
    toks.extend(AA_TO_IDX.get(res, SPECIALS["<unk>"]) for res in seq.upper())
    toks.append(SPECIALS["<eos>"])
    return toks


def pad(seq: List[int], length: int) -> List[int]:
    return seq + [SPECIALS["<pad>"]] * (length - len(seq))

print("Reading input tables...")
print("Will save to:", OUT_ROOT)

raw_activity = pd.read_csv(ACTIVITY_CSV)
activity_df = label_activity_rows(raw_activity)
print("Activity sequences after antimicrobial_activity filter:", len(activity_df))

neg_dfs = []
for p in NEG_CSVS:
    df = pd.read_csv(p, usecols=["sequence", "split"]).copy()
    neg_dfs.append(df)

neg_df = pd.concat(neg_dfs, ignore_index=True)
print("Negatives:", len(neg_df))

full_df = pd.concat([activity_df, neg_df], ignore_index=True)
full_df = full_df.drop_duplicates("sequence").copy()
print("Total unique sequences:", len(full_df))

print("Tokenising...")
encoded = [encode(s) for s in tqdm(full_df["sequence"], ncols=80)]
max_len = max(len(t) for t in encoded)
if max_len > 52:
    print(f"WARNING: Found sequence length {max_len - 2} aa > 50. Padding to {max_len} tokens.")
print("Longest sequence (incl specials):", max_len)

padded = [pad(t, max_len) for t in encoded]
mask = [[1]*len(t) + [0]*(max_len - len(t)) for t in encoded]

full_df["input_ids"] = padded
full_df["attention_mask"] = mask
full_df["labels"] = padded
features = Features({
    "input_ids":      Sequence(feature=Sequence(Value("int32"))),
    "attention_mask": Sequence(feature=Sequence(Value("int8"))),
    "labels":         Sequence(feature=Sequence(Value("int32"))),
})

OUT_ROOT.mkdir(parents=True, exist_ok=True)

for split in ("train", "val", "test"):
    split_df = full_df[full_df["split"] == split]
    if split_df.empty:
        print(f"[WARN] No rows for split '{split}' - skipping.")
        continue

    records = []
    rows = split_df.to_dict("records")
    for i in range(0, len(rows), GROUP_SIZE):
        chunk = rows[i:i+GROUP_SIZE]
        records.append({
            "input_ids":      [r["input_ids"] for r in chunk],
            "attention_mask": [r["attention_mask"] for r in chunk],
            "labels":         [r["labels"] for r in chunk],
        })

    ds = Dataset.from_list(records, features=features)
    out_dir = OUT_ROOT / split
    print(f"Saving {len(ds):,} records -> {out_dir.relative_to(AMP_DFM_ROOT)}")
    ds.save_to_disk(out_dir.as_posix())

print("Done - unconditional dataset prepared at", OUT_ROOT.relative_to(AMP_DFM_ROOT))
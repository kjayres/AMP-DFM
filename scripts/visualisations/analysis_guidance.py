#!/usr/bin/env python3
"""analysis_guidance.py

Visualise the effect of multi-objective guidance on PepDFM sampling.
Produces three kinds of figures in the output directory:
1. UMAP scatter of guided vs. unconditional peptide embeddings.
2. Descriptor-space violin plots (|net charge|, length, GRAVY, hydro-moment).
3. Trajectory plots for a small batch of guided sequences showing how they
   move in embedding space during diffusion.

The script can *optionally* generate the unconditional baseline and the
small trajectory batch on the fly so you do not need separate jobs.

Example (inside mog-dfm conda env):
    python analysis_guidance.py \
        --guided_csv ampflow/results/mog/mog_samples_scores.csv \
        --ckpt ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt \
        --out_dir ampflow/results/visualisations \
        --device cuda:0
"""
from __future__ import annotations
import argparse, json, random, math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so that `import models.*` etc. work
# even when this script is executed from a sub-directory (ampflow/…)
# ---------------------------------------------------------------------------
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, EsmModel
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# PepDFM 24-token mapping
_IDX_TO_AA = {
    4:"A",5:"C",6:"D",7:"E",8:"F",9:"G",10:"H",
    11:"I",12:"K",13:"L",14:"M",15:"N",16:"P",17:"Q",
    18:"R",19:"S",20:"T",21:"V",22:"W",23:"Y",
}
AA_TO_IDX = {v:k for k,v in _IDX_TO_AA.items()}

# ---------------------------------------------------------------------------
# Helper – ESM-2 embedding
# ---------------------------------------------------------------------------
_ESM_MODEL: EsmModel | None = None
_TOKENIZER: AutoTokenizer | None = None
def esm_embed(seqs: List[str], device: str) -> np.ndarray:
    global _ESM_MODEL, _TOKENIZER
    if _ESM_MODEL is None:
        _ESM_MODEL = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
        _ESM_MODEL.eval()
        _TOKENIZER = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    enc = _TOKENIZER(seqs, return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        rep = _ESM_MODEL(**enc).last_hidden_state  # (B,L,1280)
        mask = enc["attention_mask"].unsqueeze(-1)
        seq_rep = (rep * mask).sum(1) / mask.sum(1)  # mean-pool
    return seq_rep.cpu().numpy()

# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

def descriptors(seq: str):
    pa = ProteinAnalysis(seq or "A")  # avoid 0-len
    net_c = abs(pa.charge_at_pH(7.0))
    gravy = pa.gravy()
    length = len(seq)
    # Biopython ≥1.83 changed amino_acids_percent from a method to a property.
    try:
        aa_dict = pa.amino_acids_percent()
    except TypeError:
        aa_dict = pa.amino_acids_percent
    hyd_moment = sum(aa_dict.values()) / 20.0
    return net_c, gravy, length, hyd_moment

# ---------------------------------------------------------------------------
# Unconditional sampling helper  (very light – no guidance)
# ---------------------------------------------------------------------------

def unconditional_sample(ckpt: Path, n: int, batches: int, length: int, device: str) -> List[str]:
    from models.peptide_classifiers import load_solver  # lazy import to keep deps minimal
    from flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver

    vocab_size = 24
    step = 1.0 / 100  # 100 Euler steps – match guided run
    solver: MixtureDiscreteEulerSolver = load_solver(str(ckpt), vocab_size, device)
    per = math.ceil(n / batches)
    seqs: List[str] = []
    for _ in tqdm(range(batches), desc="Uncond batches"):
        core = torch.randint(4, vocab_size, (per, length), device=device)
        x0 = torch.cat([
            torch.zeros((per,1), dtype=torch.long, device=device),
            core,
            torch.full((per,1), 2, dtype=torch.long, device=device)
        ], 1)
        x_fin = solver.sample(x0, step_size=step)
        from ampflow.ampdfm_scripts.pepdfm_mog import detokenise  # reuse util
        seqs.extend(detokenise(row) for row in x_fin.cpu().tolist())
    return seqs[:n]

# ---------------------------------------------------------------------------
# Trajectory sampling (small batch, guided, intermediates)
# ---------------------------------------------------------------------------

def guided_trajectories(ckpt: Path, device: str, score_models, length: int = 12, batch: int = 10):
    """Sample **guided** trajectories with the same three judges used elsewhere
    (potency, haemolysis, cytotox). We need actual score models – passing an
    empty list causes torch.cat() to fail downstream.
    """
    from argparse import Namespace
    from flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver
    from models.peptide_classifiers import load_solver

    vocab_size = 24
    solver: MixtureDiscreteEulerSolver = load_solver(str(ckpt), vocab_size, device)

    # `score_models` (judges) provided by caller
    if not score_models:
        raise ValueError("guided_trajectories expects a non-empty list of judge models; pass [pot_j, hml_j, cyt_j]")

    # minimal guidance args – same defaults as pepdfm_mog
    g_args = Namespace(T=100, beta=1.0, lambda_=1.0, Phi_init=0.9, num_div=4)
    step = 1.0 / g_args.T

    # random initial tokens ------------------------------------------------
    core = torch.randint(4, vocab_size, (batch, length), device=device)
    x0 = torch.cat([
        torch.zeros((batch,1), dtype=torch.long, device=device),
        core,
        torch.full((batch,1), 2, dtype=torch.long, device=device)
    ], 1)

    x_inter = solver.multi_guidance_sample(
        args=g_args,
        x_init=x0,
        step_size=step,
        return_intermediates=True,
        verbose=False,
        score_models=score_models,
        importance=[1.0,1.0,1.0],
    )  # (steps+1, B, L)
    return x_inter  # Tensor

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="PepDFM guidance visualisations")
    default_guided = "ampflow/results/mog/generic/mog_samples_scores.csv"
    p.add_argument("--guided_csv", default=default_guided, help="Path to guided sampling scores CSV (default: %(default)s)")
    p.add_argument("--ckpt", required=True, help="Path to unconditional checkpoint")
    p.add_argument("--out_dir", required=True, help="Directory to store figures and tables")
    # Default to CPU for debugging; pass --device cuda:0 when GPU resources are available
    # Default to first CUDA GPU; override with --device cpu for CPU-only runs
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_uncond", type=int, default=2000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load guided data
    guided = pd.read_csv(args.guided_csv)
    guided["src"] = "guided"

    # ------------------------------------------------------------------
    # Generate / load unconditional baseline
    uncond_fa = out_dir / "uncond_samples.fa"
    if uncond_fa.exists():
        # load sequences from existing FASTA
        seqs = []
        with uncond_fa.open() as fh:
            for line in fh:
                if not line.startswith(">"):
                    seqs.append(line.strip())
    else:
        print("Sampling unconditional baseline …")
        seqs = unconditional_sample(Path(args.ckpt), args.n_uncond, batches=10, length=12, device=args.device)
        with uncond_fa.open("w") as fh:
            for i,s in enumerate(seqs,1):
                fh.write(f">uncond_{i}\n{s}\n")
    baseline = pd.DataFrame({"sequence": seqs})
    baseline["src"] = "uncond"

    df = pd.concat([guided[["sequence","src"]], baseline], ignore_index=True)

    # ------------------------------------------------------------------
    # Embedding + UMAP
    print("Embedding peptides with ESM-2 …")
    emb = []
    chunk = 256
    for i in tqdm(range(0, len(df), chunk)):
        emb.append(esm_embed(df.sequence.iloc[i:i+chunk].tolist(), args.device))
    emb = np.vstack(emb)

    import umap
    proj = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0).fit_transform(emb)
    df["x"], df["y"] = proj[:,0], proj[:,1]

    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df, x="x", y="y", hue="src", alpha=0.6, s=15)
    plt.title("UMAP of ESM-2 embeddings")
    plt.legend(title="set")
    plt.tight_layout()
    plt.savefig(out_dir/"umap_guided_vs_uncond.png", dpi=300)

    # ------------------------------------------------------------------
    # Property densities (potency, hemolysis, cytotox)
    # ------------------------------------------------------------------
    print("Scoring unconditional baseline …")
    from ampflow.ampdfm_scripts.pepdfm_mog import PotencyJudge, HemolysisJudge, CytotoxicityJudge
    pot_j = PotencyJudge(args.device)
    hml_j = HemolysisJudge(args.device)
    cyt_j = CytotoxicityJudge(args.device)

    if not set(["potency","hemolysis","cytotox"]).issubset(guided.columns):
        raise RuntimeError("guided CSV must contain potency/hemolysis/cytotox columns")

    # score baseline in chunks
    pot_list, hml_list, cyt_list = [], [], []
    for i in tqdm(range(0,len(baseline), chunk)):
        toks = baseline.sequence.iloc[i:i+chunk].tolist()
        # encode to PepDFM tokens using same mapping as detokenise, reversed
        AA_TO_IDX = {v:k for k,v in globals().get("_IDX_TO_AA", {}).items()}
        def encode(seq):
            toks = [0]  # <cls>
            toks += [AA_TO_IDX.get(res.upper(),3) for res in seq]
            toks.append(2)  # <eos>
            return toks
        tok_batch = [encode(s) for s in toks]
        max_len = max(len(t) for t in tok_batch)
        padded = [t + [1]*(max_len-len(t)) for t in tok_batch]
        x = torch.tensor(padded, device=args.device)
        pot_list.extend(pot_j(x).cpu().tolist())
        hml_list.extend(hml_j(x).cpu().tolist())
        cyt_list.extend(cyt_j(x).cpu().tolist())
    baseline["potency"] = pot_list
    baseline["hemolysis"] = hml_list
    baseline["cytotox"]  = cyt_list

    prop_df = pd.concat([guided[["potency","hemolysis","cytotox","src"]], baseline[["potency","hemolysis","cytotox","src"]]])

    for prop,label in [("potency","Predicted Potency"), ("hemolysis","Predicted Hemolysis"), ("cytotox","Predicted Cytotoxicity")]:
        plt.figure(figsize=(4,4))
        for src,col in [("guided","tab:pink"),("uncond","tab:blue")]:
            subset = prop_df[prop_df.src==src][prop]
            sns.kdeplot(subset, fill=True, alpha=0.4, color=col, label=src, clip=(0,1))
            plt.axvline(subset.mean(), color=col, linestyle="--")
        plt.xlabel(label); plt.ylabel("Density"); plt.legend()
        plt.tight_layout(); plt.savefig(out_dir/f"kde_{prop}.png", dpi=300); plt.close()

    # ------------------------------------------------------------------
    # Descriptor distribution shifts
    print("Computing descriptors …")
    desc_vals = np.array([descriptors(s) for s in df.sequence])
    df["net_charge"]  = desc_vals[:,0]
    df["gravy"]       = desc_vals[:,1]
    df["length"]      = desc_vals[:,2]
    df["hyd_moment"]  = desc_vals[:,3]

    for metric, label in [
        ("net_charge", "|Net charge| at pH 7"),
        ("gravy", "GRAVY"),
        ("length", "Peptide length"),
        ("hyd_moment", "Hydrophobic moment (proxy)")]:
        plt.figure(figsize=(4,4))
        sns.violinplot(data=df, x="src", y=metric, cut=0, inner="quartile")
        plt.ylabel(label); plt.xlabel("")
        plt.tight_layout()
        plt.savefig(out_dir/f"violin_{metric}.png", dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # Trajectories (small batch)
    print("Sampling 10 guided trajectories …")
    traj = guided_trajectories(Path(args.ckpt), args.device, [pot_j, hml_j, cyt_j])  # (steps+1, B, L)
    steps, B, L = traj.shape
    from ampflow.ampdfm_scripts.pepdfm_mog import detokenise
    traj_seqs = [[detokenise(traj[t,i].cpu().tolist()) for t in range(steps)] for i in range(B)]

    # embed all frames
    flat = [s for path in traj_seqs for s in path]
    traj_emb = []
    for i in tqdm(range(0,len(flat),chunk)):
        traj_emb.append(esm_embed(flat[i:i+chunk], args.device))
    traj_emb = np.vstack(traj_emb)
    traj_proj = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0).fit_transform(traj_emb)
    traj_proj = traj_proj.reshape(B, steps, 2)

    plt.figure(figsize=(6,5))
    for i in range(B):
        plt.plot(traj_proj[i,:,0], traj_proj[i,:,1], marker="o", lw=1)
    plt.title("Guided diffusion trajectories (UMAP)")
    plt.tight_layout()
    plt.savefig(out_dir/"trajectories_umap.png", dpi=300)

    # Property evolution over iterations
    print("Scoring trajectories …")
    means = {"potency":[],"hemolysis":[],"cytotox":[]}
    for t in range(steps):
        x_t = traj[t].to(args.device)
        means["potency"].append(pot_j(x_t).mean().item())
        means["hemolysis"].append(hml_j(x_t).mean().item())
        means["cytotox"].append(cyt_j(x_t).mean().item())
    iters = list(range(steps))
    for prop,label,col in [("potency","Potency","tab:green"),("hemolysis","Hemolysis","tab:red"),("cytotox","Cytotoxicity","tab:purple")]:
        plt.figure(figsize=(4,3))
        plt.plot(iters, means[prop], color=col)
        plt.xlabel("Euler iteration"); plt.ylabel(f"Mean {label}")
        plt.tight_layout(); plt.savefig(out_dir/f"curve_{prop}.png", dpi=300); plt.close()

    # save metadata
    with open(out_dir/"meta.json", "w") as fh:
        json.dump({"guided_csv": args.guided_csv, "uncond_fasta": str(uncond_fa)}, fh, indent=2)
    print("Done – figures written to", out_dir)

if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse, math, random, json, tempfile, subprocess, gzip
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../amp_dfm
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def raw_hf_collate(batch):
    out = {}
    for key in batch[0]:
        out[key] = [item[key] for item in batch]
    return out

from ampdfm.dfm.models.peptide_models import CNNModel
from ampdfm.dfm.models.peptide_models import CNNModelPep
from ampdfm.dfm.flow_matching.path import MixtureDiscreteProbPath
from ampdfm.dfm.flow_matching.path.scheduler import PolynomialConvexScheduler
from ampdfm.dfm.flow_matching.solver import MixtureDiscreteEulerSolver
from ampdfm.dfm.flow_matching.utils import ModelWrapper

from ampdfm.utils.tokenization import detokenise, CLS_IDX, EOS_IDX
from ampdfm.utils.esm_embed import get_esm_embeddings
from ampdfm.classifiers.inference import EmbeddedBooster
from ampdfm.evaluation import compute_diversity_metrics
from ampdfm.dfm.flow_matching.loss import MixturePathGeneralizedKL
import yaml

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int)
    p.add_argument("--out_dir", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--amp_val_path", type=str)
    p.add_argument("--pep_val_path", type=str)
    p.add_argument("--lev_samples", type=int)
    p.add_argument("--no_mmseqs", action="store_true")
    p.add_argument("--train_fasta_map", nargs="*", default=[])
    p.add_argument("--train_hf_map", nargs="*", default=[])
    p.add_argument("--val_map", nargs="*", default=[])
    return p.parse_args()


def sample_ampdfm(ckpt: Path, n_samples: int, device: str, seed: int) -> List[str]:
    random.seed(seed); torch.manual_seed(seed)
    state = torch.load(ckpt, map_location="cpu")
    embed_dim = state["token_embedding.weight"].shape[1]
    hidden_dim = state["linear.weight"].shape[0]
    
    model = CNNModelPep(alphabet_size=24, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    class Wrapped(ModelWrapper):
        def forward(self, x, t, **kw):
            return torch.softmax(self.model(x, t), dim=-1)

    wrapped = Wrapped(model)
    path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=2.0))
    solver = MixtureDiscreteEulerSolver(model=wrapped, path=path, vocabulary_size=24)

    collected = []
    lengths = list(range(6, 51))
    p_len = np.ones(len(lengths)) / len(lengths)

    with torch.no_grad():
        while len(collected) < n_samples:
            cur = min(256, n_samples - len(collected))
            length = random.choices(lengths, weights=p_len)[0]
            core = torch.randint(4, 24, (cur, length), device=device)
            x_init = torch.cat([
                torch.full((cur, 1), CLS_IDX, dtype=core.dtype, device=device),
                core,
                torch.full((cur, 1), EOS_IDX, dtype=core.dtype, device=device)
            ], dim=1)
            sol = solver.sample(x_init=x_init, step_size=0.01,
                                time_grid=torch.tensor([0.0, 0.999], device=device))
            collected.extend(detokenise(seq) for seq in sol.tolist())
    return collected[:n_samples]


def sample_conditional_ampdfm(ckpt: Path, cond: str, n_samples: int, device: str, seed: int) -> List[str]:
    random.seed(seed); torch.manual_seed(seed)
    cond_map = {"generic": [1,0,0,0], "ec": [1,1,0,0], "pa": [1,0,1,0], "sa": [1,0,0,1], "all": [1,1,1,1]}
    cond_vec = torch.tensor(cond_map[cond.lower()], dtype=torch.float32, device=device).unsqueeze(0)

    state = torch.load(ckpt, map_location="cpu")
    embed_dim = state["token_embedding.weight"].shape[1]
    hidden_dim = state["linear.weight"].shape[0]
    cond_dim = state["cond_proj.weight"].shape[1]
    
    model = CNNModel(alphabet_size=24, embed_dim=embed_dim, hidden_dim=hidden_dim, cond_dim=cond_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    class Wrapped(ModelWrapper):
        def forward(self, x, t, **kw):
            return torch.softmax(self.model(x, t, cond_vec=cond_vec.expand(x.shape[0], -1)), dim=-1)

    wrapped = Wrapped(model)
    path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=2.0))
    solver = MixtureDiscreteEulerSolver(model=wrapped, path=path, vocabulary_size=24)

    collected = []
    lengths = list(range(6, 51))
    p_len = np.ones(len(lengths)) / len(lengths)

    with torch.no_grad():
        while len(collected) < n_samples:
            cur = min(256, n_samples - len(collected))
            length = random.choices(lengths, weights=p_len)[0]
            core = torch.randint(4, 24, (cur, length), device=device)
            x_init = torch.cat([
                torch.full((cur, 1), CLS_IDX, dtype=core.dtype, device=device),
                core,
                torch.full((cur, 1), EOS_IDX, dtype=core.dtype, device=device)
            ], dim=1)
            sol = solver.sample(x_init=x_init, step_size=0.01,
                                time_grid=torch.tensor([0.0, 0.999], device=device))
            collected.extend(detokenise(seq) for seq in sol.tolist())
    return collected[:n_samples]


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
    try:
        keys = list(ds_loaded.keys())
        datasets = [ds_loaded["train"]] if "train" in keys else [ds_loaded[k] for k in keys]
    except:
        datasets = [ds_loaded]

    seqs = []
    for ds in datasets:
        for rec in ds:
            ids = rec.get("input_ids")
            if not ids:
                continue
            flat = [int(x) for sub in ids for x in sub] if isinstance(ids[0], (list, tuple)) else [int(x) for x in ids]
            if max(flat) >= 24:
                seqs.append(detokenise([t - 2 for t in flat]))
            else:
                seqs.append(detokenise(flat))
    return seqs


def _parse_tag_map(items: List[str]) -> Dict[str, str]:
    out = {}
    for it in items:
        if ":" in it:
            tag, path = it.split(":", 1)
            out[tag] = path
    return out

def validation_perplexity(ckpt: Path, val_path: str, cond_vec: Optional[List[int]], device: str) -> float:
    val_ds = load_from_disk(val_path)
    loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, collate_fn=raw_hf_collate)

    state = torch.load(ckpt, map_location="cpu")
    embed_dim = state["token_embedding.weight"].shape[1]
    hidden_dim = state["linear.weight"].shape[0]

    if cond_vec:
        model = CNNModel(alphabet_size=24, embed_dim=embed_dim, hidden_dim=hidden_dim, cond_dim=len(cond_vec)).to(device)
    else:
        model = CNNModelPep(alphabet_size=24, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=2.0))
    loss_fn = MixturePathGeneralizedKL(path=path)

    def _to_long_tensor(raw):
        if torch.is_tensor(raw):
            return raw.to(device)
        flat = [seq for group in raw for seq in group] if raw and isinstance(raw[0], (list, tuple)) and isinstance(raw[0][0], (list, tuple)) else raw
        if flat and torch.is_tensor(flat[0]):
            return torch.stack([t.to(device) for t in flat])
        if flat and isinstance(flat[0], (list, tuple)):
            max_len = max(len(seq) for seq in flat)
            return torch.tensor([seq + [0]*(max_len - len(seq)) for seq in flat], dtype=torch.long, device=device)
        return torch.tensor(flat, dtype=torch.long, device=device)

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            x_1 = _to_long_tensor(batch["input_ids"])
            cond = _to_long_tensor(batch["cond_vec"]) if cond_vec else None
            x_0 = torch.randint_like(x_1, high=24)
            x_0[:, 0] = x_1[:, 0]
            t = torch.rand(x_1.shape[0], device=device) * 0.999
            sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            logits = model(sample.x_t, sample.t, cond_vec=cond.float()) if cond_vec else model(sample.x_t, sample.t)
            loss = loss_fn(logits=logits, x_1=x_1, x_t=sample.x_t, t=sample.t)
            tok_count = ((x_1 != 0) & (x_1 != 1)).sum().item()
            total_loss += loss.item() * tok_count
            total_tokens += tok_count
    return math.exp(total_loss / max(total_tokens, 1))

def _resolve_antimicrobial_activity_model(species_folder: str) -> Path:
    folder = species_folder.lower()
    candidates = [
        Path("/rds/general/user/kja24/home/amp_dfm/outputs/classifiers") / "antimicrobial_activity" / folder / "model.json",
        Path("/rds/general/user/kja24/home/amp_dfm/checkpoints/classifiers") / "antimicrobial_activity" / folder / "model.json",
    ]

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    models_specs = cfg["models"]
    n_samples = args.n if args.n else cfg.get("n", 10000)
    out_dir = Path("/rds/general/user/kja24/home/amp_dfm/outputs/model_comparison")
    seed = args.seed if args.seed else cfg.get("seed", 0)
    amp_val_path = args.amp_val_path or cfg["data"]["amp_val_path"]
    pep_val_path = args.pep_val_path or cfg["data"]["pep_val_path"]
    lev_samples = args.lev_samples if args.lev_samples else n_samples
    mmseqs_flag = not args.no_mmseqs

    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    generic_classifier = EmbeddedBooster(_resolve_antimicrobial_activity_model("generic"), device=device)
    classifier_sa = EmbeddedBooster(_resolve_antimicrobial_activity_model("staphylococcus_aureus"), device=device)
    classifier_pa = EmbeddedBooster(_resolve_antimicrobial_activity_model("pseudomonas_aeruginosa"), device=device)
    classifier_ec = EmbeddedBooster(_resolve_antimicrobial_activity_model("escherichia_coli"), device=device)

    thr = 0.8
    summaries = []
    fasta_map = _parse_tag_map(args.train_fasta_map)
    hf_map = _parse_tag_map(args.train_hf_map)
    val_map = _parse_tag_map(args.val_map)
    train_cache = {}

    def _display_label(tag_raw, ckpt_path, cond):
        t = tag_raw.lower()
        if t == "ampdfm_uncond":
            return "AMP-DFM (Unconditional)"
        if t == "ampdfm_cond":
            return "AMP-DFM (Conditional)"
        return tag_raw

    def _slug(s):
        return s.replace(" ", "_").replace("(", "_").replace(")", "_").replace("/", "_").replace(",", "_")

    for spec in models_specs:
        parts = spec.split(":")
        tag, ckpt_path = parts[0], Path(parts[1])
        cond = parts[2] if len(parts) == 3 else None

        print(f"\n=== Sampling {tag} ===")
        if tag.lower() == "ampdfm_uncond":
            seqs = sample_ampdfm(ckpt_path, n_samples, device, seed)
            val_path = pep_val_path
        else:
            seqs = sample_conditional_ampdfm(ckpt_path, cond, n_samples, device, seed)
            val_path = amp_val_path

        label = _display_label(tag, ckpt_path, cond)

        print("Scoring")
        embs = get_esm_embeddings(seqs, device=device)
        amp_generic = generic_classifier.predict_from_embeddings(embs)
        amp_sa = classifier_sa.predict_from_embeddings(embs)
        amp_pa = classifier_pa.predict_from_embeddings(embs)
        amp_ec = classifier_ec.predict_from_embeddings(embs)

        pd.DataFrame({"sequence": seqs, "amp_generic": amp_generic, "amp_sa": amp_sa,
                      "amp_pa": amp_pa, "amp_ec": amp_ec}).to_csv(out_dir / f"{_slug(label)}_scores.csv", index=False)

        metrics = compute_diversity_metrics(seqs)
        pct_unique = metrics["pct_unique"]
        dup_rate = metrics["duplicate_rate"]
        shannon = metrics["positional_entropy"]
        lev_mean = metrics["lev_mean"]

        print("Validation perplexity")
        cond_map = {"generic": [1,0,0,0], "ec": [1,1,0,0], "pa": [1,0,1,0], "sa": [1,0,0,1], "all": [1,1,1,1]}
        cond_vec_for_ppl = cond_map[cond.lower()] if cond else None
        vp = val_map.get(tag, val_path)
        ppl = validation_perplexity(ckpt_path, vp, cond_vec_for_ppl, device)

        kl_aa = kl_len = exact_overlap = mmseqs80_hits = None
        lev_to_train_avg = lev_to_train_median = lev_to_train_prop_lt5 = aa_ent = None

        train_seqs = None
        if tag in train_cache:
            train_seqs = train_cache[tag]
        elif tag in fasta_map:
            try:
                train_seqs = _load_train_from_fasta(Path(fasta_map[tag]))
                train_cache[tag] = train_seqs
            except:
                pass
        elif tag in hf_map:
            try:
                train_seqs = _load_train_from_hf(hf_map[tag])
                train_cache[tag] = train_seqs
            except:
                pass

        if train_seqs:
            tmetrics = compute_diversity_metrics(seqs, train_sequences=train_seqs, lev_pairs=0)
            kl_aa = tmetrics["kl_aa"]
            kl_len = tmetrics["kl_len"]
            aa_ent = tmetrics["aa_entropy"]
            exact_overlap = len(set(seqs) & set(train_seqs))

            if lev_samples:
                import Levenshtein
                rng = np.random.default_rng(0)
                gen_sample = seqs if len(seqs) <= lev_samples else list(rng.choice(seqs, size=lev_samples, replace=False))
                train_sample = train_seqs if len(train_seqs) <= 2000 else list(rng.choice(train_seqs, size=2000, replace=False))
                dists = [min(Levenshtein.distance(g, t) for t in train_sample) for g in gen_sample]
                lev_to_train_avg = float(np.mean(dists))
                lev_to_train_median = float(np.median(dists))
                lev_to_train_prop_lt5 = float(np.mean(np.array(dists) < 5))

            if mmseqs_flag:
                try:
                    tmp = Path(tempfile.mkdtemp())
                    gen_fa = tmp / "gen.fa"
                    gen_fa.write_text("\n".join(f">g{i}\n{s}" for i, s in enumerate(seqs, 1)))
                    train_path = fasta_map[tag] if tag in fasta_map else tmp / "train.fa"
                    if tag not in fasta_map:
                        train_path.write_text("\n".join(f">t{i}\n{s}" for i, s in enumerate(train_seqs, 1)))
                    subprocess.run(["mmseqs", "createdb", str(gen_fa), str(tmp / "gen_db")], check=True)
                    subprocess.run(["mmseqs", "createdb", str(train_path), str(tmp / "train_db")], check=True)
                    res = tmp / "res"
                    subprocess.run(["mmseqs", "easy-search", str(gen_fa), str(train_path), str(res), str(tmp/"tmp"),
                                    "--min-seq-id", "0.8", "-e", "1e-4"], check=True)
                    mmseqs80_hits = sum(1 for _ in res.open())
                except:
                    pass

        hits_generic = int((amp_generic >= thr).sum())
        hits_sa = int((amp_sa >= thr).sum())
        hits_pa = int((amp_pa >= thr).sum())
        hits_ec = int((amp_ec >= thr).sum())

        summaries.append({"tag": tag, "label": label, "hits_generic": hits_generic, "hits_sa": hits_sa,
            "hits_pa": hits_pa, "hits_ec": hits_ec, "%unique": pct_unique, "duplicate_rate": dup_rate,
            "shannon": shannon, "positional_entropy": shannon, "lev_mean": lev_mean, "perplexity": ppl,
            "kl_aa": kl_aa, "kl_len": kl_len, "exact_overlap": exact_overlap, "mmseqs80_hits": mmseqs80_hits,
            "lev_to_train_avg": lev_to_train_avg, "lev_to_train_median": lev_to_train_median,
            "lev_to_train_prop_lt5": lev_to_train_prop_lt5, "aa_entropy": aa_ent})

    (out_dir / "summary_metrics.json").write_text(json.dumps(summaries, indent=2))
    print("Saved summary →", out_dir)


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""ampdfm_conditional_finetune.py

Fine-tune the ampdfm CNN with a 4-bit conditioning vector [AMP, EC, PA, SA]
on the same tokenised dataset produced by `prepare_ampdfm_dataset.py` which now
includes a `cond_vec` field. Starts from the unconditional checkpoint to keep
weights and adds a small conditioning projection.

Outputs: ampflow/ampdfm_ckpt/ampdfm_conditional_finetuned.ckpt
"""
from __future__ import annotations

import random
from pathlib import Path
import argparse
import yaml

import torch
from datasets import load_from_disk
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# Ensure project src is on PYTHONPATH when invoked via batch jobs
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../amp_dfm
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ampdfm.dfm.flow_matching.path import MixtureDiscreteProbPath
from ampdfm.dfm.flow_matching.path.scheduler import PolynomialConvexScheduler
from ampdfm.dfm.flow_matching.loss import MixturePathGeneralizedKL

from ampdfm.dfm.models.peptide_models import CNNModel
from ampdfm.utils import dataloader

# Seeding configured via YAML (training.seed); applied after config load

parser = argparse.ArgumentParser(description="Fine-tune AMP-DFM with 4-bit conditioning vector")
parser.add_argument("--config", required=True, help="Path to YAML config file")
parser.add_argument("--amp_only", action="store_true", help="Keep only AMP-positive sequences (any cond bit = 1)")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Force device (default: cuda)")
parser.add_argument("--source_dist", choices=["uniform", "mask"], default="uniform",
                    help="Source distribution for x_0 (default: uniform)")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

TRAIN_PATH = cfg["data"]["train_path"]
VAL_PATH   = cfg["data"]["val_path"]

INIT_CKPT = cfg["init"]["ckpt"]
CKPT_OUT  = cfg["output"]["ckpt_out"]
Path(CKPT_OUT).parent.mkdir(parents=True, exist_ok=True)

# Architectural constants
vocab_size  = 24
embed_dim   = 1024
hidden_dim  = 512

lr            = float(cfg.get("training", {}).get("lr", 5e-5))
epochs        = int(cfg.get("training", {}).get("epochs", 50))
warmup_epochs = int(cfg.get("training", {}).get("warmup_epochs", max(1, epochs // 10)))
batch_size    = int(cfg.get("training", {}).get("batch_size", 512))
epsilon       = float(cfg.get("training", {}).get("epsilon", 1e-3))
source_dist   = args.source_dist
poly_n        = float(cfg.get("scheduler", {}).get("polynomial_n", 2.0))

# Apply seeds from YAML
seed = int(cfg.get("training", {}).get("seed", 42))
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Resolve device after parsing; default to CPU unless explicitly set to CUDA and available
if args.device == "cuda" and torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

train_ds = load_from_disk(TRAIN_PATH)
val_ds   = load_from_disk(VAL_PATH)

if args.amp_only:
    def keep_amp(record):
        return any(sum(vec) > 0 for vec in record["cond_vec"])  # record-level gate
    train_ds = train_ds.filter(keep_amp)
    val_ds   = val_ds.filter(keep_amp)

def collate_amp_filter(batch):
    # Reuse base collate then drop negatives if amp_only
    out = dataloader.collate_fn(batch)
    if args.amp_only:
        mask = (out["cond_vec"].sum(dim=1) > 0)
        out["input_ids"] = out["input_ids"][mask]
        out["attention_mask"] = out["attention_mask"][mask]
        out["cond_vec"] = out["cond_vec"][mask]
    return out

data_module = dataloader.CustomDataModule(train_ds, val_ds, test_dataset=None, collate_fn=collate_amp_filter)
train_loader = data_module.train_dataloader()
val_loader   = data_module.val_dataloader()

model = CNNModel(alphabet_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, cond_dim=4).to(device)

# Load unconditional weights into matching submodules (token/time/conv). The
# new cond_proj is randomly initialised.
try:
    sd = torch.load(INIT_CKPT, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded init checkpoint with missing keys:", missing, "unexpected:", unexpected)
except Exception as e:
    print("[WARN] Could not load init checkpoint:", e)

path    = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=poly_n))
loss_fn = MixturePathGeneralizedKL(path=path)
optim   = torch.optim.Adam(model.parameters(), lr=lr)

warmup_lambda = lambda ep: 0.1 + 0.9 * ep / warmup_epochs if ep < warmup_epochs else 1.0
warmup_sched  = LambdaLR(optim, lr_lambda=warmup_lambda)
cosine_sched  = CosineAnnealingLR(optim, T_max=epochs - warmup_epochs, eta_min=0.1*lr)

def general_step(x_1: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
    if source_dist == "uniform":
        x_0 = torch.randint_like(x_1, high=vocab_size)
        x_0[:, 0] = x_1[:, 0]
    elif source_dist == "mask":
        x_0 = torch.zeros_like(x_1) + 3
        x_0[:, 0] = x_1[:, 0]
    else:
        raise NotImplementedError

    t      = torch.rand(x_1.size(0), device=device) * (1 - epsilon)
    sample = path.sample(t=t, x_0=x_0, x_1=x_1)
    logits = model(sample.x_t, sample.t, cond_vec=cond_vec)
    loss   = loss_fn(logits=logits, x_1=x_1, x_t=sample.x_t, t=sample.t)
    return loss

print("Starting conditional fine-tune …")
best_val = float("inf")
for epoch in range(epochs):
    model.train()
    train_losses = []
    for batch in train_loader:
        x_1 = batch["input_ids"].to(device).view(-1, batch["input_ids"].shape[-1])
        cond = batch["cond_vec"].to(device).view(-1, batch["cond_vec"].shape[-1])
        optim.zero_grad()
        loss = general_step(x_1, cond)
        loss.backward()
        optim.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x_1 = batch["input_ids"].to(device).view(-1, batch["input_ids"].shape[-1])
            cond = batch["cond_vec"].to(device).view(-1, batch["cond_vec"].shape[-1])
            loss = general_step(x_1, cond)
            val_losses.append(loss.item())

    mean_train = sum(train_losses)/len(train_losses)
    mean_val   = sum(val_losses)/len(val_losses)

    if mean_val < best_val:
        best_val = mean_val
        torch.save(model.state_dict(), CKPT_OUT)
        print(f"[Epoch {epoch}] New best ValLoss {mean_val:.4f} – saved → {CKPT_OUT}")

    if epoch < warmup_epochs:
        warmup_sched.step()
    else:
        cosine_sched.step()

    if epoch % 5 == 0 or epoch == epochs-1:
        print(f"Epoch {epoch:03d}: train {mean_train:.4f}  val {mean_val:.4f}")

print("Fine-tune complete. Best ValLoss:", best_val)



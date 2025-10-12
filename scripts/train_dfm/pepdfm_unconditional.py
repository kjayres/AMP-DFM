#!/usr/bin/env python3
"""pepdfm_unconditional.py

Train a *PepDFM-sized* CNN model **without conditioning** on the full AMP
corpus that we converted to 24-token format in
`ampflow/ampdfm_data/tokenized_pep/`.

The hyper-parameters match the original PepDFM paper:
    embed = 512, hidden = 256, 5 convolutional blocks, 24-token vocab.

Output checkpoint:  ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL

from models.peptide_models import CNNModelPep
from utils import dataloader

# ------------------------- Reproducibility ---------------------------------
random.seed(42)
torch.manual_seed(0)

# ------------------------- Paths & config ----------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = "./ampflow/ampdfm_data/tokenized_pep/train"
VAL_PATH   = "./ampflow/ampdfm_data/tokenized_pep/val"

CKPT_OUT = "./ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt"
Path(CKPT_OUT).parent.mkdir(parents=True, exist_ok=True)

vocab_size  = 24   # 0:<cls>,1:<pad>,2:<eos>,3:<unk>,4–23 AA
embed_dim   = 1024  # widened embedding dimension
hidden_dim  = 512   # wider convolutional channels

lr            = 1e-4
epochs        = 200
warmup_epochs = epochs // 10
batch_size    = 512
epsilon       = 1e-3
source_dist   = "uniform"  # same as original training

# ------------------------- Data -------------------------------------------
train_ds = load_from_disk(TRAIN_PATH)
val_ds   = load_from_disk(VAL_PATH)

data_module = dataloader.CustomDataModule(train_ds, val_ds, test_dataset=None)
train_loader = data_module.train_dataloader()
val_loader   = data_module.val_dataloader()

# ------------------------- Model & loss -----------------------------------
model = CNNModelPep(alphabet_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)

path      = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=2.0))
loss_fn   = MixturePathGeneralizedKL(path=path)
optim     = torch.optim.Adam(model.parameters(), lr=lr)

warmup_lambda = lambda ep: 0.1 + 0.9 * ep / warmup_epochs if ep < warmup_epochs else 1.0
warmup_sched  = LambdaLR(optim, lr_lambda=warmup_lambda)
cosine_sched  = CosineAnnealingLR(optim, T_max=epochs - warmup_epochs, eta_min=0.1*lr)

# ------------------------- Training step ----------------------------------

def general_step(x_1: torch.Tensor) -> torch.Tensor:
    if source_dist == "uniform":
        x_0 = torch.randint_like(x_1, high=vocab_size)
        x_0[:, 0] = x_1[:, 0]  # keep <cls> fixed
    elif source_dist == "mask":
        x_0 = torch.zeros_like(x_1) + 3  # <unk>
        x_0[:, 0] = x_1[:, 0]
    else:
        raise NotImplementedError

    t      = torch.rand(x_1.size(0), device=device) * (1 - epsilon)
    sample = path.sample(t=t, x_0=x_0, x_1=x_1)
    logits = model(sample.x_t, sample.t)
    loss   = loss_fn(logits=logits, x_1=x_1, x_t=sample.x_t, t=sample.t)
    return loss

# ------------------------- Training loop ----------------------------------
print("Starting PepDFM-size unconditional training …")

best_val = float("inf")
for epoch in range(epochs):
    model.train()
    train_losses = []
    for batch in train_loader:
        x_1 = batch["input_ids"].to(device).view(-1, batch["input_ids"].shape[-1])
        optim.zero_grad()
        loss = general_step(x_1)
        loss.backward()
        optim.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x_1 = batch["input_ids"].to(device).view(-1, batch["input_ids"].shape[-1])
            loss = general_step(x_1)
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

print("Training complete. Best ValLoss:", best_val) 
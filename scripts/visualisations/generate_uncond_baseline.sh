#!/bin/bash
#PBS -l walltime=02:30:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -N uncond_baseline
#PBS -j oe

# ------------------------------------------------------------------
# Activate the conda environment that holds all PepDFM dependencies
# ------------------------------------------------------------------
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

# Move to project root (adjust if you put the script elsewhere)
cd /rds/general/user/kja24/home/mog_dfm

CKPT="ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt"
OUT_DIR="ampflow/results/mog/baseline"

python ampflow/ampdfm_visualisations/generate_uncond_baseline.py \
    --ckpt "$CKPT" \
    --out_dir "$OUT_DIR" \
    --n 2500 \
    --device cuda:0 \
    --len_min 10 --len_max 40 \
    --seed 42

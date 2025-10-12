#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=16:mem=64gb:ngpus=1
#PBS -N pepdfm_mog_ecoli
#PBS -j oe

set -euo pipefail

# Activate conda environment ----------------------------------------------------
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

cd /rds/general/user/kja24/home/mog_dfm

CKPT_PATH="ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt"

python ampflow/ampdfm_scripts/pepdfm_mog.py \
    --ckpt "$CKPT_PATH" \
    --n_samples 2500 \
    --n_batches 10 \
    --importance 1,1,1 \
    --homopolymer_gamma 4.0 \
    --potency_variant ecoli \
    --run_tag ecoli_hp4_111_2500 \
    --device cuda:0

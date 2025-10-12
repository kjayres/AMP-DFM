#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=16:mem=64gb:ngpus=1
#PBS -N pepdfm_mog_generic
#PBS -j oe

set -euo pipefail

# Activate conda environment (user-specific path) ------------------------------
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

cd /rds/general/user/kja24/home/mog_dfm

CKPT_PATH="ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt"

python ampflow/ampdfm_scripts/pepdfm_mog.py \
    --ckpt "$CKPT_PATH" \
    --n_samples 2500 \
    --n_batches 10 \
    --run_tag generic_hp4_111_2500 \
    --importance 1,1,1 \
    --device cuda:0 \
    --homopolymer_gamma 4.0

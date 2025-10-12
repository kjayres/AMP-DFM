#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=6:mem=32gb:ngpus=1
#PBS -N pepdfm_vis
#PBS -j oe

set -euo pipefail

source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm
cd /rds/general/user/kja24/home/mog_dfm

python ampflow/ampdfm_visualisations/analysis_guidance.py \
RUN_TAG="generic"
GUIDED="ampflow/results/mog/${RUN_TAG}/mog_samples_scores.csv"
OUT_DIR="ampflow/results/visualisations/${RUN_TAG}"

python ampflow/ampdfm_visualisations/analysis_guidance.py \
    --guided_csv "$GUIDED" \
    --ckpt ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt \
    --out_dir "$OUT_DIR" \
    --device cuda:0

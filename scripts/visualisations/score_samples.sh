#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1
#PBS -N score_mog
#PBS -j oe

# ------------------------------------------------------------------
# Activate the conda environment that holds all PepDFM dependencies
# ------------------------------------------------------------------
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

# Move to project root (adjust if you put the script elsewhere)
cd /rds/general/user/kja24/home/mog_dfm

FASTA="ampflow/results/mog/mog_samples.fa"
CSV_OUT="ampflow/results/mog/mog_samples_scores.csv"

python ampflow/ampdfm_visualisations/score_samples.py \
    --fasta "$FASTA" \
    --out   "$CSV_OUT" \
    --device cuda:0   # change to cpu to run on CPU-only node

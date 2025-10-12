#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000
#PBS -N compare_pepdfm_ampdfm
#PBS -j oe
#PBS -o compare_pepdfm_ampdfm.o$PBS_JOBID

###############################################################################
# compare_pepdfm_ampdfm.sh                                                    #
#                                                                             #
# PBS script to run AMP-/Pep-DFM model comparison on Imperial HPC.          #
# Requires 1 GPU and activates the mog-dfm conda environment.               #
###############################################################################

# Exit on first error
set -euo pipefail

cd /rds/general/user/kja24/home/mog_dfm

# Activate conda env
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

export PYTHONPATH="$PWD:$PYTHONPATH"

# ---------------------- Paths (relative to repo root) -----------------------
PEP_UNCOND_CKPT="ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt"
PEP_COND_CKPT="ampflow/ampdfm_ckpt/pepdfm_conditional_finetuned.ckpt"
PEP_ORIG_CKPT="ampflow/ampdfm_ckpt/pepdfm_original_epoch200.ckpt"

# ---------------------- Run --------------------------------------------------

python ampflow/ampdfm_scripts/compare_pepdfm_ampdfm.py \
       --n 10000 \
       --models \
       pep_ours:${PEP_UNCOND_CKPT} \
       condpep:${PEP_COND_CKPT}:generic \
       pep_orig:${PEP_ORIG_CKPT} \
       --pep_val_path ampflow/ampdfm_data/tokenized_pep/val \
       --amp_val_path ampflow/ampdfm_data/tokenized_amp/val \
       --val_map \
       pep_orig:ampflow/ampdfm_data/tokenized_pep_original/val \
       pep_ours:ampflow/ampdfm_data/tokenized_pep/val \
       --train_hf_map \
       pep_orig:ampflow/ampdfm_data/tokenized_pep_original/train \
       pep_ours:ampflow/ampdfm_data/tokenized_pep/train \
       condpep:ampflow/ampdfm_data/tokenized_amp/train \
       --out_dir ampflow/results/model_panel \
       --lev_samples 2000 \
       --mmseqs \
       --seed 1234

echo "Job completed at: $(date)"
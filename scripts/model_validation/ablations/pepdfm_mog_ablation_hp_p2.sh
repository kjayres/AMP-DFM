#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=16:mem=64gb:ngpus=1
#PBS -N pepdfm_ablate_p2
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/mog_dfm/ampflow/ablations/pepdfm_mog_ablation_hp_p2.o$PBS_JOBID
#PBS -e /rds/general/user/kja24/home/mog_dfm/ampflow/ablations/pepdfm_mog_ablation_hp_p2.e$PBS_JOBID

set -euo pipefail

# Activate conda environment -------------------------------------------------
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

cd /rds/general/user/kja24/home/mog_dfm

CKPT_PATH="ampflow/ampdfm_ckpt/pepdfm_unconditional_epoch200.ckpt"

# Part 2: two ablations
IMPS=(
  "0,1,0"  # hemolysis only
  "0,0,1"  # cytotox only
)

for imp in "${IMPS[@]}"; do
  weights_compact="${imp//,/}"
  run_tag="ablations/generic_hp_ab_${weights_compact}_1k"
  echo "[INFO] Running ablation importance=${imp} → run_tag=${run_tag}"

  python ampflow/ampdfm_scripts/pepdfm_mog.py \
    --ckpt "$CKPT_PATH" \
    --n_samples 1000 \
    --n_batches 10 \
    --run_tag "$run_tag" \
    --importance "$imp" \
    --potency_variant generic \
    --device cuda:0 \
    --homopolymer_gamma 2.0
done



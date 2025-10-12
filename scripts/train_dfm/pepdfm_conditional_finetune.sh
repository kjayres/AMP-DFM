#!/bin/bash
#PBS -l walltime=02:40:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -N pepdfm_cond_ft
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/mog_dfm/ampflow/ampdfm_scripts/pepdfm_cond_ft.o$PBS_JOBID
#PBS -e /rds/general/user/kja24/home/mog_dfm/ampflow/ampdfm_scripts/pepdfm_cond_ft.e$PBS_JOBID

set -e

cd /rds/general/user/kja24/home/mog_dfm
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm
export PYTHONPATH="$PWD:$PYTHONPATH"

# Use GPU if available and restrict to AMP-positive sequences by default
python ampflow/ampdfm_scripts/pepdfm_conditional_finetune.py --device cuda --amp_only "$@"
#!/bin/bash
#PBS -l walltime=02:40:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -N ampdfm_cond_ft
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/dfm/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/dfm/

cd /rds/general/user/kja24/home/amp_dfm
source /rds/general/user/kja24/home/miniforge3/bin/activate amp-dfm

python scripts/dfm/ampdfm_conditional_finetune.py --config configs/flow_matching/ampdfm_conditional_finetune.yaml
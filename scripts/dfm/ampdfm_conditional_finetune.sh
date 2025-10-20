#!/bin/bash
#PBS -l walltime=02:40:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -N ampdfm_cond_ft
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/dfm/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/dfm/

cd /rds/general/user/kja24/home/amp_dfm
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

python /rds/general/user/kja24/home/amp_dfm/scripts/dfm/ampdfm_conditional_finetune.py --config /rds/general/user/kja24/home/amp_dfm/configs/flow_matching/ampdfm_conditional_finetune.yaml --device cuda --amp_only "$@"
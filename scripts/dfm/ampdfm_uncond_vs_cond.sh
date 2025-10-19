#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -N compare_ampdfm
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/train_dfm/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/train_dfm/

cd /rds/general/user/kja24/home/amp_dfm
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

python /rds/general/user/kja24/home/amp_dfm/scripts/train_dfm/ampdfm_uncond_vs_cond.py --config /rds/general/user/kja24/home/amp_dfm/configs/flow_matching/ampdfm_uncond_vs_cond.yaml
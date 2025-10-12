#!/bin/bash
#PBS -l walltime=37:30:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -N ampdfm_am_act_xgb_generic
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/train_judges/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/train_judges/

cd /rds/general/user/kja24/home
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

python /rds/general/user/kja24/home/amp_dfm/scripts/train_judges/train_judge.py \
  --config /rds/general/user/kja24/home/amp_dfm/configs/judges/antimicrobial_activity_xgboost.yaml



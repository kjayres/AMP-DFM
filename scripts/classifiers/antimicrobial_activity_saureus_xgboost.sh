#!/bin/bash
#PBS -l walltime=24:30:00
#PBS -l select=1:ncpus=12:mem=64gb
#PBS -N ampdfm_am_act_xgb_saureus
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/classifiers/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/classifiers/

cd /rds/general/user/kja24/home
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

python /rds/general/user/kja24/home/amp_dfm/scripts/classifiers/train_classifiers.py \
  --config /rds/general/user/kja24/home/amp_dfm/configs/classifiers/antimicrobial_activity_saureus_xgboost.yaml



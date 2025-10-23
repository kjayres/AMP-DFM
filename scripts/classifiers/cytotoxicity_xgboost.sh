#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -N ampdfm_cytotoxicity_xgb
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/classifiers/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/classifiers/

cd /rds/general/user/kja24/home/amp_dfm
source /rds/general/user/kja24/home/miniforge3/bin/activate amp-dfm

python scripts/classifiers/train_classifiers.py \
  --config configs/classifiers/cytotoxicity_xgboost.yaml

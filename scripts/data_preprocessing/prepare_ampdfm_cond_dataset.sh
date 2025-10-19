#!/bin/bash
#PBS -l walltime=00:25:00
#PBS -l select=1:ncpus=2:mem=12gb
#PBS -N prep_ampdfm_cond
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/data_preprocessing/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/data_preprocessing/

cd /rds/general/user/kja24/home

source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

python amp_dfm/scripts/data_preprocessing/prepare_ampdfm_cond_dataset.py 
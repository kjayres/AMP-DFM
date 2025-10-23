#!/bin/bash
#PBS -l walltime=00:15:00
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -N prep_ampdfm_uncond
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/data_preprocessing/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/data_preprocessing/

cd /rds/general/user/kja24/home/amp_dfm
source /rds/general/user/kja24/home/miniforge3/bin/activate amp-dfm

python scripts/data_preprocessing/prepare_ampdfm_uncond_dataset.py
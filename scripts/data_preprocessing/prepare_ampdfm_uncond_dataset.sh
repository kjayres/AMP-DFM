#!/bin/bash
#PBS -l walltime=00:15:00
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -N prep_ampdfm_uncond
#PBS -j oe

cd /rds/general/user/kja24/home

source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

python amp_dfm/scripts/data_preprocessing/prepare_ampdfm_uncond_dataset.py 
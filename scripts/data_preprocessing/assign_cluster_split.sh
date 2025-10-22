#!/bin/bash
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -N assign_splits
#PBS -j oe

cd /rds/general/user/kja24/home

source /rds/general/user/kja24/home/anaconda3/bin/activate amp-dfm

python amp_dfm/scripts/data_preprocessing/assign_cluster_split.py 
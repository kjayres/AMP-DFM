#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -N mmseqs_cluster
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/data_preprocessing/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/data_preprocessing/

cd /rds/general/user/kja24/home/amp_dfm
source /rds/general/user/kja24/home/miniforge3/bin/activate amp-dfm

python scripts/data_preprocessing/mmseqs_cluster.py
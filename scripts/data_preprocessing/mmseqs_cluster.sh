#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -N mmseqs_cluster
#PBS -j oe

cd /rds/general/user/kja24/home

source /rds/general/user/kja24/home/anaconda3/bin/activate amp-dfm

python amp_dfm/scripts/data_preprocessing/mmseqs_cluster.py 
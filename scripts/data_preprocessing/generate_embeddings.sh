#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1
#PBS -N generate_embeddings
#PBS -j oe

cd /rds/general/user/kja24/home

source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

python amp_dfm/scripts/data_preprocessing/generate_embeddings.py --device cuda --batch 32 
#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1
#PBS -N generate_embeddings
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/data_preprocessing/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/data_preprocessing/

cd /rds/general/user/kja24/home/amp_dfm
source /rds/general/user/kja24/home/miniforge3/bin/activate amp-dfm

python scripts/data_preprocessing/generate_embeddings.py --device cuda --batch 32
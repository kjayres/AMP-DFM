#!/bin/bash
#PBS -l walltime=18:00:00
#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb
#PBS -N ampdfm_mog_ecoli
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/mog/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/mog/

cd /rds/general/user/kja24/home/amp_dfm
source /rds/general/user/kja24/home/anaconda3/bin/activate amp-dfm

python /rds/general/user/kja24/home/amp_dfm/scripts/mog/ampdfm_mog.py --config /rds/general/user/kja24/home/amp_dfm/configs/mog/ampdfm_mog_ecoli.yaml

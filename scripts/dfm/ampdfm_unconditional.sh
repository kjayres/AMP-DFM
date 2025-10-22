#!/bin/bash
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -N ampdfm_unconditional
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/dfm/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/dfm/

cd /rds/general/user/kja24/home/amp_dfm
source /rds/general/user/kja24/home/anaconda3/bin/activate amp-dfm

python /rds/general/user/kja24/home/amp_dfm/scripts/dfm/ampdfm_unconditional.py --config /rds/general/user/kja24/home/amp_dfm/configs/flow_matching/ampdfm_unconditional.yaml
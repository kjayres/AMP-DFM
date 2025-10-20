#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -N ampdfm_sample
#PBS -j oe
#PBS -o /rds/general/user/kja24/home/amp_dfm/scripts/dfm/
#PBS -e /rds/general/user/kja24/home/amp_dfm/scripts/dfm/

cd /rds/general/user/kja24/home/amp_dfm
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

python /rds/general/user/kja24/home/amp_dfm/scripts/dfm/ampdfm_uncond_sample.py --config /rds/general/user/kja24/home/amp_dfm/configs/flow_matching/ampdfm_uncond_sample.yaml


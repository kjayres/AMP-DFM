#!/bin/bash
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -N pepdfm_unconditional
#PBS -j oe

set -e

cd /rds/general/user/kja24/home/mog_dfm
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm
export PYTHONPATH="$PWD:$PYTHONPATH"

python ampflow/ampdfm_scripts/pepdfm_unconditional.py 
#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -N job_name
#PBS -j oe

cd /rds/general/user/kja24/home/mog_dfm

# Activate conda env
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

# Run the python script
python amp_dfm/scripts/data_preprocessing/my_python_script.py
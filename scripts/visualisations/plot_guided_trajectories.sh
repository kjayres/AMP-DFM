#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -N guided_trajectories
#PBS -j oe

# ------------------------------------------------------------------
# Activate the conda environment that holds all PepDFM dependencies
# ------------------------------------------------------------------
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

# Move to project root
cd /rds/general/user/kja24/home/mog_dfm

# ------------------------------------------------------------------
# Run the guided trajectories analysis
# ------------------------------------------------------------------
python ampflow/ampdfm_visualisations/plot_guided_trajectories.py

echo "Guided trajectories analysis completed."

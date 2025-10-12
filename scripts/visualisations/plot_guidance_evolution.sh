#!/bin/bash
#PBS -l walltime=00:45:00
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1
#PBS -N guidance_evolution
#PBS -j oe

# ------------------------------------------------------------------
# Activate the conda environment that holds all PepDFM dependencies
# ------------------------------------------------------------------
source /rds/general/user/kja24/home/anaconda3/bin/activate mog-dfm

# Move to project root
cd /rds/general/user/kja24/home/mog_dfm

# ------------------------------------------------------------------
# Run the guidance evolution analysis
# ------------------------------------------------------------------
python ampflow/ampdfm_visualisations/plot_guidance_evolution.py

echo "Guidance evolution analysis completed."

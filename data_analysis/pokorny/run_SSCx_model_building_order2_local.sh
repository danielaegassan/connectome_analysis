#!/bin/bash -l
#SBATCH --job-name=modelbuild
#SBATCH --time=24:00:00
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=volta

source /gpfs/bbp.cscs.ch/home/pokorny/ToposampleKernel/bin/activate

python -u run_SSCx_model_building_order2_local.py

# EXAMPLE HOW TO RUN:
# sbatch run_SSCx_model_building_order2_local.sh
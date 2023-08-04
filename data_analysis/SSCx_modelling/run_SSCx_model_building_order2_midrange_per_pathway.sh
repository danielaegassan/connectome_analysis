#!/bin/bash -l
#SBATCH --job-name=model2midr
#SBATCH --time=24:00:00
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu

source /gpfs/bbp.cscs.ch/home/pokorny/ToposampleKernel/bin/activate

python -u run_SSCx_model_building_order2_midrange_per_pathway.py $1 $2 $3

# EXAMPLE HOW TO RUN:
# sbatch run_SSCx_model_building_order2_midrange_per_pathway.sh SAMPLE_SIZE <MTYPE_IDX_START> <MTYPE_IDX_END>
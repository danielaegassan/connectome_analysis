#!/bin/bash -l
#SBATCH --job-name=modelbuild
#SBATCH --time=1:00:00
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu

source ../venv/bin/activate

python -u run_batch_modelling.py ./adj_r0c10.npz ./nodes_r0c10.h5 ./run_batch_modelling.json $1 $2

# EXAMPLES HOW TO RUN WITH SBATCH:
# (1) Without data splits
# sbatch run_batch_modelling.sh
#
# (2) With N=5 data splits
# Split 1 of 5:  sbatch run_batch_modelling.sh 5 1
#        ...
# Split 5 of 5:  sbatch run_batch_modelling.sh 5 5
# Merge results: sbatch run_batch_modelling.sh 5 -1

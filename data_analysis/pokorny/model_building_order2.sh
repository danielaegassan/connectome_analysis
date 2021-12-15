#!/bin/bash -l
#SBATCH --job-name=modelbuild
#SBATCH --time=24:00:00
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu

source /gpfs/bbp.cscs.ch/home/pokorny/ToposampleKernel/bin/activate

python -u ../../library/modelling.py /gpfs/bbp.cscs.ch/project/proj102/scratch/SSCX_BioM/matrices/local_connectivity.npz /gpfs/bbp.cscs.ch/project/proj102/scratch/SSCX_BioM/matrices/neuron_info.feather model_building_config_order2_local.json $1 $2
# python -u ../../library/modelling.py /gpfs/bbp.cscs.ch/project/proj102/scratch/SSCX_BioM/matrices/local_connectivity.npz /gpfs/bbp.cscs.ch/project/proj102/scratch/SSCX_BioM/matrices/neuron_info.feather model_building_config_order2_midrange.json $1 $2
# python -u ../../library/modelling.py ../../documentation/notebooks/adj_r0c10.npz ../../documentation/notebooks/nodes_r0c10.h5 model_building_config_order2_local.json $1 $2

# EXAMPLE HOW TO RUN:
# Split   1 of 400: sbatch model_building_order2.sh 400 0
#          ...
# Split 400 of 400: sbatch model_building_order2.sh 400 399
# Merge results:    sbatch model_building_order2.sh 400 -1

# Description: Running SSCx model building from batch script (local connectivity)
# Author: C. Pokorny
# Last modified: 03/2022


# Global imports
import numpy as np
import pandas as pd
import scipy.sparse as sps
import os
import sys
import logging


# Local imports
sys.path.append('../../library')
import modelling


# Load connectivity matrix and neuron table
data_path = '/gpfs/bbp.cscs.ch/project/proj102/scratch/SSCX_BioM/matrices'
conn_fn = 'local_connectivity.npz'
nrn_fn = 'neuron_info.feather'

adj_matrix = sps.load_npz(os.path.join(data_path, conn_fn))
nrn_table = pd.read_feather(os.path.join(data_path, nrn_fn))
assert adj_matrix.shape[0] == adj_matrix.shape[1] == nrn_table.shape[0], 'ERROR: Data size mismatch!'
print(f'INFO: Loaded connectivity and properties of {nrn_table.shape[0]} neurons')


# Building multiple 2nd-order models using different random subsets of neurons
num_samples = 10
sample_sizes = [1000, 5000, 10000, 25000, 50000, 75000, 100000, 200000]
save_path = './modelling/tables'
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_fn = 'SSCx_model_params_order-2_local.h5'

# logging.getLogger().setLevel(logging.WARNING) # Global control of logging level (default: INFO; WARNING to disable outputs)

model_params_2 = []
config_dict = {'bin_size_um': 50, 'max_range_um': 1000, 'sample_size': None, 'sample_seeds': num_samples}
for idx, sample_size in enumerate(sample_sizes):
    print(f'Step {idx + 1} of {len(sample_sizes)}: Sampling {num_samples} x {sample_size} neurons')
    config_dict.update({'sample_size': sample_size})
    model_params_2.append(modelling.conn_prob_2nd_order_model(adj_matrix, nrn_table, **config_dict))
    model_params_2[idx].to_hdf(os.path.join(save_path, save_fn), key=f'sub{sample_sizes[idx]}', mode='r+' if idx else 'w') # Save to .h5 (appending)

# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Description: Running SSCx model building from batch script (midrange connectivity; per PRE pathway)
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

# logging.getLogger().setLevel(logging.WARNING) # Global control of logging level (default: INFO; WARNING to disable outputs)


# Set paths
nrn_path_pre = './modelling/nrn_tables/midrange'
nrn_fn_pre = 'nrn_info'
nrn_path_post = '/gpfs/bbp.cscs.ch/project/proj102/scratch/SSCX_BioM/matrices'
nrn_fn_post = 'neuron_info_extended'
adj_path = './modelling/adj_matrices/midrange'
adj_fn = 'adj_mat'


if __name__ == "__main__":
    if len(sys.argv) >= 2:

        sample_size = int(sys.argv[1])

        # Load m-types
        mtypes = np.load(os.path.join(nrn_path_pre, 'mtypes.npy'))

        if len(sys.argv) >= 3:
            midx_start = np.maximum(int(sys.argv[2]), 0)
        else:
            midx_start = 0

        if len(sys.argv) >= 4:
            midx_end = np.minimum(int(sys.argv[3]), len(mtypes) - 1)
        else:
            midx_end = len(mtypes) - 1

        print('////////////////////////////////////////////////////////////////////////////////////////////////////')
        print(f'INFO: Running 2nd-order model building of midrange connectivity ({sample_size} samples, {midx_start}: {mtypes[midx_start]} to {midx_end}: {mtypes[midx_end]})')
        print('////////////////////////////////////////////////////////////////////////////////////////////////////')

        # Load POST neuron table
        nrn_table_post = pd.read_feather(os.path.join(nrn_path_post, f'{nrn_fn_post}.feather'))

        # Building multiple 2nd-order models using different random subsets of neurons
        # [based on "virtual" 2D flat mapping coordinate system (ss_mapping_x/ss_mapping_y)]
        num_samples = 10 # In case of random sub-sampling, ignored otherwise
        ### sample_size = 100000 # Max. per pathway
        config_dict = {'bin_size_um': 20, 'max_range_um': 400, 'sample_size': sample_size, 'sample_seeds': num_samples, 'coord_names': ['ss_mapping_x', 'ss_mapping_y']}

        save_path = f'./modelling/tables_per_pathway_{sample_size}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_fn = 'SSCx_model_params_order-2_midrange'

        for pre_idx in range(len(mtypes))[midx_start : midx_end + 1]:
            # Load pathway-specific PRE neuron table
            nrn_table_pre = pd.read_feather(os.path.join(nrn_path_pre, f'{nrn_fn_pre}__{mtypes[pre_idx]}.feather'))

            # Load pathway-specific matrix
            adj_matrix = sps.load_npz(os.path.join(adj_path, f'{adj_fn}__{mtypes[pre_idx]}-ALL.npz'))
            assert adj_matrix.shape[0] == nrn_table_pre.shape[0] and adj_matrix.shape[1] == nrn_table_post.shape[0], 'ERROR: Data size mismatch!'
            print(f'*** {pre_idx + 1}/{len(mtypes)} *** {mtypes[pre_idx]}-ALL: {nrn_table_pre.shape[0]} x {nrn_table_post.shape[0]} neurons')

            if adj_matrix.count_nonzero() > 0:
                # Run model building
                model_params_2 = modelling.conn_prob_2nd_order_pathway_model(adj_matrix, nrn_table_pre, nrn_table_post, **config_dict)
            else:
                # Create empty (dummy) model
                model_params_2 = pd.DataFrame(modelling.build_2nd_order(np.zeros(2), np.arange(3))['model_params'], index=pd.Index([None], name='seed'))
            model_params_2.to_hdf(os.path.join(save_path, f'{save_fn}__{mtypes[pre_idx]}-ALL.h5'), key='model_params_2', mode='w') # Save to .h5

    else:
        print('Runs 2nd-order model extraction of midrange connectivity per PRE-pathway')
        print('Usage: Provide sample size, and optionally, start and end idx of m-type')
        print('       run_SSCx_model_building_order2_midrange_per_pathway SAMPLE_SIZE <MTYPE_IDX_START> <MTYPE_IDX_END>')
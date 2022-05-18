# Generate models for connectomes.
#
# Author(s): C. Pokorny
# Last modified: 12/2021


import numpy as np
import pandas as pd
import os
import scipy.optimize as opt
import scipy.sparse as sps
import scipy.spatial as spt
import matplotlib.pyplot as plt
import itertools
import progressbar
import pickle
import sys
import logging
import json

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[stream_handler])

PROB_CMAP = plt.cm.get_cmap('hot')
DATA_COLOR = 'tab:blue'
MODEL_COLOR = 'tab:red'
MODEL_COLOR2 = 'tab:olive'


###################################################################################################
# Wrapper & helper functions for model building to be used within a processing pipeline
#   w/o data/model saving, figure plotting, and data splitting
#   _Inputs_: adj_matrix, nrn_table, kwargs (bin_size_um, max_range_um, sample_size, sample_seeds)
#   _Output_: Pandas dataframe with model paramters (columns) for different seeds (rows)
###################################################################################################

def conn_prob_2nd_order_model(adj_matrix, nrn_table, **kwargs):
    """2nd-order probability model building, optionally for multiple random subsets of neurons."""

    assert 'model_order' not in kwargs.keys(), f'ERROR: Invalid argument "model_order" in kwargs!'
    kwargs.update({'model_order': 2})

    return conn_prob_model(adj_matrix, nrn_table, **kwargs)


def conn_prob_2nd_order_pathway_model(adj_matrix, nrn_table_src, nrn_table_tgt, **kwargs):
    """2nd-order probability model building for separate pathways (i.e., non-symmetric adj_matrix),
       optionally for multiple random subsets of neurons."""

    assert 'model_order' not in kwargs.keys(), f'ERROR: Invalid argument "model_order" in kwargs!'
    kwargs.update({'model_order': 2})

    return conn_prob_pathway_model(adj_matrix, nrn_table_src, nrn_table_tgt, **kwargs)


def conn_prob_3rd_order_model(adj_matrix, nrn_table, **kwargs):
    """3rd-order probability model building, optionally for multiple random subsets of neurons."""

    assert 'model_order' not in kwargs.keys(), f'ERROR: Invalid argument "model_order" in kwargs!'
    kwargs.update({'model_order': 3})

    return conn_prob_model(adj_matrix, nrn_table, **kwargs)


def conn_prob_model(adj_matrix, nrn_table, **kwargs):
    """General probability model building, optionally for multiple random subsets of neurons."""

    invalid_args = ['model_name', 'sample_seed'] # Not allowed arguments, as they will be set/used internally
    for arg in invalid_args:
        assert arg not in kwargs.keys(), f'ERROR: Invalid argument "{arg}" in kwargs!'
    kwargs.update({'model_dir': None, 'data_dir': None, 'plot_dir': None, 'do_plot': False, 'N_split': None}) # Disable plotting/saving
    model_name = None
    model_order = kwargs.pop('model_order')

    sample_size = kwargs.get('sample_size')
    if sample_size is None or sample_size >= nrn_table.shape[0]:
        sample_seeds = [None] # No randomization
        if kwargs.pop('sample_seeds', None) is not None:
            logging.warning('Using all neurons, ignoring sample seeds!')
    else:
        sample_seeds = kwargs.pop('sample_seeds', 1)

        if not isinstance(sample_seeds, list): # sample_seeds corresponds to number of seeds to generate
            sample_seeds = generate_seeds(sample_seeds, meta_seed=kwargs.pop('meta_seed', 0))
        else:
            sample_seeds = list(np.unique(sample_seeds)) # Assure that unique and sorted

    model_params = pd.DataFrame()
    for seed in sample_seeds:
        kwargs.update({'sample_seed': seed})
        _, model_dict = run_model_building(adj_matrix, nrn_table, model_name, model_order, **kwargs)
        model_params = model_params.append(pd.DataFrame(model_dict['model_params'], index=pd.Index([seed], name='seed')))

    return model_params


def conn_prob_pathway_model(adj_matrix, nrn_table_src, nrn_table_tgt, **kwargs):
    """General probability model building for separate pathways (i.e., non-symmetric adj_matrix),
       optionally for multiple random subsets of neurons."""

    invalid_args = ['model_name', 'sample_seed'] # Not allowed arguments, as they will be set/used internally
    for arg in invalid_args:
        assert arg not in kwargs.keys(), f'ERROR: Invalid argument "{arg}" in kwargs!'
    kwargs.update({'model_dir': None, 'data_dir': None, 'plot_dir': None, 'do_plot': False, 'N_split': None}) # Disable plotting/saving
    model_name = None
    model_order = kwargs.pop('model_order')

    sample_size = kwargs.get('sample_size')
    if sample_size is None or sample_size >= np.maximum(nrn_table_src.shape[0], nrn_table_tgt.shape[0]):
        sample_seeds = [None] # No randomization
        if kwargs.pop('sample_seeds', None) is not None:
            logging.warning('Using all neurons, ignoring sample seeds!')
    else:
        sample_seeds = kwargs.pop('sample_seeds', 1)

        if not isinstance(sample_seeds, list): # sample_seeds corresponds to number of seeds to generate
            sample_seeds = generate_seeds(sample_seeds, meta_seed=kwargs.pop('meta_seed', 0))
        else:
            sample_seeds = list(np.unique(sample_seeds)) # Assure that unique and sorted

    model_params = pd.DataFrame()
    for seed in sample_seeds:
        kwargs.update({'sample_seed': seed})
        _, model_dict = run_pathway_model_building(adj_matrix, nrn_table_src, nrn_table_tgt, model_name, model_order, **kwargs)
        model_params = model_params.append(pd.DataFrame(model_dict['model_params'], index=pd.Index([seed], name='seed')))

    return model_params


def generate_seeds(num_seeds, num_digits=6, meta_seed=0):
    """Helper function to generate list of unique random seeds with given number of digits."""

    assert isinstance(num_seeds, int) and num_seeds > 0, 'ERROR: Number of seeds must be a positive integer!'
    assert isinstance(num_digits, int) and num_digits > 0, 'ERROR: Number of digits must be a positive integer!'

    np.random.seed(meta_seed)
    sample_seeds = list(sorted(np.random.choice(10**num_digits - 10**(num_digits - 1), num_seeds, replace=False) + 10**(num_digits - 1))) # Sorted, 6-digit seeds

    return sample_seeds


###################################################################################################
# Main function for model building
###################################################################################################

def run_model_building(adj_matrix, nrn_table, model_name, model_order, **kwargs):
    """
    Main function for running model building, consisting of three steps:
      Data extraction, model fitting, and (optionally) data/model visualization
    """
    logging.info(f'Running order-{model_order} model building {kwargs}...')

    # Subsampling (optional)
    sample_size = kwargs.get('sample_size')
    sample_seed = kwargs.get('sample_seed')
    if sample_size is not None and sample_size > 0 and sample_size < nrn_table.shape[0]:
        logging.info(f'Subsampling to {sample_size} of {nrn_table.shape[0]} neurons (seed={sample_seed})')
        np.random.seed(sample_seed)
        sub_sel = np.random.permutation([True] * sample_size + [False] * (nrn_table.shape[0] - sample_size))
        adj_matrix = adj_matrix.tocsr()[sub_sel, :].tocsc()[:, sub_sel].tocsr()
        nrn_table = nrn_table.loc[sub_sel, :]

    # Set modelling functions
    if model_order == 2: # Distance-dependent
        fct_extract = extract_2nd_order
        fct_fit = build_2nd_order
        fct_plot = plot_2nd_order
    elif model_order == 3: # Bipolar distance-dependent
        fct_extract = extract_3rd_order
        fct_fit = build_3rd_order
        fct_plot = plot_3rd_order
    else:
        assert False, f'ERROR: Order-{model_order} model building not supported!'

    # Data splits (optional)
    N_split = kwargs.pop('N_split', None)
    part_idx = kwargs.pop('part_idx', None)
    if N_split is None:
        split_indices = None
    else:
        assert N_split > 1, 'ERROR: Number of data splits must be larger than 1!'
        split_indices = np.split(np.arange(nrn_table.shape[0]), np.cumsum([np.ceil(nrn_table.shape[0] / N_split).astype(int)] * (N_split - 1)))

    if part_idx is None or part_idx == -1: # Run data extraction and model building for all splits
        extract_only = False
        data_fn = 'data'
    else: # Run only data extraction of given part idx
        assert N_split is not None and 0 <= part_idx < N_split, 'ERROR: Part index out of range!'
        extract_only = True
        data_fn = 'data' + get_data_part_name(N_split, part_idx)

    # Extract connection probability data
    if part_idx == -1: # Special case: Load and merge results of existing parts
        assert N_split is not None, 'ERROR: Number of data splits required!'
        data_dict = merge_data(kwargs.get('data_dir'), model_name, data_fn, [get_data_part_name(N_split, p) for p in range(N_split)])
    else:
        data_dict = fct_extract(adj_matrix, nrn_table, split_indices=split_indices, part_idx=part_idx, **kwargs)
    save_data(data_dict, kwargs.get('data_dir'), model_name, data_fn)

    if extract_only: # Stop here and return data dict
        return data_dict, {}

    # Fit model
    model_dict = fct_fit(**data_dict, **kwargs)
    save_data(model_dict, kwargs.get('model_dir'), model_name, 'model')

    # Visualize data/model (optional)
    if kwargs.get('do_plot'):
        fct_plot(adj_matrix, nrn_table, model_name, **data_dict, **model_dict, **kwargs)

    return data_dict, model_dict


def run_pathway_model_building(adj_matrix, nrn_table_src, nrn_table_tgt, model_name, model_order, **kwargs):
    """
    Main function for running model building for separate pathways (i.e., non-symmetric adj_matrix),
      consisting of three steps: Data extraction, model fitting, and (optionally) data/model visualization
    """
    logging.info(f'Running order-{model_order} model building {kwargs}...')

    # Subsampling (optional)
    sample_size = kwargs.get('sample_size')
    sample_seed = kwargs.get('sample_seed')
    if sample_size is not None and sample_size > 0 and sample_size < np.maximum(nrn_table_src.shape[0], nrn_table_tgt.shape[0]):
        logging.info(f'Subsampling to {sample_size} of {nrn_table_src.shape[0]}x{nrn_table_tgt.shape[0]} neurons (seed={sample_seed})')
        np.random.seed(sample_seed)
        if sample_size < nrn_table_src.shape[0]:
            sub_sel_src = np.random.permutation([True] * sample_size + [False] * (nrn_table_src.shape[0] - sample_size))
        else:
            sub_sel_src = np.full(nrn_table_src.shape[0], True)

        if sample_size < nrn_table_tgt.shape[0]:
            sub_sel_tgt = np.random.permutation([True] * sample_size + [False] * (nrn_table_tgt.shape[0] - sample_size))
        else:
            sub_sel_tgt = np.full(nrn_table_tgt.shape[0], True)

        adj_matrix = adj_matrix.tocsr()[sub_sel_src, :].tocsc()[:, sub_sel_tgt].tocsr()
        # adj_matrix = adj_matrix[sub_sel_src, :][:, sub_sel_tgt]
        nrn_table_src = nrn_table_src.loc[sub_sel_src, :]
        nrn_table_tgt = nrn_table_tgt.loc[sub_sel_tgt, :]

    # Set modelling functions
    if model_order == 2: # Distance-dependent
        fct_extract = extract_2nd_order_pathway
        fct_fit = build_2nd_order
        fct_plot = plot_2nd_order
    else:
        assert False, f'ERROR: Order-{model_order} model building not supported!'

    # Data splits (optional)
    N_split = kwargs.pop('N_split', None)
    part_idx = kwargs.pop('part_idx', None)
    assert N_split is None and part_idx is None, 'ERROR: Data splitting not supported!'
    data_fn = 'data'

    # Extract connection probability data
    data_dict = fct_extract(adj_matrix, nrn_table_src, nrn_table_tgt, split_indices=None, part_idx=None, **kwargs)
    save_data(data_dict, kwargs.get('data_dir'), model_name, data_fn)

    # Fit model
    model_dict = fct_fit(**data_dict, **kwargs)
    save_data(model_dict, kwargs.get('model_dir'), model_name, 'model')

    # Visualize data/model (optional)
    if kwargs.get('do_plot'):
        assert False, 'ERROR: Plotting not supported!'

    return data_dict, model_dict


###################################################################################################
# Helper functions for model building
###################################################################################################

def merge_data(part_dir, model_name, spec_name, part_list):
    logging.info(f'Merging {len(part_list)} data parts...')
    
    count_conn_key = 'count_conn' # (fixed name; independent of model)
    count_all_key = 'count_all' # (fixed name; independent of model)
    p_key = None # Name of conn. prob. entry (model-dependent; starting with "p_")
    for part in part_list:
        part_file = os.path.join(part_dir, f'{model_name}__{spec_name}{part}.pickle')
        assert os.path.exists(part_file), f'ERROR: Data part "{part_file}" not found!'
        with open(part_file, 'rb') as f:
            part_dict = pickle.load(f)

        # Determine key names and initialize
        if p_key is None:
            part_keys = list(part_dict.keys())
            p_idx = np.where(np.array([str.find(key, 'p_') for key in part_keys]) == 0)[0]
            assert len(p_idx) == 1, 'ERROR: Conn. prob. entry could not be determined!'
            p_key = part_keys[p_idx[0]]
            other_keys = np.setdiff1d(part_keys, [p_key, count_conn_key, count_all_key]).tolist()
            other_dict = {key: part_dict[key] for key in other_keys}
            num_bins = len(part_dict[p_key])
            count_conn = np.zeros_like(part_dict[count_conn_key])
            count_all = np.zeros_like(part_dict[count_all_key])

        # Check consistency
        assert p_key in part_dict.keys(), f'ERROR: "{p_key}" not found in part "{part}"!'
        assert count_conn_key in part_dict.keys(), f'ERROR: "{count_conn_key}" not found in part "{part}"!'
        assert count_all_key in part_dict.keys(), f'ERROR: "{count_all_key}" not found in part "{part}"!'
        for key in other_keys:
            assert np.array_equal(other_dict[key], part_dict[key]), f'ERROR: "{key}" mismatch in part "{part}"!'
        assert part_dict[count_conn_key].shape == part_dict[count_all_key].shape == count_conn.shape == count_all.shape, f'ERROR: Bin count mismatch in part "{part}"!'

        # Add counts
        count_conn += part_dict[count_conn_key]
        count_all += part_dict[count_all_key]

    # Compute overall (merged) connection probabilities
    p_conn = np.array(count_conn / count_all)
#     p_conn[np.isnan(p_conn)] = 0.0

    # Create (merged) data dict
    data_dict = {p_key: p_conn, count_conn_key: count_conn, count_all_key: count_all, **other_dict}

    return data_dict


def save_data(save_dict, save_dir, model_name, save_spec=None):
    """Writes data/model dict to pickled data file"""
    if not save_dir:
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_spec is None:
        save_spec = ''
    else:
        save_spec = '__' + save_spec

    save_file = os.path.join(save_dir, f'{model_name}{save_spec}.pickle')
    with open(save_file, 'wb') as f:
        pickle.dump(save_dict, f)

    logging.info(f'Pickled dict written to {save_file}')


def get_data_part_name(N_split, part_idx):
    num_dig = len(str(N_split))
    return f'__part-{N_split}-{part_idx:0{num_dig}}'


def get_model_function(model, model_inputs, model_params):
    """Returns model function from string representation [so any model function can be saved to file]."""
    input_str = ','.join(model_inputs + ['model_params=model_params']) # String representation of input variables
    input_param_str = ','.join(model_inputs + list(model_params.keys())) # String representation of input variables and model parameters
    model_param_str = ','.join(model_inputs + ['**model_params']) # String representation propagating model parameters

    inner_model_str = f'lambda {input_param_str}: {model}'
    full_model_str = f'lambda {input_str}: ({inner_model_str})({model_param_str})' # Use nested lambdas to bind local variables

    model_fct = eval(full_model_str) # Build function

    # logging.info(f'Model function: {inner_model_str}')

    return model_fct


def compute_dist_matrix(src_nrn_pos, tgt_nrn_pos):
    """Computes distance matrix between pairs of neurons."""
    dist_mat = spt.distance_matrix(src_nrn_pos, tgt_nrn_pos)
    dist_mat[dist_mat == 0.0] = np.nan # Exclude autaptic connections

    return dist_mat


def compute_dist_matrix_symmetric(nrn_pos):
    """Computes symmetric distance matrix between pairs of neurons.
       Faster implementation to be used when source and target neurons
       are the same."""
    dist_mat = spt.distance.squareform(spt.distance.pdist(nrn_pos))
    dist_mat[dist_mat == 0.0] = np.nan # Exclude autaptic connections

    return dist_mat


def compute_bip_matrix(src_depths, tgt_depths):
    """
    Computes bipolar matrix between pairs of neurons based on depth difference delta_d:
      POST-synaptic neuron below (delta_d < 0) or above (delta_d > 0) PRE-synaptic neuron
    """
    bip_mat = np.sign(-np.diff(np.meshgrid(src_depths, tgt_depths, indexing='ij'), axis=0)[0, :, :])

    return bip_mat


def extract_dependent_p_conn(adj_matrix, dep_matrices, dep_bins):
    """Extract D-dimensional conn. prob. dependent on D property matrices between source-target pairs of neurons within given range of bins."""
    num_dep = len(dep_matrices)
    assert len(dep_bins) == num_dep, 'ERROR: Dependencies/bins mismatch!'
    assert np.all([dep_matrices[dim].shape == adj_matrix.shape for dim in range(num_dep)]), 'ERROR: Matrix dimension mismatch!'

    # Extract connection probability
    num_bins = [len(b) - 1 for b in dep_bins]
    bin_indices = [list(range(n)) for n in num_bins]
    count_all = np.full(num_bins, -1) # Count of all pairs of neurons for each combination of dependencies
    count_conn = np.full(num_bins, -1) # Count of connected pairs of neurons for each combination of dependencies

    logging.info(f'Extracting {num_dep}-dimensional ({"x".join([str(n) for n in num_bins])}) connection probabilities...')
    pbar = progressbar.ProgressBar(maxval=np.prod(num_bins) - 1)
    for idx in pbar(itertools.product(*bin_indices)):
        dep_sel = np.full(adj_matrix.shape, True)
        for dim in range(num_dep):
            lower = dep_bins[dim][idx[dim]]
            upper = dep_bins[dim][idx[dim] + 1]
            dep_sel = np.logical_and(dep_sel, np.logical_and(dep_matrices[dim] >= lower, (dep_matrices[dim] < upper) if idx[dim] < num_bins[dim] - 1 else (dep_matrices[dim] <= upper))) # Including last edge
        sidx, tidx = np.nonzero(dep_sel)
        count_all[idx] = np.sum(dep_sel)
        ### count_conn[idx] = np.sum(adj_matrix[sidx, tidx]) # ERROR in scipy/sparse/compressed.py if len(sidx) >= 2**31: "ValueError: could not convert integer scalar"
        # [WORKAROUND]: Split indices into parts of 2**31-1 length and sum them separately
        sidx_split = np.split(sidx, np.arange(0, len(sidx), 2**31-1)[1:])
        tidx_split = np.split(tidx, np.arange(0, len(tidx), 2**31-1)[1:])
        count_split = 0
        for s, t in zip(sidx_split, tidx_split):
            count_split = count_split + np.sum(adj_matrix[s, t])
        count_conn[idx] = count_split
    p_conn = np.array(count_conn / count_all)
#     p_conn[np.isnan(p_conn)] = 0.0

    return p_conn, count_conn, count_all


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   2nd order (distance-dependent)
###################################################################################################

def extract_2nd_order(adj_matrix, nrn_table, bin_size_um=100, max_range_um=None, coord_names=None, split_indices=None, part_idx=None, **_):
    """Extract distance-dependent connection probability (2nd order) from a sample of pairs of neurons."""

    if coord_names is None:
        coord_names = ['x', 'y', 'z'] # Default names of coordinatate system axes as in nrn_table
    if isinstance(split_indices, list):
        N_split = len(split_indices)
    else:
        N_split = 0 # Don't split
    if part_idx is not None: # Run only data extraction of given part idx
        assert 0 <= part_idx < N_split, 'ERROR: Part index out of range!'

    pos_table = nrn_table[coord_names].to_numpy()

    if N_split == 0: # Compute all at once
        # Compute distance matrix
        dist_mat = compute_dist_matrix_symmetric(pos_table)

        # Extract distance-dependent connection probabilities
        if max_range_um is None:
            max_range_um = np.nanmax(dist_mat)
        num_bins = np.ceil(max_range_um / bin_size_um).astype(int)
        dist_bins = np.arange(0, num_bins + 1) * bin_size_um

        p_conn_dist, count_conn, count_all = extract_dependent_p_conn(adj_matrix, [dist_mat], [dist_bins])

    else: # Split computation into N_split data splits (to reduce memory consumption)
        assert max_range_um is not None, f'ERROR: Max. range must be specified if data extraction splitted into {N_split} parts!'
        num_bins = np.ceil(max_range_um / bin_size_um).astype(int)
        dist_bins = np.arange(0, num_bins + 1) * bin_size_um

        count_conn = np.zeros(num_bins, dtype=int)
        count_all = np.zeros(num_bins, dtype=int)
        for sidx, split_sel in enumerate(split_indices):
            if part_idx is not None and part_idx != sidx:
                continue
            logging.info(f'<SPLIT {sidx + 1} of {N_split}>')

            # Compute distance matrix
            dist_mat_split = compute_dist_matrix(pos_table[split_sel, :], pos_table)
            
            # Extract distance-dependent connection counts
            _, count_conn_split, count_all_split = extract_dependent_p_conn(adj_matrix[split_sel, :], [dist_mat_split], [dist_bins])
            count_conn += count_conn_split
            count_all += count_all_split

        # Compute overall connection probabilities
        p_conn_dist = np.array(count_conn / count_all)
#         p_conn_dist[np.isnan(p_conn_dist)] = 0.0

    return {'p_conn_dist': p_conn_dist, 'count_conn': count_conn, 'count_all': count_all, 'dist_bins': dist_bins}


def extract_2nd_order_pathway(adj_matrix, nrn_table_src, nrn_table_tgt, bin_size_um=100, max_range_um=None, coord_names=None, split_indices=None, part_idx=None, **_):
    """Extract distance-dependent connection probability (2nd order) from a sample of pairs of neurons
       for separate pathways (i.e., non-symmetric adj_matrix)."""

    if coord_names is None:
        coord_names = ['x', 'y', 'z'] # Default names of coordinatate system axes as in nrn_table

    assert split_indices is None and part_idx is None, 'ERROR: Data splitting not supported!'

    pos_table_src = nrn_table_src[coord_names].to_numpy()
    pos_table_tgt = nrn_table_tgt[coord_names].to_numpy()

    # Compute distance matrix
    dist_mat = compute_dist_matrix(pos_table_src, pos_table_tgt)

    # Extract distance-dependent connection probabilities
    if max_range_um is None:
        max_range_um = np.nanmax(dist_mat)
    num_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_bins + 1) * bin_size_um

    p_conn_dist, count_conn, count_all = extract_dependent_p_conn(adj_matrix, [dist_mat], [dist_bins])

    return {'p_conn_dist': p_conn_dist, 'count_conn': count_conn, 'count_all': count_all, 'dist_bins': dist_bins}


def build_2nd_order(p_conn_dist, dist_bins, **_):
    """Build 2nd order model (exponential distance-dependent conn. prob.)."""
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]

    exp_model = lambda x, a, b: a * np.exp(-b * np.array(x))
    X = dist_bins[:-1][np.isfinite(p_conn_dist)] + bin_offset
    y = p_conn_dist[np.isfinite(p_conn_dist)]
    try:
        (exp_model_scale, exp_model_exponent), _ = opt.curve_fit(exp_model, X, y, p0=[0.0, 0.0])
    except:
        logging.error(f'Exception while fitting model ("{sys.exc_info()[1]}")')
        exp_model_scale = exp_model_exponent = np.nan

    logging.info(f'MODEL FIT: f(x) = {exp_model_scale:.6f} * exp(-{exp_model_exponent:.6f} * x)')

    model = 'exp_model_scale * np.exp(-exp_model_exponent * np.array(d))'
    model_inputs = ['d']
    model_params = {'exp_model_scale': exp_model_scale, 'exp_model_exponent': exp_model_exponent}

    return {'model': model, 'model_inputs': model_inputs, 'model_params': model_params}


def plot_2nd_order(adj_matrix, nrn_table, model_name, p_conn_dist, count_conn, count_all, dist_bins, model, model_inputs, model_params, plot_dir=None, **_):
    """Visualize data vs. model (2nd order)."""
    if plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    dist_model = np.linspace(dist_bins[0], dist_bins[-1], 100)

    model_str = f'f(x) = {model_params["exp_model_scale"]:.3f} * exp(-{model_params["exp_model_exponent"]:.3f} * x)'
    model_fct = get_model_function(model, model_inputs, model_params)

    plt.figure(figsize=(12, 4), dpi=300)

    # Data vs. model
    plt.subplot(1, 2, 1)
    plt.step(dist_bins, np.hstack([p_conn_dist[0], p_conn_dist]), color=DATA_COLOR, label=f'Data: N = {nrn_table.shape[0]}x{nrn_table.shape[0]} cells')
    plt.plot(dist_bins[:-1] + bin_offset, p_conn_dist, '.', color=DATA_COLOR)
    plt.plot(dist_model, model_fct(dist_model), '--', color=MODEL_COLOR, label='Model: ' + model_str)
    plt.grid()
    plt.xlabel('Distance ($\\mu$m)')
    plt.ylabel('Conn. prob.')
    plt.title('Data vs. model fit')
    plt.legend()

    # 2D connection probability (model)
    plt.subplot(1, 2, 2)
    plot_range = 500 # (um)
    r_markers = [200, 400] # (um)
    dx = np.linspace(-plot_range, plot_range, 201)
    dz = np.linspace(plot_range, -plot_range, 201)
    xv, zv = np.meshgrid(dx, dz)
    vdist = np.sqrt(xv**2 + zv**2)
    pdist = model_fct(vdist)
    plt.imshow(pdist, interpolation='bilinear', extent=(-plot_range, plot_range, -plot_range, plot_range), cmap=PROB_CMAP, vmin=0.0)
    for r in r_markers:
        plt.gca().add_patch(plt.Circle((0, 0), r, edgecolor='w', linestyle='--', fill=False))
        plt.text(0, r, f'{r} $\\mu$m', color='w', ha='center', va='bottom')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$\\Delta$x')
    plt.ylabel('$\\Delta$z')
    plt.title('2D model')
    plt.colorbar(label='Conn. prob.')

    plt.suptitle(f'Distance-dependent connection probability model (2nd order)')
    plt.tight_layout()
    if plot_dir is not None:
        out_fn = os.path.abspath(os.path.join(plot_dir, model_name + '__data_vs_model.png'))
        plt.savefig(out_fn)
        logging.info(f'Figure saved to {out_fn}')

    # Data counts
    plt.figure(figsize=(12, 4), dpi=300)
    plt.bar(dist_bins[:-1] + bin_offset, count_all, width=2.0 * bin_offset, edgecolor='k', label='All pair count')
    plt.bar(dist_bins[:-1] + bin_offset, count_conn, width=1.5 * bin_offset, label='Connection count')
    plt.gca().set_yscale('log')
    plt.xticks(dist_bins, rotation=45)
    plt.grid()
    plt.xlabel('Distance ($\\mu$m)')
    plt.ylabel('Count')
    plt.title(f'Distance-dependent connection counts (N = {nrn_table.shape[0]}x{nrn_table.shape[0]} cells)')
    plt.legend()
    plt.tight_layout()
    if plot_dir is not None:
        out_fn = os.path.abspath(os.path.join(plot_dir, model_name + '__data_counts.png'))
        plt.savefig(out_fn)
        logging.info(f'Figure saved to {out_fn}')


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   3rd order (bipolar distance-dependent)
###################################################################################################

def extract_3rd_order(adj_matrix, nrn_table, bin_size_um=100, max_range_um=None, coord_names=None, depth_name=None, split_indices=None, part_idx=None, **_):    
    """Extract distance-dependent connection probability (3rd order) from a sample of pairs of neurons."""

    if coord_names is None:
        coord_names = ['x', 'y', 'z'] # Default names of coordinatate system axes as in nrn_table
    if depth_name is None:
        depth_name = 'depth' # Default name of depth column in nrn_table
    if isinstance(split_indices, list):
        N_split = len(split_indices)
    else:
        N_split = 0 # Don't split
    if part_idx is not None: # Run only data extraction of given part idx
        assert 0 <= part_idx < N_split, 'ERROR: Part index out of range!'

    pos_table = nrn_table[coord_names].to_numpy()
    depth_table = nrn_table[depth_name].to_numpy()

    if N_split == 0: # Compute all at once
        # Compute distance matrix
        dist_mat = compute_dist_matrix_symmetric(pos_table)

        # Compute bipolar matrix (post-synaptic neuron below (delta_d < 0) or above (delta_d > 0) pre-synaptic neuron)
        bip_mat = compute_bip_matrix(depth_table, depth_table)

        # Extract bipolar distance-dependent connection probabilities
        if max_range_um is None:
            max_range_um = np.nanmax(dist_mat)
        num_dist_bins = np.ceil(max_range_um / bin_size_um).astype(int)
        dist_bins = np.arange(0, num_dist_bins + 1) * bin_size_um
        bip_bins = [np.nanmin(bip_mat), 0, np.nanmax(bip_mat)]

        p_conn_dist_bip, count_conn, count_all = extract_dependent_p_conn(adj_matrix, [dist_mat, bip_mat], [dist_bins, bip_bins])

    else: # Split computation into N_split data splits (to reduce memory consumption)
        assert max_range_um is not None, f'ERROR: Max. range must be specified if data extraction splitted into {N_split} parts!'
        num_dist_bins = np.ceil(max_range_um / bin_size_um).astype(int)
        dist_bins = np.arange(0, num_dist_bins + 1) * bin_size_um
        bip_bins = [-1, 0, 1]

        count_conn = np.zeros([num_dist_bins, 2], dtype=int)
        count_all = np.zeros([num_dist_bins, 2], dtype=int)
        for sidx, split_sel in enumerate(split_indices):
            if part_idx is not None and part_idx != sidx:
                continue
            logging.info(f'<SPLIT {sidx + 1} of {N_split}>')

            # Compute distance matrix
            dist_mat_split = compute_dist_matrix(pos_table[split_sel, :], pos_table)

            # Compute bipolar matrix (post-synaptic neuron below (delta_d < 0) or above (delta_d > 0) pre-synaptic neuron)
            bip_mat_split = compute_bip_matrix(depth_table[split_sel], depth_table)

            # Extract distance-dependent connection counts
            _, count_conn_split, count_all_split = extract_dependent_p_conn(adj_matrix[split_sel, :], [dist_mat_split, bip_mat_split], [dist_bins, bip_bins])
            count_conn += count_conn_split
            count_all += count_all_split

        # Compute overall connection probabilities
        p_conn_dist_bip = np.array(count_conn / count_all)
#         p_conn_dist_bip[np.isnan(p_conn_dist_bip)] = 0.0

    return {'p_conn_dist_bip': p_conn_dist_bip, 'count_conn': count_conn, 'count_all': count_all, 'dist_bins': dist_bins, 'bip_bins': bip_bins}


def build_3rd_order(p_conn_dist_bip, dist_bins, **_):
    """Build 3rd order model (bipolar exp. distance-dependent conn. prob.)."""
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]

    X = dist_bins[:-1][np.all(np.isfinite(p_conn_dist_bip), 1)] + bin_offset
    y = p_conn_dist_bip[np.all(np.isfinite(p_conn_dist_bip), 1), :]

    exp_model = lambda x, a, b: a * np.exp(-b * np.array(x))
    try:
        (bip_neg_exp_model_scale, bip_neg_exp_model_exponent), _ = opt.curve_fit(exp_model, X, y[:, 0], p0=[0.0, 0.0])
        (bip_pos_exp_model_scale, bip_pos_exp_model_exponent), _ = opt.curve_fit(exp_model, X, y[:, 1], p0=[0.0, 0.0])
    except:
        logging.error(f'Exception while fitting model ("{sys.exc_info()[1]}")')
        bip_neg_exp_model_scale = bip_neg_exp_model_exponent = np.nan
        bip_pos_exp_model_scale = bip_pos_exp_model_exponent = np.nan

    logging.info(f'BIPOLAR MODEL FIT: f(x, dz) = {bip_neg_exp_model_scale:.6f} * exp(-{bip_neg_exp_model_exponent:.6f} * x) if dz < 0')
    logging.info(f'                              {bip_pos_exp_model_scale:.6f} * exp(-{bip_pos_exp_model_exponent:.6f} * x) if dz > 0')
    logging.info('                              AVERAGE OF BOTH MODELS  if dz == 0')

    model = 'np.select([np.array(dz) < 0, np.array(dz) > 0, np.array(dz) == 0], [bip_neg_exp_model_scale * np.exp(-bip_neg_exp_model_exponent * np.array(d)), bip_pos_exp_model_scale * np.exp(-bip_pos_exp_model_exponent * np.array(d)), 0.5 * (bip_neg_exp_model_scale * np.exp(-bip_neg_exp_model_exponent * np.array(d)) + bip_pos_exp_model_scale * np.exp(-bip_pos_exp_model_exponent * np.array(d)))])'
    model_inputs = ['d', 'dz']
    model_params = {'bip_neg_exp_model_scale': bip_neg_exp_model_scale, 'bip_neg_exp_model_exponent': bip_neg_exp_model_exponent, 'bip_pos_exp_model_scale': bip_pos_exp_model_scale, 'bip_pos_exp_model_exponent': bip_pos_exp_model_exponent}

    return {'model': model, 'model_inputs': model_inputs, 'model_params': model_params}


def plot_3rd_order(adj_matrix, nrn_table, model_name, p_conn_dist_bip, count_conn, count_all, dist_bins, model, model_inputs, model_params, plot_dir=None, **_):
    """Visualize data vs. model (3rd order)."""
    if plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    dist_model = np.linspace(dist_bins[0], dist_bins[-1], 100)

    model_strN = f'{model_params["bip_neg_exp_model_scale"]:.3f} * exp(-{model_params["bip_neg_exp_model_exponent"]:.3f} * x)'
    model_strP = f'{model_params["bip_pos_exp_model_scale"]:.3f} * exp(-{model_params["bip_pos_exp_model_exponent"]:.3f} * x)'
    model_fct = get_model_function(model, model_inputs, model_params)

    plt.figure(figsize=(12, 4), dpi=300)

    # Data vs. model
    plt.subplot(1, 2, 1)
    bip_dist = np.concatenate((-dist_bins[:-1][::-1] - bin_offset, [0.0], dist_bins[:-1] + bin_offset))
    bip_data = np.concatenate((p_conn_dist_bip[::-1, 0], [np.nan], p_conn_dist_bip[:, 1]))
    all_bins = np.concatenate((-dist_bins[1:][::-1], [0.0], dist_bins[1:]))
    bin_data = np.concatenate((p_conn_dist_bip[::-1, 0], p_conn_dist_bip[:, 1]))
    plt.step(all_bins, np.hstack([bin_data[0], bin_data]), color=DATA_COLOR, label=f'Data: N = {nrn_table.shape[0]}x{nrn_table.shape[0]} cells')
    plt.plot(bip_dist, bip_data, '.', color=DATA_COLOR)
    plt.plot(-dist_model, model_fct(dist_model, np.sign(-dist_model)), '--', color=MODEL_COLOR, label='Model: ' + model_strN)
    plt.plot(dist_model, model_fct(dist_model, np.sign(dist_model)), '--', color=MODEL_COLOR2, label='Model: ' + model_strP)
    plt.grid()
    plt.xlabel('sign($\\Delta$z) * Distance [$\\mu$m]')
    plt.ylabel('Conn. prob.')
    plt.title('Data vs. model fit')
    plt.legend(loc='upper left', fontsize=8)

    # 2D connection probability (model)
    plt.subplot(1, 2, 2)
    plot_range = 500 # (um)
    r_markers = [200, 400] # (um)
    dx = np.linspace(-plot_range, plot_range, 201)
    dz = np.linspace(plot_range, -plot_range, 201)
    xv, zv = np.meshgrid(dx, dz)
    vdist = np.sqrt(xv**2 + zv**2)
    pdist = model_fct(vdist, np.sign(zv))
    plt.imshow(pdist, interpolation='bilinear', extent=(-plot_range, plot_range, -plot_range, plot_range), cmap=PROB_CMAP, vmin=0.0)
    plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
    for r in r_markers:
        plt.gca().add_patch(plt.Circle((0, 0), r, edgecolor='w', linestyle='--', fill=False))
        plt.text(0, r, f'{r} $\\mu$m', color='w', ha='center', va='bottom')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$\\Delta$x')
    plt.ylabel('$\\Delta$z')
    plt.title('2D model')
    plt.colorbar(label='Conn. prob.')

    plt.suptitle(f'Bipolar distance-dependent connection probability model (3rd order)')
    plt.tight_layout()
    if plot_dir is not None:
        out_fn = os.path.abspath(os.path.join(plot_dir, model_name + '__data_vs_model.png'))
        plt.savefig(out_fn)
        logging.info(f'Figure saved to {out_fn}')

    # Data counts
    bip_count = np.concatenate((count_conn[::-1, 0], [np.nan], count_conn[:, 1]))
    bip_count_all = np.concatenate((count_all[::-1, 0], [np.nan], count_all[:, 1]))
    plt.figure(figsize=(12, 4), dpi=300)
    plt.bar(bip_dist, bip_count_all, width=2.0 * bin_offset, edgecolor='k', label='All pair count')
    plt.bar(bip_dist, bip_count, width=1.5 * bin_offset, label='Connection count')
    plt.gca().set_yscale('log')
    plt.grid()
    plt.xlabel('sign($\\Delta$z) * Distance [$\\mu$m]')
    plt.ylabel('Count')
    plt.title(f'Bipolar distance-dependent connection counts (N = {nrn_table.shape[0]}x{nrn_table.shape[0]} cells)')
    plt.legend()
    plt.tight_layout()
    if plot_dir is not None:
        out_fn = os.path.abspath(os.path.join(plot_dir, model_name + '__data_counts.png'))
        plt.savefig(out_fn)
        logging.info(f'Figure saved to {out_fn}')



###################################################################################################
# Main function for running as batch script (optionally, on different data splits)
###################################################################################################

def main(adj_file, nrn_file, cfg_file, N_split=None, part_idx=None):
    """ Main function for data extraction and model building
        to be used in batch script on different data splits
    """

    # Load adjacency matrix (.npz) & neuron properties table (.h5 or .feather)
    adj_matrix = sps.load_npz(adj_file)
    if os.path.splitext(nrn_file)[-1] == '.h5':
        nrn_table = pd.read_hdf(nrn_file)
    elif os.path.splitext(nrn_file)[-1] == '.feather':
        nrn_table = pd.read_feather(nrn_file)
    else:
        assert False, f'ERROR: Neuron table format "{os.path.splitext(nrn_file)[-1]}" not supported!'

    assert adj_matrix.shape[0] == adj_matrix.shape[1] == nrn_table.shape[0], 'ERROR: Data size mismatch!'
    logging.info(f'Loaded connectivity and properties of {nrn_table.shape[0]} neurons')

    # Load config file (.json)
    with open(cfg_file, 'r') as f:
        config_dict = json.load(f)

    # Set/Overwrite data split options
    if N_split is not None:
        config_dict.update({'N_split': int(N_split)})
    if part_idx is not None:
        config_dict.update({'part_idx': int(part_idx)})

    # Run model building
    run_model_building(adj_matrix, nrn_table, **config_dict)


if __name__ == "__main__":
    main(*sys.argv[1:])

# SPDX-FileCopyrightText: 2024 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Generate models for connectomes.
#
# Author(s): C. Pokorny
# Last modified: 04/2023


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

PROB_CMAP = plt.colormaps.get_cmap('hot')
DATA_COLOR = 'tab:blue'
MODEL_COLOR = 'tab:red'
MODEL_COLOR2 = 'tab:olive'

###################################################################################################
# Wrapper function for running model building from a SLURM batch script (optionally, on different data splits)
###################################################################################################

def run_batch_model_building(adj_file, nrn_file, cfg_file, N_split=None, part_idx=None):
    """Main function for data extraction and model building to be used in a batch script on different data splits.

    Parameters
    ----------
    adj_file : str
        File name (.npz format) of scipy.sparse adjacency matrix of the circuit
    nrn_file : str
        File name (.h5 or .feather format) of pandas.DataFrame with neuron properties
    cfg_file : str
        File name (.json format) of config dict specifying the model building operation; see Notes for details
    N_split : int, optional
        Number of data splits to divide data extraction into (to reduce memory consumption)
    part_idx : int, optional
        Index of current data split (part) to extract data from
        Range:  0 .. N_split - 1 Run data extraction of given data split
               -1                Merge data splits and build model

    Returns
    -------
    None
        Nothing returned here; Data/model/figures are written to output directories as specified in `cfg_file`

    Raises
    ------
    AssertionError
        If nrn_file is not in .h5 or .feather format
    AssertionError
        If the adjacency matrix is not a square matrix matching the length of the neuron properties table
    AssertionError
        If model order not supported (supported: 2, 3)
    AssertionError
        If model fitting error occurs
    KeyError
        If name(s) of coordinates not in columns of neuron properties table

    Notes
    -----
    The adjacency matrix encodes connectivity between source (rows) and taget (columns) neurons.

    `cfg_file` must be a .json file containing a dictionary with following entries, most of which are optional:

    - `model_name` Name of the model (to be used in file names, ...)
    - `model_order` Model order (2 or 3)
    - `bin_size_um` Bin size in um for depth binning (optional; default: 100)
    - `max_range_um` Max. distance range in um to consider (optional; default: full distance range)
    - `sample_size` Size of random subset of neurons to consider (optional; default: no subsampling)
    - `sample_seed` Seed for reproducible selection of random subset of neurons (optional)
    - `coord_names` Names of the coordinates (columns in neuron properties table) based on which to compute Euclidean distance (optional; default: ["x", "y", "z"])
    - `depth_name` Name of depth coordinate (column in neuron properties table) to use in 3rd-order (bipolar) model (optional; default: "depth")
    - `model_dir` Output directory where to save the model (optional; default: no saving)
    - `data_dir` Output directory where to save the extracted data (optional; default: no saving)
    - `do_plot` Enable/disable output plotting (optional; default: no plotting)
    - `plot_dir` Output directory where to save the plots, if plotting enabled (optional; default: no saving)
    - `N_split` Number of data splits (> 1) to sequentially extract data from, to reduce memory consumption (optional; default: no splitting)
    - `part_idx` Part index (from 0 to N_split-1) to run data extraction only on a specific data split; -1 to merge existing splits and build model (optional; default: data extraction and model building for all splits)

    See Also
    --------
    run_model_building : Underlying main function for model building

    """

    # Load adjacency matrix (.npz) & neuron properties table (.h5 or .feather)
    adj = sps.load_npz(adj_file)
    if os.path.splitext(nrn_file)[-1] == '.h5':
        node_properties = pd.read_hdf(nrn_file)
    elif os.path.splitext(nrn_file)[-1] == '.feather':
        node_properties = pd.read_feather(nrn_file)
    else:
        assert False, f'ERROR: Neuron table format "{os.path.splitext(nrn_file)[-1]}" not supported!'

    assert adj.shape[0] == adj.shape[1] == node_properties.shape[0], 'ERROR: Data size mismatch!'
    logging.info(f'Loaded connectivity and properties of {node_properties.shape[0]} neurons')

    # Load config file (.json)
    with open(cfg_file, 'r') as f:
        config_dict = json.load(f)

    # Set/Overwrite data split options
    if N_split is not None:
        config_dict.update({'N_split': int(N_split)})
    if part_idx is not None:
        config_dict.update({'part_idx': int(part_idx)})

    # Run model building
    run_model_building(adj, node_properties, **config_dict)


###################################################################################################
# Wrapper & helper functions for model building to be used within a processing pipeline
#   w/o data/model saving, figure plotting, and data splitting
#   _Inputs_: adj, node_properties, kwargs (bin_size_um, max_range_um, sample_size, sample_seeds)
#   _Output_: Pandas dataframe with model paramters (columns) for different seeds (rows)
###################################################################################################

def conn_prob_2nd_order_model(adj, node_properties, **kwargs):
    """Wrapper function for 2nd-order probability model building to be used within a processing pipeline, optionally for multiple random subsets of neurons.

    Parameters
    ----------
    adj : scipy.sparse
        Sparse (symmetric) adjacency matrix of the circuit
    node_properties : pandas.DataFrame
        Data frame with neuron properties
    kwargs : dict, optional
        Additional model building settings; see Notes for details

    Returns
    -------
    pandas.DataFrame
        Data frame with model paramters (columns) for different seeds (rows)
        (No plotting and data/model/figures saving supported)

    Raises
    ------
    AssertionError
        If the adjacency matrix is not a square matrix matching the length of the neuron properties table
    AssertionError
        If invalid arguments given in kwargs which are internally used by this wrapper (like model_order, ...)
    AssertionError
        If model fitting error occurs
    AssertionError
        If sample_seeds provided as scalar but is not a positive integer
    KeyError
        If name(s) of coordinates not in columns of neuron properties table
    Warning
        If sample_seeds provided as list with duplicates
    Warning
        If sample_seeds provided but ignored because subsampling not applicable

    Notes
    -----
    The adjacency matrix encodes connectivity between source (rows) and taget (columns) neurons.

    The 2nd-order model as defined in [1]_ describes connection probabilities as a function of distance between pre- and post-synaptic neurons. Specifically, we use here an exponential distance-dependent model of the form:
    $$
    p(d) = \mbox{scale} * exp(-\mbox{exponent} * d)
    $$
    with `d` as distance in $\mu m$, and the model parameters `scale` defining the connection probability at distance zero, and `exponent` the exponent of distance-dependent decay in $\mu m^{-1}$.

    `kwargs` may contain following (optional) settings:

    - `bin_size_um` Bin size in um for depth binning (optional; default: 100)
    - `max_range_um` Max. distance range in um to consider (optional; default: full distance range)
    - `sample_size` Size of random subset of neurons to consider (optional; default: no subsampling)
    - `sample_seeds` Integer number of seeds to randomly generate, or list of specific random seeds, for reproducible selection of random subset of neurons (optional)
    - `meta_seed` Meta seed for generating N random seeds, if integer number N of sample_seeds is provided (optional; default: 0)
    - `coord_names` Names of the coordinates (columns in neuron properties table) based on which to compute Euclidean distance (optional; default: ["x", "y", "z"])
    - `N_split` Number of data splits (> 1) to sequentially extract data from, to reduce memory consumption (optional; default: no splitting)

    See Also
    --------
    conn_prob_2nd_order_pathway_model : 2nd-order model building function wrapper for different source/target node populations
    conn_prob_model : Underlying generic model building function wrapper

    References
    ----------
    .. [1] Gal E, Perin R, Markram H, London M, Segev I, "Neuron Geometry Underlies Universal Network Features in Cortical Microcircuits," bioRxiv, doi: https://doi.org/10.1101/656058.

    """

    assert 'model_order' not in kwargs.keys(), f'ERROR: Invalid argument "model_order" in kwargs!'

    return conn_prob_model(adj, node_properties, model_order=2, **kwargs)


def conn_prob_2nd_order_pathway_model(adj, node_properties_src, node_properties_tgt, **kwargs):
    """Wrapper function for 2nd-order probability model building to be used within a processing pipeline for pathways with different source and target node populations, optionally for multiple random subsets of neurons.

    Parameters
    ----------
    adj : scipy.sparse
        Sparse adjacency matrix of the circuit (may be non-symmetric)
    node_properties_src : pandas.DataFrame
        Data frame with source neuron properties (corresponding to the rows in adj)
    node_properties_tgt : pandas.DataFrame
        Data frame with target neuron properties (corresponding to the columns in adj)
    kwargs : dict, optional
        Additional model building settings; see "See Also" for details

    Returns
    -------
    pandas.DataFrame
        Data frame with model paramters (columns) for different seeds (rows)
        (No plotting and data/model/figures saving supported)

    Raises
    ------
    AssertionError
        If the rows/columns of the adjacency matrix are not matching the lengths of the source/target neuron properties tables
    AssertionError
        If invalid arguments given in kwargs which are internally used by this wrapper (like model_order, ...)
    AssertionError
        If model fitting error occurs
    AssertionError
        If data splitting selected, which is not supported for pathway model building
    AssertionError
        If sample_seeds provided as scalar but is not a positive integer
    KeyError
        If name(s) of coordinates not in columns of neuron properties table
    Warning
        If sample_seeds provided as list with duplicates
    Warning
        If sample_seeds provided but ignored because subsampling not applicable

    Notes
    -----
    The adjacency matrix encodes connectivity between source (rows) and taget (columns) neurons.

    The 2nd-order model as defined in [1]_. See "See Also" for details.

    See Also
    --------
    conn_prob_2nd_order_model : Special case of 2nd-order model building function wrapper for same source/target node population; further details to be found here

    References
    ----------
    .. [1] Gal E, Perin R, Markram H, London M, Segev I, "Neuron Geometry Underlies Universal Network Features in Cortical Microcircuits," bioRxiv, doi: https://doi.org/10.1101/656058.

    """

    assert 'model_order' not in kwargs.keys(), f'ERROR: Invalid argument "model_order" in kwargs!'

    return conn_prob_pathway_model(adj, node_properties_src, node_properties_tgt, model_order=2, **kwargs)


def conn_prob_3rd_order_model(adj, node_properties, **kwargs):
    """Wrapper function for 3rd-order probability model building to be used within a processing pipeline, optionally for multiple random subsets of neurons.

    Parameters
    ----------
    adj : scipy.sparse
        Sparse (symmetric) adjacency matrix of the circuit
    node_properties : pandas.DataFrame
        Data frame with neuron properties
    kwargs : dict, optional
        Additional model building settings; see Notes for details

    Returns
    -------
    pandas.DataFrame
        Data frame with model paramters (columns) for different seeds (rows)
        (No plotting and data/model/figures saving supported)

    Raises
    ------
    AssertionError
        If the adjacency matrix is not a square matrix matching the length of the neuron properties table
    AssertionError
        If invalid arguments given in kwargs which are internally used by this wrapper (like model_order, ...)
    AssertionError
        If model fitting error occurs
    AssertionError
        If sample_seeds provided as scalar but is not a positive integer
    KeyError
        If name(s) of coordinates not in columns of neuron properties table
    Warning
        If sample_seeds provided as list with duplicates
    Warning
        If sample_seeds provided but ignored because subsampling not applicable

    Notes
    -----
    The adjacency matrix encodes connectivity between source (rows) and taget (columns) neurons.

    The 3rd-order model as defined in [1]_ describes connection probabilities as a bipolar function of distance between pre- and post-synaptic neurons. Specifically, we use here an bipolar exponential distance-dependent model of the form:
    $$
    p(d, \Delta depth) = \mbox{scale}_N * exp(-\mbox{exponent}_N * d)~\mbox{if}~\Delta depth < 0
    $$
    $$
    p(d, \Delta depth) = \mbox{scale}_P * exp(-\mbox{exponent}_P * d)~\mbox{if}~\Delta depth > 0
    $$
    $$
    p(d, \Delta depth) = \mbox{Average of both}~\mbox{if}~\Delta depth = 0
    $$
    with `d` as distance in $\mu m$, $\Delta depth$ as difference in depth coordinate (arbitrary unit, as only sign is used; post-synaptic neuron below ($\Delta depth < 0$) or above ($\Delta depth > 0$) pre-synaptic neuron), and the model parameters `scale` defining the connection probability at distance zero, and `exponent` the exponent of distance-dependent decay in $\mu m^{-1}$ for both cases.

    `kwargs` may contain following (optional) settings:

    - `bin_size_um` Bin size in um for depth binning (optional; default: 100)
    - `max_range_um` Max. distance range in um to consider (optional; default: full distance range)
    - `sample_size` Size of random subset of neurons to consider (optional; default: no subsampling)
    - `sample_seeds` Integer number of seeds to randomly generate, or list of specific random seeds, for reproducible selection of random subset of neurons (optional)
    - `meta_seed` Meta seed for generating N random seeds, if integer number N of sample_seeds is provided (optional; default: 0)
    - `coord_names` Names of the coordinates (columns in neuron properties table) based on which to compute Euclidean distance (optional; default: ["x", "y", "z"])
    - `depth_name` Name of depth coordinate (column in neuron properties table) to use in 3rd-order (bipolar) model (optional; default: "depth")
    - `N_split` Number of data splits (> 1) to sequentially extract data from, to reduce memory consumption (optional; default: no splitting)

    See Also
    --------
    conn_prob_3rd_order_pathway_model : 3rd-order model building function wrapper for different source/target node populations
    conn_prob_model : Underlying generic model building function wrapper

    References
    ----------
    .. [1] Gal E, Perin R, Markram H, London M, Segev I, "Neuron Geometry Underlies Universal Network Features in Cortical Microcircuits," bioRxiv, doi: https://doi.org/10.1101/656058.

    """

    assert 'model_order' not in kwargs.keys(), f'ERROR: Invalid argument "model_order" in kwargs!'

    return conn_prob_model(adj, node_properties, model_order=3, **kwargs)


def conn_prob_3rd_order_pathway_model(adj, node_properties_src, node_properties_tgt, **kwargs):
    """Wrapper function for 3rd-order probability model building to be used within a processing pipeline for pathways with different source and target node populations, optionally for multiple random subsets of neurons.

    Parameters
    ----------
    adj : scipy.sparse
        Sparse adjacency matrix of the circuit (may be non-symmetric)
    node_properties_src : pandas.DataFrame
        Data frame with source neuron properties (corresponding to the rows in adj)
    node_properties_tgt : pandas.DataFrame
        Data frame with target neuron properties (corresponding to the columns in adj)
    kwargs : dict, optional
        Additional model building settings; see "See Also" for details

    Returns
    -------
    pandas.DataFrame
        Data frame with model paramters (columns) for different seeds (rows)
        (No plotting and data/model/figures saving supported)

    Raises
    ------
    AssertionError
        If the rows/columns of the adjacency matrix are not matching the lengths of the source/target neuron properties tables
    AssertionError
        If invalid arguments given in kwargs which are internally used by this wrapper (like model_order, ...)
    AssertionError
        If model fitting error occurs
    AssertionError
        If data splitting selected, which is not supported for pathway model building
    AssertionError
        If sample_seeds provided as scalar but is not a positive integer
    KeyError
        If name(s) of coordinates not in columns of neuron properties table
    Warning
        If sample_seeds provided as list with duplicates
    Warning
        If sample_seeds provided but ignored because subsampling not applicable

    Notes
    -----
    The adjacency matrix encodes connectivity between source (rows) and taget (columns) neurons.

    The 3rd-order model as defined in [1]_. See "See Also" for details.

    See Also
    --------
    conn_prob_3rd_order_model : Special case of 3rd-order model building function wrapper for same source/target node population; further details to be found here

    References
    ----------
    .. [1] Gal E, Perin R, Markram H, London M, Segev I, "Neuron Geometry Underlies Universal Network Features in Cortical Microcircuits," bioRxiv, doi: https://doi.org/10.1101/656058.

    """

    assert 'model_order' not in kwargs.keys(), f'ERROR: Invalid argument "model_order" in kwargs!'

    return conn_prob_pathway_model(adj, node_properties_src, node_properties_tgt, model_order=3, **kwargs)


def conn_prob_model(adj, node_properties, **kwargs):
    """Wrapper function for generic probability model building to be used within a processing pipeline, optionally for multiple random subsets of neurons.

    Parameters
    ----------
    adj : scipy.sparse
        Sparse (symmetric) adjacency matrix of the circuit
    node_properties : pandas.DataFrame
        Data frame with neuron properties
    kwargs : dict, optional
        Additional model building settings; see Notes for details

    Returns
    -------
    pandas.DataFrame
        Data frame with model paramters (columns) for different seeds (rows)
        (No plotting and data/model/figures saving supported)

    Raises
    ------
    AssertionError
        If the adjacency matrix is not a square matrix matching the length of the neuron properties table
    AssertionError
        If invalid arguments given in kwargs which are internally used by this wrapper
    AssertionError
        If model fitting error occurs
    AssertionError
        If sample_seeds provided as scalar but is not a positive integer
    AssertionError
        If model order not supported (supported: 2, 3)
    KeyError
        If model order not provided
    KeyError
        If name(s) of coordinates not in columns of neuron properties table
    Warning
        If sample_seeds provided as list with duplicates
    Warning
        If sample_seeds provided but ignored because subsampling not applicable

    Notes
    -----
    The adjacency matrix encodes connectivity between source (rows) and taget (columns) neurons.

    The 2nd-order and 3rd-order models as defined in [1]_ are supported. See "See Also" for details.

    `kwargs` may contain following settings, most of which are optional:

    - `model_order` Model order (2 or 3)
    - `bin_size_um` Bin size in um for depth binning (optional; default: 100)
    - `max_range_um` Max. distance range in um to consider (optional; default: full distance range)
    - `sample_size` Size of random subset of neurons to consider (optional; default: no subsampling)
    - `sample_seeds` Integer number of seeds to randomly generate, or list of specific random seeds, for reproducible selection of random subset of neurons (optional)
    - `meta_seed` Meta seed for generating N random seeds, if integer number N of sample_seeds is provided (optional; default: 0)
    - `coord_names` Names of the coordinates (columns in neuron properties table) based on which to compute Euclidean distance (optional; default: ["x", "y", "z"])
    - `depth_name` Name of depth coordinate (column in neuron properties table) to use in 3rd-order (bipolar) model (optional; default: "depth")
    - `N_split` Number of data splits (> 1) to sequentially extract data from, to reduce memory consumption (optional; default: no splitting)

    See Also
    --------
    conn_prob_2nd_order_model : 2nd-order model building function wrapper for same source/target node population
    conn_prob_3rd_order_model : 3rd-order model building function wrapper for same source/target node population
    conn_prob_pathway_model : Generic model building function wrapper for differet source/target node populations

    References
    ----------
    .. [1] Gal E, Perin R, Markram H, London M, Segev I, "Neuron Geometry Underlies Universal Network Features in Cortical Microcircuits," bioRxiv, doi: https://doi.org/10.1101/656058.

    """

    assert adj.shape[0] == adj.shape[1] == node_properties.shape[0], 'ERROR: Data size mismatch!'

    invalid_args = ['model_name', 'sample_seed', 'model_dir', 'data_dir', 'plot_dir', 'do_plot', 'part_idx'] # Not allowed arguments, as they will be set/used internally
    for arg in invalid_args:
        assert arg not in kwargs.keys(), f'ERROR: Invalid argument "{arg}" in kwargs!'
    kwargs.update({'model_dir': None, 'data_dir': None, 'plot_dir': None, 'do_plot': False, 'part_idx': None}) # Disable plotting/saving
    model_name = None
    model_order = kwargs.pop('model_order')

    sample_size = kwargs.get('sample_size')
    if sample_size is None or sample_size <= 0 or sample_size >= node_properties.shape[0]:
        sample_seeds = [None] # No randomization
        if kwargs.pop('sample_seeds', None) is not None:
            logging.warning('Using all neurons, ignoring sample seeds!')
    else:
        sample_seeds = kwargs.pop('sample_seeds', 1)

        if not isinstance(sample_seeds, list): # sample_seeds corresponds to number of seeds to generate
            sample_seeds = _generate_seeds(sample_seeds, meta_seed=kwargs.pop('meta_seed', 0))
        else:
            num_seeds = len(sample_seeds)
            sample_seeds = list(np.unique(sample_seeds)) # Assure that unique and sorted
            if len(sample_seeds) < num_seeds:
                logging.warning(f'Duplicate seeds provided!')

    model_params = pd.DataFrame()
    for seed in sample_seeds:
        kwargs.update({'sample_seed': seed})
        _, model_dict = run_model_building(adj, node_properties, model_name, model_order, **kwargs)
        model_params = pd.concat([model_params, pd.DataFrame(model_dict['model_params'], index=pd.Index([seed], name='seed'))])

    return model_params


def conn_prob_pathway_model(adj, node_properties_src, node_properties_tgt, **kwargs):
    """Wrapper function for generic probability model building to be used within a processing pipeline for pathways with different source and target node populations, optionally for multiple random subsets of neurons.

    Parameters
    ----------
    adj : scipy.sparse
        Sparse adjacency matrix of the circuit (may be non-symmetric)
    node_properties_src : pandas.DataFrame
        Data frame with source neuron properties (corresponding to the rows in adj)
    node_properties_tgt : pandas.DataFrame
        Data frame with target neuron properties (corresponding to the columns in adj)
    kwargs : dict, optional
        Additional model building settings; see "See Also" for details

    Returns
    -------
    pandas.DataFrame
        Data frame with model paramters (columns) for different seeds (rows)
        (No plotting and data/model/figures saving supported)

    Raises
    ------
    AssertionError
        If the rows/columns of the adjacency matrix are not matching the lengths of the source/target neuron properties tables
    AssertionError
        If invalid arguments given in kwargs which are internally used by this wrapper
    AssertionError
        If model fitting error occurs
    AssertionError
        If sample_seeds provided as scalar but is not a positive integer
    AssertionError
        If model order not supported (supported: 2, 3)
    AssertionError
        If data splitting selected, which is not supported for pathway model building
    KeyError
        If model order not provided
    KeyError
        If name(s) of coordinates not in columns of neuron properties table
    Warning
        If sample_seeds provided as list with duplicates
    Warning
        If sample_seeds provided but ignored because subsampling not applicable

    Notes
    -----
    The adjacency matrix encodes connectivity between source (rows) and taget (columns) neurons.

    The 2nd-order and 3rd-order models as defined in [1]_ are supported. See "See Also" for details.

    See Also
    --------
    conn_prob_model : Special case of generic model building function wrapper for same source/target node population; further details to be found here
    conn_prob_2nd_order_pathway_model : 2nd-order model building function wrapper for different source/target node population
    conn_prob_3rd_order_pathway_model : 3rd-order model building function wrapper for different source/target node population

    References
    ----------
    .. [1] Gal E, Perin R, Markram H, London M, Segev I, "Neuron Geometry Underlies Universal Network Features in Cortical Microcircuits," bioRxiv, doi: https://doi.org/10.1101/656058.

    """

    assert adj.shape[0] == node_properties_src.shape[0] and adj.shape[1] == node_properties_tgt.shape[0], 'ERROR: Data size mismatch!'

    invalid_args = ['model_name', 'sample_seed', 'model_dir', 'data_dir', 'plot_dir', 'do_plot', 'part_idx'] # Not allowed arguments, as they will be set/used internally
    for arg in invalid_args:
        assert arg not in kwargs.keys(), f'ERROR: Invalid argument "{arg}" in kwargs!'
    kwargs.update({'model_dir': None, 'data_dir': None, 'plot_dir': None, 'do_plot': False, 'part_idx': None}) # Disable plotting/saving
    model_name = None
    model_order = kwargs.pop('model_order')

    sample_size = kwargs.get('sample_size')
    if sample_size is None  or sample_size <= 0 or sample_size >= np.maximum(node_properties_src.shape[0], node_properties_tgt.shape[0]):
        sample_seeds = [None] # No randomization
        if kwargs.pop('sample_seeds', None) is not None:
            logging.warning('Using all neurons, ignoring sample seeds!')
    else:
        sample_seeds = kwargs.pop('sample_seeds', 1)

        if not isinstance(sample_seeds, list): # sample_seeds corresponds to number of seeds to generate
            sample_seeds = _generate_seeds(sample_seeds, meta_seed=kwargs.pop('meta_seed', 0))
        else:
            num_seeds = len(sample_seeds)
            sample_seeds = list(np.unique(sample_seeds)) # Assure that unique and sorted
            if len(sample_seeds) < num_seeds:
                logging.warning(f'Duplicate seeds provided!')

    model_params = pd.DataFrame()
    for seed in sample_seeds:
        kwargs.update({'sample_seed': seed})
        _, model_dict = run_pathway_model_building(adj, node_properties_src, node_properties_tgt, model_name, model_order, **kwargs)
        model_params = pd.concat([model_params, pd.DataFrame(model_dict['model_params'], index=pd.Index([seed], name='seed'))])

    return model_params


def _generate_seeds(num_seeds, num_digits=6, meta_seed=0):
    """Helper function to generate list of unique random seeds with given number of digits."""

    assert isinstance(num_seeds, int) and num_seeds > 0, 'ERROR: Number of seeds must be a positive integer!'
    assert isinstance(num_digits, int) and num_digits > 0, 'ERROR: Number of digits must be a positive integer!'

    np.random.seed(meta_seed)
    sample_seeds = list(sorted(np.random.choice(10**num_digits - 10**(num_digits - 1), num_seeds, replace=False) + 10**(num_digits - 1))) # Sorted, 6-digit seeds

    return sample_seeds


###################################################################################################
# Main function for model building
###################################################################################################

def run_model_building(adj, node_properties, model_name, model_order, **kwargs):
    """Main function for probability model building, consisting of three steps: Data extraction, model fitting, and (optionally) data/model visualization.

    Parameters
    ----------
    adj : scipy.sparse
        Sparse (symmetric) adjacency matrix of the circuit
    node_properties : pandas.DataFrame
        Data frame with neuron properties
    model_name : str
        Name of the model (to be used in file names, ...)
    model_order : int
        Model order (2 or 3)
    kwargs : dict, optional
        Additional model building settings; see Notes for details

    Returns
    -------
    dict
        Data dictionary containing extracted data points (connection probabilities) from the "extract" step; Data/figures also written to output directories as specified in kwargs
    dict
        Model dictionary containing probability model fitted to data points from "model fitting" step; Model/figures also written to output directories as specified in kwargs

    Raises
    ------
    AssertionError
        If the adjacency matrix is not a square matrix matching the length of the neuron properties table
    AssertionError
        If model order not supported (supported: 2, 3)
    AssertionError
        If model fitting error occurs
    KeyError
        If name(s) of coordinates not in columns of neuron properties table

    Notes
    -----
    The adjacency matrix encodes connectivity between source (rows) and taget (columns) neurons.

    The 2nd-order and 3rd-order models as defined in [1]_ are supported. See "See Also" for details.

    `kwargs` may contain following (optional) settings:

    - `bin_size_um` Bin size in um for depth binning (optional; default: 100)
    - `max_range_um` Max. distance range in um to consider (optional; default: full distance range)
    - `sample_size` Size of random subset of neurons to consider (optional; default: no subsampling)
    - `sample_seed` Seed for reproducible selection of random subset of neurons (optional)
    - `coord_names` Names of the coordinates (columns in neuron properties table) based on which to compute Euclidean distance (optional; default: ["x", "y", "z"])
    - `depth_name` Name of depth coordinate (column in neuron properties table) to use in 3rd-order (bipolar) model (optional; default: "depth")
    - `model_dir` Output directory where to save the model (optional; default: no saving)
    - `data_dir` Output directory where to save the extracted data (optional; default: no saving)
    - `do_plot` Enable/disable output plotting (optional; default: no plotting)
    - `plot_dir` Output directory where to save the plots, if plotting enabled (optional; default: no saving)
    - `N_split` Number of data splits (> 1) to sequentially extract data from, to reduce memory consumption (optional; default: no splitting)
    - `part_idx` Part index (from 0 to N_split-1) to run data extraction only on a specific data split; -1 to merge existing splits and build model (optional; default: data extraction and model building for all splits)

    See Also
    --------
    run_pathway_model_building : Main model building function for differet source/target node populations
    conn_prob_2nd_order_model : 2nd-order model building function wrapper for same source/target node population to be used within a processing pipeline
    conn_prob_3rd_order_model : 3rd-order model building function wrapper for same source/target node population to be used within a processing pipeline

    References
    ----------
    .. [1] Gal E, Perin R, Markram H, London M, Segev I, "Neuron Geometry Underlies Universal Network Features in Cortical Microcircuits," bioRxiv, doi: https://doi.org/10.1101/656058.

    """

    logging.info(f'Running order-{model_order} model building {kwargs}...')

    assert adj.shape[0] == adj.shape[1] == node_properties.shape[0], 'ERROR: Data size mismatch!'

    # Subsampling (optional)
    sample_size = kwargs.get('sample_size')
    sample_seed = kwargs.get('sample_seed')
    if sample_size is not None and sample_size > 0 and sample_size < node_properties.shape[0]:
        logging.info(f'Subsampling to {sample_size} of {node_properties.shape[0]} neurons (seed={sample_seed})')
        np.random.seed(sample_seed)
        sub_sel = np.random.permutation([True] * sample_size + [False] * (node_properties.shape[0] - sample_size))
        adj = adj.tocsr()[sub_sel, :].tocsc()[:, sub_sel].tocsr()
        node_properties = node_properties.loc[sub_sel, :]

    # Set modelling functions
    if model_order == 2: # Distance-dependent
        fct_extract = _extract_2nd_order
        fct_fit = _build_2nd_order
        fct_plot = _plot_2nd_order
    elif model_order == 3: # Bipolar distance-dependent
        fct_extract = _extract_3rd_order
        fct_fit = _build_3rd_order
        fct_plot = _plot_3rd_order
    else:
        assert False, f'ERROR: Order-{model_order} model building not supported!'

    # Data splits (optional)
    N_split = kwargs.pop('N_split', None)
    part_idx = kwargs.pop('part_idx', None)
    if N_split is None:
        split_indices = None
    else:
        assert N_split > 1, 'ERROR: Number of data splits must be larger than 1!'
        split_indices = np.split(np.arange(node_properties.shape[0]), np.cumsum([np.ceil(node_properties.shape[0] / N_split).astype(int)] * (N_split - 1)))

    if part_idx is None or part_idx == -1: # Run data extraction and model building for all splits
        extract_only = False
        data_fn = 'data'
    else: # Run only data extraction of given part idx
        assert N_split is not None and 0 <= part_idx < N_split, 'ERROR: Part index out of range!'
        extract_only = True
        data_fn = 'data' + _get_data_part_name(N_split, part_idx)

    # Extract connection probability data
    if part_idx == -1: # Special case: Load and merge results of existing parts
        assert N_split is not None, 'ERROR: Number of data splits required!'
        data_dict = _merge_data(kwargs.get('data_dir'), model_name, data_fn, [_get_data_part_name(N_split, p) for p in range(N_split)])
    else:
        data_dict = fct_extract(adj, node_properties, split_indices=split_indices, part_idx=part_idx, **kwargs)
    _save_data(data_dict, kwargs.get('data_dir'), model_name, data_fn)

    if extract_only: # Stop here and return data dict
        return data_dict, {}

    # Fit model
    model_dict = fct_fit(**data_dict, **kwargs)
    _save_data(model_dict, kwargs.get('model_dir'), model_name, 'model')

    # Visualize data/model (optional)
    if kwargs.get('do_plot'):
        fct_plot(adj, node_properties, model_name, **data_dict, **model_dict, **kwargs)

    return data_dict, model_dict


def run_pathway_model_building(adj, node_properties_src, node_properties_tgt, model_name, model_order, **kwargs):
    """Main function for probability model building for pathways with different source and target node populations, consisting of three steps: Data extraction, model fitting, and (optionally) data/model visualization.

    Parameters
    ----------
    adj : scipy.sparse
        Sparse adjacency matrix of the circuit (may be non-symmetric)
    node_properties_src : pandas.DataFrame
        Data frame with source neuron properties (corresponding to the rows in adj)
    node_properties_tgt : pandas.DataFrame
        Data frame with target neuron properties (corresponding to the columns in adj)
    model_name : str
        Name of the model (to be used in file names, ...)
    model_order : int
        Model order (2 or 3)
    kwargs : dict, optional
        Additional model building settings; see "See Also" for details

    Returns
    -------
    dict
        Data dictionary containing extracted data points (connection probabilities) from the "extract" step; Data/figures also written to output directories as specified in kwargs
    dict
        Model dictionary containing probability model fitted to data points from "model fitting" step; Model/figures also written to output directories as specified in kwargs

    Raises
    ------
    AssertionError
        If the rows/columns of the adjacency matrix are not matching the lengths of the source/target neuron properties tables
    AssertionError
        If model order not supported (supported: 2, 3)
    AssertionError
        If model fitting error occurs
    AssertionError
        If data splitting selected, which is not supported for pathway model building
    KeyError
        If name(s) of coordinates not in columns of neuron properties table

    Notes
    -----
    The adjacency matrix encodes connectivity between source (rows) and taget (columns) neurons.

    The 2nd-order and 3rd-order models as defined in [1]_ are supported. See "See Also" for details.

    See Also
    --------
    run_model_building : Main model building function for same source/target node populations; further details to be found here
    conn_prob_2nd_order_pathway_model : 2nd-order model building function wrapper for different source/target node populations to be used within a processing pipeline
    conn_prob_3rd_order_pathway_model : 3rd-order model building function wrapper for different source/target node populations to be used within a processing pipeline

    References
    ----------
    .. [1] Gal E, Perin R, Markram H, London M, Segev I, "Neuron Geometry Underlies Universal Network Features in Cortical Microcircuits," bioRxiv, doi: https://doi.org/10.1101/656058.

    """

    logging.info(f'Running order-{model_order} model building {kwargs}...')

    assert adj.shape[0] == node_properties_src.shape[0] and adj.shape[1] == node_properties_tgt.shape[0], 'ERROR: Data size mismatch!'

    # Subsampling (optional)
    sample_size = kwargs.get('sample_size')
    sample_seed = kwargs.get('sample_seed')
    if sample_size is not None and sample_size > 0 and sample_size < np.maximum(node_properties_src.shape[0], node_properties_tgt.shape[0]):
        logging.info(f'Subsampling to {sample_size} of {node_properties_src.shape[0]}x{node_properties_tgt.shape[0]} neurons (seed={sample_seed})')
        np.random.seed(sample_seed)
        if sample_size < node_properties_src.shape[0]:
            sub_sel_src = np.random.permutation([True] * sample_size + [False] * (node_properties_src.shape[0] - sample_size))
        else:
            sub_sel_src = np.full(node_properties_src.shape[0], True)

        if sample_size < node_properties_tgt.shape[0]:
            sub_sel_tgt = np.random.permutation([True] * sample_size + [False] * (node_properties_tgt.shape[0] - sample_size))
        else:
            sub_sel_tgt = np.full(node_properties_tgt.shape[0], True)

        adj = adj.tocsr()[sub_sel_src, :].tocsc()[:, sub_sel_tgt].tocsr()
        # adj = adj[sub_sel_src, :][:, sub_sel_tgt]
        node_properties_src = node_properties_src.loc[sub_sel_src, :]
        node_properties_tgt = node_properties_tgt.loc[sub_sel_tgt, :]

    # Set modelling functions
    if model_order == 2: # Distance-dependent
        fct_extract = _extract_2nd_order_pathway
        fct_fit = _build_2nd_order
        fct_plot = _plot_2nd_order
    elif model_order == 3: # Bipolar distance-dependent
        fct_extract = _extract_3rd_order_pathway
        fct_fit = _build_3rd_order
        fct_plot = _plot_3rd_order
    else:
        assert False, f'ERROR: Order-{model_order} model building not supported!'

    # Data splits (optional)
    N_split = kwargs.pop('N_split', None)
    part_idx = kwargs.pop('part_idx', None)
    assert N_split is None and part_idx is None, 'ERROR: Data splitting not supported!'
    data_fn = 'data'

    # Extract connection probability data
    data_dict = fct_extract(adj, node_properties_src, node_properties_tgt, split_indices=None, part_idx=None, **kwargs)
    _save_data(data_dict, kwargs.get('data_dir'), model_name, data_fn)

    # Fit model
    model_dict = fct_fit(**data_dict, **kwargs)
    _save_data(model_dict, kwargs.get('model_dir'), model_name, 'model')

    # Visualize data/model (optional)
    if kwargs.get('do_plot'):
        fct_plot(adj, [node_properties_src, node_properties_tgt], model_name, **data_dict, **model_dict, **kwargs)

    return data_dict, model_dict


###################################################################################################
# Helper functions for model building
###################################################################################################

def _merge_data(part_dir, model_name, spec_name, part_list):
    """Merges data from different data splits."""
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


def _save_data(save_dict, save_dir, model_name, save_spec=None):
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


def _get_data_part_name(N_split, part_idx):
    """Returns part name of a data split."""
    num_dig = len(str(N_split))
    return f'__part-{N_split}-{part_idx:0{num_dig}}'


def _get_model_function(model, model_inputs, model_params):
    """Returns model function from string representation [so any model function can be saved to file]."""
    input_str = ','.join(model_inputs + ['model_params=model_params']) # String representation of input variables
    input_param_str = ','.join(model_inputs + list(model_params.keys())) # String representation of input variables and model parameters
    model_param_str = ','.join(model_inputs + ['**model_params']) # String representation propagating model parameters

    inner_model_str = f'lambda {input_param_str}: {model}'
    full_model_str = f'lambda {input_str}: ({inner_model_str})({model_param_str})' # Use nested lambdas to bind local variables

    model_fct = eval(full_model_str) # Build function

    # logging.info(f'Model function: {inner_model_str}')

    return model_fct


def _compute_dist_matrix(src_nrn_pos, tgt_nrn_pos):
    """Computes distance matrix between pairs of neurons."""
    dist_mat = spt.distance_matrix(src_nrn_pos, tgt_nrn_pos)
    dist_mat[dist_mat == 0.0] = np.nan # Exclude autaptic connections

    return dist_mat


def _compute_dist_matrix_symmetric(nrn_pos):
    """Computes symmetric distance matrix between pairs of neurons.
       Faster implementation to be used when source and target neurons
       are the same."""
    dist_mat = spt.distance.squareform(spt.distance.pdist(nrn_pos))
    dist_mat[dist_mat == 0.0] = np.nan # Exclude autaptic connections

    return dist_mat


def _compute_bip_matrix(src_depths, tgt_depths):
    """
    Computes bipolar matrix between pairs of neurons based on depth difference delta_d:
      POST-synaptic neuron below (delta_d < 0) or above (delta_d > 0) PRE-synaptic neuron
    """
    bip_mat = np.sign(-np.diff(np.meshgrid(src_depths, tgt_depths, indexing='ij'), axis=0)[0, :, :])

    return bip_mat


def _extract_dependent_p_conn(adj, dep_matrices, dep_bins):
    """Extract D-dimensional conn. prob. dependent on D property matrices between source-target pairs of neurons within given range of bins."""
    num_dep = len(dep_matrices)
    assert len(dep_bins) == num_dep, 'ERROR: Dependencies/bins mismatch!'
    assert np.all([dep_matrices[dim].shape == adj.shape for dim in range(num_dep)]), 'ERROR: Matrix dimension mismatch!'

    # Extract connection probability
    num_bins = [len(b) - 1 for b in dep_bins]
    bin_indices = [list(range(n)) for n in num_bins]
    count_all = np.full(num_bins, -1) # Count of all pairs of neurons for each combination of dependencies
    count_conn = np.full(num_bins, -1) # Count of connected pairs of neurons for each combination of dependencies

    logging.info(f'Extracting {num_dep}-dimensional ({"x".join([str(n) for n in num_bins])}) connection probabilities...')
    pbar = progressbar.ProgressBar(maxval=np.prod(num_bins) - 1)
    for idx in pbar(itertools.product(*bin_indices)):
        dep_sel = np.full(adj.shape, True)
        for dim in range(num_dep):
            lower = dep_bins[dim][idx[dim]]
            upper = dep_bins[dim][idx[dim] + 1]
            dep_sel = np.logical_and(dep_sel, np.logical_and(dep_matrices[dim] >= lower, (dep_matrices[dim] < upper) if idx[dim] < num_bins[dim] - 1 else (dep_matrices[dim] <= upper))) # Including last edge
        sidx, tidx = np.nonzero(dep_sel)
        count_all[idx] = np.sum(dep_sel)
        ### count_conn[idx] = np.sum(adj[sidx, tidx]) # ERROR in scipy/sparse/compressed.py if len(sidx) >= 2**31: "ValueError: could not convert integer scalar"
        # [WORKAROUND]: Split indices into parts of 2**31-1 length and sum them separately
        sidx_split = np.split(sidx, np.arange(0, len(sidx), 2**31-1)[1:])
        tidx_split = np.split(tidx, np.arange(0, len(tidx), 2**31-1)[1:])
        count_split = 0
        for s, t in zip(sidx_split, tidx_split):
            count_split = count_split + np.sum(adj[s, t])
        count_conn[idx] = count_split
    p_conn = np.array(count_conn / count_all)
#     p_conn[np.isnan(p_conn)] = 0.0

    return p_conn, count_conn, count_all


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   2nd order (distance-dependent)
###################################################################################################

def _extract_2nd_order(adj, node_properties, bin_size_um=100, max_range_um=None, coord_names=None, split_indices=None, part_idx=None, **_):
    """Extract distance-dependent connection probability (2nd order) from a sample of pairs of neurons."""

    if coord_names is None:
        coord_names = ['x', 'y', 'z'] # Default names of coordinatate system axes as in node_properties
    if isinstance(split_indices, list):
        N_split = len(split_indices)
    else:
        N_split = 0 # Don't split
    if part_idx is not None: # Run only data extraction of given part idx
        assert 0 <= part_idx < N_split, 'ERROR: Part index out of range!'

    pos_table = node_properties[coord_names].to_numpy()

    if N_split == 0: # Compute all at once
        # Compute distance matrix
        dist_mat = _compute_dist_matrix_symmetric(pos_table)

        # Extract distance-dependent connection probabilities
        if max_range_um is None:
            max_range_um = np.nanmax(dist_mat)
        num_bins = np.ceil(max_range_um / bin_size_um).astype(int)
        dist_bins = np.arange(0, num_bins + 1) * bin_size_um

        p_conn_dist, count_conn, count_all = _extract_dependent_p_conn(adj, [dist_mat], [dist_bins])

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
            dist_mat_split = _compute_dist_matrix(pos_table[split_sel, :], pos_table)
            
            # Extract distance-dependent connection counts
            _, count_conn_split, count_all_split = _extract_dependent_p_conn(adj[split_sel, :], [dist_mat_split], [dist_bins])
            count_conn += count_conn_split
            count_all += count_all_split

        # Compute overall connection probabilities
        p_conn_dist = np.array(count_conn / count_all)
#         p_conn_dist[np.isnan(p_conn_dist)] = 0.0

    return {'p_conn_dist': p_conn_dist, 'count_conn': count_conn, 'count_all': count_all, 'dist_bins': dist_bins}


def _extract_2nd_order_pathway(adj, node_properties_src, node_properties_tgt, bin_size_um=100, max_range_um=None, coord_names=None, split_indices=None, part_idx=None, **_):
    """Extract distance-dependent connection probability (2nd order) from a sample of pairs of neurons
       for separate pathways (i.e., non-symmetric adj)."""

    if coord_names is None:
        coord_names = ['x', 'y', 'z'] # Default names of coordinatate system axes as in node_properties

    assert split_indices is None and part_idx is None, 'ERROR: Data splitting not supported!'

    pos_table_src = node_properties_src[coord_names].to_numpy()
    pos_table_tgt = node_properties_tgt[coord_names].to_numpy()

    # Compute distance matrix
    dist_mat = _compute_dist_matrix(pos_table_src, pos_table_tgt)

    # Extract distance-dependent connection probabilities
    if max_range_um is None:
        max_range_um = np.nanmax(dist_mat)
    num_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_bins + 1) * bin_size_um

    p_conn_dist, count_conn, count_all = _extract_dependent_p_conn(adj, [dist_mat], [dist_bins])

    return {'p_conn_dist': p_conn_dist, 'count_conn': count_conn, 'count_all': count_all, 'dist_bins': dist_bins}


def _build_2nd_order(p_conn_dist, dist_bins, **_):
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


def _plot_2nd_order(adj, node_properties, model_name, p_conn_dist, count_conn, count_all, dist_bins, model, model_inputs, model_params, plot_dir=None, **_):
    """Visualize data vs. model (2nd order)."""
    if plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    dist_model = np.linspace(dist_bins[0], dist_bins[-1], 100)

    model_str = f'f(x) = {model_params["exp_model_scale"]:.3f} * exp(-{model_params["exp_model_exponent"]:.3f} * x)'
    model_fct = _get_model_function(model, model_inputs, model_params)

    if isinstance(node_properties, list):
        N_pre = node_properties[0].shape[0]  # Pre-synaptic population
        N_post = node_properties[1].shape[0]  # Post-synaptic population
    else:
        N_pre = N_post = node_properties.shape[0]

    plt.figure(figsize=(12, 4), dpi=300)

    # Data vs. model
    plt.subplot(1, 2, 1)
    plt.step(dist_bins, np.hstack([p_conn_dist[0], p_conn_dist]), color=DATA_COLOR, label=f'Data: N = {N_pre}x{N_post} cells')
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
    plt.title(f'Distance-dependent connection counts (N = {N_pre}x{N_post} cells)')
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

def _extract_3rd_order(adj, node_properties, bin_size_um=100, max_range_um=None, coord_names=None, depth_name=None, split_indices=None, part_idx=None, **_):
    """Extract distance-dependent connection probability (3rd order) from a sample of pairs of neurons."""

    if coord_names is None:
        coord_names = ['x', 'y', 'z'] # Default names of coordinatate system axes as in node_properties
    if depth_name is None:
        depth_name = 'depth' # Default name of depth column in node_properties
    if isinstance(split_indices, list):
        N_split = len(split_indices)
    else:
        N_split = 0 # Don't split
    if part_idx is not None: # Run only data extraction of given part idx
        assert 0 <= part_idx < N_split, 'ERROR: Part index out of range!'

    pos_table = node_properties[coord_names].to_numpy()
    depth_table = node_properties[depth_name].to_numpy()

    if N_split == 0: # Compute all at once
        # Compute distance matrix
        dist_mat = _compute_dist_matrix_symmetric(pos_table)

        # Compute bipolar matrix (post-synaptic neuron below (delta_d < 0) or above (delta_d > 0) pre-synaptic neuron)
        bip_mat = _compute_bip_matrix(depth_table, depth_table)

        # Extract bipolar distance-dependent connection probabilities
        if max_range_um is None:
            max_range_um = np.nanmax(dist_mat)
        num_dist_bins = np.ceil(max_range_um / bin_size_um).astype(int)
        dist_bins = np.arange(0, num_dist_bins + 1) * bin_size_um
        bip_bins = [np.nanmin(bip_mat), 0, np.nanmax(bip_mat)]

        p_conn_dist_bip, count_conn, count_all = _extract_dependent_p_conn(adj, [dist_mat, bip_mat], [dist_bins, bip_bins])

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
            dist_mat_split = _compute_dist_matrix(pos_table[split_sel, :], pos_table)

            # Compute bipolar matrix (post-synaptic neuron below (delta_d < 0) or above (delta_d > 0) pre-synaptic neuron)
            bip_mat_split = _compute_bip_matrix(depth_table[split_sel], depth_table)

            # Extract distance-dependent connection counts
            _, count_conn_split, count_all_split = _extract_dependent_p_conn(adj[split_sel, :], [dist_mat_split, bip_mat_split], [dist_bins, bip_bins])
            count_conn += count_conn_split
            count_all += count_all_split

        # Compute overall connection probabilities
        p_conn_dist_bip = np.array(count_conn / count_all)
#         p_conn_dist_bip[np.isnan(p_conn_dist_bip)] = 0.0

    return {'p_conn_dist_bip': p_conn_dist_bip, 'count_conn': count_conn, 'count_all': count_all, 'dist_bins': dist_bins, 'bip_bins': bip_bins}


def _extract_3rd_order_pathway(adj, node_properties_src, node_properties_tgt, bin_size_um=100, max_range_um=None, coord_names=None, depth_name=None, split_indices=None, part_idx=None, **_):
    """Extract distance-dependent connection probability (3rd order) from a sample of pairs of neurons
       for separate pathways (i.e., non-symmetric adj)."""

    if coord_names is None:
        coord_names = ['x', 'y', 'z'] # Default names of coordinatate system axes as in node_properties
    if depth_name is None:
        depth_name = 'depth' # Default name of depth column in node_properties

    assert split_indices is None and part_idx is None, 'ERROR: Data splitting not supported!'

    pos_table_src = node_properties_src[coord_names].to_numpy()
    pos_table_tgt = node_properties_tgt[coord_names].to_numpy()
    depth_table_src = node_properties_src[depth_name].to_numpy()
    depth_table_tgt = node_properties_tgt[depth_name].to_numpy()

    # Compute distance matrix
    dist_mat = _compute_dist_matrix(pos_table_src, pos_table_tgt)

    # Compute bipolar matrix (post-synaptic neuron below (delta_d < 0) or above (delta_d > 0) pre-synaptic neuron)
    bip_mat = _compute_bip_matrix(depth_table_src, depth_table_tgt)

    # Extract bipolar distance-dependent connection probabilities
    if max_range_um is None:
        max_range_um = np.nanmax(dist_mat)
    num_dist_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_dist_bins + 1) * bin_size_um
    bip_bins = [np.nanmin(bip_mat), 0, np.nanmax(bip_mat)]

    p_conn_dist_bip, count_conn, count_all = _extract_dependent_p_conn(adj, [dist_mat, bip_mat], [dist_bins, bip_bins])

    return {'p_conn_dist_bip': p_conn_dist_bip, 'count_conn': count_conn, 'count_all': count_all, 'dist_bins': dist_bins, 'bip_bins': bip_bins}


def _build_3rd_order(p_conn_dist_bip, dist_bins, **_):
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


def _plot_3rd_order(adj, node_properties, model_name, p_conn_dist_bip, count_conn, count_all, dist_bins, model, model_inputs, model_params, plot_dir=None, **_):
    """Visualize data vs. model (3rd order)."""
    if plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    dist_model = np.linspace(dist_bins[0], dist_bins[-1], 100)

    model_strN = f'{model_params["bip_neg_exp_model_scale"]:.3f} * exp(-{model_params["bip_neg_exp_model_exponent"]:.3f} * x)'
    model_strP = f'{model_params["bip_pos_exp_model_scale"]:.3f} * exp(-{model_params["bip_pos_exp_model_exponent"]:.3f} * x)'
    model_fct = _get_model_function(model, model_inputs, model_params)

    if isinstance(node_properties, list):
        N_pre = node_properties[0].shape[0]  # Pre-synaptic population
        N_post = node_properties[1].shape[0]  # Post-synaptic population
    else:
        N_pre = N_post = node_properties.shape[0]

    plt.figure(figsize=(12, 4), dpi=300)

    # Data vs. model
    plt.subplot(1, 2, 1)
    bip_dist = np.concatenate((-dist_bins[:-1][::-1] - bin_offset, [0.0], dist_bins[:-1] + bin_offset))
    bip_data = np.concatenate((p_conn_dist_bip[::-1, 0], [np.nan], p_conn_dist_bip[:, 1]))
    all_bins = np.concatenate((-dist_bins[1:][::-1], [0.0], dist_bins[1:]))
    bin_data = np.concatenate((p_conn_dist_bip[::-1, 0], p_conn_dist_bip[:, 1]))
    plt.step(all_bins, np.hstack([bin_data[0], bin_data]), color=DATA_COLOR, label=f'Data: N = {N_pre}x{N_post} cells')
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
    plt.title(f'Bipolar distance-dependent connection counts (N = {N_pre}x{N_post} cells)')
    plt.legend()
    plt.tight_layout()
    if plot_dir is not None:
        out_fn = os.path.abspath(os.path.join(plot_dir, model_name + '__data_counts.png'))
        plt.savefig(out_fn)
        logging.info(f'Figure saved to {out_fn}')

# Generate models for connectomes.
#
# Author(s): C. Pokorny
# Last modified: 11/2021


import numpy as np
import os
import scipy.optimize as opt
import scipy.spatial as spt
import matplotlib.pyplot as plt
import itertools
import progressbar
import pickle


PROB_CMAP = plt.cm.get_cmap('hot')
DATA_COLOR = 'tab:blue'
MODEL_COLOR = 'tab:red'


def run_model_building(adj_matrix, nrn_table, model_name, model_order, **kwargs):
    """
    Main function for running model building, consisting of three steps:
      Data extraction, model fitting, and (optionally) data/model visualization
    """
    print(f'INFO: Running order-{model_order} model building {kwargs}...')

    # Subsampling (optional)
    sample_size = kwargs.get('sample_size')
    if sample_size is not None and sample_size > 0 and sample_size < nrn_table.shape[0]:
        print(f'INFO: Subsampling to {sample_size} of {nrn_table.shape[0]} neurons')
        np.random.seed(kwargs.get('sample_seed'))
        sub_sel = np.random.permutation([True] * sample_size + [False] * (nrn_table.shape[0] - sample_size))
        adj_matrix = adj_matrix.tocsr()[sub_sel, :].tocsc()[:, sub_sel].tocsr()
        nrn_table = nrn_table.loc[sub_sel, :]

    # Set modelling functions
    if model_order == 2:
        fct_extract = extract_2nd_order
        fct_fit = build_2nd_order
        fct_plot = plot_2nd_order
    else:
        assert False, f'ERROR: Order-{model_order} model building not supported!'

    # Extract connection probability data
    data_dict = fct_extract(adj_matrix, nrn_table, **kwargs)
    save_data(data_dict, kwargs.get('data_dir'), model_name, 'data')

    # Fit model
    model_dict = fct_fit(**data_dict, **kwargs)
    save_data(model_dict, kwargs.get('model_dir'), model_name, 'model')

    # Visualize data/model (optional)
    if kwargs.get('do_plot'):
        fct_plot(adj_matrix, nrn_table, model_name, **data_dict, **model_dict, **kwargs)

    return data_dict, model_dict


###################################################################################################
# Helper functions for model building
###################################################################################################

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

    print(f'INFO: Pickled dict written to {save_file}')


def get_model_function(model, model_inputs, model_params):
    """Returns model function from string representation [so any model function can be saved to file]."""
    input_str = ','.join(model_inputs + ['model_params=model_params']) # String representation of input variables
    input_param_str = ','.join(model_inputs + list(model_params.keys())) # String representation of input variables and model parameters
    model_param_str = ','.join(model_inputs + ['**model_params']) # String representation propagating model parameters

    inner_model_str = f'lambda {input_param_str}: {model}'
    full_model_str = f'lambda {input_str}: ({inner_model_str})({model_param_str})' # Use nested lambdas to bind local variables

    model_fct = eval(full_model_str) # Build function

    # print(f'INFO: Model function: {inner_model_str}')

    return model_fct


def compute_dist_matrix(src_nrn_pos, tgt_nrn_pos):
    """Computes distance matrix between pairs of neurons."""
    dist_mat = spt.distance_matrix(src_nrn_pos, tgt_nrn_pos)
    dist_mat[dist_mat == 0.0] = np.nan # Exclude autaptic connections

    return dist_mat


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

    print(f'Extracting {num_dep}-dimensional ({"x".join([str(n) for n in num_bins])}) connection probabilities...', flush=True)
    pbar = progressbar.ProgressBar(maxval=np.prod(num_bins) - 1)
    for idx in pbar(itertools.product(*bin_indices)):
        dep_sel = np.full(adj_matrix.shape, True)
        for dim in range(num_dep):
            lower = dep_bins[dim][idx[dim]]
            upper = dep_bins[dim][idx[dim] + 1]
            dep_sel = np.logical_and(dep_sel, np.logical_and(dep_matrices[dim] >= lower, (dep_matrices[dim] < upper) if idx[dim] < num_bins[dim] - 1 else (dep_matrices[dim] <= upper))) # Including last edge
        sidx, tidx = np.nonzero(dep_sel)
        count_all[idx] = np.sum(dep_sel)
        count_conn[idx] = np.sum(adj_matrix[sidx, tidx])
    p_conn = np.array(count_conn / count_all)
    p_conn[np.isnan(p_conn)] = 0.0

    return p_conn, count_conn, count_all


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   2nd order (distance-dependent)
###################################################################################################

def extract_2nd_order(adj_matrix, nrn_table, bin_size_um=100, max_range_um=None, coord_names=None, **_):
    """Extract distance-dependent connection probability (2nd order) from a sample of pairs of neurons."""

    if coord_names is None:
        coord_names = ['x', 'y', 'z'] # Default names of coordinatate system axes as in nrn_table

    pos_table = nrn_table[coord_names].to_numpy()

    # Compute distance matrix
    dist_mat = compute_dist_matrix(pos_table, pos_table)

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
    (a_opt, b_opt), _ = opt.curve_fit(exp_model, X, y, p0=[0.0, 0.0])

    print(f'MODEL FIT: f(x) = {a_opt:.3f} * exp(-{b_opt:.3f} * x)')

    model = 'a_opt * np.exp(-b_opt * np.array(d))'
    model_inputs = ['d']
    model_params = {'a_opt': a_opt, 'b_opt': b_opt}

    return {'model': model, 'model_inputs': model_inputs, 'model_params': model_params}


def plot_2nd_order(adj_matrix, nrn_table, model_name, p_conn_dist, count_conn, count_all, dist_bins, model, model_inputs, model_params, plot_dir=None, **_):
    """Visualize data vs. model (2nd order)."""
    if plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    dist_model = np.linspace(dist_bins[0], dist_bins[-1], 100)

    model_str = f'f(x) = {model_params["a_opt"]:.3f} * exp(-{model_params["b_opt"]:.3f} * x)'
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
        print(f'INFO: Figure saved to {out_fn}')

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
        print(f'INFO: Figure saved to {out_fn}')

# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy.stats import binom
from scipy.linalg import eigvals
#TODO: MODIFY THE IMPORTS TO EXTERNAL IMPORTS

from .topology import node_degree

def closeness_connected_components(adj, neuron_properties=[], directed=False, return_sum=True):
    """Compute the closeness of each connected component of more than 1 vertex
    
    Parameters
    ----------
    adj : array_like
        Adjacency matrix of the graph
    directed : bool
        If `True`, will be computed using strongly connected components and directed closeness.
    return_sum : bool
        If `True`, only one list will be returned, by summing over all the connected components.


    Returns
    -------
    array_like
        A single array( if `return_sum=True`) or a list of arrays of shape `n`, containting closeness of vertices in that component, or 0 if the vertex is not in the component. Closeness cannot be zero otherwise.

    """
    from sknetwork.ranking import Closeness
    from scipy.sparse.csgraph import connected_components

    matrix=adj.copy()
    if directed:
        n_comp, comp = connected_components(matrix, directed=True, connection="strong")
    else:
        n_comp, comp = connected_components(matrix, directed=False)
        matrix = matrix + matrix.T  # we need to make the matrix symmetric

    closeness = Closeness()  # if matrix is not symmetric automatically use directed
    n = matrix.shape[0]
    all_c = []
    for i in range(n_comp):
        c = np.zeros(n)
        idx = np.where(comp == i)[0]
        sub_mat = matrix[np.ix_(idx, idx)].tocsr()
        if sub_mat.getnnz() > 0:
            c[idx] = closeness.fit_transform(sub_mat)
            all_c.append(c)
    if return_sum:
        all_c = np.array(all_c)
        return np.sum(all_c, axis=0)
    else:
        return all_c

def communicability(adj, neuron_properties):
    pass

def closeness(adj, neuron_properties, directed=False):
    """Compute closeness centrality using sknetwork on all connected components or strongly connected
    component (if directed==True)"""
    return closeness_connected_components(adj, directed=directed)

def connected_components(adj,neuron_properties=[]):
    """Returns a list of the size of the connected components of the underlying undirected graph on sub_gids,
    if None, compute on the whole graph"""

    matrix=adj.toarray()
    matrix_und = np.where((matrix+matrix.T) >= 1, 1, 0)
    # TODO: Change the code from below to scipy implementation that seems to be faster!
    G = nx.from_numpy_matrix(matrix_und)
    return [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]

def core_number(adj, neuron_properties=[]):
    """Returns k core of directed graph, where degree of a vertex is the sum of in degree and out degree"""
    # TODO: Implement directed (k,l) core and k-core of underlying undirected graph (very similar to this)
    G = nx.from_numpy_matrix(adj.toarray())
    # Very inefficient (returns a dictionary!). TODO: Look for different implementation
    return nx.algorithms.core.core_number(G)

    # TODO: Filtered simplex counts with different weights on vertices (coreness, intersection)
    #  or on edges (strength of connection).

def density(adj, neuron_properties=[]):
    #Todo: Add #cells/volume as as possible spatial density
    adj=adj.astype('bool').astype('int')
    return adj.sum() / np.prod(adj.shape)

def __make_expected_distribution_model_first_order__(adj, direction="efferent"):
    #TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    if direction == "efferent":
        N = adj.sum(axis=1).mean()
        M = adj.shape[1]
    elif direction == "afferent":
        N = adj.sum(axis=0).mean()
        M = adj.shape[0]
    else:
        raise ValueError("Unknown direction: {0}".format(direction))
    expected = hypergeom(M, N, N)
    return expected


def distribution_number_of_common_neighbors(adj, neuron_properties=None, direction="efferent"):
    # TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    adj = adj.tocsc().astype(int)
    if direction == "efferent":
        cn = adj * adj.transpose()
    elif direction == "afferent":
        cn = adj.transpose() * adj
    else:
        raise ValueError("Unknown direction: {0}".format(direction))
    cn = np.array(cn.todense())
    cn = cn[np.triu_indices_from(cn, 1)]
    bins = np.arange(0, cn.max() + 2)
    return pd.Series(np.histogram(cn, bins=bins)[0], index=bins[:-1])


def normalized_distribution_of_common_neighbors(adj, neuron_properties=None, direction="efferent"):
    # TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    data = distribution_number_of_common_neighbors(adj, neuron_properties, direction=direction)
    expected = __make_expected_distribution_model_first_order__(adj, direction=direction)
    expected = expected.pmf(data.index) * data.sum()
    expected = pd.Series(expected, index=data.index)
    return (data - expected) / (data + expected)


def overexpression_of_common_neighbors(adj, neuron_properties=None, direction="efferent"):
    # TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    data = distribution_number_of_common_neighbors(adj, neuron_properties, direction=direction)
    data_mean = (data.index.values * data.values).sum() / data.values.sum()
    ctrl = __make_expected_distribution_model_first_order__(adj, direction=direction)
    ctrl_mean = ctrl.mean()
    return (data_mean - ctrl_mean) / (data_mean + ctrl_mean)


def common_neighbor_weight_bias(adj, neuron_properties=None, direction="efferent"):
    # TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    adj_bin = (adj.tocsc() > 0).astype(int)
    if direction == "efferent":
        cn = adj_bin * adj_bin.transpose()
    elif direction == "afferent":
        cn = adj_bin.transpose() * adj_bin

    return np.corrcoef(cn[adj > 0],
                       adj[adj > 0])[0, 1]


def common_neighbor_connectivity_bias(adj, neuron_properties=None, direction="efferent",
                                      cols_location=None, fit_log=False):
    # TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    import statsmodels.formula.api as smf
    from patsy import ModelDesc
    from scipy.spatial import distance

    if adj.dtype == bool:
        adj = adj.astype(int)

    if direction == "efferent":
        cn = adj * adj.transpose()
    elif direction == "afferent":
        cn = adj.transpose() * adj

    input_dict = {"CN": cn.toarray().flatten(),
                  "Connected": adj.astype(bool).toarray().flatten()}

    if fit_log:
        input_dict["CN"] = np.log10(input_dict["CN"] + fit_log)
    formula_str = "CN ~ Connected"
    if cols_location is not None:
        formula_str = formula_str + " + Distance"
        dmat = distance.squareform(distance.pdist(neuron_properties[cols_location].values))
        input_dict["Distance"] = dmat.flatten()
    sm_model = ModelDesc.from_formula(formula_str)

    sm_result = smf.ols(sm_model, input_dict).fit()

    pval = sm_result.pvalues.get("Connected[T.True]", 1.0)
    mdl_intercept = sm_result.params["Intercept"]
    mdl_added = sm_result.params.get("Connected[T.True]", 0.0)
    mdl_distance = sm_result.params.get("Distance", 0.0)
    return pval, mdl_added / mdl_intercept, 100 * mdl_distance / mdl_intercept

###Computing connection probabilities

def connection_probability_within(adj, neuron_properties, max_dist=200, min_dist=0,
                                  columns=["ss_flat_x", "ss_flat_y"]):
    #TODO: Does neuron_properties here assumes the right data structure?
    if isinstance(neuron_properties, tuple):
        nrn_pre, nrn_post = neuron_properties
    else:
        nrn_pre = neuron_properties;
        nrn_post = neuron_properties
    D = distance.cdist(nrn_pre[columns], nrn_post[columns])
    mask = (D > min_dist) & (D <= max_dist)  # This way with min_dist=0 a neuron with itself is excluded
    return adj[mask].mean()


def connection_probability(adj, neuron_properties):
    #TODO: Does neuron_properties here assumes the right data structure?
    exclude_diagonal = False
    if isinstance(neuron_properties, tuple):
        nrn_pre, nrn_post = neuron_properties
        if len(nrn_pre) == len(nrn_post):
            if (nrn_pre["gid"] == nrn_post["gid"]).all():
                exclude_diagonal = True
    else:
        exclude_diagonal = True

    if not exclude_diagonal:
        return adj.astype(bool).mean()
    assert adj.shape[0] == adj.shape[1], "Inconsistent shape!"
    n_pairs = adj.shape[0] * (adj.shape[1] - 1)
    return adj.astype(bool).sum() / n_pairs

#TODO ADD CODE FROM CLUSTER TO COMPUTE PROBABILITY OF CONNECTION PER PATHWAY OR ANY OTHER PROPERTIE ON THE NEURON_PROPERTY.


#TODO: Checked up to here ... clean the code below
# ##Degree based analyses
#TODO UPDATE THE RICH CLUB IMPLEMENTATION

def gini_curve(m, nrn, direction='efferent'):
    m = m.tocsc()
    if direction == 'afferent':
        degrees = np.array(m.sum(axis=0)).flatten()
    elif direction == 'efferent':
        degrees = np.array(m.sum(axis=1).flatten())
    else:
        raise Exception("Unknown value for argument direction: %s" % direction)
    cs = np.cumsum(np.flipud(sorted(degrees))).astype(float) / np.sum(degrees)
    return pd.Series(cs, index=np.linspace(0, 1, len(cs)))


def gini_coefficient(m, nrn, direction='efferent'):
    gc = gini_curve(m, nrn, direction=direction)
    A = gc.index.values
    B = gc.values
    return np.sum(np.diff(A) * (B[:-1] + B[1:]) / 2.0)


def _analytical_expected_gini_curve(m, direction='efferent'):
    if direction == 'afferent':
        N = m.shape[0] - 1
        C = m.shape[1] * N
    elif direction == 'efferent':
        N = m.shape[1] - 1
        C = m.shape[0] * N
    P = m.nnz / C
    # Only using degrees, not distribution of weigthts. TODO: Fix that
    x = np.arange(N, -1, -1)
    p = binom.pmf(x, N, P)
    A = np.cumsum(p) / p.sum()
    B = np.cumsum(p * x) / np.sum(p * x)
    return pd.Series(B, index=A)


def normalized_gini_coefficient(m, nrn, direction='efferent'):
    gc = gini_coefficient(m, nrn, direction=direction)
    ctrl = _analytical_expected_gini_curve(m, direction=direction)
    A = ctrl.index.values
    B = ctrl.values
    return 2 * (gc - np.sum(np.diff(A) * (B[:-1] + B[1:]) / 2.0))


def _bin_degrees(degrees):
    nbins = np.maximum(int(len(degrees) * 0.1), np.minimum(len(degrees), 30))
    mx = np.nanmax(degrees);
    mn = np.nanmin(degrees)
    bins = np.linspace(mn, mx + 1E-6 * (mx - mn), nbins + 1)
    degrees = np.digitize(degrees, bins=bins) - 1
    udegrees = np.arange(nbins)
    ret_x = 0.5 * (bins[:-1] + bins[1:])
    return ret_x, udegrees, degrees


def rich_club_curve(m, direction='efferent'):
    m = m.tocsc()
    if direction == 'afferent':
        degrees = np.array(m.sum(axis=0)).flatten()
    elif direction == 'efferent':
        degrees = np.array(m.sum(axis=1)).flatten()
    else:
        raise Exception("Unknown value for argument direction: %s" % direction)

    if m.dtype == bool:
        udegrees = np.arange(1, degrees.max() + 1)
        ret_x = udegrees
    else:
        ret_x, udegrees, degrees = _bin_degrees(degrees)

    edge_counter = lambda i: (degrees >= i).sum() * ((degrees >= i).sum() - 1)  # number of pot. edges
    mat_counter = lambda i: m[np.ix_(degrees >= i, degrees >= i)].sum()  # number of actual edges
    ret = (np.array([mat_counter(i) for i in udegrees]).astype(float)
           / np.array([edge_counter(i) for i in udegrees]))
    return pd.Series(ret, index=ret_x)


def efficient_rich_club_curve(M, direction="efferent", pre_calculated_richness=None, sparse_bin_set=False):
    M = M.tocoo()
    shape = M.shape
    M = pd.DataFrame.from_dict({"row": M.row, "col": M.col})
    if pre_calculated_richness is not None:
        deg = pre_calculated_richness
    elif direction == "efferent":
        deg = M["row"].value_counts()
    elif direction == "afferent":
        deg = M["col"].value_counts()
    elif direction == "both":
        D = pd.DataFrame({'row': np.zeros(shape[0]), 'col': np.zeros(shape[0])}, np.arange(shape[0])[-1::-1])
        D['row'] = D['row'] + M["row"].value_counts()
        D['col'] = D['col'] + M["col"].value_counts()
        D = D.fillna(0)
        D = D.astype(int)
        deg = D['row'] + D['col']
    else:
        raise ValueError()

    if sparse_bin_set == False:
        degree_bins = np.arange(deg.max() + 2)
    elif sparse_bin_set == True:
        degree_bins = np.unique(np.append(deg, [0, deg.max() + 1]))
    degree_bins_rv = degree_bins[-2::-1]
    nrn_degree_distribution = np.histogram(deg.values, bins=degree_bins)[0]
    nrn_cum_degrees = np.cumsum(nrn_degree_distribution[-1::-1])
    nrn_cum_pairs = nrn_cum_degrees * (nrn_cum_degrees - 1)

    deg_arr = np.zeros(shape[0], dtype=int)
    deg_arr[deg.index.values] = deg.values

    deg = None

    con_degree = np.minimum(deg_arr[M["row"].values], deg_arr[M["col"].values])
    M = None
    con_degree = np.histogram(con_degree, bins=degree_bins)[0]

    cum_degrees = np.cumsum(con_degree[-1::-1])

    return pd.Series(cum_degrees / nrn_cum_pairs, degree_bins_rv)[::-1]


def _analytical_expected_rich_club_curve(m, direction='efferent'):
    assert m.dtype == bool, "This function only works for binary matrices at the moment"
    indegree = np.array(m.sum(axis=0))[0]
    outdegree = np.array(m.sum(axis=1))[:, 0]

    if direction == 'afferent':
        degrees = indegree
    elif direction == 'efferent':
        degrees = outdegree
    else:
        raise Exception("Unknown value for argument direction: %s" % direction)

    udegrees = np.arange(1, degrees.max() + 1)
    edge_counter = lambda i: (degrees >= i).sum() * ((degrees >= i).sum() - 1)
    res_mn = []
    res_sd = []
    for deg in udegrees:
        valid = np.nonzero(degrees >= deg)[0]
        i_v = indegree[valid]
        i_sum_all = indegree.sum() - i_v
        i_sum_s = i_v.sum() - i_v
        o_v = outdegree[valid]
        S = np.array([hypergeom.stats(_ia, _is, o)
                      for _ia, _is, o in zip(i_sum_all, i_sum_s, o_v)])
        res_mn.append(np.sum(S[:, 0]) / edge_counter(deg))
        res_sd.append(np.sqrt(S[:, 1].sum()) / edge_counter(deg))  # Sum the variances, but divide the std
    df = pd.DataFrame.from_dict({"mean": np.array(res_mn),
                                 "std": np.array(res_sd)})
    df.index = udegrees
    return df


def generate_degree_based_control(M, direction="efferent"):
    """
    A shuffled version of a connectivity matrix that aims to preserve degree distributions.
    If direction = "efferent", then the out-degree is exactly preserved, while the in-degree is
    approximately preseved. Otherwise it's the other way around.
    """
    if direction == "efferent":
        M = M.tocsr()
        idxx = np.arange(M.shape[1])
        p_out = np.array(M.mean(axis=0))[0]
    elif direction == "afferent":
        M = M.tocsc()
        idxx = np.arange(M.shape[0])
        p_out = np.array(M.mean(axis=1))[:, 0]
    else:
        raise ValueError()

    for col in range(M.shape[1]):
        p = p_out.copy()
        p[col] = 0.0
        p = p / p.sum()
        a = M.indptr[col]
        b = M.indptr[col + 1]
        M.indices[a:b] = np.random.choice(idxx, b - a, p=p, replace=False)
    return M


def _randomized_control_rich_club_curve(m, direction='efferent', n=10):
    res = []
    for _ in range(n):
        m_shuf = generate_degree_based_control(m, direction=direction)
        res.append(efficient_rich_club_curve(m_shuf))
    res = pd.concat(res, axis=1)
    #TODO: Something is wrong here. rr is not defined. Should it be res?
    #      But changing rr to res causing 
    df = pd.DataFrame.from_dict(
        {
            "mean": np.nanmean(res, axis=1),
            "std": np.nanstd(res, axis=1)
        }
    )
    df.index = res.index
    return df


def normalized_rich_club_curve(m, direction='efferent', normalize='std',
                               normalize_with="shuffled", **kwargs):
    assert m.dtype == bool, "This function only works for binary matrices at the moment"
    data = efficient_rich_club_curve(m, direction=direction)
    A = data.index.values
    B = data.values
    if normalize_with == "analytical":
        ctrl = _analytical_expected_rich_club_curve(m, direction=direction)
    elif normalize_with == "shuffled":
        ctrl = _randomized_control_rich_club_curve(m, direction=direction)
    Ar = ctrl.index.values
    mn_r = ctrl["mean"].values
    sd_r = ctrl["std"].values

    if normalize == 'mean':
        return pd.Series(B[:len(mn_r)] / mn_r, index=A[:len(mn_r)])
    elif normalize == 'std':
        return pd.Series((B[:len(mn_r)] - mn_r) / sd_r, index=A[:len(mn_r)])
    else:
        raise Exception("Unknown normalization: %s" % normalize)


def rich_club_coefficient(m, **kwargs):
    Bn = normalized_rich_club_curve(m, normalize='std', **kwargs).values
    return np.nanmean(Bn)


#*************************************************************************#
#Code taken from TriDy

##
## HELPER FUNCTIONS (STRUCTURAL)
##



def nx_to_np(directed_graph):
    """Converts networkx digraph to numpy array of the adjacency matrix 

    Parameters
    ----------
    directed_graph : networkx DiGraph
        a directed graph

    Returns
    -------
    numpy array
        the adjaceny matrix of the DiGraph as a numpy array
    """
    return nx.to_numpy_array(directed_graph,dtype=int)



def np_to_nx(adjacency_matrix):
    """Converts numpy array of an adjacency matrix to a networkx digraph

    Parameters
    ----------
    adjacency_matrix : numpy array
        the adjaceny matrix of the DiGraph as a numpy array


    Returns
    -------
    networkx DiGraph
            a directed graph

    """
    return nx.from_numpy_array(adjacency_matrix,create_using=nx.DiGraph)



def largest_strongly_connected_component(adjacency_matrix):
    """Computes the largest strongly connected component of the graph with adjacency matrix adjacency_matrix,
        and returns the adjacency matrix of said component

    Parameters
    ----------
    adjacency_matrix : numpy array
        the adjaceny matrix of the DiGraph as a numpy array


    Returns
    -------
    numpy array
        The adjacency matrix of the largest strongly connected component

    """
    current_tribe_nx = np_to_nx(adjacency_matrix)
    largest_comp = max(nx.strongly_connected_components(current_tribe_nx), key=len)
    current_tribe_strong_nx = current_tribe_nx.subgraph(largest_comp)
    current_tribe_strong = nx_to_np(current_tribe_strong_nx)
    return current_tribe_strong




##
## HELPER FUNCTIONS (SPECTRAL)
##

def spectral_gap(matrix, thresh=10, param='low'):
#  In: matrix
# Out: float
    current_spectrum = spectrum_make(matrix)
    current_spectrum = spectrum_trim_and_sort(current_spectrum, threshold_decimal=thresh)
    return spectrum_param(current_spectrum, parameter=param)



def spectrum_make(matrix):
#  In: matrix
# Out: list of complex floats
    assert np.any(matrix) , 'Error (eigenvalues): matrix is empty'
    eigenvalues = eigvals(matrix)
    return eigenvalues



def spectrum_trim_and_sort(spectrum, modulus=True, threshold_decimal=10):
#  In: list of complex floats
# Out: list of unique (real or complex) floats, sorted by modulus
    if modulus:
        return np.sort(np.unique(abs(spectrum).round(decimals=threshold_decimal)))
    else:
        return np.sort(np.unique(spectrum.round(decimals=threshold_decimal)))



def spectrum_param(spectrum, parameter):
#  In: list of complex floats
# Out: float
    assert len(spectrum) != 0 , 'Error (eigenvalues): no eigenvalues (spectrum is empty)'
    if parameter == 'low':
        if spectrum[0]:
            return spectrum[0]
        else:
            assert len(spectrum) > 1 , 'Error (low spectral gap): spectrum has only zeros, cannot return nonzero eigval'
            return spectrum[1]
    elif parameter == 'high':
        assert len(spectrum) > 1 , 'Error (high spectral gap): spectrum has one eigval, cannot return difference of top two'
        return spectrum[-1]-spectrum[-2]
    elif parameter == 'radius':
        return spectrum[-1]


##
## NONSPECTRAL PARAMETER FUNCTIONS
##

def ccc(matrix):
    """Computes the classical clustering coefficient of a directed graph

    Parameters
    ----------
    matrix : numpy array
        the adjaceny matrix of a directed graph

    Returns
    -------
    float
        the clustering coefficient

    References
    ----------
    The formula is taken from the following paper.

    ..[1] G. Fagiolo, "Clustering in complex directed networks", 2006;
           [DOI: 10.1103/PhysRevE.76.026107](https://doi.org/10.1103/PhysRevE.76.026107).

    """
    deg = node_degree(matrix)[0]
    numerator = np.linalg.matrix_power(matrix+np.transpose(matrix),3)[0][0]
    denominator = 2*(deg*(deg-1)-2*reciprocal_connections_adjacency(matrix, chief_only=True))
    if denominator == 0:
        return 0
    return numerator/denominator






# degree type parameters

# tribe size

def tribe_size(matrix):
    return len(matrix)

# reciprocal connections

def reciprocal_connections(matrix, chief_only=False):
    #TODO: MERGE THIS WITH RC FUNCTION
    if chief_only:
        rc_count = np.count_nonzero(np.multiply(matrix[0],np.transpose(matrix)[0]))
    else:
        rc_count = np.count_nonzero(np.multiply(matrix,np.transpose(matrix)))//2
    return rc_count


##
## SPECTRAL PARAMETER FUNCTIONS
##


# adjacency spectrum

def asg(matrix, gap='high'):
    return spectral_gap(matrix, param=gap)


# transition probability spectrum

def tpsg(matrix, in_deg=False, gap='high'):
#  in: tribe matrix
# out: float
    current_matrix = tps_matrix(matrix, in_deg=in_deg)
    return spectral_gap(current_matrix, param=gap)

def tps_matrix(matrix, in_deg=False):
#  in: tribe matrix
# out: transition probability matrix
    current_size = len(matrix)
    if in_deg:
        degree_vector = node_degree(matrix, direction='IN').values
    else:
        degree_vector = node_degree(matrix, direction='OUT').values
    inverted_degree_vector = [0 if not d else 1/d for d in degree_vector]
    return np.matmul(np.diagflat(inverted_degree_vector),matrix)


# chung laplacian spectrum
# source 1: Laplacians and the Cheeger inequality for directed graph (Fan Chung, 2005)
# source 2: https://networkx.org/documentation/stable/reference/generated/networkx.linalg.laplacianmatrix.directed_laplacian_matrix.html

def clsg(matrix, is_strongly_conn=False, gap='low'):
#  in: tribe matrix
# out: float
    chung_laplacian_matrix = cls_matrix_fromadjacency(matrix, is_strongly_conn=is_strongly_conn)
    return spectral_gap(chung_laplacian_matrix, param=gap)

def cls_matrix_fromadjacency(matrix, is_strongly_conn=False):
#  in: numpy array
# out: numpy array
    matrix_nx = np_to_nx(matrix)
    return cls_matrix_fromdigraph(matrix_nx, matrix=matrix, matrix_given=True, is_strongly_conn=is_strongly_conn)

def cls_matrix_fromdigraph(digraph, matrix=np.array([]), matrix_given=False, is_strongly_conn=False):
#  in: networkx digraph
# out: numpy array
    digraph_sc = digraph
    matrix_sc = matrix
    # Make sure is strongly connected
    if not is_strongly_conn:
        largest_comp = max(nx.strongly_connected_components(digraph), key=len)
        digraph_sc = digraph.subgraph(largest_comp)
        matrix_sc = nx_to_np(digraph_sc)
    elif not matrix_given:
        matrix_sc = nx_to_np(digraph_sc)
    # Degeneracy: scc has size 1
    if not np.any(matrix_sc):
        return np.array([[0]])
    # Degeneracy: scc has size 2
    elif np.array_equal(matrix_sc,np.array([[0,1],[1,0]],dtype=int)):
        return np.array([[1,-0.5],[-0.5,1]])
    # No degeneracy
    else:
        return nx.directed_laplacian_matrix(digraph_sc)


# bauer laplacian spectrum
# source: Normalized graph Laplacians for directed graphs (Frank Bauer, 2012)

def blsg(matrix, reverse_flow=False, gap='high'):
#  in: tribe matrix
# out: float
    bauer_laplacian_matrix = bls_matrix(matrix, reverse_flow=reverse_flow)
    return spectral_gap(bauer_laplacian_matrix, param=gap)

def bls_matrix(matrix, reverse_flow=False):
#  in: tribe matrix
# out: bauer laplacian matrix
    #non_quasi_isolated = [i for i in range(len(matrix)) if matrix[i].any()]
    #matrix_D = np.diagflat([np.count_nonzero(matrix[nqi]) for nqi in non_quasi_isolated])
    #matrix_W = np.diagflat([np.count_nonzero(np.transpose(matrix)[nqi]) for nqi in non_quasi_isolated])
    #return np.subtract(np.eye(len(non_quasi_isolated),dtype=int),np.matmul(inv(matrix_D),matrix_W))
    current_size = len(matrix)
    return np.subtract(np.eye(current_size,dtype='float64'),tps_matrix(matrix, in_deg=(not reverse_flow)))


##
## TO BE DELETED ONCE EVERYTHING WORKS
##


# # degree
# def degree(chief_index, matrix, vertex_index=0):
#     current_tribe = tribe(chief_index, matrix)
#     return degree_adjacency(current_tribe, vertex_index=vertex_index)

# def degree_adjacency(matrix, vertex_index=0):
#     return in_degree_adjacency(matrix, vertex_index=vertex_index)+out_degree_adjacency(matrix, vertex_index=vertex_index)


# # in-degree
# def in_degree(chief_index, matrix, vertex_index=0):
#     current_tribe = tribe(chief_index, matrix)
#     return in_degree_adjacency(current_tribe, vertex_index=vertex_index)

# def in_degree_adjacency(matrix, vertex_index=0):
#     return np.count_nonzero(np.transpose(matrix)[vertex_index])


# # out-degree
# def out_degree(chief_index, matrix, vertex_index=0):
#     current_tribe = tribe(chief_index, matrix)
#     return out_degree_adjacency(current_tribe, vertex_index=vertex_index)



# def out_degree_adjacency(matrix, vertex_index=0):
#     return np.count_nonzero(matrix[vertex_index])

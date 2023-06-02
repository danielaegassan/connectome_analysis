
#TODO: MODIFY THE IMPORTS TO EXTERNAL IMPORTS


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
    import numpy as np
    from sknetwork.ranking import Closeness
    from scipy.sparse.csgraph import connected_components

    matrix=adj.toarray()
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

def centrality(self, sub_gids, kind="closeness", directed=False):
    """Compute a centrality of the graph. `kind` can be 'betweeness' or 'closeness'"""
    if kind == "closeness":
        return self.closeness(sub_gids, directed)
    else:
        ValueError("Kind must be 'closeness'!")
        #TODO:  Implement betweeness

def connected_components(adj,neuron_properties=[]):
    """Returns a list of the size of the connected components of the underlying undirected graph on sub_gids,
    if None, compute on the whole graph"""
    import networkx as nx
    import numpy as np

    matrix=adj.toarray()
    matrix_und = np.where((matrix+matrix.T) >= 1, 1, 0)
    # TODO: Change the code from below to scipy implementation that seems to be faster!
    G = nx.from_numpy_matrix(matrix_und)
    return [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]

def core_number(adj, neuron_properties=[]):
    """Returns k core of directed graph, where degree of a vertex is the sum of in degree and out degree"""
    # TODO: Implement directed (k,l) core and k-core of underlying undirected graph (very similar to this)
    import networkx
    G = networkx.from_numpy_matrix(adj.toarray())
    # Very inefficient (returns a dictionary!). TODO: Look for different implementation
    return networkx.algorithms.core.core_number(G)

    # TODO: Filtered simplex counts with different weights on vertices (coreness, intersection)
    #  or on edges (strength of connection).

def density(adj, neuron_properties=[]):
    #Todo: Add #cells/volume as as possible spatial density
    adj=adj.astype('bool').astype('int')
    return m.sum() / np.prod(m.shape)

def __make_expected_distribution_model_first_order__(adj, direction="efferent"):
    #TODO: Document, utility function used in COMMON NEIGHBOURS ANALYSIS
    from scipy.stats import hypergeom
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
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy.stats import binom
from scipy import sparse


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


def rich_club_curve(m, nrn, direction='efferent'):
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

    return pd.DataFrame(cum_degrees / nrn_cum_pairs, degree_bins_rv)


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

    df = pd.DataFrame.from_dict(
        {
            "mean": np.nanmean(rr, axis=1),
            "std": np.nanstd(rr, axis=1)
        }
    )
    df.index = res.index
    return df


def normalized_rich_club_curve(m, nrn, direction='efferent', normalize='std',
                               normalize_with="shuffled", **kwargs):
    assert m.dtype == bool, "This function only works for binary matrices at the moment"
    data = rich_club_curve(m, nrn, direction=direction)
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


def rich_club_coefficient(m, nrn, **kwargs):
    Bn = normalized_rich_club_curve(m, normalize='std', **kwargs).values
    return np.nanmean(Bn)

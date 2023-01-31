### TOPOLOGY FUNCTIONS

import sys
import resource
import tempfile
import numpy as np
import pickle
import logging
from functools import partial

from pathlib import Path
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import pandas as pd

from typing import List
# Functions that take as input a (weighted) network and give as output a topological feature.
#TODO: rc_in_simplex, filtered_simplex_counts, persistence

import numpy as np
import pandas as pd


LOG = logging.getLogger("connectome-analysis-topology")
LOG.setLevel("INFO")
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")


def _series_by_dim(from_array, name):
    """A series of counts, like simplex counts:
    one count for a given value of simplex dimension.
    """
    if from_array is None:
        return None

    dim = pd.Index(range(len(from_array)), name="dim")
    return pd.Series(from_array, name=name, index=dim)


def _frame_by_dim(from_array, name, index):
    """A dataframe of counts, like node participation:
    one count for a node and simplex dimension.
    """
    if from_array is None:
        return None

    columns = pd.Index(range(from_array.shape[1]), name=index)
    return pd.DataFrame(from_array, columns=columns).fillna(0).astype(int)


QUANTITIES = {"simplices",
              "node-participation",
              "bettis",
              "bidrectional-edges"}


def _flagser_counts(adjacency,
                    max_simplices=False,
                    count_node_participation=False,
                    list_simplices=False,
                    threads=1):
    """Call package `pyflagsercount's flagser_count` method that can be used to compute
    some analyses, getting counts of quantities such as simplices,
    or node-participation (a.k.a. `containment`)
    """
    import pyflagsercount
    adjacency = adjacency.astype(bool).astype(int)

    flagser_counts = pyflagsercount.flagser_count(adjacency,
                                                  max_simplices=max_simplices,
                                                  containment=count_node_participation,
                                                  return_simplices=list_simplices,
                                                  threads=threads)

    counts =  {"euler": flagser_counts.pop("euler"),
               "simplex_counts": _series_by_dim(flagser_counts.pop("cell_counts"),
                                                name="simplex_count"),
               "max_simplex_counts": _series_by_dim(flagser_counts.pop("max_cell_counts", None),
                                                    name="max_simplex_count"),
               "node-participation": _frame_by_dim(flagser_counts.pop("contain_counts", None),
                                                   name="node_participation", index="node"),
               "simplices": flagser_counts.pop("simplices", None)}
    counts.update(flagser_counts)
    return counts


def node_degree(adj, node_properties=[], direction=None, **kwargs):
    """Count the degree of each node in the adjacency matrix.

    TODO: CHECK THE DIRECTION OF AXII BELOW, expect adj is CSR
    """
    assert not direction or direction in ("IN", "OUT") or tuple(direction) == ("IN", "OUT"),\
        f"Invalid `direction`: {direction}"

    matrix = adj.toarray()
    index = pd.Series(range(matrix.shape[0]), name="node")
    series = lambda array: pd.Series(array, index)
    in_degree = lambda: series(matrix.sum(axis=0))
    out_degree = lambda: series(matrix.sum(axis=1))

    if not direction:
        return in_degree() + out_degree()

    if tuple(direction) == ("IN", "OUT"):
        return pd.DataFrame({"IN": in_degree(), "OUT": out_degree()})

    if tuple(direction) == ("OUT", "IN"):
        return pd.DataFrame({"OUT": out_degree(), "IN": in_degree()})

    return in_degree() if direction == "IN" else out_degree()


def simplex_counts(adj, node_properties=[],
                   max_simplices=False, threads=1,
                   **kwargs):

    #Compute simplex counts of adj
    #TODO: Change this to pyflagser_count and add options for max dim and threads,
    #Delete neuron properties from input?

    flagser_counts = _flagser_counts(adj, threads)
    return flagser_counts["simplex_counts"]


def betti_counts(adj, node_properties=[],
                 min_dim=0, max_dim=[], directed=True, coeff=2, approximation=None,
                 **kwargs):
    """..."""
    LOG.info("Compute betti counts for %s-type adjacency matrix and %s-type node properties",
             type(adj), type(node_properties))

    from pyflagser import flagser_unweighted
    import numpy as np
    adj=adj.astype('bool').astype('int') #Needed in case adj is not a 0,1 matrix
    if max_dim==[]:
        max_dim=np.inf

    if approximation==None:
        LOG.info("Run without approximation")
        bettis = flagser_unweighted(adj, min_dimension=min_dim, max_dimension=max_dim,
                                    directed=True, coeff=2,
                                    approximation=None)['betti']
    else:
        assert (all([isinstance(item,int) for item in approximation])) # asssert it's a list of integers
        approximation=np.array(approximation)
        bettis=[]

        #Make approximation vector to be of size max_dim
        if max_dim!=np.inf:
            if approximation.size-1 < max_dim:#Vector too short so pad with -1's
                approximation=np.pad(approximation,
                                     (0,max_dim-(approximation.size-1)),
                                     'constant',constant_values=-1)
            if approximation.size-1>max_dim:#Vector too long, select relevant slice
                approximation=approximation[0:max_dim+1]
            #Sanity check
            LOG.info("Correct dimensions for approximation: %s", approximation.size==max_dim+1)

        #Split approximation into sub-vectors of same value to speed up computation
        diff=approximation[1:]-approximation[:-1]
        slice_indx=np.array(np.where(diff!=0)[0])+1

        #Compute betti counts
        for dims_range in  np.split(np.arange(approximation.size),slice_indx):
            n=dims_range[0] #min dim for computation
            N=dims_range[-1] #max dim for computation
            a=approximation[n]
            if a==-1:
                a=None
            LOG.info("Run betti for dim range %s-%s with approximation %s", n,N,a)
            bettis=bettis+flagser_unweighted(adj, min_dimension=n, max_dimension=N,
                                             directed=True, coeff=2,
                                             approximation=a)['betti']

        if max_dim==np.inf:
            n=approximation.size #min dim for computation
            N=np.inf #max dim for computation
            a=None
            LOG.info("Run betti for dim range %s-%s with approximation %s",n,N,a)
            bettis=bettis+flagser_unweighted(adj, min_dimension=n, max_dimension=N,
                                             directed=True, coeff=2,
                                             approximation=a)['betti']

    return pd.Series(bettis, name="betti_count",
                     index=pd.Index(range(len(bettis)), name="dim"))


def node_participation(adj, node_properties=None):
    # Compute the number of simplices a vertex is part of
    # Input: adj adjancency matrix representing a graph with 0 in the diagonal, neuron_properties as data frame with index gid of nodes
    # Out: List L of lenght adj.shape[0] where L[i] is a list of the participation of vertex i in simplices of a given dimensio
    # TODO:  Should we merge this with simplex counts so that we don't do the computation twice?

    flagser_counts = _flagser_counts(adj, count_node_participation=True)
    return flagser_counts["node_participation"]


''''#INPUT: Address of binary file storing simplices
#OUTPUT: A list if lists L where L[i] contains the vertex ids of the i'th simplex,
#          note the simplices appear in no particular order

def binary2simplex(address, test=None, verbosity=1000000):
    """...Not used --- keeping it here as it is of interest to understanmd
    how simplices are represented on the disc by Flagser.
    """
    LOG.info("Load binary simplex info from %s", address)
    simplex_info = pd.Series(np.fromfile(address, dtype=np.uint64))
    LOG.info("Done loading binary simplex info.")

    if test:
        simplex_info = simplex_info.iloc[0:test]

    mask64 = np.uint(1) << np.uint(63)
    mask21 = np.uint64(1 << 21) - np.uint64(1)
    mask42 = (np.uint64(1 << 42) - np.uint64(1)) ^ mask21
    mask63 = ((np.uint64(1 << 63) - np.uint64(1)) ^ mask42) ^ mask21
    end = np.uint64(2 ** 21 - 1)

    def decode_vertices(integer):
        decode_vertices.ncalls += 1
        if decode_vertices.ncalls % verbosity == 0:
            mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            LOG.info("\t progress %s / %s memory %s",
                     decode_vertices.ncalls , len(simplex_info), mem_used)
        integer = np.uint64(integer)
        start = not((integer & mask64) >> np.uint64(63))
        v0 = integer & mask21
        v1 = (integer & mask42) >> np.uint64(21)
        v2 = (integer & mask63) >> np.uint64(42)
        vertices = [v for v in [v0, v1, v2] if v != end]
        return pd.Series([start, vertices], index=["start", "vertices"])
    #    vertices = [start, v0, v1, v2]
    #    return pd.Series(vertices, index=["start", 0, 1, 2])
    decode_vertices.ncalls = 0

    LOG.info("Decode the simplices into simplex vertices")
    vertices = simplex_info.apply(decode_vertices)
    LOG.info("Done decoding to simplex vertices")

    vertices = (vertices.assign(sid=np.cumsum(vertices.start))
                .reset_index(drop=True))

    simplices = (vertices.set_index("sid").vertices
                 .groupby("sid").apply(np.hstack))

    if not test:
        return simplices

    return (vertices.vertices, simplices)
'''

def list_simplices_by_dimension(adj, nodes=None, max_simplices=False,
                                verbose=False, **kwargs):
    """List all the simplices (upto a max dimension) in an adjacency matrix.
    """
    LOG.info("COMPUTE list of %s simplices by dimension", "max-" if max_simplices else "")

    N, M = adj.shape
    assert N == M, f"{N} != {M}"

    n_threads = kwargs.get("threads", kwargs.get("n_threads", 1))
    fcounts = _flagser_counts(adj, list_simplices=True, max_simplices=max_simplices,
                              threads=n_threads)
    original = fcounts["simplices"]
    coom = adj.tocoo()

    max_dim = len(original)
    dims = pd.Index(np.arange(max_dim), name="dim")
    simplices = pd.Series(original, name="simplices", index=dims)
    simplices[0] = np.reshape(np.arange(0, N), (N, 1))
    simplices[1] = np.stack([coom.row, coom.col]).T
    return simplices


def bedge_counts(adjacency, nodes=None, simplices=None, **kwargs):
    """...
    adj : Adjacency matrix N * N
    simplices : sequence of 2D arrays that contain simplices by dimension.
    ~           The Dth array will be of shape N * D
    ~           where D is the dimension of the simplices
    """
    adj = adjacency

    if simplices is None:
        LOG.info("COMPUTE `bedge_counts(...)`: No argued simplices.")
        return bedge_counts(adj, nodes, list_simplices_by_dimension(adj), **kwargs)
    else:
        LOG.info("COMPUTE `bedge_counts(...): for simplices: %s ", simplices.shape)

    dense = np.array(adjacency.toarray(), dtype=int)

    def subset_adj(simplex):
        return dense[simplex].T[simplex]

    def count_bedges(simplices_given_dim):
        """..."""
        try:
            d_simplices = simplices_given_dim.get_value()
        except AttributeError:
            d_simplices = simplices_given_dim

        if d_simplices is None or d_simplices.shape[1] == 1:
            return np.nan

        return (pd.DataFrame(d_simplices, columns=range(d_simplices.shape[1]))
                .apply(subset_adj, axis=1)
                .agg("sum"))

    return simplices.apply(count_bedges)


def convex_hull(adj, node_properties):# --> topology
    """Return the convex hull of the sub gids in the 3D space using x,y,z position for gids"""
    pass


## Filtered objects
def at_weight_edges(weighted_adj, threshold, method="strength"):
    """ Returns thresholded network on edges
    :param method: distance returns edges with weight smaller or equal than thresh
                   strength returns edges with weight larger or equal than thresh
                   assumes csr format for weighted_adj"""
    data=weighted_adj.data
    data_thresh=np.zeros(data.shape)
    if method == "strength":
        data_thresh[data>=threshold]=data[data>=threshold]
    elif method == "distance":
        data_thresh[data<=threshold]=data[data<=threshold]
    else:
        raise ValueError("Method has to be 'strength' or 'distance'")
    adj_thresh=weighted_adj.copy()
    adj_thresh.data=data_thresh
    adj_thresh.eliminate_zeros()
    return adj_thresh


def filtration_weights(adj, node_properties=None, method="strength"):
    """
    Returns the filtration weights of a given weighted adjacency matrix.
    :param method: distance smaller weights enter the filtration first
                   strength larger weights enter the filtration first

    TODO: Should there be a warning when the return is an empty array because the matrix is zero?
    """
    if method == "strength":
        return np.unique(adj.data)[::-1]

    if method == "distance":
        return np.unique(adj.data)

    raise ValueError("Method has to be 'strength' or 'distance'")


def bin_weigths(weights, n_bins=10, return_bins=False):
    '''Bins the np.array weights
    Input: np.array of floats, no of bins
    returns: bins, and binned data i.e. a np.array of the same shape as weights with entries the center value of its corresponding bin
    '''
    tol = 1e-8 #to include the max value in the last bin
    min_weight = weights.min()
    max_weight = weights.max() + tol
    step = (max_weight - min_weight) / n_bins
    bins = np.arange(min_weight, max_weight + step, step)
    digits = np.digitize(weights, bins)

    weights = (min_weight + step / 2) + (digits - 1) * step
    return (weights, bins) if return_bins else weights


def filtered_simplex_counts(adj, node_properties=None, method="strength",
                            binned=False, n_bins=10, threads=1,
                            **kwargs):
    '''Takes weighted adjancecy matrix returns data frame with filtered simplex counts where index is the weight
    method strength higher weights enter first, method distance smaller weights enter first'''
    from tqdm import tqdm
    adj = adj.copy()
    if binned==True:
        adj.data = bin_weigths(adj.data, n_bins=n_bins)

    weights = filtration_weights(adj, node_properties, method)

#    TODO: 1. Prove that the following is executed in the implementation that follows.
#    TODO: 2. If any difference, update the implementation
#    TODO: 3. Remove the reference code.
#    n_simplices = dict.fromkeys(weights)
#    for weight in tqdm(weights[::-1],total=len(weights)):
#        adj = at_weight_edges(adj, threshold=weight, method=method)
#        n_simplices[weight] = simplex_counts(adj, threads=threads)

    m = method
    def filter_weight(w):
        adj_w = at_weight_edges(adj, threshold=w, method=m)
        return simplex_counts(adj_w, threads=threads)

    n_simplices = {w: filter_weight(w) for w in weights[::-1]}
    return pd.DataFrame.from_dict(n_simplices, orient="index").fillna(0).astype(int)


def chunk_approx_and_dims(min_dim=0, max_dim=[], approximation=None):
    # Check approximation list is not too long and it's a list of integers
    assert (all([isinstance(item, int) for item in approximation])), 'approximation must be a list of integers'
    approximation = np.array(approximation)
    if max_dim == []:
        max_dim = np.inf
    assert (approximation.size - 1 <= max_dim - min_dim), "approximation list too long for the dimension range"

    # Split approximation into sub-vectors of same value to speed up computation
    diff = approximation[1:] - approximation[:-1]
    slice_indx = np.array(np.where(diff != 0)[0]) + 1
    dim_chunks = np.split(np.arange(approximation.size) + min_dim, slice_indx)
    if approximation[-1] == -1:
        dim_chunks[-1][-1] = -1
    else:
        if dim_chunks[-1][-1] < max_dim:
            dim_chunks.append([dim_chunks[-1][-1] + 1, max_dim])

    # Returned chuncked lists
    approx_chunks = []
    for j in range(len(dim_chunks)):
        if (approximation.size < max_dim - min_dim + 1) and approximation[-1] != -1 and j == len(dim_chunks) - 1:
            a = -1
        else:
            a = approximation[int(dim_chunks[j][0]) - min_dim]
        approx_chunks.append(a)
    return dim_chunks, approx_chunks


def persistence(weighted_adj, node_properties=None,
                min_dim=0, max_dim=[], directed=True, coeff=2, approximation=None,
                invert_weights=False, binned=False, n_bins=10, return_bettis=False,
                **kwargs):
    from pyflagser import flagser_weighted
    import numpy as np
    # Normalizing and binning data
    adj = weighted_adj.copy()
    if invert_weights == True:
        # Normalizing data between 0-1 and inverting order of the entries
        adj.data = (np.max(adj.data) - adj.data) / (np.max(adj.data) - np.min(adj.data))
    if binned == True:
        adj.data = bin_weigths(adj.data, n_bins=n_bins)

    # Sending computation to flagser
    # For single approximate value
    if approximation == None or isinstance(approximation, int):
        if min_dim != 0:
            LOG.info("Careful of pyflagser bug with range in dimension")
        out = flagser_weighted(adj, min_dimension=min_dim, max_dimension=max_dim, directed=True, coeff=2,
                               approximation=approximation)
        dgms = out['dgms']
        bettis = out['betti']
    # For approximate list
    else:
        # Chunk values to speed computations
        dim_chunks, approx_chunks = chunk_approx_and_dims(min_dim=min_dim, max_dim=max_dim, approximation=approximation)
        bettis = []
        dgms = []
        for dims_range, a in zip(dim_chunks, approx_chunks):
            n = dims_range[0]  # min dim for computation
            N = dims_range[-1]  # max dim for computation
            if N == -1:
                N = np.inf
            if a == -1:
                a = None
            LOG.info("Run betti for dim range %s-%s with approximation %s",n, N, a)
            if n != 0:
                LOG.info("Warning, possible bug in pyflagser when not running dimension range starting at dim 0")
            out = flagser_weighted(adj, min_dimension=n, max_dimension=N, directed=True, coeff=2, approximation=a)
            bettis = bettis + out['betti']
            dgms = dgms + out['dgms']
            LOG.info("out: %s",[out['dgms'][i].shape for i in range(len(out['dgms']))])
    if return_bettis == True:
        return dgms, bettis
    else:
        return dgms

#Tools for persistence
def num_cycles(B,D,thresh):
    #Given a persistence diagram (B,D) compute the number of cycles alive at tresh
    #Infinite bars have death values np.inf
    born=np.count_nonzero(B<=thresh)
    dead=np.count_nonzero(D<=thresh)
    return born-dead

def betti_curve(B,D):
    #Given a persistence diagram (B,D) compute its corresponding betti curve
    #Infinite bars have death values np.inf
    filt_values=np.concatenate([B,D])
    filt_values=np.unique(filt_values)
    filt_values=filt_values[filt_values!=np.inf]
    bettis=[]
    for thresh in filt_values:
        bettis.append(num_cycles(B,D,thresh))
    return filt_values,np.array(bettis)


#TODO: MODIFY THE IMPORTS TO EXTERNAL IMPORTS

###NETWORK PAIRWISE METRICS

def closeness_connected_components(adj, neuron_properties=[], directed=False, return_sum=True):
    """
    Compute the closeness of each connected component of more than 1 vertex
    :param matrix: shape (n,n)
    :param directed: if True compute using strong component and directed closeness
    :param return_sum: if True return only one list given by summing over all the component
    :return a single array( if return_sum=True) or a list of array of shape n,
    containting closeness of vertex in this component
    or 0 if vertex is not in the component, in any case closeness cant be zero otherwise
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



#What to do with these since they are all about a graph and a subgraph?

"""def degree(self, sub_gids=None, kind="in"):
    #Return in/out degrees of the subgraph, if None compute on the whole graph
    if sub_gids is not None:
        matrix = self.subarray(self.__extract_gids__(sub_gids))
    else:
        matrix = self.array
    if kind == "in":
        return np.sum(matrix, axis=0)
    elif kind == "out":
        return np.sum(matrix, axis=1)
    else:
        ValueError("Need to specify 'in' or 'out' degree!")


def density(self, sub_gids=None):
    if sub_gids is None:
        m = self.matrix
    else:
        m = self.submatrix(sub_gids)
    return m.getnnz() / np.prod(m.shape)"""


###COMMON NEIGHBOURS ANALYSIS

def __make_expected_distribution_model_first_order__(adj, direction="efferent"):
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
    data = distribution_number_of_common_neighbors(adj, neuron_properties, direction=direction)
    expected = __make_expected_distribution_model_first_order__(adj, direction=direction)
    expected = expected.pmf(data.index) * data.sum()
    expected = pd.Series(expected, index=data.index)
    return (data - expected) / (data + expected)


def overexpression_of_common_neighbors(adj, neuron_properties=None, direction="efferent"):
    data = distribution_number_of_common_neighbors(adj, neuron_properties, direction=direction)
    data_mean = (data.index.values * data.values).sum() / data.values.sum()
    ctrl = __make_expected_distribution_model_first_order__(adj, direction=direction)
    ctrl_mean = ctrl.mean()
    return (data_mean - ctrl_mean) / (data_mean + ctrl_mean)


def common_neighbor_weight_bias(adj, neuron_properties=None, direction="efferent"):
    adj_bin = (adj.tocsc() > 0).astype(int)
    if direction == "efferent":
        cn = adj_bin * adj_bin.transpose()
    elif direction == "afferent":
        cn = adj_bin.transpose() * adj_bin

    return np.corrcoef(cn[adj > 0],
                       adj[adj > 0])[0, 1]


def common_neighbor_connectivity_bias(adj, neuron_properties=None, direction="efferent",
                                      cols_location=None, fit_log=False):
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

###Computing oconnection probabilities

def connection_probability_within(adj, neuron_properties, max_dist=200, min_dist=0,
                                  columns=["ss_flat_x", "ss_flat_y"]):
    if isinstance(neuron_properties, tuple):
        nrn_pre, nrn_post = neuron_properties
    else:
        nrn_pre = neuron_properties;
        nrn_post = neuron_properties
    D = distance.cdist(nrn_pre[columns], nrn_post[columns])
    mask = (D > min_dist) & (D <= max_dist)  # This way with min_dist=0 a neuron with itself is excluded
    return adj[mask].mean()


def connection_probability(adj, neuron_properties):
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


##Degree based analyses
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

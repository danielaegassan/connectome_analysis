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


#INPUT: Address of binary file storing simplices
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

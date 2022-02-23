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
import pyflagsercount as pfc

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


def simplex_counts(adj, neuron_properties=[]):
    #Compute simplex counts of adj
    #TODO: Change this to pyflagser_count and add options for max dim and threads,
    #Delete neuron properties from input?
    from pyflagser import flagser_count_unweighted
    adj=adj.astype('bool').astype('int') #Needed in case adj is not a 0,1 matrix
    counts = np.array(flagser_count_unweighted(adj, directed=True))
    return pd.Series(counts, name="simplex_count",
                     index=pd.Index(range(len(counts)), name="dim"))


def betti_counts(adj, neuron_properties=[], min_dim=0, max_dim=[],
                 directed=True, coeff=2, approximation=None):
    from pyflagser import flagser_unweighted
    import numpy as np
    adj=adj.astype('bool').astype('int') #Needed in case adj is not a 0,1 matrix
    if max_dim==[]:
        max_dim=np.inf

    if approximation==None:
        print("Run without approximation")
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
            print("Correct dimensions for approximation:", approximation.size==max_dim+1)

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
            print("Run betti for dim range {0}-{1} with approximation {2}".format(n,N,a))
            bettis=bettis+flagser_unweighted(adj, min_dimension=n, max_dimension=N,
                                             directed=True, coeff=2,
                                             approximation=a)['betti']

        if max_dim==np.inf:
            n=approximation.size #min dim for computation
            N=np.inf #max dim for computation
            a=None
            print("Run betti for dim range {0}-{1} with approximation {2}".format(n,N,a))
            bettis=bettis+flagser_unweighted(adj, min_dimension=n, max_dimension=N,
                                             directed=True, coeff=2,
                                             approximation=a)['betti']

    return pd.Series(bettis, name="betti_count",
                     index=pd.Index(range(len(bettis)), name="dim"))


def node_participation(adj, neuron_properties):
    # Compute the number of simplices a vertex is part of
    # Input: adj adjancency matrix representing a graph with 0 in the diagonal, neuron_properties as data frame with index gid of nodes
    # Out: List L of lenght adj.shape[0] where L[i] is a list of the participation of vertex i in simplices of a given dimensio
    # TODO:  Should we merge this with simplex counts so that we don't do the computation twice?
    import pyflagsercount
    import pandas as pd
    adj = adj.astype('bool').astype('int')  # Needed in case adj is not a 0,1 matrix
    par=pyflagsercount.flagser_count(adj,containment=True,threads=1)['contain_counts']
    par_frame = pd.DataFrame(par).fillna(0).astype(int)
    par_frame.columns.name = "dim"
    return par_frame


#INPUT: Address of binary file storing simplices
#OUTPUT: A list if lists L where L[i] contains the vertex ids of the i'th simplex,
#          note the simplices appear in no particular order
def binary2simplex_MS(address):
    X = np.fromfile(address, dtype='uint64')                         #Load binary file
    S=[]                                                             #Initialise empty list for simplices

    i=0
    while i < len(X):
        b = format(X[i], '064b')                                     #Load the 64bit integer as a binary string
        if b[0] == '0':                                              #If the first bit is 0 this is the start of a new simplex
            S.append([])
        t=[int(b[-21:],2), int(b[-42:-21],2), int(b[-63:-42],2)]     #Compute the 21bit ints stored in this 64bit int:
        for j in t:
            if j != 2097151:                                         #If an int is 2^21 this means we have reached the end of the simplex, so don't add it
                S[-1].append(j)
        i+=1
    return S


def binary2simplex(address, test=None, verbosity=1000000):
    """..."""
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


def simplex_matrix(adj: sp.csc_matrix, nodes: pd.DataFrame,
                   temp_folder: Path, verbose: bool = False) -> np.array:
    """
    Returns the list of simplices in matrix form for storage. The matrix is
    a n_simplices x max_dim matrix, where n_simplices is the total number of simplices
    and max_dim is the maximum encountered simplex dimension.
    
    :param adj: Sparse csc matrix to compute the simplex list of.
    :type: sp.scs_matrix
    :param temp_folder: Relative path where to store the temporary flagser files to.
    :type: Path
    :param verbose: Whether to have the function print steps.
    :type: bool

    :return m: Matrix containing the simplices. The first row will be the indices at which each
    dimension ends. Successive rows will store simplices from each dimension.
    :rtype: np.array
    """
    def vmessage(message):
        if verbose:
            print(message)
    temp_folder.mkdir(exist_ok = True, parents=True)
    for path in temp_folder.glob("*"):
        raise FileExistsError("Found file in " + str(temp_folder.absolute()) + ". Aborting.")
    vmessage("Flagser count started execution")
    counts = pfc.flagser_count(adj, binary=str(temp_folder / "temp"), min_dim_print=1, threads = 1)
    vmessage("Flagser count completed execution")
    mdim = len(counts['cell_counts'])
    simplex_matrix = np.zeros((np.sum(counts['cell_counts'][1:]) + 1, mdim), dtype=np.int32)
    simplex_matrix[0,:] = counts['cell_counts'] #Use this both for future reference and for positioning simplices
    simplex_matrix[0,0] = 0
    simplex_matrix[0] = np.cumsum(simplex_matrix[0])
    vmessage("Parsing flagser output")
    for path in temp_folder.glob("*"):
        vmessage("Parsing " + str(path))
        simplex_list = binary2simplex(path)
        if verbose:
            simplex_list = tqdm(simplex_list, desc="Parsed simplices",
                                total = simplex_matrix[0,-1])
        for simplex in simplex_list:
             simplex_matrix[simplex_matrix[0,len(simplex)-2] + 1, :len(simplex)] = simplex
             simplex_matrix[0, len(simplex)-2]+=1
    vmessage("Generated simplex matrix")
    np.save(temp_folder/"matrix.npy", simplex_matrix)
    vmessage("Saved simplex matrix")
    return simplex_matrix


def simplices(adj, nodes=None, temp_folder=None, threads=None, **kwargs):
    """All the simplices in an adjacency matrix.

    temp_folder : A binary file will be generated here.
    """
    import tempfile
    threads = threads or 1
    LOG.warning("Compute simplices for a matrix of shape %s on %s threads",
                adj.shape, threads)

    tempfold = tempfile.TemporaryDirectory(dir=Path.cwd())
    path_temp = Path(tempfold.name)
    if list(path_temp.glob('*')):
        raise RuntimeError("Was not expecting a temporary folder to contain any files")

    counts = pfc.flagser_count(adj, binary=(path_temp / "temp-").as_posix(),
                               min_dim_print=1, threads=threads)

    sims  = pd.Series([s for p in path_temp.glob('*')
                       for s in binary2simplex(p, **kwargs)],
                      name="simplex")
    dims = sims.apply(len).apply(lambda l: l - 2).rename("dim")
    sims_dims = pd.concat([sims, dims], axis=1).set_index("dim").simplex

    tempfold.cleanup()

    return sims_dims.groupby("dim").apply(np.vstack)

def maximal_simplex_lists(adj: sp.csc_matrix, verbose: bool = False) -> List[np.array]:
    """
    Returns the list of maximal simplices in a list of matrices for storage. Each matrix is
    a n_simplices x dim matrix, where n_simplices is the total number of simplices
    with dimension dim. No temporary file needed!

    :param adj: Sparse csc matrix to compute the simplex list of.
    :type: sp.scs_matrix
    :param verbose: Whether to have the function print steps.
    :type: bool

    :return mlist: List of matrices containing the maximal simplices.
    :rtype: List[np.array]
    """
    result = pfc.flagser_count(adj, return_simplices=True, max_simplices=True, threads=1)
    coo_matrix = adj.tocoo()
    result['simplices'][1] = np.stack([coo_matrix.row, coo_matrix.col]).T
    for i in range(len(result['simplices']) - 2):
        result['simplices'][i + 2] = np.array(result['simplices'][i + 2])
    return result['simplices'][1:]


def simplex_lists(adj: sp.csc_matrix, verbose: bool = False) -> List[np.array]:
    """
    Returns the list of simplices in a list of matrices for storage. Each matrix is
    a n_simplices x dim matrix, where n_simplices is the total number of simplices
    with dimension dim. No temporary file needed!
    
    :param adj: Sparse csc matrix to compute the simplex list of.
    :type: sp.scs_matrix
    :param verbose: Whether to have the function print steps.
    :type: bool

    :return mlist: List of matrices containing the simplices. 
    :rtype: List[np.array]
    """
    result = pfc.flagser_count(adj, return_simplices=True, threads = 1)
    coo_matrix = adj.tocoo()
    result['simplices'][1] = np.stack([coo_matrix.row, coo_matrix.col]).T
    for i in range(len(result['simplices']) -2):
        result['simplices'][i+2] = np.array(result['simplices'][i+2])
    return result['simplices'][1:]


def list_simplices_by_dimension(adj, nodes=None, verbose=False, **kwargs):
    """List all the simplices (upto a max dimension) in an adjacency matrix.
    """
    N, M = adj.shape
    assert N == M, f"{N} != {M}"

    n_threads = kwargs.get("threads", kwargs.get("n_threads", 1))
    fcounts = pfc.flagser_count(adj, return_simplices=True, threads=n_threads)
    original = fcounts["simplices"]
    coom = adj.tocoo()

    max_dim = len(original)
    dims = pd.Index(np.arange(max_dim), name="dim")
    simplices = pd.Series(original, name="simplices", index=dims)
    simplices[0] = np.reshape(np.arange(0, N), (N, 1))
    simplices[1] = np.stack([coom.row, coom.col]).T
    return simplices


def fetch_analysis(a, among, for_key, adjacency, nodes=None, **kwargs):
    """Either get results analysis `for_key` from the pipeline, or compute it!

    Arguments
    among :: the analyses to fetch from.
    """
    a, computation = a

    try:
        toc = among[a]
    except TypeError as terror:
        raise TypeError(f"Expecting a dict mapping name --> TOC for annalysis, NOT {among}\n"
                        "{terror}")
    except KeyError:
        LOG.warning("Analysis %s seems not to have been computed already.")
        pass
    else:
        return toc.loc[for_key]

    return computation(adj, nodes, **kwargs)


def bedge_counts_0(adjacency, nodes=None, key=None, simplices=None, **kwargs):
    """
    Function to count bidirectional edges in all positions for all simplices of a matrix.

    :param adj: adjacency matrix of the whole graph.
    :type: sp.csc_matrix
    :param simplex_matrix_list: list of arrays containing the simplices. Each array is
    a  n_simplices x simplex_dim array. Same output as simplex_matrix_list.
    :type: List[np.ndarray]

    :return bedge_counts: list of 2d matrices with bidirectional edge counts per position.
    :rtype: List[np.ndarray]
    """

    def extract_subm(simplex: np.ndarray, adj: np.ndarray)->np.ndarray:
        return adj[simplex].T[simplex]

    adj_dense = np.array(adj.toarray()) #Did not test sparse matrix performance
    bedge_counts = []
    for matrix in simplices:
        bedge_counts.append(
            np.sum(
                np.apply_along_axis(
                    partial(extract_subm, adj = adj_dense),
                    1,
                    matrix,
                ), axis = 0
            )
        )
    return bedge_counts


def bedge_counts(adjacency, nodes=None, simplices=None, **kwargs):
    """
    Adapted from `bedge_counts` implementation by MS.

    adj : Adjacency matrix N * N
    simplices : sequence of 2D arrays that contain simplices by dimension.
    ~           The Dth array will be of shape N * D
    ~           where D is the dimension of the simplices
    ~           and N the number of such matrices in the graph.
    """
    dense = np.array(adjacency.toarray(), dtype=int)

    def subset_adj(simplex):
        """Adjacency matrix subset to the nodes in a simplex.
        """
        return dense[simplex].T[simplex].astype(int)

    def collect_adjacencies(of_simplices_given_dimension):
        return of_simplices_given_dimension.sum(axis=0)

    if simplices  is None:
        simplices = list_simplices_by_dimension(adjacency)

    return (simplices
            .apply(lambda simps_d: np.apply_along_axis(subset_adj, 1, simps_d))
            .apply(collect_adjacencies))


def convex_hull(adj, neuron_properties):
    """Return the convex hull of the sub gids in the 3D space using x,y,z position for gids"""
    pass


## Filtered objects
def at_weight_edges(weighted_adj, threshold, method="strength"):
    #TODO: Efficient implementation with sparse matrices.
    #TODO: Filtration on vertices
    """ Returns thresholded network on edges
    :param method: distance returns edges with weight smaller or equal than thresh
                   strength returns edges with weight larger or equal than tresh"""
    adj=adj.toarray()
    adj_thresh=np.zeros(adj.shape)
    if method == "strength":
        adj_tresh[adj_thresh>=threshold]=adj[adj_thresh>=threshold]
    elif method == "distance":
        adj_tresh[adj_thresh<=threshold]=adj[adj_thresh<=threshold]
    else:
        raise ValueError("Method has to be 'strength' or 'distance'")
    return adj_thresh


def filtration_weights(weighted_adj, neuron_properties=[],method="strength"):
    #Todo: Should there be a warning when the return is an empty array because the matrix is zero?
    """Returns the filtration weights of a given weighted matrix.
    :param method:distance smaller weights enter the filtration first
                  strength larger weights enter the filtration first"""
    if method == "strength":
        return np.unique(adj.data)[::-1]
    elif method == "distance":
        return np.unique(adj.data)
    else:
       raise ValueError("Method has to be 'strength' or 'distance'")


def filtered_simplex_counts(weighted_adj, neuron_properties=[],method="strength"):
    simplex_counts_filtered=[]
    for weight in filtration_weights(weighted_adj,neuron_properties=[],method=method):
        adj=at_weight_edges(weighted_adj,neuron_properties=[],threshold=weight,method=method)
        simplex_counts_filtered.append(simplex_counts(adj))
    return  simplex_counts

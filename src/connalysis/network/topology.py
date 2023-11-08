# Network analysis functions based on topological constructions
#
# Author(s): D. Egas Santander, M. Santoro, JP. Smith, V. Sood
# Last modified: 03/2023

#TODO unweighted: convex hull, combinatorial ricci,
#TODO weighted: filtered_simplex_counts, persistence,


import resource
import numpy as np
import pandas as pd
import logging
import scipy.sparse as sp
import pyflagsercount
import pyflagser
import math

from .local import neighbourhood

#Imports not used as global imports, check what can be removed.
# import sys
# import tempfile
# import pickle
# from functools import partial
# from pathlib import Path
# from tqdm import tqdm
# from typing import List


LOG = logging.getLogger("connectome-analysis-topology")
LOG.setLevel("INFO")
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")

#######################################################
################# UNWEIGHTED NETWORKS #################
#######################################################

def rc_submatrix(adj):
    """Returns the symmetric submatrix of reciprocal connections of adj
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.

    Returns
    -------
    sparse matrix
        symmetric matrix of the same dtype as adj of reciprocal connections
    """
    adj=sp.csr_matrix(adj)
    if np.count_nonzero(adj.diagonal()) != 0:
        logging.warning('The diagonal is non-zero and this may lead to errors!')
    mask=adj.copy().astype('bool')
    mask=(mask.multiply(mask.T))
    mask.eliminate_zeros
    return adj.multiply(mask).astype(adj.dtype)

def underlying_undirected_matrix(adj):
    """Returns the symmetric matrix of undirected connections of `adj`.
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry in `adj[i][j]` implies there is an edge from vertex `i` to vertex `j`.

    Returns
    -------
    sparse boolean matrix
        Corresponding to the symmetric underlying undirected graph
    """
    adj=sp.csr_matrix(adj)
    if np.count_nonzero(adj.diagonal()) != 0:
        logging.warning('The diagonal is non-zero and this may lead to errors!')
    return (adj+adj.T).astype('bool')


def _series_by_dim(from_array, name_index=None, index=None, name=None):
    """A series of counts, like simplex counts:
    one count for a given value of simplex dimension.
    """
    if from_array is None:
        return None
    if index is None:
        index = pd.Index(range(len(from_array)), name=name_index)
    else:
        assert len(index)==len(from_array), "array and index are not the same length"
        index = pd.Index(index, name=name_index)
    return pd.Series(from_array, index=index, name=name)


def _frame_by_dim(from_array, no_columns, name, index):
    """A dataframe of counts, like node participation:
    one count for a node and simplex dimension.
    """
    if from_array is None:
        return None
    #Todo add method for when no_columns is not given
    columns = pd.Index(range(no_columns), name=index)
    return pd.DataFrame(from_array, columns=columns).fillna(0).astype(int)


QUANTITIES = {"simplices",
              "node-participation",
              "bettis",
              "bidrectional-edges"}

def _flagser_counts(adjacency,
                    max_simplices=False,
                    count_node_participation=False,
                    list_simplices=False,
                    threads=1,max_dim=-1, edge_containment=False):
    """Call package `pyflagsercount's flagser_count` method that can be used to compute
    some analyses, getting counts of quantities such as simplices,
    or node-participation (a.k.a. `containment`)
    """
    import pyflagsercount
    adjacency = sp.csr_matrix(adjacency.astype(bool).astype(int))
    if np.count_nonzero(adjacency.diagonal()) != 0:
        logging.warning('The diagonal is non-zero!  Non-zero entries in the diagonal will be ignored.')


    flagser_counts = pyflagsercount.flagser_count(adjacency,
                                                  max_simplices=max_simplices,
                                                  containment=count_node_participation,
                                                  return_simplices=list_simplices,
                                                  threads=threads,max_dim=max_dim,
                                                  edge_containment=edge_containment)

    counts =  {"euler": flagser_counts.pop("euler"),
               "simplex_counts": _series_by_dim(flagser_counts.pop("cell_counts"),
                                                name="simplex_count", name_index="dim"),
               "max_simplex_counts": _series_by_dim(flagser_counts.pop("max_cell_counts", None),
                                                    name="max_simplex_count", name_index="dim"),
               "simplices": flagser_counts.pop("simplices", None)}
    if counts["max_simplex_counts"] is None:
        max_dim_participation=counts["simplex_counts"].shape[0]
    else:
        max_dim_participation=counts["max_simplex_counts"].shape[0]
    counts["node_participation"]= _frame_by_dim(flagser_counts.pop("contain_counts", None),max_dim_participation,
                                                name="node_participation", index="node")
    counts.update(flagser_counts)
    return counts


def node_degree(adj, node_properties=None, direction=None, weighted=False, **kwargs):
    """Compute degree of nodes in network adj
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j
        of weight adj[i,j].
    node_properties : data frame
        Data frame of neuron properties in adj. Only necessary if used in conjunction with TAP or connectome utilities.
    direction : string or tuple of strings
        Direction for which to compute the degree

        'IN' - In degree

        'OUT'- Out degree

        None or ('IN', 'OUT') - Total degree i.e. IN+OUT

    Returns
    -------
    series or data frame

    Raises
    ------
    Warning
        If adj has non-zero entries in the diagonal
    AssertionError
        If direction is invalid
    """
    assert not direction or direction in ("IN", "OUT") or tuple(direction) == ("IN", "OUT"),\
        f"Invalid `direction`: {direction}"

    if not isinstance(adj, np. ndarray):
        matrix = adj.toarray()
    else:
        matrix=adj.copy()
    if not weighted:
        matrix=matrix.astype('bool')
    if np.count_nonzero(np.diag(matrix)) != 0:
        logging.warning('The diagonal is non-zero!  This may cause errors in the analysis')
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

def node_k_degree(adj, node_properties=None, direction=("IN", "OUT"), max_dim=-1, **kwargs):
    #TODO: Generalize from one population to another
    """Compute generalized degree of nodes in network adj.  The k-(in/out)-degree of a node v is the number of
    k-simplices with all its nodes mapping to/from the node v.
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j
        of weight adj[i,j].  The matrix can be asymmetric, but must have 0 in the diagonal.
    node_properties : dataframe
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    direction : string
        Direction for which to compute the degree

        'IN' - In degree

        'OUT'- Out degree

        (’IN’, ’OUT’) - both
    max_dim : int
        Maximal dimension for which to compute the degree max_dim >=2 or -1 in
        which case it computes all dimensions.

    Returns
    -------
    data frame
        Table of of k-(in/out)-degrees

    Raises
    ------
    Warning
        If adj has non-zero entries in the diagonal which are ignored in the analysis
    AssertionError
        If direction is invalid
    AssertionError
        If not max_dim >1

    Notes
    -----
    Note that the k-in-degree of a node v is the number of (k+1) simplices the node v is a sink of.
    Dually, the k-out-degree of a node v is the number of (k+1) simplices the node v is a source of.
    """
    matrix = sp.csr_matrix(adj)
    assert (max_dim > 1) or (max_dim==-1), "max_dim should be >=2"
    assert direction in ("IN", "OUT") or tuple(direction) == ("IN", "OUT"), \
        f"Invalid `direction`: {direction}"
    if np.count_nonzero(matrix.diagonal()) != 0:
        logging.warning('The diagonal is non-zero!  Non-zero entries in the diagonal will be ignored.')
    import pyflagsercount
    flagser_out = pyflagsercount.flagser_count(matrix, return_simplices=True, max_dim=max_dim)
    max_dim_possible = len(flagser_out['cell_counts']) - 1
    if max_dim==-1:
        max_dim = max_dim_possible
    elif max_dim > max_dim_possible:
        logging.warning("The maximum dimension selected is not attained")
        max_dim = max_dim_possible
    if (max_dim <= 1) and (max_dim!=-1):
        print("There are no simplices of dimension 2 or higher")
    else:
        index = pd.Series(range(matrix.shape[0]), name="node")
        generalized_degree = pd.DataFrame(index=index)
        for dim in np.arange(2, max_dim + 1):
            if "OUT" in direction:
                # getting source participation across dimensions
                x, y = np.unique(np.array(flagser_out['simplices'][dim])[:, 0], return_counts=True)
                generalized_degree[f'{dim}_out_degree'] = pd.Series(y, index=x)
            if "IN" in direction:
                # getting sink participation across dimensions
                x, y = np.unique(np.array(flagser_out['simplices'][dim])[:, dim], return_counts=True)
                generalized_degree[f'{dim}_in_degree'] = pd.Series(y, index=x)
        return generalized_degree.fillna(0)


def simplex_counts(adj, node_properties=None,max_simplices=False,
                   threads=1,max_dim=-1, simplex_type='directed', **kwargs):
    # TODO: ADD TRANSPOSE
    """Compute the number of simplex motifs in the network adj.
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j
        of weight adj[i,j].  The matrix can be asymmetric, but must have 0 in the diagonal.
    node_properties : dataframe
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    max_simplices : bool
        If False counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.
    simplex_type: string
        Type of simplex to consider (See Notes):

        ’directed’ - directed simplices

        ’undirected’ - simplices in the underlying undirected graph

        ’reciprocal’ - simplices in the undirected graph of reciprocal connections

    Returns
    -------
    series
        simplex counts

    Raises
    ------
    AssertionError
        If adj has non-zero entries in the diagonal which can produce errors.
    AssertionError
        If adj is not square.

    Notes
    -----
    A directed simplex of dimension k in adj is a set of (k+1) nodes which are all to all connected in a feedforward manner.
    That is, they can be ordered from 0 to k such that there is an edge from i to j whenever i < j.

    An undirected simplex of dimension k in adj is a set of (k+1) nodes in adj which are all to all connected.  That is, they
    are all to all connected in the underlying undirected graph of adj.  In the literature this is also called a (k+1)-clique
    of the underlying undirected graph.

    A reciprocal simplex of dimension k in adj is a set of (k+1) nodes in adj which are all to all reciprocally connected.
    That is, they are all to all connected in the undirected graph of reciprocal connections of adj.  In the literature this is
    also called a (k+1)-clique of the undirected graph of reciprocal connections.
    """
    adj=sp.csr_matrix(adj)
    assert np.count_nonzero(adj.diagonal()) == 0, 'The diagonal of the matrix is non-zero and this may lead to errors!'
    N, M = adj.shape
    assert N == M, 'Dimension mismatch. The matrix must be square.'


    #Symmetrize matrix if simplex_type is not 'directed'
    if simplex_type=='undirected':
        adj=sp.triu(underlying_undirected_matrix(adj)) #symmtrize and keep upper triangular only
    elif simplex_type=="reciprocal":
        adj=sp.triu(rc_submatrix(adj)) #symmtrize and keep upper triangular only

    flagser_counts = _flagser_counts(adj, threads=threads, max_simplices=max_simplices, max_dim=max_dim)
    if max_simplices:
        return flagser_counts["max_simplex_counts"]
    else:
        return flagser_counts["simplex_counts"]

def normalized_simplex_counts(adj, node_properties=None,
                   max_simplices=False, threads=1,max_dim=-1,
                   **kwargs):
    """Compute the ratio of directed/undirected simplex counts normalized to be between 0 and 1.
    See simplex_counts and undirected_simplex_counts for details.
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j
        of weight adj[i,j].  The matrix can be asymmetric, but must have 0 in the diagonal.
    node_properties : dataframe
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    max_simplices : bool
        If False counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.

    Returns
    -------
    panda series
        Normalized simplex counts

    Raises
    ------
    AssertionError
        If adj has non-zero entries in the diagonal which can produce errors.

    Notes
    -----
    Maybe we should say why we choose this metric"""

    from scipy.special import factorial
    denominator=simplex_counts(adj, node_properties=node_properties,max_simplices=max_simplices,
                                          threads=threads,max_dim=max_dim,simplex_type='undirected', **kwargs).to_numpy()
    #Global maximum dimension since every directed simplex has an underlying undirected one of the same dimension
    max_dim_global=denominator.size
    #Maximum number of possible directed simplices for each undirected simplex across dimensions
    max_possible_directed=np.array([factorial(i+1) for i in np.arange(max_dim_global)])
    denominator=np.multiply(denominator, max_possible_directed)
    numerator=simplex_counts(adj, node_properties=node_properties,max_simplices=max_simplices,
                             threads=threads,max_dim=max_dim,simple_type='directed', **kwargs).to_numpy()
    numerator=np.pad(numerator, (0, max_dim_global-len(numerator)), 'constant', constant_values=0)
    return _series_by_dim(np.divide(numerator,denominator)[1:],name="normalized_simplex_counts",
                          index=np.arange(1,max_dim_global), name_index="dim")


def node_participation(adj, node_properties=None, max_simplices=False,
                       threads=1,max_dim=-1,simplex_type='directed',**kwargs):
    """Compute the number of simplex motifs in the network adj each node is part of.
    See simplex_counts for details.
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    node_properties : dataframe
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    max_simplices : bool
        If False (default) counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.
    simplex_type : string
        Type of simplex to consider:

        ’directed’ - directed simplices

        ’undirected’ - simplices in the underlying undirected graph

        ’reciprocal’ - simplices in the undirected graph of reciprocal connections

    Returns
    -------
    data frame
        Indexed by the nodes in adj and with columns de dimension for which node participation is counted

    Raises
    -------
    AssertionError
        If adj has non-zero entries in the diagonal which can produce errors.
    AssertionError
        If adj is not square.
    """

    adj=sp.csr_matrix(adj).astype('bool')
    assert np.count_nonzero(adj.diagonal()) == 0, 'The diagonal of the matrix is non-zero and this may lead to errors!'
    N, M = adj.shape
    assert N == M, 'Dimension mismatch. The matrix must be square.'


    #Symmetrize matrix if simplex_type is not 'directed'
    if simplex_type=='undirected':
        adj=sp.triu(underlying_undirected_matrix(adj)) #symmtrize and keep upper triangular only
    elif simplex_type=="reciprocal":
        adj=sp.triu(rc_submatrix(adj)) #symmtrize and keep upper triangular only

    flagser_counts = _flagser_counts(adj, count_node_participation=True, threads=threads,
                                     max_simplices=max_simplices, max_dim=max_dim)
    return flagser_counts["node_participation"]

def edge_participation(adj, node_properties=None, max_simplices=False,
                       threads=1,max_dim=-1,simplex_type='directed', return_simplex_counts=False, verbose=False, **kwargs):
    """Compute the number of simplex motifs in the network adj each edge is part of.
    See simplex_counts for details.
    Parameters
    ----------
    adj : 2d array or sparse matrix
        Adjacency matrix of the directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    node_properties : dataframe
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    max_simplices : bool
        If False (default) counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.
    simplex_type : string
        Type of simplex to consider:

        ’directed’ - directed simplices

        ’undirected’ - simplices in the underlying undirected graph

        ’reciprocal’ - simplices in the undirected graph of reciprocal connections

    Returns
    -------
    data frame
        Indexed by the edges in adj and with columns de dimension for which node participation is counted

    Raises
    -------
    AssertionError
        If adj has non-zero entries in the diagonal which can produce errors.
    AssertionError
        If adj is not square.
    """

    adj=sp.csr_matrix(adj).astype('bool')
    assert np.count_nonzero(adj.diagonal()) == 0, 'The diagonal of the matrix is non-zero and this may lead to errors!'
    N, M = adj.shape
    assert N == M, 'Dimension mismatch. The matrix must be square.'


    #Symmetrize matrix if simplex_type is not 'directed'
    if simplex_type=='undirected':
        adj=sp.triu(underlying_undirected_matrix(adj)) #symmtrize and keep upper triangular only
    elif simplex_type=="reciprocal":
        adj=sp.triu(rc_submatrix(adj)) #symmtrize and keep upper triangular only

    flagser_out = pyflagsercount.flagser_count(adj, edge_containment=True, threads=threads,
                                     max_simplices=max_simplices, max_dim=max_dim)
    logging.info("Done running flagser")
    e_contain = pd.DataFrame.from_dict(flagser_out['edge_contain_counts'], orient="index").fillna(0).astype(int)
    if return_simplex_counts:
        return e_contain, flagser_out["cell_counts"]
    else:
        return e_contain

def list_simplices_by_dimension(adj, node_properties=None, max_simplices=False,max_dim=-1,nodes=None,
                                verbose=False, simplex_type='directed', **kwargs):
    """List all simplex motifs in the network adj.
    Parameters
    ----------
    adj : 2d (N,N)-array or sparse matrix
        Adjacency matrix of a directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    node_properties :  data frame
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    max_simplices : bool
        If False counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.
    simplex_type : string
        Type of simplex to consider:

        ’directed’ - directed simplices

        ’undirected’ - simplices in the underlying undirected graph

        ’reciprocal’ - simplices in the undirected graph of reciprocal connections
    nodes : 1d array or None(default)
        Restrict to list only the simplices whose source node is in nodes.  If None list all simplices

    Returns
    -------
    series
        Simplex lists indexed per dimension.  The dimension k entry is a (no. of k-simplices, k+1)-array
        is given, where each row denotes a simplex.

    Raises
    ------
    AssertionError
        If adj has non-zero entries in the diagonal which can produce errors.
    AssertionError
        If adj is not square.
    AssertionError
        If nodes is not a subarray of np.arange(N)

    See Also
    --------
    simplex_counts : A function that counts the simplices instead of listing them and has descriptions of the
    simplex types.
    """
    LOG.info("COMPUTE list of %ssimplices by dimension", "max-" if max_simplices else "")

    import pyflagsercount

    adj=sp.csr_matrix(adj)
    assert np.count_nonzero(adj.diagonal()) == 0, 'The diagonal of the matrix is non-zero and this may lead to errors!'
    N, M = adj.shape
    assert N == M, 'Dimension mismatch. The matrix must be square.'
    if not nodes is None:
        assert np.isin(nodes,np.arange(N)).all(), "nodes must be a subarray of the nodes of the matrix"

    #Symmetrize matrix if simplex_type is not 'directed'
    if simplex_type=='undirected':
        adj=sp.triu(underlying_undirected_matrix(adj)) #symmtrize and keep upper triangular only
    elif simplex_type=="reciprocal":
        adj=sp.triu(rc_submatrix(adj)) #symmtrize and keep upper triangular only

    n_threads = kwargs.get("threads", kwargs.get("n_threads", 1))


    # Only the simplices that have sources stored in this temporary file will be considered
    if not nodes is None:
        import tempfile
        import os
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        vertices_todo = tmp_file.name + ".npy"
        np.save(vertices_todo, nodes, allow_pickle=False)
    else:
        vertices_todo=''

    #Generate simplex_list
    original=pyflagsercount.flagser_count(adj, max_simplices=max_simplices,threads=n_threads,max_dim=max_dim,
                                      vertices_todo=vertices_todo, return_simplices=True)['simplices']

    #Remove temporary file
    if not nodes is None:
        os.remove(vertices_todo)

    #Format output
    max_dim = len(original)
    dims = pd.Index(np.arange(max_dim), name="dim")
    simplices = pd.Series(original, name="simplices", index=dims).apply(np.array)
    #When counting all simplices flagser doesn't list dim 0 and 1 because they correspond to vertices and edges
    if not max_simplices:
        if nodes is None:
            nodes=np.arange(0, N)
        coom = adj.tocoo()
        simplices[0] = np.reshape(nodes, (nodes.size, 1))
        mask=np.isin(coom.row,nodes)
        simplices[1] = np.stack([coom.row[mask], coom.col[mask]]).T
    return simplices

def in_degree_from_pop(adj, source_pop, max_simplices=False,threads=1, max_dim=-1, ** kwargs):
    # TODO: DO THE OUTDEGREE VERSION
    # TODO: Get participation directly from flagsercount via vertices to do?
    """Compute generalized in-degree of nodes source_pop onto the rest of the nodes in adj.
    Parameters
    ----------
    adj: 2d (N,N)-array or sparse matrix
        Adjacency matrix of a directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    source_pop: list of indices of the source population, must be a subset of ``np.arange(0, adj.shape[0])``
    max_simplices : bool
        If False counts all simplices.
        If True counts only maximal simplices.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.
        Particularly useful for large or dense graphs.

    Returns
    -------
    Data frame
        Table of k-in-degrees from source_pop indexed by the target population.

    Raises
    ------
    AssertionError
        If adj restricted to source_pop has non-zero entries in the diagonal which can produce errors.
    """
    adj=adj.tocsr()
    source_pop = np.sort(source_pop)
    target_pop = np.setdiff1d(np.arange(adj.shape[0]), source_pop)
    adj_source = adj[np.ix_(source_pop, source_pop)]
    adj_cross = adj[np.ix_(source_pop, target_pop)]
    degs=cross_col_k_in_degree(adj_cross, adj_source,
                                 max_simplices=max_simplices,threads=threads, max_dim=max_dim, **kwargs)
    degs.index=target_pop
    return degs

def cross_col_k_in_degree(adj_cross, adj_source, max_simplices=False,
                          threads=1,max_dim=-1,**kwargs):
    #TODO: DO THE OUTDEGREE VERSION
    #TODO: Get participation directly from flagsercount via vertices to do?
    """Compute generalized in-degree of nodes in adj_target from nodes in adj_source.
    The k-in-degree of a node v is the number of k-simplices in adj_source with all its nodes mapping to v
    through edges in adj_cross.
    Parameters
    ----------
    adj_cross : (n,m) array or sparse matrix
        Matrix of connections from the nodes in adj_n to the target population.
        n is the number of nodes in adj_source and m is the number of nodes in adj_target.
        A non-zero entry adj_cross[i,j] implies there is an edge from i-th node of adj_source
        to the j-th node of adj_target.
    adj_source : (n, n)-array or sparse matrix
        Adjacency matrix of the source network where n is the number of nodes in the source network.
        A non-zero entry adj_source[i,j] implies there is an edge from node i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    max_simplices : bool
        If False counts all simplices.
        If True counts only maximal simplices.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.
        Particularly useful for large or dense graphs.

    Returns
    -------
    Data frame
        Table of cross-k-in-degrees indexed by the m nodes in the target population.

    Raises
    ------
    AssertionError
        If adj_source has non-zero entries in the diagonal which can produce errors.
    """
    adj_source=sp.csr_matrix(adj_source).astype('bool')
    adj_cross=sp.csr_matrix(adj_cross).astype('bool')
    assert np.count_nonzero(adj_source.diagonal()) == 0, \
    'The diagonal of the source matrix is non-zero and this may lead to errors!'
    assert adj_source.shape[0] == adj_source.shape[1], \
    'Dimension mismatch. The source matrix must be square.'
    assert adj_source.shape[0] == adj_cross.shape[0], \
    'Dimension mismatch. The source matrix and cross matrix must have the same number of rows.'

    n_source = adj_source.shape[0] #Size of the source population
    n_target = adj_cross.shape[1] #Size of the target population
    # Building a square matrix [[adj_source, adj_cross], [0,0]]
    adj=sp.bmat([[adj_source, adj_cross],
                 [sp.csr_matrix((n_target, n_source), dtype='bool'),
                  sp.csr_matrix((n_target, n_target), dtype='bool')]])
    # Transposing to restrict computation to ``source nodes'' in adj_target in flagsercount
    adj=adj.T
    nodes=np.arange(n_source, n_source+n_target) #nodes on target population
    slist=list_simplices_by_dimension(adj, max_simplices=max_simplices, max_dim=max_dim,nodes=nodes,
                                      simplex_type='directed',verbose=False,threads=threads,**kwargs)

    #Count participation as a source in transposed matrix i.e. participation as sink in the original
    cross_col_deg=pd.DataFrame(columns=slist.index[1:], index=nodes)
    for dim in slist.index[1:]:
        index,deg=np.unique(slist[dim][:,0],return_counts=True)
        cross_col_deg[dim].loc[index]=deg
    cross_col_deg=cross_col_deg.fillna(0)
    return cross_col_deg


def betti_counts(adj, node_properties=None,
                 min_dim=0, max_dim=[], simplex_type='directed', approximation=None,
                 **kwargs):
    """Count betti counts of flag complex of adj.  Type of flag complex is given by simplex_type.

    Parameters
    ----------
    adj : 2d (N,N)-array or sparse matrix
        Adjacency matrix of a directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.  Matrix will be cast to 0,1 entries so weights
        will be ignored.
    node_properties :  data frame
        Data frame of neuron properties in adj.  Only necessary if used in conjunction with TAP or connectome utilities.
    min_dim : int
        Minimal dimension from which betti counts are computed.
        The default min_dim = 0 (counting number of connected components).
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = [] counts betti numbers up to the maximal dimension of the complex.
    simplex_type : string
        Type of flag complex to consider, given by the type of simplices it is built on.
        Possible types are:

        ’directed’ - directed simplices (directed flag complex)

        ’undirected’ - simplices in the underlying undirected graph (clique complex of the underlying undirected graph)

        ’reciprocal’ - simplices in the undirected graph of reciprocal connections (clique complex of the
        undirected graph of reciprocal connections.)
    approximation : list of integers  or None
        Approximation parameter for the computation of the betti numbers.  Useful for large networks.
        If None all betti numbers are computed exactly.
        Otherwise, min_dim must be 0 and approximation but be a list of positive integers or -1.
        The list approximation is either extended by -1 entries on the right or sliced from [0:max_dim+1] to obtain
        a list of length max_dim.  Each entry of the list denotes the approximation value for the betti computation
        of that dimension if -1 approximation in that dimension is set to None.

        If the approximation value at a given dimension is `a` flagser skips cells creating columns in the reduction
        matrix with more than `a` entries.  This is useful for hard problems.  For large, sparse networks a good value
        if often `100,00`.  If set to `1` that dimension will be virtually ignored.  See [1]_

    Returns
    -------
    series
        Betti counts indexed per dimension from min_dim to max_dim.

    Raises
    ------
    AssertionError
        If adj has non-zero entries in the diagonal which can produce errors.
    AssertionError
        If adj is not square.
    AssertionError
        If approximation != None and min_dim != 0.

    See Also
    --------
    [simplex_counts](network.md#src.connalysis.network.topology.simplex_counts) :
    A function that counts the simplices forming the complex from which bettis are count.
    Simplex types are described there in detail.

    References
    ----------
    For details about the approximation algorithm see

    ..[1] D. Luetgehetmann, "Documentation of the C++ flagser library";
           [GitHub: luetge/flagser](https://github.com/luetge/flagser/blob/master/docs/documentation_flagser.pdf).

    """
    LOG.info("Compute betti counts for %s-type adjacency matrix and %s-type node properties",
             type(adj), type(node_properties))

    from pyflagser import flagser_unweighted

    #Checking matrix
    adj = sp.csr_matrix(adj).astype(bool).astype('int')
    assert np.count_nonzero(adj.diagonal()) == 0, 'The diagonal of the matrix is non-zero and this may lead to errors!'
    N, M = adj.shape
    assert N == M, 'Dimension mismatch. The matrix must be square.'
    assert not((not approximation is None) and (min_dim!=0)), \
        'For approximation != None, min_dim must be set to 0.  \nLower dimensions can be ignored by setting approximation to 1 on those dimensions'

    # Symmetrize matrix if simplex_type is not 'directed'
    if simplex_type == 'undirected':
        adj = sp.triu(underlying_undirected_matrix(adj))  # symmtrize and keep upper triangular only
    elif simplex_type == "reciprocal":
        adj = sp.triu(rc_submatrix(adj))  # symmtrize and keep upper triangular only
    #Computing bettis
    if max_dim==[]:
        max_dim=np.inf

    if approximation==None:
        LOG.info("Run without approximation")
        bettis = flagser_unweighted(adj, min_dimension=min_dim, max_dimension=max_dim,
                                    directed=True, coeff=2,
                                    approximation=None)['betti']
    else:
        assert (all([isinstance(item,int) for item in approximation])) # assert it's a list of integers
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
                     index=pd.Index(np.arange(min_dim, len(bettis)+min_dim), name="dim"))


def _binary2simplex(address, test=None, verbosity=1000000):
    """...Not used --- keeping it here as it is of interest to understanmd
    how simplices are represented on the disc by Flagser.
    #INPUT: Address of binary file storing simplices
    #OUTPUT: A list if lists L where L[i] contains the vertex ids of the i'th simplex,
    #          note the simplices appear in no particular order

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


def _generate_abstract_edges_in_simplices(dim, position="all"):
    import itertools
    """Generate indices of edges in a simplex with nodes 0, 1, ... dim

    Parameters
    ----------
    dim : int
        Dimension of the simplex
    position: str
        Position of the edges to extract

        'all': all edges of the simplex

        'spine': edges along the spine of the simplex

    Returns
    -------
    list
        list of pairs of nodes indexing the edges selected
    """
    if position == "all":
        edges_abstract = np.array(list(itertools.combinations(range(dim + 1), 2)))
    elif position == "spine":
        edges_abstract = np.array([[i, i + 1] for i in range(dim)])
    return edges_abstract


#def extract_submatrix_of_simplices(simplex_list, N, position="all"):
def extract_submatrix_from_simplex_list(simplex_list, N, position="spine"):
    """Generate binary submatrix of NxN matrix of edges in simplex list.

    Parameters
    ----------
    simplex list: 2d-array
        Array of dimension (no. of simplices, dimension).
        Each row corresponds to a list of nodes on a simplex
        indexed by the order of the nodes in an NxN matrix.
    N: int
        Number of nodes in original graph defining the NxN matrix.
    position: str
        Position of the edges to extract

        'all': all edges of the simplex (it is more efficient to do this with ``extract_submatrix_from_edge_par``)

        'spine': edges along the spine of the simplex
        (only makes sense for directed simplices)

    Returns
    -------
    coo bool matrix
        Matrix with of shape (N,N) with entries `True` corresponding to edges in simplices.
    """
    if position=="all":
        logging.warning("For edges in any position it is more efficient to use extract_submatrices_from_edge_participation")
    if simplex_list.shape[0] == 0:
        return sp.csr_matrix((N, N), dtype=bool)  # no simplices in this dimension
    else:
        dim = simplex_list.shape[1] - 1
        edges_abstract = _generate_abstract_edges_in_simplices(dim,
                                                               position=position)  # abstract list of edges to extract from each simplex
        edges = np.unique(np.concatenate([simplex_list[:, edge] for edge in edges_abstract]), axis=0)
        return (sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(N, N))).astype(bool)


def extract_submatrices_from_edge_participation(edge_par, N, dims=None, thresh=1):
    """Generate binary submatrix of an NxN matrix of edges with edge participation greater than thresh.

    Parameters
    ----------
    edge_par: DataFrame
        DataFrame with edge participation values formatted as the output of ``edge_participation`` i.e.,
        indexed by the edges in adj and with columns de dimension for which edge participation is counted
    dims: list of ints
        dimensions of the simplices to consider, if ``None`` all positive dimensions are considered
    N: int
        Number of nodes in original graph defining the NxN matrix.
        The indices of edge_par must be smaller than N
    thresh: int
        Threshold value at which to consider an edge.  If thresh=1 all edges that
        elong to at least one simplex are considered.

    Returns
    -------
    dict
        keys: dimensions
        values: sparse binary matrices in coo format of shape (N,N) with entries `True` corresponding to edges
        that belong to at least thresh simplices of the dimension in its corresponding key.
    """

    if dims is None: dims = edge_par.columns
    dims = np.array(dims)
    assert np.isin(dims,
                   edge_par.columns).all(), "The array dims shoulds be a subset of the columns of edge participation"
    mats = {}
    # Reformat edge_participation (Maybe already do this in the output of edge_participation?)
    df = edge_par.set_index(keys=pd.MultiIndex.from_tuples(edge_par.index)).reset_index(names=["row", "col"])
    for dim in dims:
        edges = df[df[dim] >= thresh]
        mats[f'dimension_{dim}'] = sp.coo_matrix((np.ones(len(edges)), (edges.row, edges.col)), shape=(N, N)).astype(
            bool)
    return mats

def get_k_skeleta_graph(adj=None, max_simplices=False, dimensions=None, simplex_type='directed',
                        simplex_list=None, N=None, position="all",
                        **kwargs):
    # TODO: Deprecate or update to use edge_participation
    # Choose only some dimensions???
    # check max dim is consistent with simplex_list only used if adj is given and must be >0
    # adj only used is simplex list is none
    # Add requirement to give adj is direction is undirected and multiply adj by mat!!!
    """Return the edges of the (maximal) k-skeleton of the flag complex of adj for all k<= max_dim in the position determined
    by position.
    If simplex list are provided, it will compute the edges directly from these and not use adj,
    in which case N (the number of rows and columns of adj) is required.
    If simplex lists are not provided they will be calculated with for the flag complex whose type is determined by
    simplex_type as for simplex_counts.

    Parameters
    ----------
    adj : (N,N)-array or sparse matrix
        Adjacency matrix of a directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    max_simplices : bool
        If False counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    dimensions : list of ints
        Dimensions `k` for which the `k`-skeleta is computed, if None all dimensions are computed.
    simplex_type : string
        Type of simplex to consider if computed from adj:

        ’directed’ - directed simplices

        ’undirected’ - simplices in the underlying undirected graph

        ’reciprocal’ - simplices in the undirected graph of reciprocal connections
    simplex list: series
        Series 2d-arrays indexed by dimension.
        Each array is of dimension (no. of simplices, dimension).
        Each row corresponds to a list of nodes on a simplex.
        If provided adj will be ignored but N will be required.
    N: int
        Number of nodes in original graph.
    position: str
        Position of the edges to extract

        'all': all edges of the simplex

        'spine': edges along the spine of the simplex
        (only makes sense if simplices are directed)

    Returns
    -------
    dict
        Dictionary with keys dimensions and values boolean (N,N) matrices with entries `True`
        corresponding to edges in (maximal) simplices of that dimension.

    Raises
    ------
    AssertionError
        If neither adj nor simplex_list are provided
    AssertionError
        If N <= than an entry in the simplex list
    AssertionError
        If a dimension is required that is not an index in the simplex list

    Notes
    ------
    In order to list k-simplices and thus the k-skeleton, flagsercount needs to list all lower
    dimensional simplices anyhow.

    """

    assert not (adj is None and simplex_list is None), "Either adj or simplex_list need to be provided"

    if dimensions == None:
        max_dim = -1
    else:
        max_dim = np.max(np.array(dimensions))

    if (simplex_list is None):  # Compute simplex lists if not provided
        simplex_list = list_simplices_by_dimension(adj, node_properties=None,
                                                   max_simplices=max_simplices, max_dim=max_dim,
                                                   simplex_type=simplex_type,
                                                   nodes=None, verbose=False, **kwargs)
        N = adj.shape[0]
    else:
        assert isinstance(N, int), 'If simplex list are provide N must be provided'
        assert N > np.nanmax(simplex_list.explode().explode()), \
            "N must be larger than all the entries in the simplex list"
        assert (dimensions == None) or np.isin(dimensions, simplex_list.index).all(), \
            f'Some requested dimensions={dimensions} are not in the simplex lists index={simplex_list.index.to_numpy()}'
    # Extract 'k'-skeleton
    dims = simplex_list.index[simplex_list.index != 0]  # Doesn't make sense to look at the 0-skeleton
    if dimensions != None:
        dims = dims[np.isin(dims, dimensions)]
    skeleton_mats = {f'dimension_{dim}': None for dim in dims}
    for dim in dims:
        mat = extract_submatrix_of_simplices(simplex_list[dim], N, position=position)
        if simplex_type in ('undirected', 'reciprocal'):
            mat = (mat + mat.T).astype(bool)
        skeleton_mats[f'dimension_{dim}'] = mat
    return skeleton_mats



def count_rc_edges_skeleta(adj=None, max_dim=-1, max_simplices=False,
                            N=None, simplex_list=None, edge_par=None,
                           simplex_type='directed',position="all",
                           return_mats=False,threads=8, **kwargs):
    """Count the edges and reciprocal edges in the k-``skeleta`` of the directed flag complex of adj for all
    k<= max_dim. If simplex list are provided, it will compute the skeleta directly from these and not use adj.

    Parameters
    ----------
    adj : (N,N)-array or sparse matrix
        Adjacency matrix of a directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    max_simplices : bool
        If False counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.
    simplex list: series
        Series 2d-arrays indexed by dimension.
        Each array is of dimension (no. of simplices, dimension).
        Each row corresponds to a list of nodes on a simplex.
        If provided adj will be ignored but N will be required.
    edge_par: DataFrame
        DataFrame with edge participation values formatted as the output of ``edge_participation`` i.e.,
        indexed by the edges in adj and with columns de dimension for which edge participation is counted
    N: int
        Number of nodes in original graph.
    simple_type: str
        See [simplex_counts](network.md#src.connalysis.network.topology.simplex_counts)
    position: str
        Position of the edges to extract

        'all': all edges of the simplex

        'spine': edges along the spine of the simplex
        (only makes sense if simplices are directed)
    return_mats : bool
        If True return the matrices of the underlying graphs of the k-skeleta as in
        get_k_skeleta_graph
    threads: int
        Number of threads into which to parallelize the computation

    Returns
    -------
    data frame, (dict)
        data frame with index dimensions and columns number of (rc) edges in the corresponding skeleta
        if return_mats==True, also return the graphs of the k-skeleta as in get_k_skeleta_graph.

    Raises
    ------
    AssertionError
        If neither adj nor precomputed simplex_list or edge_par values are provided
    AssertionError
        If N <= than an entry in the simplex list

    See Also
    --------
    [edge_participation](network.md#src.connalysis.network.topology.edge_participation) :
    A function that counts the number of times an edge is part of a simplex.

    [simplex_counts](network.md#src.connalysis.network.topology.simplex_counts) :
    A function that counts the simplices forming the complex from which bettis are count,
    where simplex types are described in detail.
    """

    # Check if enough inputs are provided
    if (position == 'spine') and (simplex_list is not None):
        not_precomputed = False
    elif (position == 'all') and (edge_par is not None):
        not_precomputed = False
    else:
        not_precomputed = True

    assert not (adj is None and not_precomputed), "Either adj or simplex_list/edge_participation need to be provided"

    if not_precomputed:
        N = adj.shape[0]
        if position == "spine":  # Compute simplex since they are not provided
            simplex_list = list_simplices_by_dimension(adj, node_properties=None,
                                                       max_simplices=max_simplices, max_dim=max_dim,
                                                       simplex_type='directed',
                                                       nodes=None, verbose=False, **kwargs)
        elif position == "all":  # More efficient than going from simplex lists if the position is not important
            edge_par, simplex_counts = edge_participation(adj, node_properties=None, max_simplices=max_simplices,
                                                          threads=threads, max_dim=max_dim, simplex_type=simplex_type,
                                                          return_simplex_counts=True)

    else:
        if position == "spine":
            assert N > np.nanmax(simplex_list.explode().explode()), \
                "N must be larger than all the entries in the simplex list"

    # Extract 'k'-skeleton and count (rc-)edges
    if position == "spine":
        dims = simplex_list.index[simplex_list.index != 0]  # Doesn't make sense to look at the 0-skeleton
    elif position == "all":
        dims = pd.Index(edge_par.drop(0, axis=1, errors="ignore").columns, name="dim")
    print(dims)
    edge_counts = pd.DataFrame(index=dims, columns=["number_of_edges", "number_of_rc_edges", "rc/edges_percent"])
    if return_mats == True:
        skeleton_mats = {f'dimension_{dim}': None for dim in dims}
    print(dims)  # Delete me?

    if position == "spine":
        for dim in dims:
            if simplex_list[dim].size > 0:
                assert N > np.max(simplex_list[dim]), \
                    "N must be larger than all the entries in the simplex list"
                mat = extract_submatrix_from_simplex_list(simplex_list[dim], N, position=position)
                edges = mat.sum();
                rc_edges = rc_submatrix(mat).sum()
                edge_counts["number_of_edges"].loc[dim] = edges
                edge_counts["number_of_rc_edges"].loc[dim] = rc_edges
                edge_counts["rc/edges_percent"].loc[dim] = (rc_edges * 100) / edges
            else:
                edge_counts["number_of_edges"].loc[dim] = 0
            if return_mats == True:
                skeleton_mats[f'dimension_{dim}'] = mat
    elif position == "all":
        skeleton_mats = extract_submatrices_from_edge_participation(edge_par, N, dims=dims, thresh=1)
        for dim in dims:
            edges = skeleton_mats[f'dimension_{dim}'].sum()
            edge_counts["number_of_edges"].loc[dim] = edges
            if edges > 0:
                rc_edges = rc_submatrix(skeleton_mats[f'dimension_{dim}']).sum()
                edge_counts["number_of_rc_edges"].loc[dim] = rc_edges
                edge_counts["rc/edges_percent"].loc[dim] = (rc_edges * 100) / edges
            else:
                edge_counts["number_of_rc_edges"].loc[dim] = 0
                edge_counts["rc/edges_percent"].loc[dim] = 0

    if return_mats == True:
        return edge_counts, skeleton_mats
    else:
        return edge_counts


def bedge_counts(adjacency, simplices=None,
                 max_simplices = False, max_dim = -1, simplex_type = 'directed', ** kwargs):
    """Counts the total number of edges in each position on the subgraphs defined by the nodes
    of the simplices listed in simplices.  If a simplex list is not passed, simplices are computed on
    the adjacency matrix.

    Parameters
    ----------
    adjacency : (N,N)-array or sparse matrix
        Adjacency matrix of a directed network.  A non-zero entry adj[i,j] implies there is an edge from i to j.
        The matrix can be asymmetric, but must have 0 in the diagonal.
    simplices : series
        Series  of 2d-arrays indexed by dimension.
        Each array is of dimension (no. of simplices, dimension).
        Each row corresponds to a list of nodes on a simplex.
    max_simplices : bool
        If False counts all simplices in adj.
        If True counts only maximal simplices i.e., simplex motifs that are not contained in higher dimensional ones.
    max_dim : int
        Maximal dimension up to which simplex motifs are counted.
        The default max_dim = -1 counts all existing dimensions.  Particularly useful for large or dense graphs.
    simplex_type: str
        See [simplex_counts](network.md#src.connalysis.network.topology.simplex_counts)

    Returns
    -------
    series
        pandas series with index dimensions values (dim+1, dim+1) arrays.  The (i,j) entry counts the number of edges
        from node i to node j on all the subgraphs of adjacency on the nodes of the simplices listed.  See notes.

    Notes
    -------
    Every directed $k$-simplex $[v_o, v_1, \\ldots, v_k]$ defines as subgraph of the adjacency matrix, with edges
    $v_i \\to v_j$ whenever $i\leq j$, but also possibly with ''reverse'' edges.  One can represent this structure
    with a non-symmetric $(k+1, k+1)$-matrix with `1`'s for every edge in the subgraph.  The output of this function
    gives for each dimension the sum of all these matrices over all the simplices provided in `simplices` or over
    all the simplices in the adjacency matrix if none is provided.  The lower triangular part of these matrices is
    therefore a metric of recurrence within simplices, or "higher dimensional recurrence".
    In particular, in dimension 1 it is the number of reciprocal edges in the network.
    """

    adj = adjacency

    if simplices is None:
        LOG.info("COMPUTE `bedge_counts(...)`: No argued simplices.")
        return bedge_counts(adj,
                            list_simplices_by_dimension(adj, max_simplices = max_simplices,
                                                        max_dim = max_dim, simplex_type = simplex_type, ** kwargs))
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


#TRIAD ANALYSIS
label_edges = np.arange(3 * 3).reshape(3, 3)  # Indexing edges on a 3x3 matrix going from left to right and up to down
def _get_triad_id(edges):
    """Given a the edges on a graph on nodes {0,1,2}
    return a list of edges indexed from 0 to 8 as in label_edges.

    Parameters
    ----------
    edges : tuple of pairs
        Each pair is of the form $(i,j)$ where $i, j \\in \\{0,1,2}$.

    Returns
    -------
    list of integers between 0 and 8 indexing the edges as in label_edges
    """
    row, col = tuple(zip(*edges))
    return tuple(np.sort(label_edges[row, col]))

# We list all connected triads by hand as sort them as in Gal et al., 2017
connected_triads = {
    # On two edges
    0: ((0, 2), (1, 0)),
    1: ((0, 2), (1, 2)),
    2: ((0, 1), (0, 2)),
    # On three edges
    3: ((0, 1), (1, 2), (0, 2)),
    4: ((0, 1), (0, 2), (1, 0)),
    5: ((0, 1), (1, 0), (2, 0)),
    6: ((0, 1), (1, 2), (2, 0)),
    # On four edges
    7: ((0, 1), (0, 2), (1, 0), (2, 0)),
    8: ((0, 1), (0, 2), (1, 0), (2, 1)),
    9: ((0, 1), (0, 2), (1, 0), (1, 2)),
    10: ((0, 1), (0, 2), (1, 2), (2, 1)),
    # On 5 edge
    11: ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0)),
    # On 6 edges
    12: ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
}

# Exhaustive dictionary of triads (digraphs on 3 nodes) represented by their list of edges as indexed in
# label_edges.  The 3-cycle needs to be entered twice because all edges have the same in-out degree but there
# are two graphs in the isomorphism class.
triad_dict = {_get_triad_id(connected_triads[i]): i for i in range(13)}
triad_dict[(2, 3, 7)] = 6
# Size of isomorphism class of each triad type i.e., number of permutation of the vertices giving the same graph
triad_combinations = np.array([6, 3, 3,  # 2 edges
                               6, 6, 6, 2,  # 3 edges
                               3, 6, 3, 3,  # 4 edges
                               6,  # 5 edges
                               1])  # 3-clique
def count_triads_fully_connected(adj, max_num_sampled=5000000, return_normalized=False):
    """Counts the numbers of each triadic motif in the matrix adj.

    Parameters
    ----------
    adj : 2d-array
        Adjacency matrix of a directed network.
    max_num_sampled : int
        The maximal number of connected triads classified. If the number of
        connected triads is higher than that, only the specified number is sampled at random and
        classified. The final counts are extrapolated as (actual_num_triads/ max_num_sampled) * counts.
    return_normalized : bool
        If True return the triad counts divided by the size of each isomorphism class.  That is, the total counts
        divided by the following array:

        $[6, 3, 3, 6, 6, 6, 2, 3, 6, 3, 3, 6, 1].$

    Returns
    -------
    1d-array
        The counts of the various triadic motifs in adj as ordered in Figure 5 [1]_.

    Notes
    ------
    Only connectected motifs are counted, i.e. motifs with less than 2 connections or only a single bidirectional
    connection are not counted. The connected motifs are ordered as in Figure 5 [1]_.

    References
    -------

    ..[1] Gal, Eyal, et al.
    ["Rich cell-type-specific network topology in neocortical microcircuitry."](https://www.nature.com/articles/nn.4576)
    Nature neuroscience 20.7 (2017): 1004-1013.

    """

    # Functions to indetify triads
    def canonical_sort(M):
        """Sorts row/columns of the matrix adj using the lexicographical order of the
        tuple (out_degree, in_degree).

        Parameters
        ----------
        M : 2d-array
            Adjacency matrix of a directed network.

        Returns
        -------
        2d-array
            the matrix adj with rows/columns sorted
        """
        in_degree = np.sum(M, axis=0)
        out_degree = np.sum(M, axis=1)
        idx = np.argsort(-in_degree - 10 * out_degree)
        return M[:, idx][idx]

    def identify_motif(M):
        """
        Identifies the connected directed digraph on three nodes M as on in the full classification
        list given in the dictionary triad_dict.

        Parameters
        ----------
        M : array
            A (3,3) array describing a directed connected digraph on three nodes.

        Returns
        -------
        The index of the motif as indexed in the dictiroanry triad_dict which follows the
        ordering of Gal et al., 2017
        """
        triad_code = tuple(np.nonzero(canonical_sort(M).flatten())[0])
        return triad_dict[triad_code]

    # Finding and counting triads
    import time
    adj = adj.toarray()  # Casting to array makes finding triads an order of magnitude faster
    t0 = time.time()
    undirected_adj = underlying_undirected_matrix(adj).toarray()
    # Matrix with i,j entries number of undirected paths between i and j in adj
    path_counts = np.triu(undirected_adj @ undirected_adj, 1)
    connected_pairs = np.nonzero(path_counts)
    triads = set()
    print("Testing {0} potential triadic pairs".format(len(connected_pairs[0])))
    for x, y in zip(*connected_pairs):
        # zs = np.nonzero((undirected_adj.getrow(x).multiply(undirected_adj.getrow(y))).toarray()[0])[0]
        zs = np.nonzero(undirected_adj[x] & undirected_adj[y])[0]
        for z in zs:
            triads.add(tuple(sorted([x, y, z])))
    triads = list(triads)
    print("Time spent finding triads: {0}".format(time.time() - t0))
    print("Found {0} connected triads".format(len(triads)))
    t0 = time.time()
    counts = np.zeros(np.max(list(triad_dict.values())) + 1)
    sample_idx = np.random.choice(len(triads),
                                  np.minimum(max_num_sampled, len(triads)),
                                  replace=False)
    for idx in sample_idx:
        triad = triads[idx]
        motif_id = identify_motif(adj[:, triad][triad, :])
        counts[motif_id] += 1
    print("Time spent classifying triads: {0}".format(time.time() - t0))
    if return_normalized:
        return (((len(triads) / len(sample_idx)) * counts).astype(int)) / triad_combinations
    else:
        return (((len(triads) / len(sample_idx)) * counts).astype(int))




def _convex_hull(adj, node_properties):# --> topology
    """Return the convex hull of the sub gids in the 3D space using x,y,z position for gids"""
    pass

def get_all_simplices_from_max(max_simplices):
    """Takes the list of maximal simplices are returns the list of all simplices.

        Parameters
        ----------
        max_simplices : list
            A list of lists of tuples. Where max_simplices[k] is a list of the 0 dimensional maximal simplices,
            where each simplex is a tuple of the vertices of the simplex

        Returns
        -------
        list
            A list of lists of tuples. Of the same format as the inputted list but now contains all simplices.
        """
    simplices = list(max_simplices)
    for k in range(len(max_simplices)-1,0,-1):
        print(max_simplices[k])
        for simplex in simplices[k]:
            for s in range(k,-1,-1):
                x = tuple(simplex[:s]+simplex[s+1:])
                if x not in simplices[k-1]:
                    simplices[k-1].append(x)

    return simplices


def simplicial_rich_club_curve(M, maximal=False, sparse_bin_set=False):
    """Computes the simplicial rich club curve of a network.
           Where the i'th entry is the density of the subnetwork induced by the vertices that are contained in
           more than i (maximal) simplices.

        Parameters
        ----------
        adj : 2d-array
            Adjacency matrix of a directed network.
        max_simplices : bool
            If true then vertex participation is the number of maximal simplices each vertex is contained in.
        sparse_bin_set : bool
            If true then consecutive entries with same rich club coefficient are grouped into bins together,

        Returns
        -------
        pandas.Series
            Where the i'th entry is the rich club coefficient of the network induced by all vertices which are
            contained in more that i (maximal) simplices

    """
    import pyflagsercount
    from .classic import efficient_rich_club_curve
    vertex_par = pd.DataFrame(pyflagsercount.flagser_count(M, max_simplices=maximal, containment=True)['contain_counts']).replace(np.nan,0).astype(int)
    return pd.DataFrame([efficient_rich_club_curve(M, pre_calculated_richness=vertex_par[i]) for i in range(vertex_par.shape[1])]).transpose().dropna(how='all')



#######################################################
################# METHODS FROM TRIDY ##################
#######################################################


def euler_characteristic(matrix):
    """Computes the Euler characteristic of the flag complex of the graph with adjacency matrix matrix

        Parameters
        ----------
        matrix : 2d-array
            Adjacency matrix of a directed network.

        Returns
        -------
        integer
            The Euler characteristic of the flag complex of matrix

    """
    return pyflagsercount.flagser_count(M)['euler']



def tcc(center, matrix):
    """Computes the transitive clustering coefficient of the graph induced by 
            the neighbourhood of center in matrix

        Parameters
        ----------
        center : int
            The index of the vertex to be considered
        matrix : 2d-array
            Adjacency matrix of a directed network.

        Returns
        -------
        float
            The transitive cluster coefficient of the neighbourhood of center

    """
    current_nhbd = neighbourhood(center, matrix)
    return tcc_adjacency(current_nhbd)


def tcc_adjacency(matrix):
    """Computes the transitive clustering coefficient of matrix

        Parameters
        ----------
        matrix : 2d-array
            Adjacency matrix of a directed network.

        Returns
        -------
        float
            The transitive cluster coefficient of matrix

    """
    outdeg = np.count_nonzero(matrix[0])
    indeg = np.count_nonzero(np.transpose(matrix)[0])
    repdeg = reciprocal_connections_adjacency(matrix, chief_only=True)
    totdeg = indeg+outdeg
    chief_containment = node_participation(matrix,max_dim=2).iloc[0]
    numerator = 0 if len(chief_containment) < 3 else chief_containment[2]
    denominator = (totdeg*(totdeg-1)-(indeg*outdeg+repdeg))
    if denominator == 0:
        return 0
    return numerator/denominator


def dc(center, matrix, coeff_index=2):
    """Computes the density coefficient of the graph induced by 
            the neighbourhood of center in matrix

        Parameters
        ----------
        center : int
            The index of the vertex to be considered
        matrix : 2d-array
            Adjacency matrix of a directed network.
        coeff_index : int
             The dimension to be computed, default=2

        Returns
        -------
        float
            The density coefficient of the neighbourhood of center

    """
    current_nhbd = neighbourhood(center, matrix)
    return dc_adjacency(current_nhbd, coeff_index=coeff_index)



def dc_adjacency(matrix, coeff_index=2):
    """Computes the density coefficient of the matrix

        Parameters
        ----------
        matrix : 2d-array
            Adjacency matrix of a directed network.
        coeff_index : int
             The dimension to be computed, default=2

        Returns
        -------
        float
            The density coefficient of the flag complex of the graph with adjacency matrix matrix
    """
    assert coeff_index >= 2, 'Assertion error: Density coefficient must be at least 2'
    flagser_output = node_participation(matrix,max_dim=coeff_index).iloc[0]
    if len(flagser_output) <= coeff_index:
        density_coeff = 0
    elif flagser_output[coeff_index] == 0:
        density_coeff = 0
    else:
        numerator = coeff_index*flagser_output[coeff_index]
        denominator = (coeff_index+1)*(len(matrix)-coeff_index)*flagser_output[coeff_index-1]
        if denominator == 0:
            density_coeff = 0
        else:
            density_coeff = numerator/denominator
    return density_coeff




#I have kept this for now for backwards compatability, in case it is used anywhere in existing code
def normalised_simplex_count(matrix, dim=2):
    """Computes the normalised simplex count of the flag complex of the graph with adjacency matrix matrix

        Parameters
        ----------
        matrix : 2d-array
            Adjacency matrix of a directed network.
        dim : int
             The dimension to be computed, default=2

        Returns
        -------
        float
            The normalised simplex count of dimension dim of the flag complex of matrix

    """
    return normalized_simplex_counts(matrix)[dim]



def nbc(matrix):
    """Computes the normalised Betti coefficient of the graph with adjacency matrix matrix

        Parameters
        ----------
        matrix : 2d-array
            Adjacency matrix of a directed network.

        Returns
        -------
        float
            The normalised Betti coefficient of the graph with adjacency matrix matrix

    """
    flagser_output = pyflagser.flagser_unweighted(matrix, directed=True)
    cells = flagser_output['cell_count']
    bettis = flagser_output['betti']
    while (cells[-1] == 0) and (len(cells) > 1):
        cells = cells[:-1]
    while (bettis[-1] == 0) and (len(bettis) > 1):
        bettis = bettis[:-1]
    normalized_betti_list = [(i+1)*bettis[i]/cells[i] for i in range(min(len(bettis),len(cells)))]
    return sum(normalized_betti_list)

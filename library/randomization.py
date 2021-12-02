# Functions that implement random controls of a network.  Three class of models

# Shuffle: Random controls implementing by shuffling edges.  Number of edges remain constant.
# Probability: Random controls implemented by assigning a probability for each edge to be part of the control.
# Other:  Different kind of control model e.g. for reciprocal connections

#Models to implement:

#- Erdos Renyi
#- Erdos Renyi corrected reciprocal connections
#- DAG plus reciprocal connections
#- Stochastic block model
#- Distance dependent
#- Distance dependent with depth dependence

####### IMPORTS #######################
import numpy as np
import scipy.sparse as sp
import generateModel as gm

from .resources.randomization import (
    bidirectional_edges,
    adjust_bidirectional_connections,
    add_bidirectional_connections,
    half_matrix
)

######generateModel versions###########
#Erdos-Renyi model:
#Input: n, p, threads
#    n = number of vertices (int)
#    p = edge probability (double)
#    threads = number of threads to use (int)
def run_ER(n,p,threads):
    return gm.ER(n,p,threads)

#Stochastic Block Model:
#Input: n, pathways, mtypes, threads
#    n = number of vertices (int)
#    M = M[i][j] entry is probability of edge between mtype i and mtype j (numpy array, shape=(m,m), dtype=float64), where m is number of mtypes
#    mtypes = i'th entry is mtype of vertex i (numpy array, shape=(n,), dtype=uint8)
#    threads = number of threads to use (int)
def run_SBM(n, pathways, mtypes, threads):
    return gm.SBM(n, pathways, mtypes, threads)

#Distance Dependant 2nd Order:
#Input: n, a, b, xyz, threads
#    n = number of vertices (int)
#    a = coefficient of probability function (double)
#    b = coefficient of probability function (double)
#    xyz = the coordinates of the vertices, (numpy array, shape=(n,3), dtype=float64)
#    threads = number of threads to use (int)
def run_DD2(n,a,b,xyz,threads)
    return gm.DD2(n,a,b,xyz,threads)

#Distance Dependant 3rd Order:
#Input: n, a, b, xyz, threads
#    n = number of vertices (int)
#    a1 and a2 = coefficients of probability function for dz<0 (double)
#    a2 and b2 = coefficient of probability function for dz>0 (double)
#    xyz = the coordinates of the vertices, (numpy array, shape=(n,3), dtype=float64)
#    depths = i'th entry is depth of vertex i (numpy array, shape=(n,), dtype=float64)
#    threads = number of threads to use (int)
def run_DD3(n,a1,b1,a2,b2,xyz,depths,threads):
    return gm.DD3(n,a1,b1,a2,b2,xyz,depths,threads)


####### SHUFFLE #######################
def ER_shuffle(adj, neuron_properties=[]):
    #Creates an ER control by shuffling entries away from the diagonal in adj
    #TODO: Re-implement this using only sparse matrices
    n=adj.get_shape()[0]
    adj=adj.toarray()
    not_diag=np.concatenate([adj[np.triu_indices(n,k=1)],adj[np.tril_indices(n,k=-1)]])#Entries off the diagonal
    np.random.shuffle(not_diag)
    adj[np.triu_indices(n,k=1)]=not_diag[0:n*(n-1)//2]
    adj[np.tril_indices(n,k=-1)]=not_diag[n*(n-1)//2:]
    return sp.csr_matrix(adj)

####### PROBABILITY ##################

####### OTHER ########################
def adjusted_ER(sparse_matrix: sp.csc_matrix, generator_seed:int) -> sp.csc_matrix:
    """
    Function to generate an ER with adjusted bidirectional connections.

    :param sparse_matrix: Sparse input matrix.
    :type: sp.csc_matrix
    :param generator_seed: Numpy generator seed.
    :type: int

    :return adjER_matrix: Adjusted ER model.
    :rtype: sp.csc_matrix
    """
    generator = np.random.default_rng(generator_seed)
    target_bedges = int(bidirectional_edges(sparse_matrix).count_nonzero() / 2)
    ER_matrix = ER_shuffle(sparse_matrix).tocsc()
    return adjust_bidirectional_connections(ER_matrix, target_bedges, generator)

def underlying_model(sparse_matrix: sp.csc_matrix, generator_seed: int):
    """
    Function to generate the underlying control model, obtained by turning
    the graph into a DAG (by making it undirected and using the GIDs to give
    directions) and adding bidirectional connections.

    :param sparse_matrix: Sparse input matrix.
    :type: sp.csc_matrix
    :param generator_seed: Numpy generator seed.
    :type: int

    :return und_matrix: Underlying model.
    :rtype: sp.csc_matrix
    """
    generator = np.random.default_rng(generator_seed)
    target_bedges = int(bidirectional_edges(sparse_matrix).count_nonzero() / 2)
    ut_matrix = sp.triu(sparse_matrix + sparse_matrix.T)
    return add_bidirectional_connections(ut_matrix, target_bedges, generator)


def bishuffled_model(sparse_matrix: sp.csc_matrix, generator_seed: int) -> sp.csc_matrix:
    """
    Function to generate the bishuffled control model, obtained by removing
    bidirectional edges by assigning a direction (according to GID order)
    to exactly half of them and the other direction to the other half.
    Probability-based direction assignment (original thesis) was not implemented
    to make the algorithm more performant, maybe to be used for SSCX.

    :param sparse_matrix: Sparse input matrix.
    :type: sp.csc_matrix
    :param generator_seed: Numpy generator seed.
    :type: int

    :return und_matrix: Bishuffled model.
    :rtype: sp.csc_matrix
    """
    generator = np.random.default_rng(generator_seed)
    ut_bedges = sp.triu(bidirectional_edges(sparse_matrix))
    target_bedges = ut_bedges.count_nonzero()
    bedges1, bedges2 = half_matrix(ut_bedges, generator)
    return add_bidirectional_connections(sparse_matrix - bedges1 - bedges2.T, target_bedges, generator)

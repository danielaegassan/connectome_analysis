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
import logging

import numpy as np
import scipy.sparse as sp

import generateModel as gm

LOG = logging.getLogger("connectome-analysis-randomization")
LOG.setLevel("INFO")
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")
######generateModel versions###########
def run_ER(n,p,threads):
    """
    Erdos-Renyi model

    Input: n, p, threads
       n = number of vertices (int)
       p = edge probability (double)
       threads = number of threads to use (int)

    Output: coo matrix
    """
    return gm.ER(n,p,threads)


def run_SBM(n, pathways, mtypes, threads):
    """
    Stochastic Block Model

    Input: n, pathways, mtypes, threads
       n = number of vertices (int)
       pathways = pathways[i][j] entry is probability of edge between mtype i and mtype j (numpy array, shape=(m,m), dtype=float64), where m is number of mtypes
       mtypes = i'th entry is mtype of vertex i (numpy array, shape=(n,), dtype=uint8)
       threads = number of threads to use (int)

    Output: coo matrix
    """
    return gm.SBM(n, pathways, mtypes, threads)


def run_DD2(n,a,b,xyz,threads):
    """
    Distance Dependant 2nd Order

    Input: n, a, b, xyz, threads
       n = number of vertices (int)
       a = coefficient of probability function (double)
       b = coefficient of probability function (double)
       xyz = the coordinates of the vertices, (numpy array, shape=(n,3), dtype=float64)
       threads = number of threads to use (int)

    Output: coo matrix
    """
    return gm.DD2(n,a,b,xyz,threads)

def run_DD2_model(adj, nrn, threads=8, **configdict):
    "Wrapper of distance dependence modelling together with graph generation"
    model_params = modelling.conn_prob_2nd_order_model(adj_matrix, nrn_table, **config_dict)
    n = adj.shape[0]
    a = model_params.mean(axis=0)['exp_model_scale']
    b = -model_params.mean(axis=0)['exp_model_exponent']
    xyz = nrn_table.loc[:,['x', 'y', 'z']].to_numpy() #Make and assert that checks these columns exist!
    C=gm.DD2(n,a,b,xyz,threads)
    i=C['row']
    j=C['col']
    data=np.ones(len(i))
    return sp.coo_matrix((data, (i, j)), [n,n])


def run_DD3(n,a1,b1,a2,b2,xyz,depths,threads):
    """
    Distance Dependant 3rd Order

    Input: n, a, b, xyz, threads
       n = number of vertices (int)
       a1 and a2 = coefficients of probability function for dz<0 (double)
       a2 and b2 = coefficient of probability function for dz>0 (double)
       xyz = the coordinates of the vertices, (numpy array, shape=(n,3), dtype=float64)
       depths = i'th entry is depth of vertex i (numpy array, shape=(n,), dtype=float64)
       threads = number of threads to use (int)

    Output: coo matrix
    """
    return gm.DD3(n,a1,b1,a2,b2,xyz,depths,threads)


#######_ SHUFFLE #######################

def seed_random_state(shuffler, seeder=np.random.seed):
    """Decorate a connectivity shuffler to seed it's random-state before execution.

    It is expected that the generator can be seeded calling `seeder(seed)`.
    """
    def seed_and_run_method(adj, neuron_properties=[], seed=None, **kwargs):
        """Reinitialize numpy random state using the value of seed among `kwargs`.
        doing nothing if no `seed` provided --- expecting an external initialization.
        """
        if seed is None:
            LOG.warning("No seed among keyword arguments")
        else:
            seeder(seed)

        return shuffler(adj, neuron_properties, **kwargs)

    return seed_and_run_method


def run_DD2_block_pre(n,pathways,mtypes,xyz,threads):
    """
    Distance Dependant Stochastic Block Model (pre synaptic only)

    Input: n, pathways, mtypes, xyz, threads
       n = number of vertices (int)
       pathways = pathways[i] is a pair (a,b) of coefficients for DD2 probability function (numpy array, shape=(m,2), dtype=double), where m is number of mtypes
       mtypes = i'th entry is mtype of vertex i (numpy array, shape=(n,), dtype=uint8)
       xyz = the coordinates of the vertices, (numpy array, shape=(n,3), dtype=float64)
       threads = number of threads to use (int)

    Output: coo matrix
    """
    return gm.DD2_block_pre(n,pathways,mtypes,xyz,threads)


def run_DD2_block(n,pathways,mtypes,xyz,threads):
    """
    Distance Dependant Stochastic Block Model

    Input: n, pathways, mtypes, xyz, threads
       n = number of vertices (int)
       pathways = pathways[i][j] is a pair (a,b) of coefficients for DD2 probability function (numpy array, shape=(m,m,2), dtype=double), where m is number of mtypes
       mtypes = i'th entry is mtype of vertex i (numpy array, shape=(n,), dtype=uint8)
       xyz = the coordinates of the vertices, (numpy array, shape=(n,3), dtype=float64)
       threads = number of threads to use (int)

    Output: coo matrix
    """
    return gm.DD2_block(n,pathways,mtypes,xyz,threads)


####### SHUFFLE #######################
@seed_random_state
def ER_shuffle(adj, neuron_properties=[]):
    """
    #Creates an ER control by shuffling entries away from the diagonal in adj
    TODO: Re-implement this using only sparse matrices
    """
    n = adj.get_shape()[0]
    adj = adj.toarray()
    LOG.info("Shuffle %s edges following Erdos-Renyi", adj.sum())
    above_diagonal = adj[np.triu_indices(n, k=1)]
    below_diagonal = adj[np.tril_indices(n, k=-1)]
    off_diagonal = np.concatenate([above_diagonal, below_diagonal])

    np.random.shuffle(off_diagonal)
    adj[np.triu_indices(n,k=1)] = off_diagonal[0:n*(n-1)//2]
    adj[np.tril_indices(n,k=-1)] = off_diagonal[n*(n-1)//2:]
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
    from .resources.randomization import bidrectional_edges, adjust_bidirectional_connections
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
    from .resources.randomization import bidrectional_edges, add_bidirectional_connections
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
    from .resources.randomization import bidrectional_edges, add_bidirectional_connections, half_matrix
    generator = np.random.default_rng(generator_seed)
    ut_bedges = sp.triu(bidirectional_edges(sparse_matrix))
    target_bedges = ut_bedges.count_nonzero()
    bedges1, bedges2 = half_matrix(ut_bedges, generator)
    return add_bidirectional_connections(sparse_matrix - bedges1 - bedges2.T, target_bedges, generator)


def configuration_model(sparse_matrix: sp.coo_matrix, generator_seed: int):
    """
    Function to generate the configuration control model, obtained by
    shuffling the row and column of coo format independently, to create
    new coo matrix, then removing any multiple edges and loops.

    :param sparse_matrix: Sparse input matrix.
    :type: sp.coo_matrix
    :param generator_seed: Numpy generator seed.
    :type: int

    :return CM_matrix: Configuration model.
    :rtype: sp.csr_matrix
    """
    generator = np.random.default_rng(generator_seed)
    R = sparse_matrix.row
    C = sparse_matrix.col
    generator.shuffle(R)
    generator.shuffle(C)
    CM_matrix = sp.coo_matrix(([1]*len(R),(R,C)),shape=sparse_matrix.shape).tocsr()
    CM_matrix.setdiag(0)
    CM_matrix.eliminate_zeros()
    return CM_matrix

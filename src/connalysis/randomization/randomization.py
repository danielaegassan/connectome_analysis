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

import generate_model as gm

LOG = logging.getLogger("connectome-analysis-randomization")
LOG.setLevel("INFO")
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")
######generate_model versions###########
def run_ER(n, p, threads=8, seed=(None,None)):
    """Creates an Erdos Renyi digraph.

    Parameters
    ----------
    n : int
        Number of vertices
    p : float
        Edge probablity, must satisfy $0 \\le p \\le 1$
    threads : int
        Number of parallel threads to be used
    seed : pair of ints
        Random seed to be used, if none is provided a seed is randomly selected

    Returns
    -------
    dict
        The edge list of the new digraph as a dictionary
        with keys 'row' and 'col'. Where (row[i],col[i]) is a directed edge
        of the digraph, for all i.

    Examples
    --------
    Setting n=3 and p=1 gives the complete digraph on 3 vertices:
    >>> connalysis.randomization.run_ER(3,1)
    {'row': [0, 0, 1, 1, 2, 2], 'col': [1, 2, 0, 2, 0, 1]}

    Raises
    ------
    AssertionError
        If p is not between 0 and 1

    """
    assert (p >= 0 and p <= 1), "p must be between 0 and 1"
    if seed[0]==None or seed[1]==None:
        return gm.ER(n,p,threads)
    else:
        return gm.ER(n,p,threads,seed[0],seed[1])


def run_SBM(n, probs, blocks, threads=8, seed=(None,None)):
    """Creates a random digraph using the stochastic block model.

    Parameters
    ----------
    n : int
        Number of vertices
    probs : numpy array of floats
        shape=(m,m) where m is the number of blocks.
        probs[i][j] is probability of an edge between block i and block j
    blocks : numpy array of ints
        shape=(n,). The i'th entry gives to which block vertex i belongs.
    threads : int
        Number of parallel threads to be used
    seed : pair of ints
        Random seed to be used, if none is provided a seed is randomly selected

    Returns
    -------
    dict
        The edge list of the new digraph as a dictionary
        with keys 'row' and 'col'. Where (row[i],col[i]) is a directed edge
        of the digraph, for all i.

    Examples
    --------
    To create an SBM digraph on 4 vertices where the even to
    odd, or odd to even, vertices connect with high probablity (p=0.9)
    and the even to evens or odd to odds connect with low probability (p=0.1):
    >>> connalysis.randomization.run_SBM(4,np.array([[0.1,0.9],[0.9,0.1]]),np.array([0,1,0,1]))
    {'row': [0, 0, 1, 1, 1, 2, 2, 3, 3], 'col': [1, 3, 0, 2, 3, 1, 3, 0, 2]


    Raises
    ------
    TypeError
        If blocks contains non-integers

    References
    ----------
    [1] P.W. Holland, K. Laskey, S. Leinhardt,
    ["Stochastic Blockmodels: First Steps"](https://www.sciencedirect.com/science/article/pii/0378873383900217),
    Soc Networks, 5-2, pp. 109-137, 1982

    """

    if seed[0]==None or seed[1]==None:
        return gm.SBM(n, probs, blocks, threads)
    else:
        return gm.SBM(n, probs, blocks, threads, seed[0], seed[1])


def run_DD2(n,a,b,xyz,threads=8, seed=(None,None)):
    """Creates a random digraph using the 2nd-order probability model.

    Parameters
    ----------
    n : int
        Number of vertices
    a : float
        Coefficient of probability function
    b : float
        Absolute value of power of exponent in probability function
    xyz : (n,3)-numpy array of floats
        Co-ordinates of vertices in $\mathbb{R}^3$
    threads : int
        Number of parallel threads to be used
    seed : pair of ints
        Random seed to be used, if none is provided a seed is randomly selected

    Returns
    -------
    dict
        The edge list of the new digraph as a dictionary
        with keys 'row' and 'col'. Where (row[i],col[i]) is a directed edge
        of the digraph, for all i.

    Examples
    --------
    TODO

    See Also
    --------
    [conn_prob_2nd_order_model](modelling.md#src.connalysis.modelling.modelling.conn_prob_2nd_order_model) : A variant of this function for neurons

    References
    ----------
    [1] TODO

    """
    if seed[0]==None or seed[1]==None:
        return gm.DD2(n,a,b,xyz,threads)
    else:
        return gm.DD2(n,a,b,xyz,threads,seed[0],seed[1])

def run_DD2_model(adj, node_properties,
                  model_params_dd2=None, #an analysis that could be loaded from the pipeline
                  coord_names= ['x', 'y', 'z'],
                  threads=8, return_params=False, **config_dict):
    """
    Wrapper generating a random control graph based on 2nd order distance dependence model
    Input:
    adj: original adjacency matrix, if model_params have already been computed can pass empty matrix of the right size
    node_properties: DataFrame with information on the vertices of adj, it must have columns corresponding to the names
    the coordinates to be used for distance computation.  Default ['x', 'y', 'z']
    configdict: Add me --> to generate parameters of 2nd order distance model
    model_params: optional input of pre-computed model parameters, data frame with rows corresponding to seeds of model estimation
    (single row if subsampling is not used) and columns:
    exp_model_scale and exp_model_exponent for the model parameters.  See modelling.conn_prob_2nd_order_model for details.

    Output: scipy coo matrix, optional model_parameters
    """

    if model_params_dd2 is None:
        from .import modelling
        #TODO:  What to do if coord_names are also given in configdict and do not match coord_names?
        config_dict["coord_names"]=coord_names
        model_params_dd2 = modelling.conn_prob_2nd_order_model(adj, node_properties,**config_dict)

    LOG.info("Run DD2 model with parameters: \n%s", model_params_dd2)

    n = adj.shape[0]
    a = model_params_dd2.mean(axis=0)['exp_model_scale']
    b = model_params_dd2.mean(axis=0)['exp_model_exponent']
    xyz = node_properties.loc[:,coord_names].to_numpy() #Make and assert that checks these columns exist!
    if len(coord_names)<3: #Extend by zeros if lower dimensional data was used to compute distance
        xyz=np.hstack([xyz,np.zeros((xyz.shape[0],3-xyz.shape[1]))])
    C=gm.DD2(n,a,b,xyz,threads)
    i=C['row']
    j=C['col']
    data=np.ones(len(i))
    if return_params==True:
        return sp.coo_matrix((data, (i, j)), [n,n]), model_params_dd2
    else:
        return sp.coo_matrix((data, (i, j)), [n,n])


def run_DD3(n,a1,b1,a2,b2,xyz,depths,threads=8, seed=(None,None)):
    """Creates a random digraph using the 2nd-order probability model.

    Parameters
    ----------
    n : int
        Number of vertices
    a1 : float
        Coefficient of probability function for negative depth
    b1 : float
        Absolute value of power of exponent in probability function for negative depth
    a2 : float
        Coefficient of probability function for positive depth
    b2 : float
        Absolute value of power of exponent in probability function for positive depth
    xyz : (n,3)-numpy array of floats
        Co-ordinates of vertices in $\mathbb{R}^3$
    threads : int
        Number of parallel threads to be used
    seed : pair of ints
        Random seed to be used, if none is provided a seed is randomly selected

    Returns
    -------
    dict
        The edge list of the new digraph as a dictionary
        with keys 'row' and 'col'. Where (row[i],col[i]) is a directed edge
        of the digraph, for all i.

    Examples
    --------
    TODO

    See Also
    --------
    [conn_prob_3rd_order_model](modelling.md#src.connalysis.modelling.modelling.conn_prob_3rd_order_model) : A variant of this function for neurons

    References
    ----------
    [1] TODO

    """
    if seed[0]==None or seed[1]==None:
        return gm.DD3(n,a1,b1,a2,b2,xyz,depths,threads)
    else:
        return gm.DD3(n,a1,b1,a2,b2,xyz,depths,threads,seed[0],seed[1])


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


def run_DD2_block_pre(n, probs, blocks, xyz, threads=8, seed=(None,None)):
    """Creates a random digraph using a combination of the stochastic block model
       and the 2nd order distance dependent model. Such that the probability of an edge
       is given by the distance dependent equation, but the parameters of that equation
       vary depending on the block of the source of the edge.
       # TODO:  Add this to tutorials

    Parameters
    ----------
    n : int
        Number of vertices
    probs : numpy array of floats
        shape=(m,2) where m is the number of blocks.
        probs[i][0] is the coefficient of the distance dependent equation (value a) for source vertex i and
        probs[i][0] is the absolute value of power of exponent in the distance dependent equation (value b)
    blocks : numpy array of ints
        shape=(n,). The i'th entry is which block vertex i belongs to.
    xyz : (n,3)-numpy array of floats
        Co-ordinates of vertices in $\mathbb{R}^3$
    threads : int
        Number of parallel threads to be used
    seed : pair of ints
        Random seed to be used, if none is provided a seed is randomly selected

    Returns
    -------
    dict
        The edge list of the new digraph as a dictionary
        with keys 'row' and 'col'. Where (row[i],col[i]) is a directed edge
        of the digraph, for all i.

    Examples
    --------
    TODO


    Raises
    ------
    TypeError
        If blocks contains non-integers

    See Also
    --------
    run_SBM: Function which runs the stochastic block model

    run_DD2 : Function which runs the 2nd distance dependent model

    run_DD2_block : Similar function that also accounts for the block of the target vertex

    """

    if seed[0]==None or seed[1]==None:
        return gm.DD2_block_pre(n, probs, blocks, xyz, threads)
    else:
        gm.DD2_block_pre(n, probs, blocks, xyz, threads, seed[0], seed[1])


def run_DD2_block(n, probs, blocks, xyz, threads, seed=(None,None)):
    """Creates a random digraph using a combination of the stochastic block model
       and the 2nd order distance dependent model. Such that the probability of an edge
       is given by the distance dependent equation, but the parameters of that equation
       vary depending on the block of the source of the edge and block of the target.

    Parameters
    ----------
    n : int
        Number of vertices
    probs : numpy array of floats
        shape=(m,m,2) where m is the number of blocks. For source vertex i and target vertex j
        probs[i][j][0] is the coefficient of the distance dependent equation (value a) and
        probs[i][j][0] is the absolute value of power of exponent in the distance dependent equation (value b)
    blocks : numpy array of ints
        shape=(n,). The i'th entry is which block vertex i belongs to.
    xyz : (n,3)-numpy array of floats
        Co-ordinates of vertices in $\mathbb{R}^3$
    threads : int
        Number of parallel threads to be used
    seed : pair of ints
        Random seed to be used, if none is provided a seed is randomly selected

    Returns
    -------
    dict
        The edge list of the new digraph as a dictionary
        with keys 'row' and 'col'. Where (row[i],col[i]) is a directed edge
        of the digraph, for all i.

    Examples
    --------
    TODO


    Raises
    ------
    TypeError
        If blocks contains non-integers

    See Also
    --------
    run_DD2 : Function which runs the 2nd distance dependent model

    run_SBM: Function which runs the stochastic block model

    run_DD2_block_pre : Similar function that only accounts for the block of the source vertex

    """
    if seed[0]==None or seed[1]==None:
        return gm.DD2_block(n, probs, blocks, xyz, threads)
    else:
        return gm.DD2_block(n, probs, blocks, xyz, threads, seed[0], seed[1])

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
# Functions that implement random controls of a network.  There are two general types of models

# Probability: Random controls implemented by assigning a probability for each edge to be part of the control.
# Shuffle: Random controls implementing by shuffling edges according to certain rules.

# IMPORTS #
import logging
import numpy as np
import scipy.sparse as sp
import bigrandomgraphs as gm
LOG = logging.getLogger("connectome-analysis-randomization")
LOG.setLevel("INFO")
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")

#######################################################
################# PROBABILITY MODELS  #################
#######################################################

def _dict_to_coo(adj, N):
    # Utility function to format dict into scipy.sparse.coo
    return sp.coo_matrix((np.ones(len(adj['row'])), (adj['row'], adj['col'])), shape=(N, N))


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
        adj = gm.ER(n,p,threads)
    else:
        adj = gm.ER(n,p,threads,seed[0],seed[1])
    return _dict_to_coo(adj,n)



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
        adj = gm.SBM(n, probs, blocks, threads)
    else:
        adj = gm.SBM(n, probs, blocks, threads, seed[0], seed[1])
    return _dict_to_coo(adj, n)


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

    See Also
    --------
    [conn_prob_2nd_order_model](modelling.md#src.connalysis.modelling.modelling.conn_prob_2nd_order_model) :
    The modelling function from which the parameters ``a`` and ``b``can be obtained.


    """
    if seed[0]==None or seed[1]==None:
        adj = gm.DD2(n,a,b,xyz,threads)
    else:
        adj = gm.DD2(n,a,b,xyz,threads,seed[0],seed[1])
    return _dict_to_coo(adj,n)

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


    See Also
    --------
    [conn_prob_3rd_order_model](modelling.md#src.connalysis.modelling.modelling.conn_prob_3rd_order_model) :
    The modelling function from which the parameters ``a1/a2`` and ``b1/b2``can be obtained.



    """
    if seed[0]==None or seed[1]==None:
        adj = gm.DD3(n,a1,b1,a2,b2,xyz,depths,threads)
    else:
        adj = gm.DD3(n,a1,b1,a2,b2,xyz,depths,threads,seed[0],seed[1])
    return _dict_to_coo(adj,n)


def run_DD2_block_pre(n, probs, blocks, xyz, threads=8, seed=(None,None)):
    """Creates a random digraph using a combination of the stochastic block model
       and the 2nd order distance dependent model. Such that the probability of an edge
       is given by the distance dependent equation, but the parameters of that equation
       vary depending on the block of the source of the edge.

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


    Raises
    ------
    TypeError
        If blocks contains non-integers

    See Also
    --------
    [run_SBM](randomization.md#src.connalysis.randomization.randomization.run_SBM):
    Function which runs the stochastic block model

    [run_DD2](randomization.md#src.connalysis.randomization.randomization.run_DD2) :
    Function which runs the 2nd distance dependent model

    [run_DD2_block](randomization.md#src.connalysis.randomization.randomization.run_DD2_block) :
    Similar function that also accounts for the block of the target vertex

    """

    if seed[0]==None or seed[1]==None:
        adj = gm.DD2_block_pre(n, probs, blocks, xyz, threads)
    else:
        adj = gm.DD2_block_pre(n, probs, blocks, xyz, threads, seed[0], seed[1])
    return _dict_to_coo(adj,n)


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


    Raises
    ------
    TypeError
        If blocks contains non-integers

    See Also
    --------
    [run_DD2](randomization.md#src.connalysis.randomization.randomization.run_DD2) :
    Function which runs the 2nd distance dependent model

    [run_SBM](randomization.md#src.connalysis.randomization.randomization.run_SBM) :
    Function which runs the stochastic block model

    [run_DD2_block_pre](randomization.md#src.connalysis.randomization.randomization.run_DD2_block_pre) :
    Similar function that only accounts for the block of the source vertex

    """
    if seed[0]==None or seed[1]==None:
        adj = gm.DD2_block(n, probs, blocks, xyz, threads)
    else:
        adj = gm.DD2_block(n, probs, blocks, xyz, threads, seed[0], seed[1])
    return _dict_to_coo(adj,n)


#######################################################
################### SHUFFLE MODELS  ###################
#######################################################
def _seed_random_state(shuffler, seeder=np.random.seed):
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


####### SHUFFLE #######################
@_seed_random_state
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


def configuration_model(adj, seed = None):
    """Function to generate the configuration control model, obtained by
    shuffling the row and column of coo format independently, to create
    new coo matrix, then removing any multiple edges and loops.

    Parameters
    ----------
    adj : coo-matrix
        Adjacency matrix of a directed network.
    seed : int
        Random seed to be used

    Returns
    -------
    csr matrix
        Configuration model control of adj

    See Also
    --------
    [run_SBM](randomization.md#src.connalysis.randomization.randomization.run_SBM) :
    Function which runs the stochastic block model

    [run_DD2](randomization.md#src.connalysis.randomization.randomization.run_DD2) :
    Function which runs the 2nd distance dependent model
    """
    adj=adj.tocoo()
    generator = np.random.default_rng(seed)
    R = adj.row
    C = adj.col
    generator.shuffle(R)
    generator.shuffle(C)
    CM_matrix = sp.coo_matrix(([1]*len(R),(R,C)),shape=adj.shape).tocsr()
    CM_matrix.setdiag(0)
    CM_matrix.eliminate_zeros()
    return CM_matrix

def adjusted_ER(adj, seed=None):
    """Function to generate an Erdos  Renyi model with adjusted bidirectional connections.

    Parameters
    ----------
    adj :  csc_matrix
        Adjacency matrix of a directed network.
    seed : int
        Random seed to be used

    Returns
    -------
    csc_matrix
        Erdos Renyi shuffled control with additional reciprocal connections added at random
        to match the number of reciprocal connections of the original matrix.

    See Also
    --------
    [underlying_model](randomization.md#src.connalysis.randomization.randomization.underlying_model) :
    Function which returns a digraph with the same  underlying undirected graph
    and same number of reciprocal connections

    [bishuffled_model](randomization.md#src.connalysis.randomization.randomization.bishuffled_model) :
    Function which returns a digraph with shuffled reciprocal connections
    """
    from connalysis.network.topology import rc_submatrix
    from .rand_utils import adjust_bidirectional_connections
    generator = np.random.default_rng(seed)
    ER_matrix = ER_shuffle(adj, seed=seed).tocsc()
    bedges_to_add = int(rc_submatrix(adj).count_nonzero() -rc_submatrix(ER_matrix).count_nonzero())//2
    if bedges_to_add >= 0:
        return adjust_bidirectional_connections(ER_matrix, bedges_to_add, generator)
    else:
        LOG.info("Erdos-Renyi control has more reciprocal connections than original, so they are not adjusted.")
        return ER_matrix

def underlying_model(adj, seed: int=None):
    """Function to generate a digraph with the same  underlying undirected graph as adj
        and the same number of reciprocal connections

    Parameters
    ----------
    adj : csc_matrix
        Adjacency matrix of a directed network.
    seed : int
        Random seed to be used

    Returns
    -------
    csc_matrix
        Digraph with the same  underlying undirected graph as adj and the same number of reciprocal connections

    See Also
    --------
    [adjusted_ER](randomization.md#src.connalysis.randomization.randomization.adjusted_ER) :
    Function to generate an Erdos  Renyi model with adjusted bidirectional connections

    [bishuffled_model](randomization.md#src.connalysis.randomization.randomization.bishuffled_model) :
    Function which returns a digraph with shuffled reciprocal connections
    """
    from connalysis.network.topology import rc_submatrix
    from .rand_utils import  add_bidirectional_connections
    generator = np.random.default_rng(seed)
    target_bedges = int(rc_submatrix(adj).count_nonzero() / 2)
    ut_matrix = sp.triu(adj + adj.T)
    return add_bidirectional_connections(ut_matrix, target_bedges, generator)

def bishuffled_model(adj, seed = None):
    """Function to generate a digraph with shuffled reciprocal connections

    Parameters
    ----------
    adj : csc_matrix
        Adjacency matrix of a directed network.
    seed : int
        Random seed to be used

    Returns
    -------
    csc_matrix
        Digraph with shuffled reciprocal connections

    See Also
    --------
    [adjusted_ER](randomization.md#src.connalysis.randomization.randomization.adjusted_ER) :
    Function to generate an Erdos  Renyi model with adjusted bidirectional connections

    [underlying_model](randomization.md#src.connalysis.randomization.randomization.underlying_model) :
    Function which returns a digraph with the same  underlying undirected graph
    and same number of reciprocal connections
    """
    from connalysis.network.topology import rc_submatrix
    from .rand_utils import  add_bidirectional_connections, half_matrix
    generator = np.random.default_rng(seed)
    ut_bedges = sp.triu(rc_submatrix(adj))
    target_bedges = ut_bedges.count_nonzero()
    bedges1, bedges2 = half_matrix(ut_bedges, generator)
    return add_bidirectional_connections(adj - bedges1 - bedges2.T, target_bedges, generator)

#######################################################
################ GRAPH MODIFICATIONS  #################
#######################################################
def add_rc_connections_skeleta(adj,factors,dimensions=None, skeleta=None, threads=8, seed=0, return_skeleta=False):
    """Function to add reciprocal connections at random to adj on the skeleta of maximal simplices of adj

    Parameters
    ----------
    adj : sparse matrix
        Adjacency matrix of a directed network
    factors: int or dict
        Factor by which to multiply the reciprocal connections on the ``k``-skeleta of adj.  If factors is an int
        the same factor is used on all dimensions.  Otherwise, factors can be a dictionary with keys dimensions
        and values the factor by which to multiply the number of reciprocal connections on that dimensions.
    dimensions: array
        The dimensions at which to increase the number of reciprocal connections.  If ``None`` then all dimensions
        will be used
    skeleta: dict
        Dictionary with keys f'dimension_{dim}' for dim in dimensions and values binary sparse sub-matrices of adj
        on which reciprocal connections will be added.
    threads: int
        Number of threads on which to parallelize the skeleta computation if not pre-computed
    seed : int
        Random seed to be used to selecte edges that will become reciprocal

    Returns
    -------
    csc_matrix, dict
        Digraph with add reciprocal connections
        If return_skeleta=True it also returns the skeleta of maximal simplices of adj in the dimensions selected

    """
    adj=adj.tocsr()
    from connalysis.network.topology import rc_submatrix
    from .rand_utils import add_bidirectional_connections
    # Compute skeleton graphs if not precomputed
    if skeleta is None:
        from connalysis.network.topology import get_k_skeleta_graph
        max_simplices=True # Add option for all simplices?
        skeleta=get_k_skeleta_graph(adj, max_simplices=max_simplices, dimensions=dimensions, threads=threads)
    # Restrict to dimensions that contain simplices
    if dimensions is None:
        dimensions = np.array([int(key[10:]) for key in skeleta.keys()])
    else:
        dimensions = np.intersect1d(np.array([int(key[10:]) for key in skeleta.keys()]), dimensions)
    # Generate mapping between factors and dimensions or check the one provided
    if isinstance(factors,int):
        factors={dim:factors for dim in dimensions}
    else:
        assert isinstance(factors, dict), "factors must be int or dictionary"
        assert np.isin(dimensions, np.array(list(factors.keys()))).all(), "all dimensions must be a key in factors"

    # Add bidirectional connections
    generator=np.random.default_rng(seed)
    rc_add={dim:((factors[dim]-1)*(rc_submatrix(skeleta[f'dimension_{dim}']).sum()))//2 for dim in dimensions}
    print("Number of reciprocal connections added per dimension"); display(rc_add) # Remove or add verbose option
    M=adj.copy()
    for dim in dimensions:
        M+=add_bidirectional_connections(skeleta[f'dimension_{dim}'], rc_add[dim], generator).astype(bool)
    if return_skeleta:
        return M, skeleta
    else:
        return M

def add_rc_connections(adj,n_rc, seed=0):
    """Function to turn a fixed amount of unidirectional connections of adj into reciprocal connections.

    Parameters
    ----------
    adj : sparse matrix
        Adjacency matrix of a directed network
    n_rc : Number of reciprocal connections to be added
    seed : int
        Random seed to be used to selecte edges that will become reciprocal

    Returns
    -------
    matrix
        Digraph with n_rc more edges than adj, all of which form reciprocal connections
    """
    # TODO: Move this function from utils and change dependencies
    from .rand_utils import add_bidirectional_connections
    # Add bidirectional connections
    generator=np.random.default_rng(seed)
    return add_bidirectional_connections(adj, n_rc, generator).astype(bool)

def add_connections(adj,nc, seed=0,sparse_mode=True, max_iter=30):
    """Function add connections at random

    Parameters
    ----------
    adj : matrix
        Adjacency matrix of a directed network
    nc : Number of connections to be added
    seed : int
        Random seed to be used to selecte edges that will become reciprocal
    sparse_mode: bool
        If sparse_mode is ``True`` the matrix is generated iteratively restricting to a sparse format.
        If ``False`` adj is converted to dense and edges are added in a single step

    Returns
    -------
    bool matrix
        Digraph with nc more edges than adj
    """
    adj=adj.astype(bool)
    # Add bidirectional connections
    if sparse_mode:
        # TODO: Search for more efficient way to do this
        N=adj.shape[0]; E=adj.sum(); k=0 # Number nodes, target edges and iteration counter
        while adj.sum()< E +nc: #target number of edges
            if k>max_iter:
                print("More than max_it iterations tried, increase number of iterations or try sparse_mode =False")
                break
            den=(E+nc-adj.sum())/(N*N) #density of matrix added
            generator = np.random.default_rng(seed)
            A=sp.random(*adj.shape, density=den, format='csr', dtype = 'bool', random_state = generator)
            A.setdiag(0)
            adj=adj+A; k+=1
        adj.eliminate_zeros()
    else:
        if sp.issparse(adj): adj=adj.toarray()
        ul_ind = np.where(np.eye(*adj.shape) == 0) # non-diagonal indices
        zero_ind=np.where(adj[ul_ind]==0)
        generator = np.random.default_rng(seed)
        selection=zero_ind[0][generator.choice(zero_ind[0].shape[0], replace=False, size=nc)]
        adj[(ul_ind[0][selection],ul_ind[1][selection])]=1
    return adj
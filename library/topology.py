import scipy.sparse as sp
import numpy as np
import pandas as pd

from functools import partial
from typing import List
# Functions that take as input a (weighted) network and give as output a topological feature.
#TODO: rc_in_simplex, filtered_simplex_counts, persitence

def simplex_counts(adj, neuron_properties=[], max_simplices=False,threads=1):

    #Compute simplex counts of adj
    #TODO: Change this to pyflagser_count and add options for max dim and threads,
    #Delete neuron properties from input?
    import pyflagsercount
    adj=adj.astype('bool').astype('int') #Needed in case adj is not a 0,1 matrix
    if max_simplices==False:
        return pyflagsercount.flagser_count(adj, max_simplices=max_simplices,threads=threads)["cell_counts"]
    if max_simplices==True:
        return pyflagsercount.flagser_count(adj, max_simplices=max_simplices,threads=threads)



def betti_counts(adj, neuron_properties=[], min_dim=0, max_dim=[], directed=True, coeff=2, approximation=None):
    from pyflagser import flagser_unweighted
    import numpy as np
    adj=adj.astype('bool').astype('int') #Needed in case adj is not a 0,1 matrix
    if max_dim==[]:
        max_dim=np.inf

    if approximation==None:
        print("Run without approximation")
        return flagser_unweighted(adj, min_dimension=min_dim,
                                  max_dimension=max_dim, directed=True, coeff=2, approximation=None)['betti']
    else:
        assert (all([isinstance(item,int) for item in approximation])) # asssert it's a list of integers
        approximation=np.array(approximation)
        bettis=[]

        #Make approximation vector to be of size max_dim
        if max_dim!=np.inf:
            if approximation.size-1 < max_dim:#Vector too short so pad with -1's
                approximation=np.pad(approximation,(0,max_dim-(approximation.size-1)),'constant',constant_values=-1)
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
            bettis=bettis+flagser_unweighted(adj, min_dimension=n,
                                        max_dimension=N, directed=True, coeff=2, approximation=a)['betti']

        if max_dim==np.inf:
            n=approximation.size #min dim for computation
            N=np.inf #max dim for computation
            a=None
            print("Run betti for dim range {0}-{1} with approximation {2}".format(n,N,a))
            bettis=bettis+flagser_unweighted(adj, min_dimension=n,
                                        max_dimension=N, directed=True, coeff=2, approximation=a)['betti']

        return bettis

def node_participation(adj, neuron_properties):
    # Compute the number of simplices a vertex is part of
    # Input: adj adjancency matrix representing a graph with 0 in the diagonal, neuron_properties as data frame with index gid of nodes
    # Out: List L of lenght adj.shape[0] where L[i] is a list of the participation of vertex i in simplices of a given dimensio
    # TODO:  Should we merge this with simplex counts so that we don't do the computation twice?
    import pyflagsercount
    import pandas as pd
    adj = adj.astype('bool').astype('int')  # Needed in case adj is not a 0,1 matrix
    par=pyflagsercount.flagser_count(M,containment=True,threads=1)['contain_counts']
    par = {i: par[i] for i in np.arange(len(par))}
    par=pd.DataFrame.from_dict(par, orient="index").fillna(0).astype(int)
    return par


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
    import pyflagsercount as pfc
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
    import pyflagsercount as pfc
    result = pfc.flagser_count(adj, return_simplices=True, threads = 1)
    coo_matrix = adj.tocoo()
    result['simplices'][1] = np.stack([coo_matrix.row, coo_matrix.col]).T
    for i in range(len(result['simplices']) -2):
        result['simplices'][i+2] = np.array(result['simplices'][i+2])
    return result['simplices'][1:]


def bedge_counts(adj: sp.csc_matrix, simplex_matrix_list: List[np.ndarray]) -> List[np.ndarray]:
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
    for matrix in simplex_matrix_list:
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

def convex_hull(adj, neuron_properties):# --> topology
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

def filtration_weights(weighted_adj, neuron_properties=[],method="strength"):
    #Todo: Should there be a warning when the return is an empty array because the matrix is zero?
    """Returns the filtration weights of a given weighted matrix.
    :param method:distance smaller weights enter the filtration first
                  strength larger weights enter the filtration first"""
    if method == "strength":
        return np.unique(weighted_adj.data)[::-1]
    elif method == "distance":
        return np.unique(weighted_adj.data)
    else:
       raise ValueError("Method has to be 'strength' or 'distance'")

def filtered_simplex_counts(weighted_adj, neuron_properties=[],method="strength"):
    '''Takes weighted adjancecy matrix returns data frame with filtered simplex counts where index is the weight
    method strength higher weights enter first, method distance smaller weights enter first'''
    from tqdm import tqdm
    weights=filtration_weights(weighted_adj,neuron_properties=[],method=method)
    simplex_counts_filtered=dict.fromkeys(weights)
    for weight in tqdm(weights,total=len(weights)):
        adj=at_weight_edges(weighted_adj,threshold=weight,method=method)
        simplex_counts_filtered[weight]=simplex_counts(adj)
    simplex_counts_filtered=pd.DataFrame.from_dict(simplex_counts_filtered,orient="index").fillna(0).astype(int)
    return  simplex_counts_filtered

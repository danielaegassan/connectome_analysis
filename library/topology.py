import scipy.sparse as sp
import numpy as np
import pyflagsercount as pfc
import pickle

from pathlib import Path
from tqdm import tqdm
from typing import List
# Functions that take as input a (weighted) network and give as output a topological feature.
#TODO: rc_in_simplex, filtered_simplex_counts, persitence

import numpy as np
import pandas as pd


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
def binary2simplex(address):
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

def simplex_matrix(adj: sp.csc_matrix, temp_folder: Path, verbose: bool = False) -> np.array:
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
            simplex_list = tqdm(simplex_list, desc="Parsed simplices", total = simplex_matrix[0,-1])
        for simplex in simplex_list:
             simplex_matrix[simplex_matrix[0,len(simplex)-2] + 1, :len(simplex)] = simplex
             simplex_matrix[0, len(simplex)-2]+=1
    vmessage("Generated simplex matrix")
    np.save(temp_folder/"matrix.npy", simplex_matrix)
    vmessage("Saved simplex matrix")
    return simplex_matrix


def simplex_matrix_list(adj: sp.csc_matrix, temp_folder: Path, verbose: bool = False) -> List[np.array]:
    """
    Returns the list of simplices in a list of matrices for storage. Each matrix is
    a n_simplices x dim matrix, where n_simplices is the total number of simplices
    with dimension dim.
    
    :param adj: Sparse csc matrix to compute the simplex list of.
    :type: sp.scs_matrix
    :param temp_folder: Relative path where to store the temporary flagser files to.
    :type: Path
    :param verbose: Whether to have the function print steps.
    :type: bool

    :return mlist: List of matrices containing the simplices. 
    :rtype: List[np.array]
    """
    def vmessage(message):
        if verbose:
            print(message)
    temp_folder.mkdir(exist_ok = True, parents=True)
    for path in temp_folder.glob("*"):
        raise FileExistsError("Found file in " + str(temp_folder.absolute()) + ". Aborting.")
    vmessage("Flagser count started execution")
    counts = pfc.flagser_count(adj, binary=str(temp_folder / "temp"), min_dim_print=1, threads = 1)
    pointers = np.zeros((len(counts['cell_counts'])-1,), dtype=int)
    vmessage("Flagser count completed execution")
    mdim = len(counts['cell_counts'])
    simplex_matrix_list = []
    for dim, dim_length in enumerate(counts['cell_counts'][1:]):
        simplex_matrix_list.append(np.zeros((dim_length, dim+2), dtype=np.int32))
    vmessage("Parsing flagser output")
    for path in temp_folder.glob("*"):
        vmessage("Parsing " + str(path))
        simplex_list = binary2simplex(path)
        if verbose:
            simplex_list = tqdm(simplex_list, desc="Parsed simplices", total = np.sum(counts['cell_counts'][1:]))
        for simplex in simplex_list:
             sdim = len(simplex)-2
             simplex_matrix_list[sdim][pointers[sdim], :] = simplex
             pointers[sdim]+=1
    vmessage("Generated simplex matrix list")
    pickle.dump(simplex_matrix_list, (temp_folder / "ml.pkl").open('wb'))
    vmessage("Saved simplex matrix list")
    return simplex_matrix_list

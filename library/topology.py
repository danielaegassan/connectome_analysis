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
    par = {i: par[i] for i in np.arange(len(par))}
    return par
    par=pd.DataFrame.from_dict(par, orient="index").fillna(0).astype(int)
    return par
    par=par.join(neuron_properties['gid'])
    return par





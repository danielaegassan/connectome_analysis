# Functions that take as input a (weighted) network and give as output a topological feature.
#TODO: rc_in_simplex, filtered_simplex_counts, persitence


def simplex_counts(adj, neuron_properties=[]):
    #Compute simplex counts of adj
    #TODO: Change this to pyflagser_count and add options for max dim and threads,
    #Delete neuron properties from input?
    from pyflagser import flagser_count_unweighted
    adj=adj.astype('bool').astype('int') #Needed in case adj is not a 0,1 matrix
    return flagser_count_unweighted(adj, directed=True)

def betti_counts(adj, neuron_properties=[], min_dim=0, max_dim=[], directed=True, coeff=2, approximation=None):
    #Compute betti counts of adj which represents a direted graph parameters as in pyflagsercount
    #Delete neuron properties from input?
    from pyflagser import flagser_unweighted
    import numpy as np
    if max_dim==[]:
        max_dim=np.inf
    adj=adj.astype('bool').astype('int') #Needed in case adj is not a 0,1 matrix
    return flagser_unweighted(adj, min_dimension=min_dim,
                                    max_dimension=max_dim, directed=True, coeff=2, approximation=None)['betti']

def node_participation(adj, neuron_properties):
    # Compute the number of simplices a vertex is part of
    # Input: adj adjancency matrix representing a graph with 0 in the diagonal, neuron_properties as data frame with index gid of nodes
    # Out: List L of lenght adj.shape[0] where L[i] is a list of the participation of vertex i in simplices of a given dimensio
    # TODO:  Change pyflagsercontain so that it takes as input sparse matrices
    import pyflagsercount
    import pandas as pd
    adj = adj.astype('bool').astype('int')  # Needed in case adj is not a 0,1 matrix
    par=pyflagsercount.flagser_count(M,containment="delete_me")['contain_counts'][0]
    par = {i: par[i] for i in np.arange(len(par))}
    par=pd.DataFrame.from_dict(par, orient="index").fillna(0).astype(int)
    par=par.join(ninfo['gid'])
    return par




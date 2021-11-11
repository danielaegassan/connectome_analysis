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


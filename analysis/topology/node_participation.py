def node_participation(adj, neuron_properties):
    # Compute the number of simplices a vertex is part of
    # Input: adj adjancency matrix representing a graph with 0 in the diagonal, neuron_properties as data frame with index gid of nodes
    # Out: List L of lenght adj.shape[0] where L[i] is a list of the participation of vertex i in simplices of a given dimensio
    # TODO:  Change pyflagsercontain so that it takes as input sparse matrices
    from pyflagsercontain import flagser_count
    import pandas as pd
    adj = adj.astype('bool').astype('int')  # Needed in case adj is not a 0,1 matrix
    par = flagser_count(adj.toarray())
    par = {i: par[i] for i in np.arange(len(par))}
    par=pd.DataFrame.from_dict(par, orient="index").fillna(0)
    par=par.join(ninfo['gid'])
    return par


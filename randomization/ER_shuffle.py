def ER_shuffle(adj, neuron_properties=[]):
    #Creates an ER control by shuffling entries away from the diagonal in adj
    #TODO: Re-implement this using only sparse matrices
    import scipy.sparse
    n=adj.get_shape()[0]
    adj=adj.toarray()
    not_diag=np.concatenate([adj[np.triu_indices(n,k=1)],adj[np.tril_indices(n,k=-1)]])#Entries off the diagonal
    np.random.shuffle(not_diag)
    adj[np.triu_indices(n,k=1)]=not_diag[0:n*(n-1)//2]
    adj[np.tril_indices(n,k=-1)]=not_diag[n*(n-1)//2:]
    return scipy.sparse.csr_matrix(adj)
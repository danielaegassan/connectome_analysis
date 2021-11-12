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
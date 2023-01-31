#Moved to src/connalysis/network
import numpy

from scipy.spatial import distance


def connection_probability_within(adj, neuron_properties, max_dist=200, min_dist=0,
                                  columns=["ss_flat_x", "ss_flat_y"]):
    if isinstance(neuron_properties, tuple):
        nrn_pre, nrn_post = neuron_properties
    else:
        nrn_pre = neuron_properties; nrn_post = neuron_properties
    D = distance.cdist(nrn_pre[columns], nrn_post[columns])
    mask = (D > min_dist) & (D <= max_dist)  # This way with min_dist=0 a neuron with itself is excluded
    return adj[mask].mean()

def connection_probability(adj, neuron_properties):
    exclude_diagonal = False
    if isinstance(neuron_properties, tuple):
        nrn_pre, nrn_post = neuron_properties
        if len(nrn_pre) == len(nrn_post):
            if (nrn_pre["gid"] == nrn_post["gid"]).all():
                exclude_diagonal = True
    else:
        exclude_diagonal = True
    
    if not exclude_diagonal:
        return adj.astype(bool).mean()
    assert adj.shape[0] == adj.shape[1], "Inconsistent shape!"
    n_pairs = adj.shape[0] * (adj.shape[1] - 1)
    return adj.astype(bool).sum() / n_pairs

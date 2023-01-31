#Moved to src/connalysis/network
#-*- coding: utf-8 -*-
"""
Network metrics on a graph
authors: Daniela Egas Santander, Nicolas Ninin
last modified: 12.2021
"""

def closeness_connected_components(adj, neuron_properties=[], directed=False, return_sum=True):
    """
    Compute the closeness of each connected component of more than 1 vertex
    :param matrix: shape (n,n)
    :param directed: if True compute using strong component and directed closeness
    :param return_sum: if True return only one list given by summing over all the component
    :return a single array( if return_sum=True) or a list of array of shape n,
    containting closeness of vertex in this component
    or 0 if vertex is not in the component, in any case closeness cant be zero otherwise
    """
    import numpy as np
    from sknetwork.ranking import Closeness
    from scipy.sparse.csgraph import connected_components

    matrix=adj.toarray()
    if directed:
        n_comp, comp = connected_components(matrix, directed=True, connection="strong")
    else:
        n_comp, comp = connected_components(matrix, directed=False)
        matrix = matrix + matrix.T  # we need to make the matrix symmetric

    closeness = Closeness()  # if matrix is not symmetric automatically use directed
    n = matrix.shape[0]
    all_c = []
    for i in range(n_comp):
        c = np.zeros(n)
        idx = np.where(comp == i)[0]
        sub_mat = matrix[np.ix_(idx, idx)].tocsr()
        if sub_mat.getnnz() > 0:
            c[idx] = closeness.fit_transform(sub_mat)
            all_c.append(c)
    if return_sum:
        all_c = np.array(all_c)
        return np.sum(all_c, axis=0)
    else:
        return all_c

def communicability(adj, neuron_properties):
    pass

def closeness(adj, neuron_properties, directed=False):
    """Compute closeness centrality using sknetwork on all connected components or strongly connected
    component (if directed==True)"""
    return closeness_connected_components(adj, directed=directed)

def centrality(self, sub_gids, kind="closeness", directed=False):
    """Compute a centrality of the graph. `kind` can be 'betweeness' or 'closeness'"""
    if kind == "closeness":
        return self.closeness(sub_gids, directed)
    else:
        ValueError("Kind must be 'closeness'!")
        #TODO:  Implement betweeness

def connected_components(adj,neuron_properties=[]):
    """Returns a list of the size of the connected components of the underlying undirected graph on sub_gids,
    if None, compute on the whole graph"""
    import networkx as nx
    import numpy as np

    matrix=adj.toarray()
    matrix_und = np.where((matrix+matrix.T) >= 1, 1, 0)
    # TODO: Change the code from below to scipy implementation that seems to be faster!
    G = nx.from_numpy_matrix(matrix_und)
    return [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]

def core_number(adj, neuron_properties=[]):
    """Returns k core of directed graph, where degree of a vertex is the sum of in degree and out degree"""
    # TODO: Implement directed (k,l) core and k-core of underlying undirected graph (very similar to this)
    import networkx
    G = networkx.from_numpy_matrix(adj.toarray())
    # Very inefficient (returns a dictionary!). TODO: Look for different implementation
    return networkx.algorithms.core.core_number(G)

    # TODO: Filtered simplex counts with different weights on vertices (coreness, intersection)
    #  or on edges (strength of connection).



#What to do with these since they are all about a graph and a subgraph?

"""def degree(self, sub_gids=None, kind="in"):
    """Return in/out degrees of the subgraph, if None compute on the whole graph"""
    if sub_gids is not None:
        matrix = self.subarray(self.__extract_gids__(sub_gids))
    else:
        matrix = self.array
    if kind == "in":
        return np.sum(matrix, axis=0)
    elif kind == "out":
        return np.sum(matrix, axis=1)
    else:
        ValueError("Need to specify 'in' or 'out' degree!")


def density(self, sub_gids=None):
    if sub_gids is None:
        m = self.matrix
    else:
        m = self.submatrix(sub_gids)
    return m.getnnz() / np.prod(m.shape)"""




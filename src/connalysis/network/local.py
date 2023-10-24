# Network analysis functions restricted to neighborhoods

# Author(s): D. Egas Santander, JP. Smith, J. Lazovskis
# Last modified: 10/2023

import numpy as np
import scipy.sparse as sp
def neighbours(v, matrix):
    #TODO: ADD the option of including or not the chief/center
    """Computes the neighbours of v in graph with adjacency matrix matrix

    Parameters
    ----------
    v : int
        the index of the vertex
    matrix : matrix
        the adjacency matrix of the graph

    Returns
    -------
    list
        the list of neighbours of v in matrix
    """
    neighbours = np.unique(np.concatenate((np.nonzero(matrix[v])[0],np.nonzero(np.transpose(matrix)[v])[0])))
    neighbours.sort(kind='mergesort')
    return np.concatenate((np.array([v]),neighbours))
def neighbourhood(v, matrix): #Previous name # def tribe(v, matrix):
    # TODO: ADD incoming and outgoing neighborhood
    # TODO: ADD sparse version
    # TODO: DO we need to add the chief at the start for this pipeline?
    """Computes the matrix induced by the neighbours of v in graph with adjacency matrix matrix

    Parameters
    ----------
    v : int
        the index of the vertex
    matrix : matrix
        the adjacency matrix of the graph

    Returns
    -------
    matrix
        the adjaceny matrix of the neighbourhood of v in matrix
    """
    print(matrix.getformat())
    nhbd = neighbours(v, matrix)
    return matrix[np.ix_(nhbd,nhbd)]

def local_property(adj, func, centers, kwargs):
    nbhd_values=[]
    for center in centers:
        nbhd = neighbourhood(center, adj)
        nbhd_values.append(func(nbhd, kwargs))
    return nbhd_values

#TODO: Make this work without using the dataframe
# def top_chiefs(parameter, number=50, order_by_ascending=False):
# #  In: string, integer, boolean
# # Out: list of integers
#     return df.sort_values(by=[parameter],ascending=order_by_ascending)[:number].index.values

# def top_nbhds(parameter, number=50, order_by_ascending=False, matrix=adj):
# #  In: string, integer, boolean, matrix
# # Out: list of matrices
#     top_chief_list = top_chiefs(parameter, number=number, order_by_ascending=order_by_ascending)
#     return [neighbourhood(i, matrix=matrix) for i in top_chief_list]



#TODO: Not sure what this does, and it's not used. Should we include it?
# def new_nbhds(nbhd_list, index_range):
# #  In: list of list of integers
# # Out: list of list of integers
#     new_list = []
#     choice_vector = range(index_range)
#     for nbhd in nbhd_list:
#         new_neighbours = np.random.choice(choice_vector, size=len(nbhd)-1, replace=False)
#         while nbhd[0] in new_neighbours:
#             new_neighbours = np.random.choice(choice_vector, size=len(nbhd)-1, replace=False)
#         new_list.append(np.hstack((nbhd[0], new_neighbours)))
#     return new_list

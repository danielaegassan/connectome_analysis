# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Network analysis functions restricted to neighborhoods

# Author(s): D. Egas Santander, M. Reimann, JP. Smith, J. Lazovskis
# Last modified: 10/2023

import numpy as np
import scipy.sparse as sp
import pandas as pd


def neighborhood_indices(M, pre=True, post=True, all_nodes=True, centers=None):
    """Computes the indices of the neighbors of the nodes listed in centers

        Parameters
        ----------
        M : sparse matrix or 2d array
            The adjacency matrix of the graph
        pre : bool
            If ``True`` compute the nodes mapping to the nodes in centers (the in-neighbors of the centers)
        post : bool
            If ``True`` compute the nodes that the centers map to (the out-neighbors of the centers)
        all_nodes : bool
            If ``True`` compute the neighbors of all nodes in M, if ``False`` compute only the neighbors of the nodes
            listed in centers
        centers : 1d-array
            The indices of the nodes for which the neighbors need to be computed.  This entry is ignored if
            all_nodes is ``True`` and required if all_nodes is ``False``.

        Returns
        -------
        data frame
            indices: range from 0 to ``M.shape[0]`` if all_nodes is set to ``True``, otherwise centers.

            values: the neighbors of each center in the indices.

        Raises
        ------
        AssertionError
            If the matrix M is not square
        AssertionError
            If both pre and post are ``False``
        AssertionError
            If all_nodes is ``False`` but centers are not provided

    """
    assert M.shape[0] == M.shape[1], "The matrix is not square"
    assert np.logical_or(pre, post), "At least one of the pre/post parameters must be True"
    if all_nodes: centers = np.arange(M.shape[0])
    assert centers is not None, "If all_nodes == False and array of centers must be provided"

    M = M.tocoo() if sp.issparse(M) else sp.coo_matrix(M)

    base_df = pd.DataFrame({"row": M.row, "col": M.col})
    idxx = pd.Index(centers, name="center")
    nb_df = pd.Series([[]] * centers.shape[0], index=idxx, name="neighbors")
    # Restriction to requested centers
    if not all_nodes: res_df = base_df.apply(lambda _x: np.isin(_x, centers))

    if post:
        df = base_df[res_df['row']] if not all_nodes else base_df
        new_df = df.rename(columns={"row": "center", "col": "neighbors"}).groupby("center")["neighbors"].apply(list)
        nb_df = nb_df.combine(new_df, np.union1d, fill_value=[])
    if pre:
        df = base_df[res_df['col']] if not all_nodes else base_df
        new_df = df.rename(columns={"col": "center", "row": "neighbors"}).groupby("center")["neighbors"].apply(list)
        nb_df = nb_df.combine(new_df, np.union1d, fill_value=[])
    return nb_df.apply(lambda _x: _x.astype(int))


def submat_at_ind(M, ind):
    """Computes the submatrix of M on the nondes indexed by ind

    Parameters
    ----------
    M : matrix
        the adjacency matrix of the graph
    ind : 1d-array
        the indices on which to slice the matrix M

    Returns
    -------
    matrix
        the adjaceny matrix of the submatrix of M on the nodes in ind
    """
    if sp.issparse(M): M=M.tocsr()
    return  M[ind][:, ind]

def neighborhood(adj, v, pre=True, post=True, include_center=True, return_neighbors=False):
    """Gets the neighborhood of v in adj
            Parameters
            ----------
            adj : sparse matrix or 2d array
                The adjacency matrix of the graph
            pre : bool
                If ``True`` compute the submatrix on the nodes mapping to v (the in-neighbors of v)
            post : bool
                If ``True`` compute the submatrix on the nodes mapping to v (the out-neighbors of v)
            include_center : bool
                If ``True`` include v in the neighborhood
            return_neighbors : bool
                If ``True`` also return the indices in adj of the neighbors of v

            Returns
            -------
            matrix (sparse if adj is sparse)
                If pre = post = ``True`` it returns the full neighbohood of v

                If pre = ``True`` and post = ``False``it returns the in-neighborhood of v

                If pre = ``False`` and post = ``True``it returns the out-neighborhood of v

                If include_center = ``True``, then v is the first indexed node in the matrix,
                else it is excluded
        """
    nb_df=neighborhood_indices(adj, pre=pre, post=post, all_nodes=False, centers=np.array([v]))
    if sp.issparse(adj): adj=adj.tocsr()
    nb_ind = nb_df.loc[v]
    if include_center:
        nb_ind = np.append(v, nb_ind)
    if return_neighbors:
        return submat_at_ind(adj, nb_ind), nb_ind
    else:
        return submat_at_ind(adj, nb_ind)

def neighborhood_of_set_indices(M, node_set, pre=True, post=True):
    """Computes the indices of the neighbors of the nodes in node_set

            Parameters
            ----------
            M : sparse matrix or 2d array
                The adjacency matrix of the graph
            pre : bool
                If ``True`` compute the nodes mapping to the nodes in node_set (the in-neighbors of the node_set)
            post : bool
                If ``True`` compute the nodes that the centers map to node_set (the out-neighbors of the node_set)
            node_set : 1d-array
                The indices of the nodes for which the neighbors need to be computed.

            Returns
            -------
            array
                indices of the neighbhors of node_set

            Raises
            ------
            AssertionError
                If both pre and post are ``False``
    """
    assert np.logical_or(pre, post), "At least one of the pre/post parameters must be True"
    return np.unique(np.concatenate(neighborhood_indices(M,pre=pre,post=post,all_nodes=False,centers=node_set).values))

def neighborhood_of_set(M, node_set, pre=True, post=True, include_centers=True, return_neighbors=False):
    """Gets the neighborhood of the nodes in node_set
            Parameters
            ----------
            M : sparse matrix or 2d array
                The adjacency matrix of the graph
            node_set : array
                The indices of the nodes of which the neighborhood will be computed
            pre : bool
                If ``True`` compute the submatrix on the nodes mapping to node_set (the in-neighbors of node_set)
            post : bool
                If ``True`` compute the submatrix on the nodes that node_set maps to (the out-neighbors of node_set)
            include_center : bool
                If ``True`` include node_set in the graph
            return_neighbors : bool
                If ``True`` also return the indices in M of the neighbors of node_set

            Returns
            -------
            matrix (sparse if M is sparse)
                If pre = post = ``True`` it returns the full neighbohood of node_set

                If pre = ``True`` and post = ``False``it returns the in-neighborhood of node_set

                If pre = ``False`` and post = ``True``it returns the out-neighborhood of node_set
    """
    nodes=neighborhood_of_set_indices(M, node_set, pre, post)
    if include_centers:
        nodes=np.unique(np.concatenate([node_set, nodesA]))
    if isinstance(M, sp.coo_matrix): M=M.tocsr()
    # Slicing in different ways depending on matrix type
    if isinstance(M, np.ndarray): nbd= M[np.ix_(nodes, nodes)]
    if isinstance(M, sp.csr_matrix): nbd= M[nodes].tocsc()[:, nodes]
    if isinstance(M, sp.csc_matrix): nbd= M[:,nodes].tocsr()[nodes]
    if return_neighbors: return nbd, nodes
    else: return nbd

def property_at_neighborhoods(adj, func, pre=True, post=True,include_center=True,
                             all_nodes=True, centers=None, **kwargs):
    """Computes the property func on the neighborhoods of the centers within adj

            Parameters
            ----------
            adj : sparse matrix or 2d array
                The adjacency matrix of the graph
            func : function
                Function computing a network theoretic property e.g., degree or simplex counts
            pre : bool
                If ``True`` include the nodes mapping to the nodes in centers
            post : bool
                If ``True`` include the nodes that the centers map to
            include_center : bool
                If ``True`` include the centers
            all_nodes : bool
                If ``True`` compute func on the neighborhoods of all nodes in adj,

                If ``False`` compute it only the neighborhoods of the nodes listed in centers
            centers : 1d-array
                The indices of the nodes to consider.  This entry is ignored if
                all_nodes is ``True`` and required if all_nodes is ``False``.

            Returns
            -------
            dict
                keys: range from 0 to ``M.shape[0]`` if all_nodes is set to ``True``, otherwise centers

                values: the output of ``func`` in each neighborhood of the nodes in centers
    """
    nb_df=neighborhood_indices(adj, pre=pre, post=post, all_nodes=all_nodes, centers=centers)
    if sp.issparse(adj): adj=adj.tocsr()# TODO: Add warning of format!
    nbhd_values={}
    for center in nb_df.index:
        nb_ind = nb_df.loc[center]
        if include_center: nb_ind=np.append(center, nb_ind)
        nbhd=submat_at_ind(adj, nb_ind)
        #nbhd_values.loc[center]=func(nbhd, kwargs)
        nbhd_values[center]=func(nbhd, **kwargs)
    return nbhd_values


def properties_at_neighborhoods(adj, func_config, pre=True, post=True, include_center=True,
                                all_nodes=True, centers=None):
    """Computes the properties in func_config on the neighborhoods of the centers within adj

            Parameters
            ----------
            adj : sparse matrix or 2d array
                The adjacency matrix of the graph
            func_config : dict
                Configuration dictionary of functions to be computed on neihgborhoods
            pre : bool
                If ``True`` include the nodes mapping to the nodes in centers
            post : bool
                If ``True`` include the nodes that the centers map to
            include_center : bool
                If ``True`` include the centers
            all_nodes : bool
                If ``True`` compute func on the neighborhoods of all nodes in adj,

                If ``False`` compute it only the neighborhoods of the nodes listed in centers
            centers : 1d-array
                The indices of the nodes to consider.  This entry is ignored if
                all_nodes is ``True`` and required if all_nodes is ``False``.

            Returns
            -------
            dict
                keys: keys of func_config

                values: dict with

                    keys: range from 0 to ``M.shape[0]`` if all_nodes is set to ``True``, otherwise centers

                    values: the output of ``func`` in each neighborhood of the nodes in centers
    """
    nb_df = neighborhood_indices(adj, pre=pre, post=post, all_nodes=all_nodes, centers=centers)
    if sp.issparse(adj): adj = adj.tocsr()  # TODO: Add warning of format?
    nbhd_values = {key: {} for key in func_config.keys()}
    for center in nb_df.index:
        nb_ind = nb_df.loc[center]
        if include_center: nb_ind = np.append(center, nb_ind)
        nbhd = submat_at_ind(adj, nb_ind)
        for key in func_config.keys():
            nbhd_values[key][center] = func_config[key]['function'](nbhd, **func_config[key]['kwargs'])
    return nbhd_values

#### OLD CODE BELOW

def neighbours(v, matrix):
    #TODO: Delete redundant
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
    # TODO: Needs to be deleted
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
    #print(matrix.getformat())
    nhbd = neighbours(v, matrix)
    return matrix[np.ix_(nhbd,nhbd)]
#
# def local_property(adj, func, centers, kwargs):
#     nbhd_values=[]
#     for center in centers:
#         nbhd = neighbourhood(center, adj)
#         nbhd_values.append(func(nbhd, kwargs))
#     return nbhd_values

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


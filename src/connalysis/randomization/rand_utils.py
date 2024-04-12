# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import scipy.sparse as sp

from typing import List, Union
from connalysis.network.topology import rc_submatrix


def subset_matrix(sparse_matrix: sp.coo_matrix, selection: np.ndarray) -> sp.coo_matrix:
    """
    Returns a subset of the matrix given a selection of edges.

    :param sparse_matrix: Sparse input matrix in coo format.
    :type: sp.coo_matrix
    :param selection: Array with indices corresponding to edges to select from coo.
    :type: np.ndarray

    :return subset_matrix: Matrix containing edges specified by selection.
    :rtype: sp.coo_matrix
    """
    return sp.coo_matrix((np.ones(len(selection)), (sparse_matrix.row[selection], sparse_matrix.col[selection])),
                         shape=(sparse_matrix.shape))


def subsampled_matrix(sparse_matrix: sp.coo_matrix, n: int, generator: np.random.Generator) -> sp.coo_matrix:
    """
    Returns a random subsample of the matrix of a given size.

    :param sparse_matrix: Sparse input matrix in coo format.
    :type: sp.coo_matrix
    :param n: Size of the subsample.
    :type: int
    :param generator: Numpy generator to properly use randomness.
    :type: np.random.Generator

    :return subsample_matrix: Matrix containing n edges.
    :rtype: sp.coo_matrix
    """
    selection = generator.choice(len(sparse_matrix.col), replace=False, size=n)
    return subset_matrix(sparse_matrix, selection)


def non_overlapping_subsampled_matrices(sparse_matrix: sp.coo_matrix, ns: List[int],
                                        generator: np.random.Generator) -> List[sp.coo_matrix]:
    """Function to retrieve an arbitrary amount of non-overlapping subsampled matrices of given edge count.

    :param sparse_matrix: Sparse input matrix in coo format.
    :type: sp.coo_matrix
    :param ns: Sizes of the subsamples.
    :type: List[int]
    :param generator: Numpy generator to properly use randomness.
    :type: np.random.Generator

    :return subsample_matrices: Matrices containing ns edges.
    :rtype: List[sp.coo_matrix]
    """
    selection = generator.choice(len(sparse_matrix.col), replace=False, size=sum(ns))  # Samples some edges
    ls = np.append(0, np.cumsum(ns))
    return [subset_matrix(sparse_matrix, selection[ls[i]:ls[i + 1]]) for i in range(len(ls) - 1)]


def half_matrix(sparse_matrix: sp.coo_matrix, generator: np.random.Generator) -> List[sp.coo_matrix]:
    """
    Split the matrix edges exactly in half. Returns one matrix per half.

    :param sparse_matrix: Sparse input matrix in coo format.
    :type: sp.coo_matrix
    :param generator: Numpy generator to properly use randomness.
    :type: np.random.Generator

    :return subsample_matrices: List of two matrices, one per half.
    :rtype: List[sp.coo_matrix]
    """
    permutation = generator.permutation(len(sparse_matrix.col))
    return subset_matrix(sparse_matrix, permutation[:int(len(permutation) / 2)]), subset_matrix(sparse_matrix,
                                                                                                permutation[int(len(
                                                                                                    permutation) / 2):])


def adjust_bidirectional_connections(sparse_matrix: sp.csc_matrix, bedges_to_add: int,
                                     generator: np.random.Generator) -> sp.csc_matrix:
    """
    Turn a fixed amount of directional connections into bidirectional connections.

    :param sparse_matrix: Sparse input matrix in coo format.
    :type: sp.coo_matrix
    :param bedges_to_add: Number of directional edges to transform into bedges.
    :type: int
    :param generator: Numpy generator to properly use randomness.
    :type: np.random.Generator

    :return adjusted_matrix: Matrix with adjusted bedges
    :rtype: sp.csc_matrix
    """
    bedge = rc_submatrix(sparse_matrix)
    dedge = sparse_matrix - bedge
    del bedge
    dedge_coo = dedge.tocoo(copy=False)  # Easier to subsample
    dedges_to_bedge, dedges_to_remove = non_overlapping_subsampled_matrices(dedge_coo,
                                                                            [bedges_to_add, bedges_to_add],
                                                                            generator)  # Coo matrices are auto casted before sum
    return sparse_matrix + dedges_to_bedge.T - dedges_to_remove


def add_bidirectional_connections(sparse_matrix: sp.csc_matrix, bedges_to_add: int,
                                  generator: np.random.Generator) -> sp.csc_matrix:
    """
    Turn a fixed amount of directional connections into bidirectional connections.

    :param sparse_matrix: Sparse input matrix in coo format.
    :type: sp.coo_matrix
    :param bedges_to_add: Number bedges to add.
    :type: int
    :param generator: Numpy generator to properly use randomness.
    :type: np.random.Generator

    :return adjusted_matrix: Matrix with adjusted bedges.
    :rtype: sp.csc_matrix
    """
    bedge = rc_submatrix(sparse_matrix)
    dedge = sparse_matrix - bedge
    del bedge
    dedge = dedge.tocoo(copy=False)  # Easier to subsample
    dedge = subsampled_matrix(dedge, bedges_to_add, generator)
    return sparse_matrix + dedge.T

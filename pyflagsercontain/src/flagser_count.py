"""Implementation of the python API for the cell count of the flagser C++ library."""

import numpy as np
from pyflagsercontain import compute_cell_count

def flagser_count(adjacency_matrix):
    return compute_cell_count(adjacency_matrix.shape[0], np.transpose(np.array(np.nonzero(adjacency_matrix))))

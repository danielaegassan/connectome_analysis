#TODO: Design and add small topological tests
def test_flagser():
    from connalysis.network import simplex_counts
    from scipy import sparse
    A = sparse.random(100, 100, density=0.1)
    A.setdiag(0)
    A.eliminate_zeros()
    simplex_counts(A)

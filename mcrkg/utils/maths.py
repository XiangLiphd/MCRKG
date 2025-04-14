import numpy as np
from scipy import sparse


def normalize_sparse_adj(A, sparse_type='coo'):
    in_degree = np.array(A.sum(1)).reshape(-1)
    in_degree[in_degree == 0] = 1e-5
    d_inv = sparse.diags(1 / in_degree)
    A = getattr(d_inv.dot(A), 'to' + sparse_type)()
    return A

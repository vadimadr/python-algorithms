import numpy as np


def matrix_product(lhs, rhs):
    M, N = lhs.shape
    N, Q = rhs.shape
    res = np.zeros((M, Q), lhs.dtype)

    for i in range(N):
        for j in range(M):
            for k in range(Q):
                res[j, k] += lhs[j, i] * rhs[i, k]
    return res

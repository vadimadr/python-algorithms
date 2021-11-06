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


def LUP_decomposition(A, tol=1e-6):
    """Factorize matrix A = PLU
    using Gaussian Elimination in O(n^3)
    L - is lower triangular, U - upper triangular, and P - is permutation matrix.
    """
    A = A.copy()
    n, m = A.shape
    p = np.arange(n)
    for jc in range(min(n, m)):
        # select pivot: maximize |U_{j,j}|
        max_j = jc
        for ir in range(jc + 1, n):
            if abs(A[p[ir], jc]) > abs(A[p[max_j], jc]):
                max_j = ir

        # swap jc and max_j rows
        p[max_j], p[jc] = p[jc], p[max_j]

        if abs(A[p[jc], jc]) < tol:
            # skip current row
            # matrix is degenerate!
            continue

        for ir in range(jc + 1, n):
            A[p[ir], jc] = A[p[ir], jc] / A[p[jc], jc]
            for k in range(jc + 1, m):
                A[p[ir], k] -= A[p[ir], jc] * A[p[jc], k]

    # extract L and U matrices (optional step: can use matrix A itself)
    L, U = np.eye(n), np.zeros((n, m))
    for i in range(n):
        for j in range(min(i, m)):
            L[i, j] = A[p[i], j]
        for j in range(i, m):
            U[i, j] = A[p[i], j]

    return L, U, p

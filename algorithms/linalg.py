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


def LUP_decomposition(A, tol=1e-6, count_perm=False):
    """Factorize matrix A = PLU
    using Gaussian Elimination in O(n^3)
    L - is lower triangular, U - upper triangular, and P - is permutation matrix.
    """
    A = A.copy()
    n, m = A.shape
    p = np.arange(n)
    n_perm = 0
    for jc in range(min(n, m)):
        # select pivot: maximize |U_{j,j}|
        max_j = jc
        for ir in range(jc + 1, n):
            if abs(A[p[ir], jc]) > abs(A[p[max_j], jc]):
                max_j = ir

        # swap jc and max_j rows
        if p[max_j] != p[jc]:
            p[max_j], p[jc] = p[jc], p[max_j]
            n_perm += 1

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

    if count_perm:
        return L, U, p, n_perm
    return L, U, p


def det(X):
    _, u, _, np = LUP_decomposition(X, count_perm=True)
    res = 1
    for i in range(X.shape[0]):
        res *= u[i, i]
    return res * (-1) ** np


def linsolve(A, B, tol=1e-6):
    """Solve AX = B

    If system is underparametrized, then find some root
    """
    assert A.shape[0] == B.shape[0], "Incorrect dimensions of equation"
    assert A.shape[0] <= A.shape[1], "equation is over-parametrized"
    B = B.reshape((B.shape[0], -1))  # make B always 2 dim
    _, d = A.shape
    n, m = B.shape

    l, u, p = LUP_decomposition(A)
    x = np.zeros((d, m))
    for k in range(m):

        # forward substitution (solve Ly = b)
        for i in range(min(d, n)):
            x[i, k] = B[p[i], k]
            for j in range(min(d, i)):
                x[i, k] -= l[i, j] * x[j, k]

        # backward substitution (solve Ux = y)
        for i in reversed(range(min(d, n))):
            if abs(u[i, i]) < tol:
                # system is degenerate
                return False
            for j in range(i + 1, d):
                x[i, k] -= u[i, j] * x[j, k]
            x[i, k] /= u[i, i]

    return x


def matrix_invert(X):
    assert X.shape[0] == X.shape[1]
    n = X.shape[0]
    P = np.eye(n)
    return linsolve(X, P)


def matrix_rank(X):
    _, U, _ = LUP_decomposition(X)
    r = 0
    for i in range(min(X.shape[0], X.shape[1])):
        if abs(U[i, i]) >= 1e-6:
            r += 1
    return r


def QR_decomposition(A):
    """Perform orthogonalization of A using
    (stable modification of) Gram–Schmidt process (MGS)

    Decomposes A = Q * R into product of orthonormal matrix Q
    and upper-triangular matrix R
    """
    _, m = A.shape
    Q, R = A.copy(), np.eye(m)
    for j in range(m):
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], Q[:, j])
            Q[:, j] -= R[i, j] * Q[:, i]
        R[j, j] = np.sqrt(np.dot(Q[:, j], Q[:, j]))
        if np.abs(R[j, j]) < 1e-6:
            continue
        Q[:, j] /= R[j, j]
    return Q, R


def QR_algorithm_eig(A, max_iters=1000, max_rel_err=1e-6):
    """Finds eigenvectors and corresponding eigenvalues
    using QR algorithm.
    """
    eig_w = np.zeros(A.shape[0])
    eig_v = np.eye(A.shape[0])
    for _ in range(max_iters):
        Q, R = QR_decomposition(A)
        A = R @ Q

        delta = np.abs((np.diag(A) / eig_w - 1))
        print(delta)

        eig_v = eig_v @ Q
        eig_w = np.diag(A)
        if np.min(delta) < max_rel_err:
            break
    return eig_w, eig_v

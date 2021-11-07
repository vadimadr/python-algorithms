import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite, floats, integers, one_of

from algorithms.linalg import (
    LUP_decomposition,
    det,
    linsolve,
    matrix_invert,
    matrix_product,
    matrix_rank,
    QR_decomposition,
    QR_algorithm_eig,
)


def test_matrix_product():
    a = np.random.random((15, 20))
    b = np.random.random((20, 10))
    c = a.dot(b)
    d = matrix_product(a, b)
    assert np.allclose(c, d)


@composite
def st_floats(draw, mag=100, eps=1e-6):
    """Floats with moderate values"""
    x = draw(floats(allow_nan=False, max_value=mag, min_value=-mag))
    if abs(x) < eps:
        return 0
    return x


@composite
def st_matrix(draw, maxn=30, mag=100, square=False, min_det=None):
    n, m = draw(integers(1, maxn)), draw(integers(1, maxn))
    if square:
        m = n

    elements = one_of(
        integers(min_value=-mag, max_value=mag),
        st_floats(),
    )

    A = draw(arrays(np.float64, shape=(n, m), elements=elements))

    if min_det is not None:
        assume(abs(np.linalg.det(A)) > min_det)

    return A


class TestLUP:
    def check_lup(self, x):
        x = np.array(x)
        l, u, p = LUP_decomposition(x)
        n, m = x.shape
        assert l.shape == (n, n)
        assert u.shape == (n, m)
        np.testing.assert_allclose(l @ u, x[p], atol=1e-4)

        # check L is lower triangular
        # check U is upper triangular

    def test_gauss(self):
        # must work with every permutation of rows
        self.check_lup([[0.0], [0.0], [0.0]])
        self.check_lup([[1, 2, 3], [7.0, 8, 9], [4, 5, 6]])
        self.check_lup([[1, 2, 3], [4, 5, 6], [7.0, 8, 9]])
        self.check_lup([[4, 5, 6], [7.0, 8, 9], [1, 2, 3]])
        self.check_lup([[4, 5, 6], [1, 2, 3], [7.0, 8, 9]])
        self.check_lup([[7.0, 8, 9], [4, 5, 6], [1, 2, 3]])
        self.check_lup([[7.0, 8, 9], [1, 2, 3], [4, 5, 6]])
        self.check_lup(np.array([[1, 2], [3, 4]], np.float32))
        self.check_lup([[1.0], [1.0]])
        self.check_lup([[1.0, 1.0]])

    @given(st_matrix())
    def test_lup(self, x):
        self.check_lup(x)


@given(st_matrix(square=True, maxn=15, mag=3))
def test_det(m):
    assert pytest.approx(np.linalg.det(m), rel=1e-2) == det(m)


@composite
def st_linear_equations(draw, maxn=30, mag=3):
    A = draw(st_matrix(maxn=maxn, mag=mag))
    assume(A.shape[0] <= A.shape[1])

    s_el = st_floats(mag=mag)
    b = draw(arrays(np.float64, shape=(A.shape[0], 1), elements=s_el))

    return A, b


@given(st_linear_equations())
def test_linsolve(data):
    A, b = data
    x = linsolve(A, b)
    if x is False:
        return
    b0 = A @ x
    np.testing.assert_allclose(b0, b, atol=1e-4)


@composite
def st_invertible_matrix(draw, maxn=30, mag=3):
    A = draw(st_matrix(maxn=maxn, mag=mag, square=True, min_det=0.01))
    return A


@given(st_invertible_matrix())
def test_matrix_invert(X):
    X_inv = matrix_invert(X)
    I = np.eye(X.shape[0])
    np.testing.assert_allclose(X_inv @ X, I, atol=1e-5)


@given(st_matrix())
def test_matrix_rank(X):
    assert matrix_rank(X) == np.linalg.matrix_rank(X)


@given(st_invertible_matrix())
def test_QR_decomposition(x):
    q, r = QR_decomposition(x)

    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[0]), atol=1e-7)
    np.testing.assert_allclose(q @ r, x, atol=1e-7)


@pytest.mark.parametrize(
    "x",
    [
        np.array([[1.0, 0.0], [1.0, 1.0]]),
        np.array([[1, 0.5], [0.5, 1]]),
    ],
)
def test_eigenvals(x):
    w, v = QR_algorithm_eig(x)

    for i in range(x.shape[0]):
        np.testing.assert_allclose(w[i] * v[:, i], x @ v[:, i], atol=1e-4)

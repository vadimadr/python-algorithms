import numpy as np

from algorithms.linalg import matrix_product, LUP_decomposition
from hypothesis.strategies import composite, integers, one_of, floats
from hypothesis.extra.numpy import arrays
from hypothesis import given


def test_matrix_product():
    a = np.random.random((15, 20))
    b = np.random.random((20, 10))
    c = a.dot(b)
    d = matrix_product(a, b)
    assert np.allclose(c, d)


@composite
def st_floats(draw, mod=100, eps=1e6):
    """Floats with moderate values"""
    x = draw(floats(allow_nan=False, max_value=mod, min_value=-mod, width=32))
    if abs(x) < eps:
        return 0
    return x


@composite
def st_matrix(draw, maxn=30):
    n, m = draw(integers(1, maxn)), draw(integers(1, maxn))

    elements = one_of(
        integers(min_value=100, max_value=100),
        st_floats(),
    )

    A = draw(arrays(np.float32, shape=(n, m), elements=elements))
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

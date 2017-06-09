import numpy as np

from algorithms.linalg import matrix_product


def test_matrix_product():
    a = np.random.random((15, 20))
    b = np.random.random((20, 10))
    c = a.dot(b)
    d = matrix_product(a, b)
    assert np.allclose(c, d)

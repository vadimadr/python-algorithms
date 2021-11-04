from math import cos, exp, sin

from algorithms.numerical_analysis import newton_raphson_root, ternary_search
from tests.utils import float_eq


def test_newton_root():
    f1 = lambda x: cos(x) - x ** 3
    df1 = lambda x: -sin(x) - 3 * x ** 2
    assert float_eq(newton_raphson_root(f1, 0, 0.5, df1), 0.865474033102)
    assert float_eq(newton_raphson_root(f1, 0, 0.5, None), 0.865474033102)

    f2 = lambda x: x ** 5 - 2 * x
    df2 = lambda x: 5 * x ** 4 - 2
    assert float_eq(newton_raphson_root(f2, -3, 0, df2), -1.423605848552331)
    assert float_eq(newton_raphson_root(f2, -3, 0, None), -1.423605848552331)


def test_optimization_ternary():
    f1 = lambda x: (x - 3) ** 2 + 2.4
    assert float_eq(ternary_search(f1, 0, 10), 3)

    f2 = lambda x: exp(-((x - 1.5) ** 2))
    assert float_eq(ternary_search(f2, -5, 5, min_=False), 1.5)

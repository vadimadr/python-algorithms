from algorithms.geometry import Vec2, orthogonal, Vec3, l2
from tests.utils import float_eq


def test_dot2():
    x0 = Vec2(4, 2)
    y0 = Vec2(-1, 2)
    y1 = Vec2(-1, 0)
    assert orthogonal(x0, y0)
    assert not orthogonal(x0, y1)

    x1 = Vec2(1.5, 3.1)
    y2 = Vec2(1, -15 / 31)
    y3 = Vec2(3, 2.5)
    assert orthogonal(x1, y2)
    assert not orthogonal(x1, y3)


def test_dot3():
    x0 = Vec3(1, 2, 3)
    y0 = Vec3(0, 3, -2)
    y1 = Vec3(0, 2, -1)
    assert orthogonal(x0, y0)
    assert not orthogonal(x0, y1)

    x1 = Vec3(1.3, 2.7, 0.1)
    y2 = Vec3(1.5, -5 / 6, 3)
    y3 = Vec3(1.5, 0.6, .3)
    assert orthogonal(x1, y2)
    assert not orthogonal(x1, y3)


def test_l2():
    x0 = Vec2(1.5, 4)
    assert float_eq(l2(x0), 4.2720018)

    x1 = Vec3(1.3, 7.2, 3.4)
    assert float_eq(l2(x1), 8.0678373806)

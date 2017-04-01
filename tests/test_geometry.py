from hypothesis import given
from hypothesis.strategies import floats, tuples, integers
from math import hypot, isclose

from algorithms.geometry import Vec2, orthogonal, Vec3, l2, orientation, \
    vec2_prod, vec3_prod, line
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


def test_triple_orientation():
    p0 = Vec2(-3, 2)
    p1 = Vec2(-5, 3)
    p2 = Vec2(-1, -1)

    assert orientation(p0, p1, p2) > 0
    assert orientation(p1, p0, p2) < 0
    assert orientation(p0, p1, p1) == 0

    e0 = Vec3(1, 0, 0)
    e1 = Vec3(0, 1, 0)
    e2 = Vec3(0, 0, 1)
    assert orientation(e0, e1, e2) > 0
    assert orientation(e0, e2, e1) < 0


def test_vec2_prod():
    a = Vec2(4, 0)
    b = Vec2(2, 2)
    assert vec2_prod(a, b) == 8


def test_vec3_prod():
    a = Vec3(1, 2.5, 3)
    b = Vec3(.7, 1.4, 0)
    c = vec3_prod(a, b)
    assert orthogonal(a, c)
    assert orthogonal(b, c)
    assert orientation(a, b, c) > 0


reals = floats(min_value=-10000, max_value=10000)
ints = integers(-10000, 10000)


@given(tuples(reals, reals, reals, reals))
def test_line(t):
    px, py, qx, qy = t
    p = Vec2(px, py)
    q = Vec2(qx, qy)
    a, b, c = line(p, q)
    assert isclose(a * px + b * py + c, 0, abs_tol=1e-6)


@given(tuples(ints, ints, ints, ints))
def test_line(t):
    px, py, qx, qy = t
    p = Vec2(px, py)
    q = Vec2(qx, qy)
    a, b, c = line(p, q, 'gcd')
    assert a * px + b * py + c == 0

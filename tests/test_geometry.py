from hypothesis import given
from hypothesis.strategies import floats, tuples, integers
from math import hypot, isclose, sqrt

from algorithms.geometry import Vec2, orthogonal, Vec3, l2, orientation, \
    vec2_prod, vec3_prod, line, Line2, distance_to_line, line_projection, \
    line_parallel, line_same, line_intersect, segment_intersection, \
    segment_union_measure, segment_cover, polygon_area, points_inside, \
    convex_polygon, circle_line_intersection
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
def test_line_int(t):
    px, py, qx, qy = t
    p = Vec2(px, py)
    q = Vec2(qx, qy)
    a, b, c = line(p, q, 'gcd')
    assert isinstance(a, int)
    assert isinstance(b, int)
    assert isinstance(c, int)
    assert a * px + b * py + c == 0


def test_distance_to_line():
    l = Line2(2.3, 1.4, 4)
    p = Vec2(0.5, -2)
    assert isclose(distance_to_line(l, p), 0.872768, abs_tol=1e-6)


def test_line_projection():
    l = Line2(-2, 6, -30)  # y = (1/3)*x + 5
    p = Vec2(10, 5)
    prp = line_projection(l, p)
    assert abs(hypot(prp.x - p.x, prp.y - p.y) - sqrt(10)) < 1e-6
    assert abs(prp.x - 9.0) < 1e-6 and abs(prp.y - 8.0) < 1e-6


def test_line_parallel():
    l1 = Line2(1, 1, 3)
    l2 = Line2(2, 2, 8)
    l3 = Line2(1, 2, 3)
    assert line_parallel(l1, l2)
    assert not line_parallel(l1, l3)


def test_line_same():
    l1 = Line2(1, 1, 3)
    l2 = Line2(2, 2, 6)
    l3 = Line2(2, 2, 8)
    assert line_same(l1, l2)
    assert not line_same(l1, l3)


def test_line_intersect():
    l1 = Line2(0, 1, 2)
    l2 = Line2(1, 0, 3)
    intersect = line_intersect(l1, l2)
    assert abs(intersect.x - -3) < 1e-6 and abs(intersect.y - -2) < 1e-6

    l1 = Line2(2, 2, 6)
    l2 = Line2(2, 2, 8)

    assert not line_intersect(l1, l2)


def test_segment_intersect():
    # normal
    a, b, c, d = Vec2(1, 1), Vec2(4, 2), Vec2(5, -2), Vec2(2, 2)
    assert segment_intersection(a, b, c, d)
    # not cross
    a, b, c, d = Vec2(1, 1), Vec2(4, 2), Vec2(5, -2), Vec2(2, 1)
    assert not segment_intersection(a, b, c, d)
    # vertical
    a, b, c, d = Vec2(1, 1), Vec2(4, 2), Vec2(3, 0), Vec2(3, 5)
    assert segment_intersection(a, b, c, d)
    # horiz
    a, b, c, d = Vec2(1, 1), Vec2(4, 2), Vec2(1, 1), Vec2(6, 1)
    assert segment_intersection(a, b, c, d)
    # one point
    a, b, c, d = Vec2(1, 1), Vec2(4, 2), Vec2(1, 1), Vec2(1, 1)
    assert segment_intersection(a, b, c, d)
    # two point
    a, b, c, d = Vec2(1, 1), Vec2(1, 1), Vec2(1, 1), Vec2(1, 1)
    assert segment_intersection(a, b, c, d)


def test_segment_union_measure():
    xs = [(1, 2), (1.5, 3)]
    assert abs(segment_union_measure(xs) - 2) < 1e-6


def test_segment_cover():
    xs = [(3, 7), (1, 8), (5, 6), (2, 4)]
    assert segment_cover(xs) == [4, 6]
    ys = [(0, 4.5)]
    assert segment_cover(xs, ys) == [6]


def test_polygon_area():
    xs = [Vec2(1, 2), Vec2(3, 5), Vec2(6, 5), Vec2(3, 7), Vec2(-2, 4)]
    s = polygon_area(xs)
    assert abs(s - 14.5) < 1e-6


def test_points_inside():
    xs = [Vec2(0, 0), Vec2(3, 0), Vec2(3, 3), Vec2(0, 3)]
    assert points_inside(xs) == 4
    xs = [Vec2(1, 2), Vec2(3, 5), Vec2(6, 5), Vec2(3, 7), Vec2(-2, 4)]
    assert points_inside(xs) == 11


def test_convex_polygon():
    xs = [Vec2(1, 1), Vec2(2, 2), Vec2(3, 1), Vec2(3, 3), Vec2(1, 3)]
    assert not convex_polygon(xs)
    xs = [Vec2(2, 1), Vec2(3, 1), Vec2(4, 2), Vec2(3, 3), Vec2(2, 3),
        Vec2(1, 2)]
    assert convex_polygon(xs)


def test_circle_line_intersect():
    p = Vec2(4, 5)
    r = 3
    l1 = line(Vec2(10, 6), Vec2(3, 12))
    l2 = line(Vec2(6, 1), Vec2(0.5, 8))
    l3 = line(Vec2(1, 0), Vec2(1, 2))

    n1, p1 = circle_line_intersection(p, r, l1)
    n2, p2 = circle_line_intersection(p, r, l2)
    n3, p3 = circle_line_intersection(p, r, l3)
    assert n1 == 0
    assert n2 == 2
    assert n3 == 1

    assert abs((p3[0].x - p.x)**2 + (p3[0].y - p.y)**2 - r*r) < 1e-6
    assert abs((p2[0].x - p.x)**2 + (p2[0].y - p.y)**2 - r*r) < 1e-6
    assert abs((p2[1].x - p.x)**2 + (p2[1].y - p.y)**2 - r*r) < 1e-6

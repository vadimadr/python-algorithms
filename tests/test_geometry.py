from math import hypot, isclose, sqrt

from hypothesis import given
from hypothesis.strategies import floats, integers, tuples

from algorithms.geometry import (
    Line2,
    Vec2,
    Vec3,
    angle_cmp,
    circle_intersection,
    circle_line_intersection,
    convex_hull,
    convex_polygon,
    distance_to_line,
    l2,
    line,
    line_intersect,
    line_parallel,
    line_projection,
    line_same,
    orientation,
    orthogonal,
    point_inside_convex_polygon,
    points_inside,
    polygon_area,
    segment_cover,
    segment_intersection,
    segment_union_measure,
    vec2_prod,
    vec3_prod,
)
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
    y3 = Vec3(1.5, 0.6, 0.3)
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
    b = Vec3(0.7, 1.4, 0)
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
    a, b, c = line(p, q, "gcd")
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
    xs = [Vec2(2, 1), Vec2(3, 1), Vec2(4, 2), Vec2(3, 3), Vec2(2, 3), Vec2(1, 2)]
    assert convex_polygon(xs)


def point_on_line(l, p0):
    return abs(l.a * p0.x + l.b * p0.y + l.c) < 1e-6


def point_on_circle(p, p0, r):
    return abs((p0.x - p.x) ** 2 + (p0.y - p.y) ** 2 - r * r) < 1e-6


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

    assert point_on_circle(p, p3[0], r)
    assert point_on_circle(p, p2[0], r)
    assert point_on_circle(p, p2[1], r)

    assert point_on_line(l3, p3[0])
    assert point_on_line(l2, p2[0])
    assert point_on_line(l2, p2[1])


def test_circle_circle_intersect():
    p0 = Vec2(4, 5)
    r0 = 3

    p1, r1 = Vec2(8, 6), 4

    n1, ip1 = circle_intersection(p0, r0, p1, r1)

    assert n1 == 2
    assert point_on_circle(p0, ip1[0], r0)
    assert point_on_circle(p0, ip1[1], r0)
    assert point_on_circle(p1, ip1[0], r1)
    assert point_on_circle(p1, ip1[1], r1)


def test_convex_hull():
    p1 = [(0, 0), (0, 1)]
    p2 = [(0, 0), (1, 1), (2, 0)]
    p3 = [(0, 0), (2, 0), (0, 2), (2, 2), (1, 1)]

    cp1 = sorted(convex_hull([Vec2(*p) for p in p1]))
    cp2 = sorted(convex_hull([Vec2(*p) for p in p2]))
    cp3 = sorted(convex_hull([Vec2(*p) for p in p3]))
    assert cp1 == p1
    assert cp2 == p2
    assert cp3 == sorted([(0, 0), (2, 0), (0, 2), (2, 2)])


def test_angle_cmp():
    # points sorted in counter-clockwise order
    pts = [
        Vec2(-2, 0),
        Vec2(-1, -2),
        Vec2(0, -3),
        Vec2(1, -2),
        Vec2(1, 1),
        Vec2(1, 2),
        Vec2(0, 3),
        Vec2(-1, 2),
    ]

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert angle_cmp(pts[i], pts[j])


def test_point_inside_convex_polygon():
    # tested on 166B
    poly = [(0, 0), (4, 0), (6, 5), (4, 5), (0, 3)]
    poly = [Vec2(*p) for p in poly]
    inside = [(1, 1), (2, 2), (1, 3), (3, 3), (2, 1)]
    inside = [Vec2(*p) for p in inside]
    outsie = [(0, 0), (1, 0), (10, 0), (0, 3), (-3, 1), (8, 8), (5, 5), (6, 4)]
    outsie = [Vec2(*p) for p in outsie]

    for p in inside:
        assert point_inside_convex_polygon(poly, p)

    for p in outsie:
        assert not point_inside_convex_polygon(poly, p)

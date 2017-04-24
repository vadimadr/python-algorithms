from collections import namedtuple, deque

from math import sqrt, gcd, hypot, inf

eps = 1e-14
Vec2 = namedtuple('Vec2', ['x', 'y'])
Vec3 = namedtuple('Vec3', ['x', 'y', 'z'])
Line2 = namedtuple('Line2', ['a', 'b', 'c'])


def det2(a, b, c, d):
    """Determinant of 2x2 matrix"""
    return a * d - b * c


def dot(v, u):
    """inner aka scalar aka dot product"""
    if isinstance(v, Vec2):
        return v.x * u.x + v.y * u.y
    else:
        return v.x * u.x + v.y * u.y + v.z * u.z


def l2(x):
    return sqrt(dot(x, x))


def orthogonal(v, u):
    return abs(dot(v, u)) < eps


def vec2_prod(a, b):
    """pseudo cross product
    a x b = |a||b|sin(θ) = det([a1 a2; b1 b2])
    θ - is the rotation (counter clockwise) angle from a to b

    It is oriented area of the parallelogram btw a, b
    """
    return det2(a.x, a.y, b.x, b.y)


def vec3_prod(a, b):
    """cross product
    a x b = c, c is the vector with following properties:

    |c| = area (S) of the parallelogram btw a, b
    c is orthogonal to both a and b
    a, b, c - is right oriented

    c = det([e0 e1 e2; a0 a1 a2; b0 b1 b2]), where (e0, e1, e2) - is
    orthonormal basis
    """
    x = a.y * b.z - a.z * b.y
    y = a.z * b.x - a.x * b.z
    z = a.x * b.y - a.y * b.x
    return Vec3(x, y, z)


def orientation(a, b, c):
    """Test if a,b,c is a right handed triple (i.e. rotation from a to b
    seen from c is counterclockwise)

    Returns triple product d = det([a; b; c])

    if d > 0 then triple is right-handed (counter clockwise)
    if d < 0 then triple is left-handed (clockwise)
    if d = 0 then triple is coplanar (are in the same plane)
    """
    if isinstance(a, Vec2):
        # 2d space. Let z = 1, then apply Laplace expansion to first row
        return a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)

    return a.x * b.y * c.z \
        + a.y * b.z * c.x \
        + a.z * b.x * c.y \
        - a.x * b.z * c.y \
        - a.y * b.x * c.z \
        - a.z * b.y * c.x


def line(p: Vec2, q: Vec2, normed='unit'):
    """Find line equation for ax + by + c = 0, through given points p and q"""
    # Solve [p - q, x] == 0
    a = q.y - p.y
    b = p.x - q.x
    c = -a * p.x - b * p.y

    if normed is None:
        return a, b, c
    if normed == 'unit':
        # unit length normal vector
        z = sqrt(a * a + b * b)
        if abs(z) > eps:
            a, b, c = a / z, b / z, c / z
    if normed == 'gcd':
        # if p and q are integers, then a, b, c are integers
        z = gcd(gcd(a, b), c)
        if z != 0:
            a, b, c = a // z, b // z, c // z
    if normed is not None and a < 0:
        a, b, c = -a, -b, -c

    return Line2(a, b, c)


def distance_to_line(l: Line2, p: Vec2):
    """d = distance from p to l

    let q - some point on l, then d = (p-q, n)/|n|
    c = - qx - qy
    (p-q, n) = a*(px - qx) + b*(py - qy) = a*px + b*py + c
    """
    a, b, c = l
    x, y = p
    z = hypot(a, b)
    return abs(a * x + b * y + c) / z


def line_projection(l: Line2, p: Vec2):
    """Orthogonal projection of p on l

    p = Prp + Ortp
    PrP = p - Ortp = p - d / |n| *  n
    """
    a, b, c = l
    x, y = p
    prx = (b * (b * x - a * y) - a * c) / (a * a + b * b)
    pry = (a * (a * y - b * x) - b * c) / (a * a + b * b)
    return Vec2(prx, pry)


def line_parallel(l1: Line2, l2: Line2):
    """Test if lines l1 and l2 are parallel in R2

    Lines are parallel iff normal vectors are collinear
    """
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    n1, n2 = Vec2(a1, b1), Vec2(a2, b2)  # normal vectors
    return abs(vec2_prod(n1, n2)) < eps


def line_same(l1: Line2, l2: Line2):
    """Test if lines are the same

    Lines are the same if a,b,c are proportional with same ratio
    """
    a1, b1, c1 = l1
    a2, b2, c2 = l2

    # a, b, c are proportional iff all dets are equal to zero
    c_ = abs(det2(a1, b1, a2, b2)) < eps and \
        abs(det2(a1, c1, a2, c2)) < eps and \
        abs(det2(b1, c1, b2, c2)) < eps

    return c_


def line_intersect(l1: Line2, l2: Line2):
    """Test if lines are intersects and find the intersection point"""
    a1, b1, c1 = l1
    a2, b2, c2 = l2

    # Solve the linear system using Cramer's rule
    z = det2(a1, b1, a2, b2)
    if abs(z) < eps:
        return False
    x = -det2(c1, b1, c2, b2) / z
    y = -det2(a1, c1, a2, c2) / z
    return Vec2(x, y)


def bb_test(a, b, c, d):
    """1d Bounding box test"""
    a, b = sorted((a, b))
    c, d = sorted((c, d))
    return max(a, c) <= min(b, d) + eps


def in_range(l, r, x):
    """Test if x in [l, r]"""
    return min(l, r) <= x + eps and x <= max(l, r) + eps


def segment_intersection(a: Vec2, b: Vec2, c: Vec2, d: Vec2):
    """Find segments AB and CD intersection

    Returns
    --------
    False if do not intersects
    Vec2 if intersection is the one point
    (Vec2, Vec2) if intersection is segment
    """
    # check if bounding boxes are cross first
    if not bb_test(a.x, b.x, c.x, d.x) or not bb_test(a.y, b.y, c.y, d.y):
        return False
    l1 = line(a, b)
    l2 = line(c, d)

    # check if one of segments is a point
    point_test = all(map(lambda x: abs(x) < eps, l1)) or \
        all(map(lambda x: abs(x) < eps, l2))

    if line_parallel(l1, l2):
        if not line_same(l1, l2):
            return False
        a, b = sorted((a, b))
        c, d = sorted((c, d))
        l = max((a, c))
        r = max((b, d))

        return (l, r) if not point_test else l

    else:
        p = line_intersect(l1, l2)
        # check if intersection point is inside segment
        c_ = in_range(a.x, b.x, p.x) and \
            in_range(a.y, b.y, p.y) and \
            in_range(c.x, d.x, p.x) and \
            in_range(c.y, d.y, p.y)

        return p if c_ else False


def segment_union_measure(xs):
    """Returns length of union of segments"""
    p = []
    for a, b in xs:
        # p[i][1] = point is right
        p.append((a, False))
        p.append((b, True))

    p = sorted(p)
    m = 0
    c = 0
    for i in range(len(p)):
        if c and i:
            m += p[i][0] - p[i - 1][0]
        c += 1 if p[i][1] else -1
    return m


def segment_cover(xs, ys=None):
    """Covers all segments with minimal set of points (each segment is
    covered with at least one point)

    Parameters
    -----------
    xs : List
        segments to be covered (objective)
    ys : List or None
        open intervals that can not be covered (point can not be placed
        inside it)
    """
    if not ys:
        ys = []

    # ps[i] = (x, objective?, right?, i_seg)
    ps = []
    for i, x in enumerate(xs):
        ps.append((x[0], True, False, i))
        ps.append((x[1], True, True, i))

    for i, y in enumerate(ys):
        ps.append((y[0], False, False, i))
        ps.append((y[1], False, True, i))

    ps = sorted(ps)
    covered = [False] * len(xs)
    d = deque()  # queue of not covered segments
    coverage = []

    last_free = None
    inclusion_rate = 0  # nesting level of not objective segments
    for p in ps:
        x, objective, right, n = p

        # not objective
        if not objective and not right:
            inclusion_rate += 1
            if inclusion_rate == 1:
                # or x - eps if ys are segments
                last_free = x
        elif not objective and right:
            inclusion_rate -= 1

        # objective
        if objective and not right:
            d.append(n)
        elif objective and not covered[n]:
            if inclusion_rate == 0:  # point can be placed here
                # cover current and all opened segments
                coverage.append(x)
                # flush stack
                for el in d:
                    covered[el] = True
                d.clear()
            else:
                if d and xs[d[0]][0] <= last_free:
                    coverage.append(last_free)
                while d and xs[d[0]][0] <= last_free:
                    covered[d.popleft()] = True
    return coverage


def polygon_area(pts):
    """Returns area of polygon bounded by edges (pts[i], pts[i+1]) Polygon
    may not be convex

    Let O be some point. Then S = Sum of areas of all triangles OAB, where AB
    - some edge of input polygon

    Parameters
    ------------------
    pts: List[Vec2]
        vertices of polygon in clockwise order
    """
    s = 0
    for i in range(len(pts)):
        s += vec2_prod(pts[i], pts[i - 1])
        # or use areas of trapezoids with bases y1 and y2 instead
        # s += (pts[i].x - pts[i-1].x)*(pts[i].y + pts[i-1].y)
    return abs(s) / 2


def points_inside(pts):
    """Returns number of points with integer coordinates inside a polygon (
    not in the border)

    Use Pick's formula s = i + b/2 - 1
    where i - points inside, b - points on the border
    """
    s = 0
    b = 0
    for i in range(len(pts)):
        s += vec2_prod(pts[i], pts[i - 1])
        # number of points on segment
        b += gcd(pts[i].x - pts[i - 1].x, pts[i].y + pts[i - 1].y)
    return (abs(s) - b + 2) // 2


def convex_polygon(pts):
    """Checks if polygon is convex"""
    s = None
    for i in range(len(pts)):
        v1 = Vec2(pts[i - 2].x - pts[i - 1].x, pts[i - 2].y - pts[i - 1].y)
        v2 = Vec2(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y)
        prod = vec2_prod(v1, v2)
        s = prod if s is None else s
        if prod * s < 0:
            return False
    return True


def circle_line_intersection(p, r, l):
    """Finds intersection of a line with a circle
    
    Parameters
    -----------
    p : Vec2
        Center of a circle
    r 
        radius of a circle
    l : Line2
    
    Returns
    ---------
    n : int
        number of intersection points
    pts
        intersection points
    """
    # move circle and line to the origin
    l0 = Line2(l.a, l.b, l.c + l.a * p.x + l.b * p.y)
    a, b, c = l0
    # closest to the origin point on line
    x0 = Vec2(-a * c / (a * a + b * b) + p.x, -b * c / (a * a + b * b) + p.y)
    # distance from line to the origin
    d = abs(c) / hypot(a, b)
    if d > r:
        return 0, ()
    elif abs(d - r) < eps:
        return 1, (x0,)
    else:
        # distance from x0 to intersection points
        d0 = sqrt(r * r - d * d)
        # (-b, a) is collinear with l
        dx = -b * d0 / hypot(a, b)
        dy = a * d0 / hypot(a, b)
        return 2, (Vec2(x0.x + dx, x0.y + dy), Vec2(x0.x - dx, x0.y - dy))


def circle_intersection(p1, r1, p2, r2):
    """Finds intersection of two circles with centers p1 and p2 and radius 
    r1 and r2
    
    Method
    -------
    Assume first circle is at the origin, then solve the system of equations:
    x^2 + y^2 = r1^2
    (x - x0)^2 + (y - y0)^2 = r2^2
    
    and subtract the first equation from the second:
    x (-2x0) + y (-2y0) + (x0^2 + y0^2 + r1^2 - r2^2) = 0
    """
    # centers are identical
    if abs(p1.x - p2.x) < 1e-6 and abs(p1.y - p2.y) < 1e-6:
        if abs(r1 - r2) < 1e-6:
            return inf, ()  # circles are identical
        else:
            return 0, ()  # circles do not intersect

    p2 = Vec2(p2.x - p1.x, p2.y - p1.y)
    a = -2 * p2.x
    b = -2 * p2.y
    c = p2.x ** 2 + p2.y ** 2 + r1 ** 2 - r2 ** 2
    n, ips = circle_line_intersection(Vec2(0, 0), r1, Line2(a, b, c))

    return n, [Vec2(x + p1.x, y + p1.y) for x, y in ips]

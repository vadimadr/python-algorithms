from collections import namedtuple

from math import sqrt, gcd, hypot

eps = 1e-14
Vec2 = namedtuple('Vec2', ['x', 'y'])
Vec3 = namedtuple('Vec3', ['x', 'y', 'z'])
Line2 = namedtuple('Line2', ['a', 'b', 'c'])


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
    return a.x * b.y - a.y * b.x


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



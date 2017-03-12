from collections import namedtuple

from math import sqrt

eps = 1e-14
Vec2 = namedtuple('Vec2', ['x', 'y'])
Vec3 = namedtuple('Vec3', ['x', 'y', 'z'])


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

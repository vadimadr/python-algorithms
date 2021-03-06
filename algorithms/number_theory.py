from functools import reduce
from itertools import chain
from math import sqrt
from operator import mul


def even(n):
    # n ~ 2*k
    return n % 2 == 0


def odd(n):
    # n ~ 2*k + 1
    return n % 2 != 0


def gcd(a, b):
    """
    Greatest common divisor (greatest common factor)

    Notes
    ---------

    Euclidean algorithm:

    a > b > r_1 > r_2 > ... > r_n
    a = b*q + r
    b = r_1*q_1 + r_2
    ...
    r_n-1 = r_n*q_n

    gcd(a,b) = gcd(b,r)
    gcd(a,0) = a

    """

    while b != 0:
        a, b = b, a % b
    return a


def lcm(a, b):
    """
    Least common multiplier
    """
    return abs(a * b) // gcd(a, b)


def extended_euclidian(a, b):
    """
    returns (x, y, d) such that:
    x*a + y*b = d, d = gcd(a,b)

    r_1 = a - b * q_1
    r_2 = b - r_1 * q_2
    ...
    r_n = r_n-2 - r_n-1*q_n = a * x_n + b * y_n

    x_n = x_n-2 - x_n-1 * q_n
    y_n = y_n-2 - y_n-1 * q_n
    """

    # Naive version:
    # x2, y2 = 1, 0
    # x1, y1 = 0, 1
    # while b > 0:
    #     q, a, b = a // b, b, a % b
    #     x = x2 - x1 * q
    #     y = y2 - y1 * q
    #     x2, x1 = x1, x
    #     y2, y1 = y1, y
    # return x2, y2, a

    # Recursive version O(log^2(a))
    # suppose we know x1, y1 for (b, a%b) and a%b = a - b*q
    # then b*x1 + (a%b)*y1 = a*y1 + b*(x1 - y1*q)
    if a == b == 0:
        return 0, 0, 0
    if b == 0:
        return 1, 0, a
    x, y, d = extended_euclidian(b, a % b)
    return y, x - y * (a // b), d


def binomial(n, k):
    """
    Chose k objects from n.

    (n, k) = n!/(k!(n-k)!)

    Pascal's rule:
    (n + 1, k) = (n, k) + (n, k - 1)
    (k, k) = 0
    """
    # if k > n:
    #     return 0
    # if k == n or k == 0:
    #     return 1
    # return binomial(n - 1, k) + binomial(n - 1, k - 1)

    res = 1

    for i in range(1, k + 1):
        res = res * (n - i + 1) // i

    return res


def binomial_table(n, k):
    """
    Pascal's triangle:
    (n, k) = C[n][k]
    """
    C = [[0] * (i + 1) for i in range(n + 1)]

    for i in range(n + 1):
        for j in range((min(i, k) + 1)):
            if j == 0 or j == i:
                C[i][j] = 1
            else:
                C[i][j] = C[i - 1][j - 1] + C[i - 1][j]
    return C[n][k]


def factorial(n):
    """
    Returns n! = n(n-1)(n-2)...
    """
    return reduce(mul, range(1, n), 1)


def fibonacci(n):
    """
    Returns nth fibonacci number F(n)

    F(n) = F(n-2) + F(n-1)
    """
    k, m = 1, 1

    if n < 2:
        return n

    for i in range(2, n):
        k, m = m, k + m

    return m


def isqrt(x):
    # isqrt(x) = floor(sqrt(x))
    return int(sqrt(x))


def coprime(a, b):
    """
    Check if two integers are coprime. Integers are coprime if the only
    integer that divides both of them is 1. That is gcd(a,b) = 1.

    Parameters
    ----------
    a : int
        input values

    b : int
        input value

    Returns
    -------
    bool
        Whether integers are coprime

    """
    return gcd(a, b) == 1


def is_prime(n: int) -> bool:
    if n in (2, 3):
        return True
    if n < 2 or n % 2 == 0 or n % 3 == 0:
        return False

    # all numbers of the form (6n +- 1)
    for q in range(5, isqrt(n) + 1, 6):
        if n % q == 0 or n % (q + 2) == 0:
            return False

    return True


def sieve(n):
    """
    Get all primes <= n
    """
    s = [True] * (n + 1)
    for i in range(2, isqrt(n) + 1):
        if s[i]:
            for j in range(i + i, n + 1, i):
                s[j] = False
    return [i for i in range(2, n + 1) if s[i]]


def factorize(n):
    """
    Prime decomposition

    Decomposes integer n into
    n = p1^a1 * p2^a2 * pn^an

    where p_i are primes and a_i are their exponents

    Parameters
    ----------
    n : int
        integer to factorize

    Returns
    -------
    factors : list
        list of the prime factors, together with their exponents

    Examples
    --------
    >>> factorize(2434500)
    [(2, 2), (3, 2), (5, 3), (541, 1)]

    """

    if n in (0, 1):
        return [(n, 1)]

    factors = []

    if n < 0:
        factors.append((-1, 1))
        n = - n

    # check 2, 3, then all integers in form q = 6k +- 1
    for q in chain((2, 3), range(5, isqrt(n) + 1, 6)):
        # q = 6k - 1
        a = 0
        while n % q == 0:
            # q is prime because n already divided by its prime factors
            n //= q
            a += 1
        if a > 0:
            factors.append((q, a))

        # 6k + 1
        q += 2
        a = 0
        while n % q == 0:
            # q is prime because n already divided by its prime factors
            n //= q
            a += 1
        if a > 0:
            factors.append((q, a))

    if n != 1:
        factors.append((n, 1))

    return factors


def prime_pi(n):
    """
    Number of primes <= n
    """
    if n < 2:
        return 0

    primes = sieve(n)
    return len(primes)


def euler_phi(n):
    """
    Number of coprimes <= n
    """

    if n == 1:
        return 1

    phi = n
    for p, a in factorize(n):
        phi -= phi // p

    return phi


def binpow(x, r):
    """Binary exponential algorithm"""

    # recursive implementation:
    # if even(r):
    #     ans = binpow(x, r // 2)
    #     return ans * ans
    # else:
    #     return binpow(x, r - 1) * x
    ans = 1
    while r > 0:
        if odd(r):
            ans *= x
        x *= x
        r = r // 2
    return ans


def linear_diophantine(a, b, c):
    """Solve ax + by = c, where x, y are integers

    1. solution exists iff c % gcd(a,b) = 0
    2. all solutions have form (x0 + b'k, y0 - a'k)

    Returns
    -------
    None if no solutions exists
    (x0, y0, a', b') otherwise
    """
    # d = pa + qb
    p, q, d = extended_euclidian(a, b)
    if d == 0 or c % d != 0:
        return None
    # ax + by = c <=> a'x + b'y = c'
    a, b, c = a // d, b // d, c // d
    return p * c, q * c, a, b

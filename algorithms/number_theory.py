from functools import reduce
from itertools import chain
from math import sqrt
from operator import mul
from random import randint


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
        n = -n

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


def linear_sieve(max_n):
    """Computes all primes < max_n and lits of
    all smallest factors in O(max_n) time

    Returns
    -------
    primes: list of all primes < max_n
    smallest_factors: list such that if q = smallest_factors[n],
        then n % q == 0 and q is minimal.
    """
    smallest_factors = [0] * max_n
    primes = []

    for i in range(2, max_n):
        if smallest_factors[i] == 0:
            smallest_factors[i] = i
            primes.append(i)

        for p in primes:
            if p > smallest_factors[i] or i * p >= max_n:
                break
            smallest_factors[i * p] = p
    return primes, smallest_factors


primes, spf = linear_sieve(int(1e6))


def powmod(x, k, m):
    """Computes x ^ k (mod m)
    using binary exponentiation in O(lg k) time"""
    ans = 1
    while k > 0:
        if odd(k):
            ans = ans * x % m
            k -= 1
        else:
            x = x * x % m
            k /= 2
    return ans


def factor_twos(x):
    """Represent x = 2^s * d"""
    d, s = x, 0
    while even(d):
        d >>= 1
        s += 1
    return d, s


def fermat_test(n, a):
    if n == 2:
        return True
    return powmod(a, n - 1, n) == 1


def fermat_strong_test(n, a):
    """Performs Fermat Strong Test with base ans
    Returns True if n is probable prime

    For a composite integer n it returns True with probability ~ 1/4
    For a prime integer n it always returns True
    """
    if n == 2:
        return True
    # n - 1 = d * 2 ^ s
    d, s = factor_twos(n - 1)

    # by Fermat theorem, if n is prime then
    # (a^d - 1)(a^d + 1)(a^2d + 1)(a^4d + 1)...(a^2^(s-1)d + 1) = 0 (mod n)
    a = powmod(a, d, n)
    if a == 1 or a == n - 1:
        return True
    for _ in range(s):
        a = a * a % n
        if a == n - 1:
            return True
    return False


def jacobi(a, n):
    """Returns:
    0 if a is not coprime to n (gcd(a,n) != 1)
    -1 if a is NOT perfect square modulo n
    1 if a maybe perfect square
    """
    if n == 1:
        return 1
    if a % n == 0:
        return 0
    if a < 0:
        return jacobi(a + n * (1 + (-a - 1) // n), n)

    # jacobi of (2^k/n)
    a1, k = factor_twos(a)
    j_even = 1 if n % 8 == 1 or n % 8 == 7 else -1
    if even(k):
        j_even = 1

    # law of quadratic reciprocity
    if n % 4 == 3 and a1 % 4 == 3:
        return -j_even * jacobi(n % a1, a1)
    return j_even * jacobi(n % a1, a1)


def lucas_strong_test(n, p, q):
    if even(n):
        return n == 2

    D = p ** 2 - 4 * q

    def div2(x):
        """Performs division by 2 modulo n"""
        if odd(x):
            x += n
        return x // 2 % n

    def lucas_double(u_k, v_k, k):
        # computes U_k, V_k -> U_2k, V_2k
        return u_k * v_k % n, (v_k * v_k + -2 * powmod(q, k, n)) % n

    def lucas_sum(u_k, v_k, u_m, v_m):
        # computes U_{k+m}, V_{k+m}
        u_km = div2(u_k * v_m + u_m * v_k)
        v_km = div2(v_k * v_m + D * u_k * u_m)
        return u_km, v_km

    # n - J(D/n) = 2^s*d
    d, s = factor_twos(n - jacobi(D, n))

    # compute U_d, V_d
    # representing d as binary number
    u, v = 0, 2  # u0, v0
    u_k, v_k, k, d_rem = 1, p, 1, d  # u1, v1
    while d_rem:
        if d_rem & 1:
            u, v = lucas_sum(u, v, u_k, v_k)
        u_k, v_k = lucas_double(u_k, v_k, k)
        k *= 2
        d_rem >>= 1

    if u == 0:
        return True

    for _ in range(s + 1):
        if v == 0:
            return True
        u, v = lucas_double(u, v, d)
        d *= 2

    # If Q != +-1 we may additionally check congruences
    # V_{n+1} = 2Q (mod n) and Q^(n+1)/2 = Q*J(Q, n) (mod n)
    # (Frobenius probable prime)
    return False


def lucas_selfridge_test(n):
    """Lucas strong test with Selfridge parameters"""
    if isqrt(n) ** 2 == n:
        return False
    D = 5
    while jacobi(D, n) != -1:
        if D > 0:
            D = -(D + 2)
        else:
            D = -(D - 2)
    return lucas_strong_test(n, 1, (1 - D) // 4)


# correctly checks all primes < 3 * 10^9
BASES_1 = [2, 3, 5, 7]
# correctly checks all primes < 9 * 10^18
BASES_2 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def Miller_Rabin_test(n, bases=None, num_trials=5):
    if bases is None:
        # with random base probability of false positive is 1/4!
        # bases = [randint(2, n - 1) for _ in range(num_trials)]
        if n <= 1e9:
            bases = BASES_1
        else:
            bases = BASES_2

    for a in bases:
        if n == a:
            continue
        if n % a == 0:
            return False
        if not fermat_strong_test(n, a):
            return False
    return True


def Ballie_PSW_test(n, max_trivial_trials=100):
    """BPSW probable primality test.

    There is no known pseudo-primes that pass this test.
    Lower bound for potential pseudoprime is 10^10000.
    """
    for i in range(max_trivial_trials):
        if primes[i] == n:
            return True
        if n % primes[i] == 0:
            return False
        if primes[i] ** 2 >= n:
            return True
    if not fermat_strong_test(n, 2):
        return False
    if not lucas_selfridge_test(n):
        return False
    return True


def Pollard_rho_Floyd(n, x0=2, c=1):
    """Pollard's Rho method
    Attempts to find a divisor using pollard rho method
    with Floyd cycle finding algorithm.

    Returns either any prime factor of n or n itself.

    If n = p*q then algorithm exits approximately in O(p^(1/4))
    """

    def f(x):
        return (x * x + c) % n

    x, y, g = x0, x0, 1
    while g == 1:
        x = f(x)
        y = f(f(y))
        g = gcd(abs(x - y), n)
    return g


def Pollard_rho_factor(n, check_prime=False):
    """Runs Pollard Rho method with different
    initial values and intercepts (x0, c) until
    some factor is found"""
    if check_prime and Miller_Rabin_test(n):
        return n

    if even(n):
        return 2

    while True:
        x0 = randint(2, n - 1)
        c = randint(1, n - 1)
        g = Pollard_rho_Floyd(n, x0, c)
        if g != n:
            return g


def Pollard_pm1(n, primes, max_B=1000000):
    """Pollard's p - 1 method
    Attempts to find some B-powersmooth factor of n

    Fcator p is B-powersmooth if p - 1 = p1^d1 * ... * pn^dn
    and max(p1, ..., pn) < B

    Parameters
    ----------
    n : int
        integer to factorize
    primes : list
        sorted list of primes < max_B
    max_B : int
        maximal powersmoothness of extracted factor
    """
    B = 10
    g = 1
    while B < max_B and g < n:
        a = randint(2, n - 2)
        g = gcd(a, n)
        if g != 1:
            return g
        for p in primes:
            if p >= B:
                break
            pd = 1  # p^d
            while pd * p <= B:
                pd *= p
            a = powmod(a, pd, n)
            g = gcd(a - 1, n)
            if g != 1 and g != n:
                return g
        B *= 2
    return 1


def log_modulo(a, b, m):
    """Computes discrete logarithm i.e.
    finds x such that a^x = b (mod m)

    Uses Shanks algorithms which takes O(sqrt(m)) time
    """
    # find x in form x = np - q for some (n, p)
    # => a^x = b ~ a^np = a^q * b
    a, b = a % m, b % m
    n = isqrt(m) + 1
    # compute all a^q * b
    rhs = {}
    for q in range(n + 1):
        rhs[b] = q
        b = b * a % m

    an = powmod(a, n, m)
    lhs = 1
    for p in range(1, n + 1):
        lhs = lhs * an % m
        if lhs in rhs:
            return n * p - rhs[lhs]
    # some numbers do not have discrete log (example: log_2 3 (mod 7))
    return None


def primitive_root(n):
    """Finds generator of group of integers modulo p.
    i.e. finds integer `g` such that for each `a` exists some `k`:
    a = g^k (mod n)

    g exists if and only if:
    1) n is p^d  2) n is 2*p^d for 3) n <= 4 (Gauss theorem)
    """
    if n == 2:
        return 1
    phi = euler_phi(n)
    phi_divisors = factorize(phi)
    for g in range(2, n + 1):
        if gcd(g, n) != 1:
            continue
        for d, _ in phi_divisors:
            if powmod(g, phi // d, n) == 1:
                # g can not be a generator
                break
        else:
            return g

    # generator not found
    # Gauss theorem conditions did not met
    return None


def kth_root_modulo(a, k, n):
    """Computes discrete kth root
    i.e. find x such that x^k = a (mod n)"""
    # x = g^y => x^k = (g^k)^y = a
    # => y = log_{g^k} a (mod n)
    g = primitive_root(n)
    gk = powmod(g, k, n)
    y = log_modulo(gk, a, n)
    return powmod(g, y, n)

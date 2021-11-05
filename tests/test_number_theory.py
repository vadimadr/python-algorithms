from bisect import bisect_left
import os
from functools import reduce

import pytest
from hypothesis import assume, given
from hypothesis.strategies import integers, composite

from algorithms.number_theory import (
    Ballie_PSW_test,
    binomial,
    binpow,
    euler_phi,
    even,
    extended_euclidian,
    factorize,
    fermat_strong_test,
    fibonacci,
    gcd,
    is_prime,
    jacobi,
    kth_root_modulo,
    linear_diophantine,
    linear_sieve,
    log_modulo,
    lucas_selfridge_test,
    Miller_Rabin_test,
    odd,
    Pollard_pm1,
    Pollard_rho_factor,
    powmod,
    primitive_root,
    sieve,
)

_dir = os.path.dirname(__file__)

with open(os.path.join(_dir, "data/primes.dat")) as src:
    PRIME_LIST = list(map(int, src))

with open(os.path.join(_dir, "data/notprimes.dat")) as src:
    NON_PRIMES = list(map(int, src))


@pytest.fixture(scope="module")
def primes():
    return PRIME_LIST


@pytest.fixture
def nonprimes():
    return NON_PRIMES


class TestIs_prime:
    def test_is_prime(self, primes):
        for p in primes:
            assert is_prime(p), "Wrong result with p = %d" % p

    def test_non_prime(self, nonprimes):
        for p in nonprimes:
            assert not is_prime(p), "Wrong result with p = %d" % p


class TestFactor:
    def test_zero(self):
        assert factorize(0) == [(0, 1)]

    def test_negative(self):
        assert factorize(-6) == [(-1, 1), (2, 1), (3, 1)]

    def test_prime(self):
        assert factorize(7) == [(7, 1)]

    def test_simple(self, primes):
        ps = [3, 4, 5, 6]
        factors = [(primes[i], 1) for i in ps]
        n = reduce(lambda a, b: a * b[0], factors, 1)

        assert factorize(n) == factors

    def test_large(self, primes):
        ps = [4213, 5261, 8974]
        factors = [(primes[i], 1) for i in ps]
        n = reduce(lambda a, b: a * b[0], factors, 1)

        assert factorize(n) == factors

    def test_pollard_rho(self, nonprimes):
        for np in nonprimes:
            d = Pollard_rho_factor(np)
            assert d != np and np % d == 0

    def test_pollard_pm1(self, nonprimes):
        primes, _ = linear_sieve(500)
        for np in nonprimes:
            d = Pollard_pm1(np, primes)
            assert d != np and np % d == 0


class TestSieve:
    def test_sieve(self):
        sieve(2) == [2]
        sieve(3) == [2, 3]
        sieve(5) == [2, 3, 5]

    def test_sieve2(self):
        with open(os.path.join(_dir, "data/primes.dat")) as src:
            lines = src.readlines()
            primes = [int(n.strip()) for n in lines]
            maxp = primes[-1]
            sieve_primes = sieve(maxp)
            sieve_primes == primes

    def test_linear_sieve(self, primes, nonprimes):
        sieve_pr, spf = linear_sieve(primes[-1] + 1)
        assert sieve_pr == primes
        assert spf[11 * 13] == 11

        for np in nonprimes[:100]:
            factor = spf[np]
            assert factor != np and factor > 1
            assert np % factor == 0

        for pr in primes[:100]:
            assert spf[pr] == pr


class TestPrimality:
    def test_fermat_strong(self, primes, nonprimes):
        assert fermat_strong_test(11, 2)

        # always returns true for primes
        for pr in primes[:100]:
            assert fermat_strong_test(pr, 2)

        for comp in nonprimes[:100]:
            # works well up to 3277 with base 2
            assert not fermat_strong_test(comp, 2)

    def test_miller_rabin(self, primes, nonprimes):
        assert Miller_Rabin_test(11, [2, 3])
        assert not Miller_Rabin_test(121, [2, 3])

        for pr in primes[:100]:
            assert Miller_Rabin_test(pr, [2, 3])
        for cp in nonprimes[:100]:
            assert not Miller_Rabin_test(cp, [2, 3])

    def test_lucas_selfridge(self, primes, nonprimes):
        for p in primes[:100]:
            assert lucas_selfridge_test(p)
        for np in nonprimes[:30]:
            assert not lucas_selfridge_test(np)

    def test_bpsw(self, primes, nonprimes):
        for p in primes[:100]:
            assert Ballie_PSW_test(p)
        for np in nonprimes[:100]:
            assert not Ballie_PSW_test(np)


@pytest.mark.parametrize(
    "a,b,expected",
    ((20, 100, 20), (15, 0, 15), (13, 13, 13), (37, 600, 1), (624129, 2061517, 18913)),
)
def test_gcd(a, b, expected):
    assert gcd(a, b) == expected


@given(integers(), integers())
def test_gcd_invariant(a, b):
    d = gcd(a, b)
    if a == 0 and b == 0:
        assert d == 0
        return
    assert d != 0
    assert a % d == 0
    assert b % d == 0


@given(integers(), integers())
def test_extended_gcd_invariant(a, b):
    p, q, d = extended_euclidian(a, b)
    assert d == gcd(a, b)
    if d == 0:
        assert p == q == 0
        return
    assert d != 0
    assert a % d == 0
    assert b % d == 0
    assert a * p + b * q == d


def test_even():
    assert even(2)
    assert even(0)
    assert odd(3)


def test_binomial():
    assert binomial(5, 0) == 1
    assert binomial(5, 1) == 5
    assert binomial(5, 3) == 10
    assert binomial(5, 7) == 0
    assert binomial(23, 11) == 1352078


def test_fib():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(2) == 1
    assert fibonacci(13) == 233


def test_euler_phi():
    assert euler_phi(1) == 1
    assert euler_phi(2) == 1
    assert euler_phi(3) == 2
    assert euler_phi(25) == 20
    assert euler_phi(4242) == 1200


def test_binpow():
    assert binpow(2, 0) == 2 ** 0
    assert binpow(2, 1) == 2 ** 1
    assert binpow(2, 2) == 2 ** 2
    assert binpow(2, 3) == 2 ** 3
    assert binpow(2, 10) == 2 ** 10
    assert binpow(3, 13) == 3 ** 13


def test_powmod():
    assert powmod(2, 0, 10) == 2 ** 0 % 10
    assert powmod(2, 1, 10) == 2 ** 1 % 10
    assert powmod(2, 2, 10) == 2 ** 2 % 10
    assert powmod(2, 3, 10) == 2 ** 3 % 10
    assert powmod(2, 4, 10) == 2 ** 4 % 10
    assert powmod(2, 5, 10) == 2 ** 5 % 10


def test_jacobi_symbol():
    pr, spf = linear_sieve(100)
    # defined only for odd modulo
    for n in range(1, 100, 2):
        for a in range(1, n):
            # compute jacobi symbol using
            # multiplicative property and euler's criterion
            b, j = n, 1
            while b != 1:
                legendre = powmod(a, (spf[b] - 1) // 2, spf[b])
                if legendre == spf[b] - 1:
                    legendre = -1
                j *= legendre
                b //= spf[b]
            assert j == jacobi(a, n)

    assert jacobi(-1, 7) == jacobi(6, 7)
    assert jacobi(-1, 5) == jacobi(4, 5)
    assert jacobi(5 + 7, 7) == jacobi(5, 7)
    assert jacobi(4 + 7, 7) == jacobi(4, 7)


@given(integers(), integers(), integers())
def test_linear_diophantine(a, b, c):
    assume(a != 0 and b != 0)
    solution = linear_diophantine(a, b, c)
    d = gcd(a, b)
    if d == 0 or c % d != 0:
        assert solution is None
        return
    x0, y0, a0, b0 = solution
    for k in range(1, 10):
        assert (x0 + b0 * k) * a + (y0 - a0 * k) * b == c


def test_lind():
    ans = linear_diophantine(1, 1, 0)
    assert ans is not None


@pytest.mark.parametrize("mod", [7, 11, 13])
def test_discrete_log(mod):
    for a in range(2, mod):
        for x in range(mod - 1):
            b = a ** x % mod
            x_comp = log_modulo(a, b, mod)
            assert a ** x_comp % mod == b


@composite
def st_primes(draw, max_prime=100):
    max_ind = bisect_left(PRIME_LIST, max_prime)
    i = draw(integers(0, max_ind))
    return PRIME_LIST[i]


@composite
def st_discrete_log(draw):
    mod = draw(st_primes(100))
    a = draw(integers(1, mod - 1))
    x = draw(integers(0, mod - 1))
    return a, x, mod


@given(st_discrete_log())
def test_discrete_log_hyp(data):
    a, x, mod = data
    a %= mod
    x %= mod
    b = powmod(a, x, mod)
    x = log_modulo(a, b, mod)
    assert powmod(a, x, mod) == b


@pytest.mark.parametrize(
    "n,g",
    [
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 2),
        (6, 5),
        (7, 3),
        (9, 2),
        (10, 3),
        (11, 2),
        (29, 2),
        (137, 3),
        (5233, 10),
    ],
)
def test_primitive_root(n, g):
    assert primitive_root(n) == g


@composite
def st_kth_root(draw):
    m = draw(st_primes())
    assume(m > 2)
    k = draw(integers(2, m - 1))
    a = draw(integers(2, m - 1))
    x = powmod(a, k, m)
    assume(x != 1)
    return x, k, m, a


@pytest.mark.parametrize("x,k,m", [(4, 2, 17), (4, 2, 5), (2, 2, 7)])
def test_kth_root(x, k, m):
    sqx = kth_root_modulo(x, k, m)
    assert powmod(sqx, k, m) == x


@given(st_kth_root())
def test_kth_root_hyp(data):
    x, k, m, a = data
    a_comp = kth_root_modulo(x, k, m)
    assert powmod(a_comp, k, m) == x

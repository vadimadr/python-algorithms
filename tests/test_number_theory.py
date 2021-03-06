import os
from functools import reduce
from unittest import TestCase

import pytest
from hypothesis import given, assume
from hypothesis.strategies import integers

from algorithms.number_theory import (binomial, euler_phi, even, factorize,
                                      fibonacci, gcd, is_prime, odd, sieve, binpow, linear_diophantine,
                                      extended_euclidian)

_dir = os.path.dirname(__file__)


@pytest.fixture
def primes():
    with open(os.path.join(_dir, 'data/primes.dat')) as src:
        return list(map(int, src))


@pytest.fixture
def nonprimes():
    with open(os.path.join(_dir, 'data/notprimes.dat')) \
            as \
            src:
        return list(map(int, src))


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


class TestSieve(TestCase):

    def test_sieve(self):
        self.assertEqual(sieve(2), [2])
        self.assertEqual(sieve(3), [2, 3])
        self.assertEqual(sieve(5), [2, 3, 5])

    def test_sieve2(self):
        with open(os.path.join(_dir, 'data/primes.dat')) \
                as src:
            lines = src.readlines()
            primes = [int(n.strip()) for n in lines]
            maxp = primes[-1]
            sieve_primes = sieve(maxp)
            self.assertEqual(sieve_primes, primes)


@pytest.mark.parametrize("a,b,expected", (
        (20, 100, 20), (15, 0, 15), (13, 13, 13), (37, 600, 1),
        (624129, 2061517, 18913)))
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

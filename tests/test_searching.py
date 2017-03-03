from unittest.case import TestCase

import pytest
from hypothesis import given
from hypothesis.strategies import text

from algorithms.searching import (bruteforce_substr, equal_range, kmp_substr,
                                  lower_bound, prefix, upper_bound)
from tests.utils import substring_pair


class TestBinSearch(TestCase):
    def setUp(self):
        self.cases = [([], 1, 0, 0),
                      ([1], 0, 0, 0),
                      ([1], 1, 1, 0),
                      ([1], 2, 1, 1),
                      ([1, 1], 0, 0, 0),
                      ([1, 1], 1, 2, 0),
                      ([1, 1], 2, 2, 2),
                      ([1, 1, 1], 0, 0, 0),
                      ([1, 1, 1], 1, 3, 0),
                      ([1, 1, 1], 2, 3, 3),
                      ([1, 1, 1, 1], 0, 0, 0),
                      ([1, 1, 1, 1], 1, 4, 0),
                      ([1, 1, 1, 1], 2, 4, 4),
                      ([1, 2], 0, 0, 0),
                      ([1, 2], 1, 1, 0),
                      ([1, 2], 1.5, 1, 1),
                      ([1, 2], 2, 2, 1),
                      ([1, 2], 3, 2, 2),
                      ([1, 1, 2, 2], 0, 0, 0),
                      ([1, 1, 2, 2], 1, 2, 0),
                      ([1, 1, 2, 2], 1.5, 2, 2),
                      ([1, 1, 2, 2], 2, 4, 2),
                      ([1, 1, 2, 2], 3, 4, 4),
                      ([1, 2, 3], 0, 0, 0),
                      ([1, 2, 3], 1, 1, 0),
                      ([1, 2, 3], 1.5, 1, 1),
                      ([1, 2, 3], 2, 2, 1),
                      ([1, 2, 3], 2.5, 2, 2),
                      ([1, 2, 3], 3, 3, 2),
                      ([1, 2, 3], 4, 3, 3),
                      ([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 0, 0, 0),
                      ([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 1, 1, 0),
                      ([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 1.5, 1, 1),
                      ([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 2, 3, 1),
                      ([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 2.5, 3, 3),
                      ([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 3, 6, 3),
                      ([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 3.5, 6, 6),
                      ([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 4, 10, 6),
                      ([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 5, 10, 10)]

    def testPredefined(self):
        for a, val, hi, lo in self.cases:
            self.assertEqual(hi, upper_bound(a, val, 0, len(a)))
            self.assertEqual(lo, lower_bound(a, val, 0, len(a)))

    def testNormal(self):
        a = [1, 2, 3, 4, 4, 4, 5, 6, 7]
        b = equal_range(a, 4, 0, len(a))
        self.assertEqual(b[0], 3)
        self.assertEqual(b[1], 6)

    def testFirst(self):
        a = [1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        b = equal_range(a, 1, 0, len(a))
        self.assertEqual(b[0], 0)
        self.assertEqual(b[1], 3)

    def testLast(self):
        a = [1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 8, 8]
        b = equal_range(a, 8, 0, len(a))
        self.assertEqual(b[0], a.index(8))
        self.assertEqual(b[1], len(a))

    def testMissing(self):
        a = [1, 2, 3, 4, 5, 6, 8, 9, 0]
        b = equal_range(a, 7, 0, len(a))
        self.assertEqual(b[0], a.index(8))
        self.assertEqual(b[0], b[1])


substr_cases = [('aaa', 'a', 0),
                ('aaa', '', 0),
                ('', '', 0),
                ('abcdef', 'def', 3),
                ('computer', 'muter', -1),
                ('stringmatchingmat', 'ingmat', 3),
                ('videobox', 'videobox', 0),
                ('uw', 'uw', 0),
                ('orb', 'fqh', -1),
                ('gvdq', 'q', 3),
                ('xnaf', 'xnaf', 0),
                ('qwo', 'mgl', -1),
                ('lnh', 'rhy', -1),
                ('rimbo', 'ri', 0),
                ('fgskh', 'qspdy', -1),
                ('dwpo', 'dwpo', 0),
                ('idabfmgh', 'da', 1),
                ('mptkcmuhe', 'tkcmu', 2),
                ('hjkfysdbx', 'j', 1),
                ('jfxfz', 'poiin', -1),
                ('hqigdon', 'ycywhie', -1),
                ('qpjshvtx', 'dftrszyr', -1),
                ('fidxla', 'idx', 1),
                ('dqzbppwrwh', 'dqzb', 0),
                ('gqxgfbmxrf', 'mxr', 6),
                ('ypxiij', 'hntlfk', -1),
                ('paddthceqirlndjpojjpsodmrohzjkexocqdhpdy',
                 'lhtixjeaiybwzpgqeuujaxkwablyyzdntuevhjlj', -1),
                ('utkibcybixucoglcnjjlcoocdikaplizbgapbhity',
                 'utkibcybixucoglcnjjlcoo', 0),
                ('shpxozhkbnzhhycaqojoctjmcejskpufpehrcar', 'caqojoc', 14),
                ('mzsiahlihrabgbdtyqhcwdoramscbxysdzqanuiiswpvryscs',
                 'wwuryvxljfggbnxscimmgebsmzvpcdmlytpgfygggsxbxxazj', -1),
                ('jgwhosbkdscjwrabzhivzmqhmexepzuvomrtngaqykmuvqgrme',
                 'mgzydphujvsqgpvsomejpqyxxjdbsamsipeceiufowljbllihb', -1),
                ('qabworhuozfyqecgqvg', 'orhuozfyqecgqvg', 4),
                ('guzbhoyomkddggslvyigrzmxwqpqajxofeuznlwiijua',
                 'iwdodbzjfdvxdcstwqwlxfipryjtdfztzwagvrqdabrm', -1),
                ]


@pytest.fixture(scope="class", params=[bruteforce_substr, kmp_substr],
                ids=["brute", "kmp"])
def substr_method(request):
    request.cls.substr = staticmethod(request.param)


@pytest.mark.usefixtures("substr_method")
class TestSubstr:
    @pytest.mark.parametrize('t,s,e', substr_cases)
    def test_predefined(self, t, s, e):
        assert self.substr(s, t, 0, len(t)) == e

    @given(substring_pair())
    def test_substr_auto(self, substr_pair):
        t, s = substr_pair
        assert bruteforce_substr(s, t, 0, len(t)) == t.find(s)


@pytest.mark.parametrize('s,w', [
    ('ababaca', [0, 0, 1, 2, 3, 0, 1]),
    ('aabaaab', [0, 1, 0, 1, 2, 2, 3]),
    ('abcabcd', [0, 0, 0, 1, 2, 3, 0]),
    ('abcdabcabcdabcdab', [0, 0, 0, 0, 1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6])
])
def test_prefix(s, w):
    assert prefix(s) == w


@given(text('abcdef'))
def test_prefix_invariant(s):
    p = prefix(s)

    for i, f in enumerate(p):
        # prefix same as suffix
        assert s[:f] == s[i - f + 1:i + 1]

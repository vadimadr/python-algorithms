import string

import pytest
from hypothesis import given
from hypothesis.strategies import text

from algorithms.strings import (
    boyer_moore_substr,
    bruteforce_substr,
    kmp_substr,
    prefix,
    prefix_hash,
    robin_karp_substr,
)
from tests.utils import substring_pair

substr_cases = [
    ("aaa", "a", 0),
    ("aaa", "", 0),
    ("", "", 0),
    ("abcdef", "def", 3),
    ("computer", "muter", -1),
    ("stringmatchingmat", "ingmat", 3),
    ("videobox", "videobox", 0),
    ("uw", "uw", 0),
    ("orb", "fqh", -1),
    ("gvdq", "q", 3),
    ("xnaf", "xnaf", 0),
    ("qwo", "mgl", -1),
    ("lnh", "rhy", -1),
    ("rimbo", "ri", 0),
    ("fgskh", "qspdy", -1),
    ("dwpo", "dwpo", 0),
    ("idabfmgh", "da", 1),
    ("mptkcmuhe", "tkcmu", 2),
    ("hjkfysdbx", "j", 1),
    ("jfxfz", "poiin", -1),
    ("hqigdon", "ycywhie", -1),
    ("qpjshvtx", "dftrszyr", -1),
    ("fidxla", "idx", 1),
    ("dqzbppwrwh", "dqzb", 0),
    ("gqxgfbmxrf", "mxr", 6),
    ("ypxiij", "hntlfk", -1),
    (
        "paddthceqirlndjpojjpsodmrohzjkexocqdhpdy",
        "lhtixjeaiybwzpgqeuujaxkwablyyzdntuevhjlj",
        -1,
    ),
    ("utkibcybixucoglcnjjlcoocdikaplizbgapbhity", "utkibcybixucoglcnjjlcoo", 0),
    ("shpxozhkbnzhhycaqojoctjmcejskpufpehrcar", "caqojoc", 14),
    (
        "mzsiahlihrabgbdtyqhcwdoramscbxysdzqanuiiswpvryscs",
        "wwuryvxljfggbnxscimmgebsmzvpcdmlytpgfygggsxbxxazj",
        -1,
    ),
    (
        "jgwhosbkdscjwrabzhivzmqhmexepzuvomrtngaqykmuvqgrme",
        "mgzydphujvsqgpvsomejpqyxxjdbsamsipeceiufowljbllihb",
        -1,
    ),
    ("qabworhuozfyqecgqvg", "orhuozfyqecgqvg", 4),
    (
        "guzbhoyomkddggslvyigrzmxwqpqajxofeuznlwiijua",
        "iwdodbzjfdvxdcstwqwlxfipryjtdfztzwagvrqdabrm",
        -1,
    ),
]


@pytest.fixture(
    scope="class",
    params=[bruteforce_substr, kmp_substr, robin_karp_substr, boyer_moore_substr],
    ids=["brute", "kmp", "robin_karp", "bm"],
)
def substr_method(request):
    request.cls.substr = staticmethod(request.param)


@pytest.mark.usefixtures("substr_method")
class TestSubstr:
    @pytest.mark.parametrize("t,s,e", substr_cases)
    def test_predefined(self, t, s, e):
        assert self.substr(s, t, 0, len(t)) == e

    @given(substring_pair())
    def test_substr_auto(self, substr_pair):
        t, s = substr_pair
        assert bruteforce_substr(s, t, 0, len(t)) == t.find(s)


@pytest.mark.parametrize(
    "s,w",
    [
        ("ababaca", [0, 0, 1, 2, 3, 0, 1]),
        ("aabaaab", [0, 1, 0, 1, 2, 2, 3]),
        ("abcabcd", [0, 0, 0, 1, 2, 3, 0]),
        ("abcdabcabcdabcdab", [0, 0, 0, 0, 1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6]),
    ],
)
def test_prefix(s, w):
    assert prefix(s) == w


@given(text("abcdef"))
def test_prefix_invariant(s):
    p = prefix(s)

    for i, f in enumerate(p):
        # prefix same as suffix
        assert s[:f] == s[i - f + 1 : i + 1]


@given(text(string.ascii_lowercase + string.ascii_uppercase))
def test_prefix_hash(s):
    assert prefix_hash("", []) == []
    p = []
    h = prefix_hash(s, p)
    for i in range(len(s)):
        si = s[: i + 1]
        hi = prefix_hash(si, p)
        assert h[i] == hi[-1]

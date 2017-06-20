from unittest.case import TestCase

from algorithms.searching import (equal_range, longest_common_subsequence,
                                  longest_increasing_subsequence, lower_bound,
                                  maximum_sum_subarray, maximum_sum_subarray2,
                                  upper_bound)


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


def test_longest_increasing_subsequence():
    a0 = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    assert longest_increasing_subsequence(a0) == [0, 2, 6, 9, 11, 15]

    a1 = [3, 2, 6, 4, 5, 1]
    assert longest_increasing_subsequence(a1) == [2, 4, 5]


def test_max_subarray():
    a0 = [1, 2, 3, 4, 5]
    l0, r0, s0 = 0, len(a0), sum(a0)
    assert maximum_sum_subarray(a0) == (l0, r0, s0)

    a1 = [-1, 2, 4, -3, 5, 2, -5, 2]
    l1, r1 = 1, 6
    s1 = sum(a1[l1:r1])
    assert maximum_sum_subarray(a1) == (l1, r1, s1)

    a2 = [-1, -2, -3]
    l2, r2 = 0, 1
    s2 = sum(a2[l2:r2])
    assert maximum_sum_subarray(a2) == (l2, r2, s2)


def test_max_subarray2():
    a0 = [1, 2, 3, 4, 5]
    l0, r0, s0 = 0, len(a0), sum(a0)
    assert maximum_sum_subarray2(a0) == (l0, r0, s0)

    a1 = [-1, 2, 4, -3, 5, 2, -5, 2]
    l1, r1 = 1, 6
    s1 = sum(a1[l1:r1])
    assert maximum_sum_subarray2(a1) == (l1, r1, s1)

    a2 = [-1, -2, -3]
    l2, r2 = 0, 1
    s2 = sum(a2[l2:r2])
    assert maximum_sum_subarray2(a2) == (l2, r2, s2)


def test_longest_common_subsequence():
    a = "xmjyauz"
    b = "mzjawxu"
    assert longest_common_subsequence(a, b) == ['m', 'j', 'a', 'u']

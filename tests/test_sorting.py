from unittest import TestCase

import numpy as np

from algorithms.sorting import *
from algorithms.sorting.merge import merge_lists, merge_n_lists, merge_sort


def is_sorted(seq, start=0, end=None):
    end = end or len(seq) - 1
    for i in range(start + 1, end + 1):
        if seq[i - 1] > seq[i]:
            return False
    return True


def sorting_wrapper(fn):
    def wrap(a, *args, **kwargs):
        fn(a, 0, len(a), *args, **kwargs)
        return list(a)

    return wrap


class BaseSortTest(TestCase):

    def sort_method(a, l, r):
        a[:] = sorted(a)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sort = sorting_wrapper(self.__class__.sort_method)

    def assertSorted(self, seq):
        self.assertTrue(is_sorted(seq))

    def testAlreadySorted(self):
        xs = np.arange(16)  # 1 2 ... 15
        self.sort(xs)
        self.assertSorted(xs)

    def testReversed(self):
        xs = np.arange(16, 0, -1)  # 16 15 ... 1
        self.sort(xs)
        self.assertSorted(xs)

    def testRepeat(self):
        xs = np.full((16,), 5, dtype=int)  # 5 5 ... 5
        self.sort(xs)
        self.assertSorted(xs)

    def testSingle(self):
        xs = np.array([1])
        self.sort(xs)
        self.assertSorted(xs)

    def testRandom1(self):
        np.random.seed(123)
        xs = np.random.randint(100, size=64)
        self.sort(xs)
        self.assertSorted(xs)

    def testRandom2(self):
        np.random.seed(456)
        xs = np.random.randint(100, size=64)
        self.sort(xs)
        self.assertSorted(xs)

    def testRandom3(self):
        np.random.seed(111)
        xs = np.random.randint(100, size=64)
        self.sort(xs)
        self.assertSorted(xs)

    def testRandom4(self):
        np.random.seed(888)
        xs = np.random.randint(1000, size=256)
        self.sort(xs)
        self.assertSorted(xs)

    def testRandom5(self):
        np.random.seed(333)
        xs = np.random.rand(128)
        self.sort(xs)
        self.assertSorted(xs)


class BaseBubbleSortTest(BaseSortTest):
    sort_method = bubble_sort


class BaseQSortTest(BaseSortTest):
    sort_method = quick_sort


class BaseInsertionSortTest(BaseSortTest):
    sort_method = insertion_sort


class BaseSelectionSortTest(BaseSortTest):
    sort_method = selection_sort


class BaseHeapSortTest(BaseSortTest):
    sort_method = heap_sort


class TestMerge(TestCase):

    def testEqual(self):
        xs = [1] * 10
        ys = [1] * 10
        res = merge_lists(xs, ys)
        self.assertTrue(is_sorted(res))

    def testEqual1(self):
        xs = [1] * 10
        ys = [5] * 10
        res = merge_lists(xs, ys)
        self.assertTrue(is_sorted(res))

    def testEqual2(self):
        xs = [5] * 10
        ys = [1] * 10
        res = merge_lists(xs, ys)
        self.assertTrue(is_sorted(res))

    def testAsc(self):
        xs = list(range(1, 16))
        ys = list(range(16, 31))
        res = merge_lists(xs, ys)
        self.assertTrue(is_sorted(res))

    def testCross(self):
        xs = list(range(1, 16))
        ys = list(range(8, 23))
        res = merge_lists(xs, ys)
        self.assertTrue(is_sorted(res))

    def testDiffSize(self):
        xs = list(range(1, 20))
        ys = list(range(50, 200))
        res = merge_lists(xs, ys)
        self.assertTrue(is_sorted(res))

    def testDiffSize1(self):
        xs = list(range(50, 200))
        ys = list(range(1, 20))
        res = merge_lists(xs, ys)
        self.assertTrue(is_sorted(res))

    def testRandom1(self):
        np.random.seed(111)
        xs = sorted(np.random.randint(100, size=64))
        ys = sorted(np.random.randint(100, size=65))
        res = merge_lists(xs, ys)
        self.assertTrue(is_sorted(res))

    def testRandom2(self):
        np.random.seed(123)
        xs = sorted(np.random.randint(100, size=64))
        ys = sorted(np.random.randint(100, size=65))
        res = merge_lists(xs, ys)
        self.assertTrue(is_sorted(res))

    def testRandom3(self):
        np.random.seed(555)
        xs = sorted(np.random.randint(100, size=65))
        ys = sorted(np.random.randint(100, size=64))
        res = merge_lists(xs, ys)
        self.assertTrue(is_sorted(res))

    def testRandom4(self):
        np.random.seed(333)
        xs = sorted(np.random.randint(100, size=64))
        ys = sorted(np.random.randint(100, size=65))
        res = merge_lists(xs, ys)
        self.assertTrue(is_sorted(res))


class TestMergeN(TestCase):

    def testRandom1(self):
        np.random.seed(456)
        nlists = np.random.randint(5, 32)
        dims = [np.random.randint(5, 16) for i in range(nlists)]
        lists = [sorted(np.random.randint(0, 50, size=i)) for i in dims]
        res = merge_n_lists(lists)
        self.assertTrue(is_sorted(res))

    def testRandom2(self):
        np.random.seed(111)
        nlists = np.random.randint(5, 32)
        dims = [np.random.randint(5, 16) for i in range(nlists)]
        lists = [sorted(np.random.randint(0, 50, size=i)) for i in dims]
        res = merge_n_lists(lists)
        self.assertTrue(is_sorted(res))

    def testRandom3(self):
        np.random.seed(222)
        nlists = np.random.randint(5, 32)
        dims = [np.random.randint(5, 16) for i in range(nlists)]
        lists = [sorted(np.random.randint(0, 50, size=i)) for i in dims]
        res = merge_n_lists(lists)
        self.assertTrue(is_sorted(res))

    def testRandom4(self):
        np.random.seed(333)
        nlists = np.random.randint(5, 32)
        dims = [np.random.randint(5, 16) for i in range(nlists)]
        lists = [sorted(np.random.randint(0, 50, size=i)) for i in dims]
        res = merge_n_lists(lists)
        self.assertTrue(is_sorted(res))

    def testRandom5(self):
        np.random.seed(123)
        nlists = np.random.randint(5, 32)
        dims = [np.random.randint(5, 16) for i in range(nlists)]
        lists = [sorted(np.random.randint(0, 50, size=i)) for i in dims]
        res = merge_n_lists(lists)
        self.assertTrue(is_sorted(res))


class BaseMergeSortTest(BaseSortTest):
    sort_method = merge_sort

import numpy as np
from operator import add


class FenwickTree:
    """Binary indexed tree. Answers associative [0, r) queries on O(log n)

    Tree nodes store cumulative frequency of elements in [g(i), i],
    where g(i) is some function that satisfies 0 <= g(i) <= i

    This implementation uses g(i) that remove least significant bit:
    g(1011) = 1010, g(1010) = 1000

    Another popular choice (when we want to make 0-based indexation)
    is to use g(i) that replaces all trailing 1-bits with 0-bits:
    g(1011) = 1000, g(1101) = 1100
    """

    def __init__(self, arr, op=add, initial=0):
        self.initial = initial
        self.function = op
        self.tree = np.full(len(arr) + 1, initial)  # use 1-based indexation for simplicity

        for i in range(len(arr)):
            self.update(i, arr[i])

    def query(self, r):
        """Split input range [0, r) into non-overlapping sub-ranges with decreasing lengths:
        [0, g(g(...) - 1)), ..., [g(g(r) - 1), [g(r), r)
        """
        i = r
        result = self.initial
        while i > 0:
            result = self.function(result, self.tree[i])
            i -= i & -i  # or i = (i & (i + 1)) - 1
        return result

    def update(self, i, x):
        """Update all segments that intersect with i. i.e. update all j s.t g(j) <= i <= j"""
        i += 1
        while i < len(self.tree):
            self.tree[i] = self.function(self.tree[i], x)
            i += i & -i  # or i = (i | (i + 1))

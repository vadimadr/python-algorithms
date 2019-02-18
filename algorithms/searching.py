from math import inf


def lower_bound(a, val, lo, hi):
    """
    find insertion position for val using binary search
    lower (left) and upper (right) bounds of 3:
    1 2 3 3 3 4 5
        ^     ^
    if there is no val in [lo, hi) then lower_bound == upper_bound
    """
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo


def lower_bound2(a, val, start, end):
    """This implementation of binary search is less error-prone
    while still working in O(log n) time. It uses simple linear search
    with decreasing step size.
    """
    step = end - start
    pos = start - 1
    while step > 0:
        while pos + step < end and a[pos + step] < val:
            pos += step
        step //= 2
    return pos + 1


def upper_bound(a, val, lo, hi):
    while lo < hi:
        mid = (lo + hi) // 2
        if val < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def equal_range(a, val, start, end):
    lo = lower_bound(a, val, start, end)
    hi = upper_bound(a, val, start, end)
    return lo, hi


def longest_increasing_subsequence(a):
    # d[i] is the lowest number that increasing subsequence of length i ends
    # with.
    d = [inf] * len(a)
    p = [-1] * len(a)  # index predecessor of a[i]
    p_i = [-1] * len(a)  # index of element d[i]
    d[0] = -inf
    n_largest = 0
    for i in range(len(a)):
        k = upper_bound(d, a[i], 0, len(d))
        if d[k - 1] < a[i] < d[k]:
            d[k] = a[i]
            p_i[k] = i
            p[i] = p_i[k - 1]
            n_largest = max(n_largest, k)

    path = []
    i = p_i[n_largest]
    while i != -1:
        path.append(a[i])
        i = p[i]
    path.reverse()
    return path


def maximum_sum_subarray(a):
    """returns subarray [l, r) with maximal sum s"""
    s = 0  # prefix sum
    min_s, min_i = 0, -1  # minimum on s[0..r - 1]
    l, r, max_s = 0, 1, a[0]
    for i, e in enumerate(a):
        s += e
        # suppose i is right boundary,
        # then l - 1 is the minimum on s[0..r - 1]
        if s - min_s > max_s:
            l = min_i + 1
            r = i + 1
            max_s = s - min_s
        if s < min_s:
            min_i = i
            min_s = s

    return l, r, max_s


def maximum_sum_subarray2(a):
    """another algorithm for O(n) max sum subarray"""
    s, cur_l = 0, 0  # current sum
    l, r, max_s = 0, 1, a[0]
    for i, e in enumerate(a):
        s += e
        # better to reset l index
        if s < e:
            s = e
            cur_l = i
        if s > max_s:
            max_s = s
            l = cur_l
            r = i + 1
    return l, r, max_s


def longest_common_subsequence(a, b):
    """Returns longest common subsequence of a and b in O(n*m)
    Z is a subsequence of a if Z == a with some elements removed.

    properties of Z (optimal substructure):
    1. a[-1] == b[-1] => Z[:-1] == LCS(a[:-1], b[:-1])
    2. a[-1] != b[-1] and a[-1] != Z[-1] => Z == LCS(a[:-1], b)
    2. a[-1] != b[-1] and b[-1] != Z[-1] => Z == LCS(a, b[:-1])
    """
    n, m = len(a), len(b)
    lcs = [[0] * (m + 1) for _ in range(n + 1)]  # lcs of prefixes
    p = [[(0, 0)] * (m + 1) for _ in range(n + 1)]

    # lcs[n][m] == 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
                p[i][j] = (i - 1, j - 1)
            elif lcs[i - 1][j] > lcs[i][j - 1]:
                lcs[i][j] = lcs[i - 1][j]
                p[i][j] = (i - 1, j)
            else:
                lcs[i][j] = lcs[i][j - 1]
                p[i][j] = (i, j - 1)

    z = []
    i, j = n, m
    while i > 0 and j > 0:
        if p[i][j] == (i - 1, j - 1):
            z.append(a[i - 1])
        i, j = p[i][j]
    z.reverse()
    return z

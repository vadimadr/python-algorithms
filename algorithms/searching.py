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
    n = end - start
    while n > 0:
        mid = start + n // 2
        if a[mid] < val:
            start = mid + 1
            n = n - (n // 2) - 1
        else:
            n //= 2
    return start


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

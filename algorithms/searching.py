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

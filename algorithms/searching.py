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


def bruteforce_substr(s, t, start, end):
    n = end - start
    m = len(s)
    for i in range(n - m + 1):
        if t[i: i + m] == s:
            return i
    return -1


def prefix(s):
    """
    s = s[:i+1] - i-th prefix
    1. p[i] = max k < |s|: s[:k] == s[i-k+1:i+1]
    2. s[:p[k]] == s[...:i+1]  - prefix of prefix is suffix
    3. s[p[i]] == s[i+1] => p[i+1] = p[i] + 1
    """
    p = [0] * len(s)
    for i in range(1, len(s)):
        k = p[i - 1]
        # find max k such that s[k] == s[i]
        while k > 0 and s[k] != s[i]:
            # try lower k (use 2nd prop.)
            k = p[k - 1]
        # try to increase max prefix (3rd prop.)
        if s[k] == s[i]:
            k += 1
        p[i] = k
    return p


def kmp_substr(s, t, start, end):
    """
    compute prefix for s # t
    '#' is needed to prevent growing after s (e.g aa#a... )
    complexity: O(|s+t|)
    """
    if len(s) == 0:
        return 0
    p, k = prefix(s), 0
    for i in range(start, end):
        while k > 0 and s[k] != t[i]:
            k = p[k - 1]
        if s[k] == t[i]:
            k += 1
        if k == len(s):
            return i - k + 1
    return -1

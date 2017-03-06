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
    m = len(s)
    for i in range(start, end + 1):
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


def prefix_hash(s, p, P=53):
    """
    Compute hashes for every prefix. Only [a-zA-Z] symbols are supported

    Parameters
    -------------
    s : str
        input string
    p : List[int]
        all powers of P. p[i] = P ** i

    Returns
    ----------
    h : List[int]
        h[i] = hash(s[:i + 1])
    """
    h = [0] * len(s)
    p_ = len(p)
    s_ = len(s) - len(p)

    # increase p buf if needed
    if s_ > 0:
        p.extend([1] * s_)

    if p_ == 0:
        p_ += 1  # p**0 = 1

    for i in range(p_, s_):
        # p[-1] = 1
        p[i] = p[i - 1] * P

    for i, c in enumerate(s):
        if c.islower():
            code = ord(c) - ord('a')
        else:
            code = ord(c) - ord('A')
        h[i] = h[i - 1] + (code + 1) * p[i]
    return h


def robin_karp_substr(s, t, start, end, P=53):
    if s == '':
        return 0
    t = t[start:end]
    p = []
    ht = prefix_hash(t, p, P)
    hs = prefix_hash(s, p, P)

    n = len(t)
    m = len(s)
    for i in range(n - m + 1):
        # get h = hash(t[:i+1])*p[i]
        if i == 0:
            h = ht[i + m - 1]
        else:
            h = ht[i + m - 1] - ht[i - 1]

        # compare h and hash(s)*p[i]
        if h == (hs[-1] * p[i]):
            # check for collision
            if bruteforce_substr(s, t, i, i + m) != -1:
                return i

    return -1


def boyer_moore_substr(s, t, start, end):
    """
    Boyer - Moore substring algorithm

    Informal description:
    Align s and t. Compare s[i] and t[i] from right end. Let u = s[i + 1:] -
    characters that match t (u == t[j + i + 1:...]) and t[i + j] != s[i]
    then we must slide s by one of the rules:

    1. Match t[i + j] = c and rightmost s[k] = c
    2. If u is substring of s (except for u itself) than math u in t and
    rightmost substring of s == u. (This is the minimal shift to match u again)
    3. If no substrings found then match longest prefix of s that match its
    suffix with suffix of u in t

    """
    m = len(s)

    if m == 0:
        return 0

    stop_table = {}

    # create table for bad character heuristic
    # can give negative offsets
    for i in range(m - 1):  # skip last char
        # no s[i] in tail of s
        stop_table[s[i]] = i + 1

    # create table for good suffix heuristic
    # Boyer–Moore–Horspool (simplified version of this) does not use it.
    p = prefix(s)
    pr = prefix(''.join(reversed(s)))
    suffshift = [m - p[-1]] * m
    for i in range(m):
        j = m - pr[i] - 1
        suffshift[j] = min(i - pr[i] + 1, suffshift[j])

    # compare right to left
    # slide left to right
    j = start + m - 1
    while j < end:
        i = m - 1
        while s[i] == t[j]:
            if i == 0:
                return j
            i -= 1
            j -= 1
        # chose the best heuristics among two
        j += max(i - stop_table.get(t[j], m), suffshift[i])
    return -1

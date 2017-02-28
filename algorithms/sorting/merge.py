from .simple import insertion_sort


def merge2(a, lo, mid, hi):
    n = (hi - lo + 1)
    b = [0] * n
    q, p = lo, mid
    for i in range(n):
        # either second array is exhausted
        # or next element in the left array is lt. in one in the right
        if p > hi or q < mid and a[q] <= a[p]:
            b[i] = a[q]
            q += 1
        else:
            b[i] = a[p]
            p += 1

    a[lo:hi + 1] = b


def merge_lists(xs, ys):
    res = xs + ys
    merge2(res, 0, len(xs), len(res) - 1)
    return res


def merge_n(a, run):
    """
    merge all runs into one array
    e.g [1,2,3] + [10,20,80] + [5,7,8,9]

    :param a: array of sorted runs
    :param run: run[i] - begging of the ith run (sorted subseq)
    run[-1] == index of the next to the rightmost element in the range
    """

    # number of elements in the range
    n = run[-1] - run[0]
    # temporary array
    b = [0] * n
    # number of runs
    nrun = len(run) - 1

    # TODO: Smart temporary array creation (get rid of last copy)

    # unless all runs are merged
    while nrun > 1:
        nrun = 1
        for k in range(1, nrun, 2):
            lo, mid, hi = run[k - 1:k + 2]  # bounds
            p, q = lo, mid  # pointers to the next elements
            run[nrun] = hi
            nrun += 1
            for i in range(n):
                if p > hi or q < mid and a[q] <= a[p]:
                    b[i] = a[q]
                    q += 1
                else:
                    b[i] = a[p]
                    p += 1
        a, b = b, a
    b[:] = a[:]


MIN_MERGE = 8


def merge_sort(a, l, r):
    runs = []

    for k in range(l, r + 1, MIN_MERGE):
        insertion_sort(a, k, min(r, k + MIN_MERGE))
        runs.append(k)
    runs.append(r + 1)
    merge_n(a, runs)


def merge_n_lists(lsts):
    k = 0
    runs = []
    res = []
    for l in lsts:
        res.extend(l)
        runs.append(k)
        k += len(l)
    runs.append(k - 1)
    merge_n(res, runs)
    return res

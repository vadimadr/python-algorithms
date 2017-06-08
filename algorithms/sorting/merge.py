from .simple import insertion_sort


def merge(a, lo, mid, hi, buf):
    q, p = lo, mid
    for i in range(lo, hi):
        # either second array is exhausted
        # or next element in the left array is lt. in one in the right
        if p >= hi or q < mid and a[q] < a[p]:
            buf[i] = a[q]
            q += 1
        else:
            buf[i] = a[p]
            p += 1


def merge_inplace(a, lo, mid, hi):
    # TODO: inplace merge w/o buffer. Currently works as merge function adapter
    buf = [0] * (hi - lo)
    merge(a, lo, mid, hi, buf)
    a[lo:hi] = buf


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


def merge_lists(xs, ys):
    res = xs + ys
    merge_inplace(res, 0, len(xs), len(res))
    return res


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


MIN_MERGE = 8


def merge_sort(arr, lo, hi):
    buf = [0] * (hi - lo)
    swapped = False
    m = MIN_MERGE  # size of minimal sorted subarray
    # optional step. Also works when m = 1
    for k in range(lo, hi, MIN_MERGE):
        insertion_sort(arr, k, min(hi, k + MIN_MERGE))

    while m < len(arr):
        for k in range(lo, hi - m, 2 * m):
            merge(arr, lo, lo + m, min(lo + 2 * m, hi), buf)
        swapped = not swapped
        arr, buf = buf, arr
        m *= 2

    if swapped:
        buf[lo:hi] = arr[:]

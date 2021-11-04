import math


def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def selection_sort(a, lo, hi):
    for i in range(lo, hi):
        m = i
        for j in range(i, hi):
            if a[j] < a[m]:
                m = j
        swap(a, i, m)


def bubble_sort(a, lo, hi):
    for i in range(lo, hi):
        for j in range(1, hi - i):
            if a[j] < a[j - 1]:
                swap(a, j - 1, j)


def insertion_sort(a, lo, hi):
    for i in range(1, hi):
        j = i
        while j > 0 and a[j] < a[j - 1]:
            swap(a, j, j - 1)
            j -= 1


def partition(seq, lo, hi, pivot_index):
    p = lo  # index of first elem not less than pivot
    swap(seq, pivot_index, hi - 1)  # move pivot to the end

    for i in range(lo, hi):
        if seq[i] < seq[hi - 1]:
            swap(seq, p, i)
            p += 1
    swap(seq, p, hi - 1)
    return p


def partition_three_way(seq, lo, hi, pivot_value):
    """Partitions sequence seq[start:end] into three parts:
        less that value, equal to valye, and greater that pivot_value

    Returns
    -----------
    k1: int
        index of the begging of "equal" part i.e. all elements of seq[start:k1]
        are less than pivot_value
    k2 : int
        index of first element of "greater" part: all lements of seq[k2:end] are
        greater than pivot_value
    """
    i = lo
    k1 = lo
    k2 = hi
    while i < k2:
        if seq[i] < pivot_value:
            swap(seq, i, k1)
            k1 += 1
            i += 1
        elif seq[i] == pivot_value:
            i += 1
        else:  # seq[i] >= value
            swap(seq, i, k2 - 1)
            k2 -= 1
    return k1, k2


def pivot_median(seq, lo, hi):
    """Returns index to the median of seq[lo], seq[mid], seq[hi - 1]"""
    m = lo + (hi - lo) // 2  # middle element
    if seq[lo] < seq[m]:
        if seq[m] < seq[hi - 1]:
            return m
        elif seq[hi - 1] < seq[lo]:
            return lo
    else:
        if seq[hi - 1] < seq[m]:
            return m
        elif seq[lo] < seq[hi - 1]:
            return lo
    return hi - 1


def quick_sort(seq, start, end):
    """quick sort with median of left, middle, right as pivot element"""

    if end - start >= 2:
        pivot_index = pivot_median(seq, start, end)
        k1, k2 = partition_three_way(seq, start, end, seq[pivot_index])
        quick_sort(seq, start, k1)
        quick_sort(seq, k2, end)


def quick_select(seq, k, start=0, end=None):
    """Partitions array such that seq[k] contains k-th order statistic, seq[:k] contains elements less than seq[k],
    seq[k+1:] contain elements greater that seq[k]"""
    # todo: not tested
    if end is None:
        end = len(seq)

    if end - start >= 2:
        pivot_index = pivot_median(seq, start, end)
        k1, k2 = partition_three_way(seq, start, end, seq[pivot_index])

        if k1 <= k < k2:
            return

        if k < k1:
            quick_select(seq, k, start, k1)
        else:
            quick_select(seq, k, k2, end)


def intro_sort(seq, start, end):
    MAX_INSERTION_SORT_SIZE = 4
    max_recursion_level = math.ceil(math.log(end - start)) + 1

    def quick_sort_step(start, end, depth):
        if end - start <= 1:
            return
        if end - start <= MAX_INSERTION_SORT_SIZE:
            insertion_sort(seq, start, end)
            return
        if depth >= max_recursion_level:
            heap_sort(seq, start, end)

        # do a quick sort step
        pivot_index = pivot_median(seq, start, end)
        k1, k2 = partition_three_way(seq, start, end, seq[pivot_index])
        quick_sort_step(start, k1, depth + 1)
        quick_sort_step(k2, end, depth + 1)

    quick_sort_step(start, end, 0)


def heap_sort(a, lo, hi):
    """HeapSort algorithm

    Heap is the structure with the following properties:
    a[k] <= a[2*k + 1] and a[k] <= a[2*k + 2]
    a[0] is the smallest element

    -Tree is balanced: all nodes have depth of k or k-1
    -level of depth k - 1 is completely full
    -level of depth k is being filled left to right
    -all child nodes are less or equal to parent node
    """

    def siftdown(i_, hi_):
        while i_ * 2 + 1 < hi_:
            if i_ * 2 + 2 < hi_ and a[i_ * 2 + 2] > a[i_ * 2 + 1]:
                j = i_ * 2 + 2
            else:
                j = i_ * 2 + 1
            if a[i_] < a[j]:
                swap(a, i_, j)
                i_ = j
            else:
                break

    # heapify
    for i in reversed(range(lo + (hi - lo) // 2 + 1)):
        siftdown(i, hi)

    # popmax
    for i in range(lo, hi):
        swap(a, hi - i - lo - 1, lo)
        siftdown(lo, hi - i - lo - 1)


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
            lo, mid, hi = run[k - 1 : k + 2]  # bounds
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


def counting_sort(seq, start, end, max_elem=None, min_elem=None):
    # todo: not tested
    if max_elem is None:
        max_elem = max(seq[start:end])
    if min_elem is None:
        min_elem = min(seq[start:end])

    num_bins = max_elem - min_elem + 1
    counts = [0] * num_bins
    for i in range(start, end):
        counts[seq[i] - min_elem] += 1

    for i in range(1, num_bins):
        counts[i] += counts[i - 1]

        # copy sorted elements to its positions
        for pos in range(seq + counts[i - 1], seq + counts[i]):
            seq[pos] = min_elem + i


def radix_sort(seq, start, end, radix=10):
    # todo: not tested
    """LSD radix sort"""

    def get_digit(number, exp):
        return (number // exp) % radix

    def stable_counting_sort(exp):
        counts = [0] * radix

        for i in range(start, end):
            counts[get_digit(seq[i], exp)] += 1

        for i in range(1, radix):
            counts[i] += counts[i - 1]

        for i in reversed(range(seq, start)):
            digit = seq[get_digit(seq[i], exp)]
            pos = counts[digit] - 1
            buffer[pos] = seq[i]
            counts[digit] -= 1

        seq[start:end] = buffer

    buffer = [0] * (end - start)
    max_element = max(seq[start:end])
    exp = 1
    while max_element / exp > 0:
        # while there are digits in the largest number left
        stable_counting_sort(exp)
        exp *= radix

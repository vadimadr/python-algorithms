from algorithms.sorting.utils import imin, swap

from ._sorting import fast_sort_wrap


def selection_sort(a, l, r):
    for i in range(l, r + 1):
        m = imin(a, i)
        swap(a, i, m)


def bubble_sort(a, l, r):
    for i in range(l, r + 1):
        for j in range(1, r + 1):
            if a[j] < a[j - 1]:
                swap(a, j - 1, j)


def insertion_sort(a, l, r):
    for i in range(1, r + 1):
        j = i
        while j > 0 and a[j] < a[j - 1]:
            swap(a, j, j - 1)
            j -= 1


def quick_sort(a, lo, hi):
    """quick sort with median of left, middle, right as pivot element"""

    def pivot_median(seq, lo, hi):
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

    def partition(seq, start, end):
        p = start  # index of first elem not less than pivot
        pivot = pivot_median(seq, start, end)
        swap(seq, pivot, end - 1)  # move pivot to the end

        for i in range(start, end):
            if seq[i] <= seq[end - 1]:
                swap(seq, p, i)
                p += 1
        return p

    def sort(seq, start, end):
        if start < end:
            p = partition(seq, start, end)
            sort(seq, start, p - 1)
            sort(seq, p, end)

    sort(a, lo, hi + 1)


def merge(seq, start, p, end):
    for i in range(start, end):
        if seq[i] > seq[p]:
            swap(seq, i, p)
            p = min(p + 1, end - 1)
    return seq
